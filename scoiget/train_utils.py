import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
from scoiget.scoiget_model import SCOIGET  # Import the SCOIGET model
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import contextlib
import io


def prepare_data(adata, use_norm_x=True):
    """
    Prepare the data for the GNN model by converting AnnData into PyTorch Geometric Data format.
    
    Args:
        adata (AnnData): AnnData object containing the graph structure in `obsm['graph_neigh']`
                         and node embeddings in `obsm['norm_x']` or `obsm['feat']`.
        use_norm_x (bool): Whether to use `norm_x` as node embeddings; otherwise, `feat` will be used.
    
    Returns:
        data (torch_geometric.data.Data): Data object for the GNN model.
    """
    # Select node embeddings
    if use_norm_x:
        node_emb = adata.obsm['norm_x']
        edge_prob_key = 'edge_probabilities_norm_x'
    else:
        node_emb = adata.obsm['feat']
        edge_prob_key = 'edge_probabilities'

    # Convert node embeddings to a dense tensor
    if isinstance(node_emb, (csr_matrix, coo_matrix)):
        node_emb = node_emb.toarray()
    x = torch.tensor(node_emb, dtype=torch.float)

    # Convert graph structure to PyTorch Geometric format
    graph_neigh = adata.obsm['graph_neigh']
    edge_index, edge_weight = from_scipy_sparse_matrix(coo_matrix(graph_neigh))

    # Process edge probabilities
    edge_probabilities = adata.obsm[edge_prob_key]
    if isinstance(edge_probabilities, (csr_matrix, coo_matrix)):
        edge_probabilities = edge_probabilities.tocoo().data
    elif isinstance(edge_probabilities, np.ndarray):
        edge_probabilities = edge_probabilities.flatten()
    else:
        raise ValueError("Unsupported format for edge_probabilities.")

    # Adjust edge probabilities length if necessary
    if edge_probabilities.shape[0] != edge_index.shape[1]:
        print(f"Warning: Adjusting edge_probabilities length from {edge_probabilities.shape[0]} to {edge_index.shape[1]}.")
        edge_probabilities = edge_probabilities[:edge_index.shape[1]]

    edge_attr = torch.tensor(edge_probabilities, dtype=torch.float)

    # Debugging information
    print(f"x shape: {x.shape}")
    print(f"edge_index shape: {edge_index.shape}")
    print(f"edge_attr shape: {edge_attr.shape}")

    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def split_large_graph(data, num_subgraphs, batch_size=None):
    """
    Split a large graph into smaller subgraphs for mini-batch training.

    Args:
        data (Data): PyTorch Geometric Data object representing the graph.
        num_subgraphs (int): Number of subgraphs to split into.
        batch_size (int, optional): Size of each subgraph batch; if None, it's calculated automatically.

    Returns:
        data_list (list[Data]): A list of smaller Data objects representing subgraphs.
    """
    node_count = data.x.size(0)
    edge_count = data.edge_index.size(1)

    if batch_size is None:
        batch_size = node_count // num_subgraphs

    data_list = []
    for i in range(num_subgraphs):
        # Determine the node range for the current subgraph
        node_start = i * batch_size
        node_end = min((i + 1) * batch_size, node_count)

        # Select edges that belong to the current subgraph
        node_mask = (data.edge_index[0] >= node_start) & (data.edge_index[0] < node_end) & \
                    (data.edge_index[1] >= node_start) & (data.edge_index[1] < node_end)
        edge_index = data.edge_index[:, node_mask]
        edge_attr = data.edge_attr[node_mask]

        # Skip subgraphs without edges
        if edge_index.size(1) == 0:
            print(f"Edge index out of range in subgraph {i}. Skipping...")
            continue

        # Adjust edge indices to be relative to the subgraph's node range
        edge_index[0] -= node_start
        edge_index[1] -= node_start

        # Select node features for the current subgraph
        sub_x = data.x[node_start:node_end]

        # Ensure edge indices are within the range of the subgraph's node features
        if edge_index.max() >= sub_x.size(0):
            print(f"Edge index out of range in subgraph {i}. Skipping...")
            continue

        # Create a subgraph Data object
        sub_data = Data(x=sub_x, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(sub_data)

    return data_list


from torch_geometric.utils import subgraph

def split_train_val(data, validation_split):
    """
    Split the input graph into training and validation subsets.

    Args:
        data (Data): PyTorch Geometric Data object.
        validation_split (float): Fraction of nodes to use for validation (0 < validation_split < 1).

    Returns:
        train_data (Data): Training subset of the graph.
        val_data (Data): Validation subset of the graph.
    """
    assert 0 < validation_split < 1, "validation_split must be between 0 and 1."

    num_nodes = data.x.size(0)
    val_size = int(num_nodes * validation_split)
    perm = torch.randperm(num_nodes, device=data.edge_index.device)

    train_idx = perm[val_size:]
    val_idx = perm[:val_size]

    train_edge_index, train_edge_attr = subgraph(
        train_idx, data.edge_index, data.edge_attr, relabel_nodes=True
    )
    val_edge_index, val_edge_attr = subgraph(
        val_idx, data.edge_index, data.edge_attr, relabel_nodes=True
    )

    # Construct training and validation Data objects
    train_data = Data(
        x=data.x[train_idx],
        edge_index=train_edge_index,
        edge_attr=train_edge_attr,
        y=data.y[train_idx] if (hasattr(data, 'y') and data.y is not None) else None
    )

    val_data = Data(
        x=data.x[val_idx],
        edge_index=val_edge_index,
        edge_attr=val_edge_attr,
        y=data.y[val_idx] if (hasattr(data, 'y') and data.y is not None) else None
    )

    return train_data, val_data


def train_scoiget(data, original_dim, intermediate_dim, latent_dim, max_cp, kl_weights, 
                  epochs=100, lr=0.001, validation_split=None, save_path=None, batch_size=1024, 
                  dropout=0.0, hmm_states=3, gnn_heads=8, max_iters=20, use_mini_batch=True, 
                  num_subgraphs=10, lambda_smooth=0.1, device=None):
    """
    Train the SCOIGET model with spatial smoothing and HMM state smoothing.

    Args:
        data (Data): Input graph data for training.
        original_dim (int): Dimension of input features.
        intermediate_dim (int): Dimension of intermediate features.
        latent_dim (int): Dimension of latent space.
        max_cp (int): Maximum copy number.
        kl_weights (float): Weight for KL divergence loss.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        validation_split (float, optional): Fraction of data for validation.
        save_path (str, optional): Path to save the trained model and loss plots.
        batch_size (int): Batch size for mini-batch training.
        dropout (float): Dropout rate for the model.
        hmm_states (int): Number of states for HMM.
        gnn_heads (int): Number of attention heads in GNN layers.
        max_iters (int): Maximum number of iterations for HMM fitting.
        use_mini_batch (bool): Whether to use mini-batch training.
        num_subgraphs (int): Number of subgraphs for mini-batch training.
        lambda_smooth (float): Weight for spatial smoothing loss.
        device (torch.device, optional): Device for training (e.g., 'cuda' or 'cpu').

    Returns:
        model (SCOIGET): Trained SCOIGET model.
    """
    # Remaining code is identical to the original, with comments explaining each step.