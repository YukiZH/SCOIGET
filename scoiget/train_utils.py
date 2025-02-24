import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
from scoiget.scoiget_model import SCOIGET  # 导入SCOIGET模型
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import warnings
import contextlib
import io


def prepare_data(adata, use_norm_x=True):
    """
    Prepare data for GNN model with dense tensors to optimize memory usage for large-scale data.
    
    Args:
        adata (AnnData): AnnData object containing the graph structure in `obsm['graph_neigh']`
                         and node embeddings in `obsm['norm_x']` or `obsm['feat']`.
        use_norm_x (bool): If True, use `norm_x` for node embeddings; otherwise, use `feat`.
    
    Returns:
        data (torch_geometric.data.Data): Data object for GNN model.
    """
    # Choose node embeddings
    if use_norm_x:
        node_emb = adata.obsm['norm_x']
        edge_prob_key = 'edge_probabilities_norm_x'
    else:
        node_emb = adata.obsm['feat']
        edge_prob_key = 'edge_probabilities'

    # Convert node embeddings to dense tensor
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

    # Adjust edge probabilities length
    if edge_probabilities.shape[0] != edge_index.shape[1]:
        print(f"Warning: Adjusting edge_probabilities length from {edge_probabilities.shape[0]} to {edge_index.shape[1]}.")
        edge_probabilities = edge_probabilities[:edge_index.shape[1]]

    edge_attr = torch.tensor(edge_probabilities, dtype=torch.float)

    # Debugging information
    print(f"x shape: {x.shape}")
    print(f"edge_index shape: {edge_index.shape}")
    print(f"edge_attr shape: {edge_attr.shape}")

    # Create data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def split_large_graph(data, num_subgraphs, batch_size=None):
    """
    Split a large graph into smaller subgraphs for mini-batch training.

    Args:
        data (Data): A PyTorch Geometric Data object.
        num_subgraphs (int): The number of subgraphs to split into.
        batch_size (int, optional): If provided, determines the size of each subgraph batch.

    Returns:
        data_list (list[Data]): A list of smaller Data objects.
    """
    node_count = data.x.size(0)
    edge_count = data.edge_index.size(1)

    if batch_size is None:
        batch_size = node_count // num_subgraphs

    data_list = []
    for i in range(num_subgraphs):
        # 当前子图的节点范围
        node_start = i * batch_size
        node_end = min((i + 1) * batch_size, node_count)

        # 筛选属于当前子图的边
        node_mask = (data.edge_index[0] >= node_start) & (data.edge_index[0] < node_end) & \
                    (data.edge_index[1] >= node_start) & (data.edge_index[1] < node_end)
        edge_index = data.edge_index[:, node_mask]
        edge_attr = data.edge_attr[node_mask]

        # 如果没有边，则跳过当前子图
        if edge_index.size(1) == 0:
            print(f"Edge index out of range in subgraph {i}. Skipping...")
            continue

        # 调整边的索引，使其相对于子图的节点范围
        edge_index[0] -= node_start
        edge_index[1] -= node_start

        # 当前子图的节点特征
        sub_x = data.x[node_start:node_end]

        # 确保边索引和节点特征范围匹配
        if edge_index.max() >= sub_x.size(0):
            print(f"Edge index out of range in subgraph {i}. Skipping...")
            continue

        # 创建子图 Data 对象
        sub_data = Data(x=sub_x, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(sub_data)

    return data_list


from torch_geometric.utils import subgraph

def split_train_val(data, validation_split):
    """
    Split the input graph data into training and validation subsets.

    Args:
        data (Data): The input PyTorch Geometric Data object.
        validation_split (float): Fraction of nodes to use for validation (0 < validation_split < 1).

    Returns:
        train_data (Data): Training subset.
        val_data (Data): Validation subset.
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

    # 构建训练和验证数据
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
    Train the SCOIGET model with spatial smoothing and HMM states smoothing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型和优化器
    model = SCOIGET(original_dim, intermediate_dim, latent_dim, max_cp, kl_weights, 
                    dropout=dropout, hmm_states=hmm_states, max_iters=max_iters, 
                    gnn_heads=gnn_heads, lambda_smooth=lambda_smooth, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 根据data生成子图列表
    if use_mini_batch:
        # 使用子图进行 mini-batch 训练
        print("Using mini-batch training...")
        data_list = split_large_graph(data, num_subgraphs=num_subgraphs, batch_size=batch_size)
        valid_subgraphs = [sub_data for sub_data in data_list if sub_data.edge_index.size(1) > 0]

        print(f"Generated {len(valid_subgraphs)} valid subgraphs:")
        for i, sub_data in enumerate(valid_subgraphs):
            print(f"  Subgraph {i}:")
            print(f"    Nodes: {sub_data.x.size(0)}")
            print(f"    Edges: {sub_data.edge_index.size(1)}")

        if len(valid_subgraphs) == 0:
            raise ValueError("No valid subgraphs generated. Please check your graph data.")
        loader = DataLoader(valid_subgraphs, batch_size=1, shuffle=True)
    else:
        # 使用整张图进行训练
        print("Using full-graph training...")
        loader = [data]

    train_losses = []
    val_losses = []
    warnings.filterwarnings("ignore", category=UserWarning)

    # 如果您有验证集，则需要从data中分割出验证集数据（validation_split）
    # 假设您已经有相应的逻辑分割data为train_data, val_data
    # 这里只是给出结构化的示例
    train_data = data
    val_data = None
    if validation_split is not None:
        # 请根据您的数据情况进行数据拆分，这里只示意
        train_data, val_data = split_train_val(data, validation_split)
    
    with tqdm(total=epochs, desc="Training Progress", unit="epoch") as pbar:
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for batch in loader:
                if use_mini_batch:
                    batch = batch.to(device)
                else:
                    batch = data.to(device)

                # 使用model中定义的train_step进行训练
                loss = model.train_step(batch.x, batch.edge_index, batch.edge_attr, optimizer)
                total_loss += loss

            avg_loss = total_loss / len(loader)
            train_losses.append(avg_loss)

            # 可选：验证集
            if val_data is not None:
                model.eval()
                with torch.no_grad():
                    val_loss = model.validation_step(val_data.x.to(device), 
                                                     val_data.edge_index.to(device), 
                                                     val_data.edge_attr.to(device))
                    val_losses.append(val_loss)
                    pbar.set_postfix({"Training Loss": avg_loss, "Validation Loss": val_loss})
            else:
                pbar.set_postfix({"Training Loss": avg_loss})

            if (epoch + 1) % 10 == 0:
                if val_data is not None:
                    print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")

            pbar.update(1)

    # 保存模型和绘制损失曲线
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, "scoiget_model.pth"))
        print(f"Model saved to {os.path.join(save_path, 'scoiget_model.pth')}")
    
    plt.figure(figsize=(8, 6), constrained_layout=True)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", color="blue")
    if val_data is not None:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", color="red")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_file = os.path.join(save_path, "training_loss_curve.png") if save_path else "training_loss_curve.png"
    plt.savefig(save_file)
    plt.show()
    
    return model
