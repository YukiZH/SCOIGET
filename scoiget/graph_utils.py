import numpy as np
import torch
import ot
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.special import softmax


def get_x_bin_data(adata, bin_size):
    """
    Perform binning of gene expression data from an AnnData object and add the binned data to `adata.uns['binned_data']`.
    
    Args:
        adata (AnnData): AnnData object containing gene expression data.
        bin_size (int): Number of genes per bin.
    
    Returns:
        AnnData: Updated AnnData object with binned data added to `adata.uns['binned_data'].obsm`.
    """
    # Extract data from AnnData object
    binned_data = adata.uns['binned_data']
    try:
        x = binned_data.X.todense()
    except AttributeError:
        x = binned_data.X
    
    # Ensure the number of columns is a multiple of bin_size
    n_vars = x.shape[1]
    remainder = n_vars % bin_size
    if remainder != 0:
        x = np.pad(x, ((0, 0), (0, bin_size - remainder)), mode='constant', constant_values=0)
    
    # Perform binning and calculate mean values
    x_bin = x.reshape(-1, bin_size).mean(axis=1).reshape(x.shape[0], -1)
    
    # Add binned data to `obsm` of binned_data
    binned_data.obsm['X_bin'] = x_bin
    return adata


def get_x_bin_data_torch(adata, bin_size, batch_size=1000):
    """
    Perform binning of gene expression data using GPU acceleration with PyTorch.
    
    Args:
        adata (AnnData): AnnData object containing gene expression data.
        bin_size (int): Number of genes per bin.
        batch_size (int): Number of rows to process in each batch.
    
    Returns:
        AnnData: Updated AnnData object with binned data added to `adata.uns['binned_data'].obsm`.
    """
    binned_data = adata.uns['binned_data']
    x = binned_data.X

    if not issparse(x):
        x = csr_matrix(x)

    n_rows, n_vars = x.shape
    remainder = n_vars % bin_size
    if remainder != 0:
        padding = csr_matrix((n_rows, bin_size - remainder))
        x = csr_matrix(np.hstack([x.toarray(), padding.toarray()]))

    n_bins = x.shape[1] // bin_size
    x_bin = torch.zeros((n_rows, n_bins), device='cuda')  # Allocate GPU tensor

    # Process data in batches
    for i in range(0, n_rows, batch_size):
        batch_x = torch.tensor(x[i:i + batch_size].toarray(), device='cuda')

        # Compute mean for each bin
        for j in range(n_bins):
            start_idx = j * bin_size
            end_idx = (j + 1) * bin_size
            x_bin[i:i + batch_size, j] = batch_x[:, start_idx:end_idx].mean(dim=1)

    # Move binned data back to CPU and convert to sparse matrix
    x_bin_cpu = x_bin.cpu().numpy()
    binned_data.obsm['X_bin'] = csr_matrix(x_bin_cpu)
    return adata


def construct_spatial_knn_graph(adata, n_neighbors):
    """
    Construct a k-nearest neighbors graph based on spatial coordinates.
    
    Args:
        adata (AnnData): AnnData object containing spatial coordinates in `obsm['spatial']`.
        n_neighbors (int): Number of nearest neighbors to consider for each spot.
    
    Effect:
        Updates `adata.obsm` with the adjacency matrix under the key 'graph_neigh'.
    """
    # Retrieve spatial coordinates
    positions = adata.obsm['spatial']  

    # Use NearestNeighbors to find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(positions)
    distances, indices = nbrs.kneighbors(positions)
    
    # Initialize sparse adjacency matrix
    n_spots = positions.shape[0]
    rows = np.repeat(np.arange(n_spots), n_neighbors)
    cols = indices[:, 1:].flatten()  # Exclude self-neighbors
    data = np.ones(rows.shape[0])

    # Create sparse adjacency matrix
    adjacency_matrix = csr_matrix((data, (rows, cols)), shape=(n_spots, n_spots))

    # Symmetrize adjacency matrix
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    adjacency_matrix[adjacency_matrix > 1] = 1  # Ensure binary adjacency

    # Store adjacency matrix in AnnData object
    adata.obsm['graph_neigh'] = adjacency_matrix


def compute_edge_weights_and_probabilities(adata, use_norm_x=True, n_neighbors=5):
    """
    Compute edge weights and probabilities for a graph using k-nearest neighbors and PCA.
    
    Args:
        adata (AnnData): AnnData object containing node embeddings in `obsm['norm_x']` or `obsm['feat']`.
        use_norm_x (bool): Whether to use `norm_x` for node embeddings; otherwise, `feat` is used.
        n_neighbors (int): Number of nearest neighbors to consider for each node.
    
    Effect:
        Updates `adata.obsm` with edge weights and probabilities in sparse format.
    """
    # Retrieve and preprocess node embeddings
    node_emb = adata.obsm['norm_x'] if use_norm_x else adata.obsm['feat']
    scaler = MaxAbsScaler()

    # Check if embeddings are sparse, and convert to dense if necessary
    if issparse(node_emb):
        embedding = scaler.fit_transform(node_emb.toarray())
    else:
        embedding = scaler.fit_transform(node_emb)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=32, random_state=42)
    embedding = pca.fit_transform(embedding)

    # Perform k-NN search using NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    # Initialize sparse matrices for edge weights and probabilities
    n_spots = embedding.shape[0]
    edge_weights = lil_matrix((n_spots, n_spots), dtype=float)
    edge_probabilities = lil_matrix((n_spots, n_spots), dtype=float)

    # Populate edge weights matrix
    for i in range(n_spots):
        neighbors = indices[i, 1:]  # Exclude self
        dist = distances[i, 1:]
        edge_weights[i, neighbors] = dist

    # Compute softmax probabilities based on edge weights in graph_neigh
    graph_neigh = adata.obsm['graph_neigh']
    for i in range(n_spots):
        neighbors = graph_neigh[i].nonzero()[1]  # Use edges in graph_neigh
        if len(neighbors) > 0:
            non_zero_weights = edge_weights[i, neighbors].toarray().flatten()
            softmax_weights = softmax(non_zero_weights)
            edge_probabilities[i, neighbors] = softmax_weights

    # Store edge weights and probabilities in AnnData object
    adata.obsm['edge_weights_norm_x' if use_norm_x else 'edge_weights'] = edge_weights.tocsr()
    adata.obsm['edge_probabilities_norm_x' if use_norm_x else 'edge_probabilities'] = edge_probabilities.tocsr()