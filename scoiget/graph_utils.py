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
        AnnData: The updated AnnData object with binned data added to `adata.uns['binned_data'].obsm`.
    """
    # Extract data from adata
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
    # Perform binning and compute the mean
    x_bin = x.reshape(-1, bin_size).copy().mean(axis=1).reshape(x.shape[0], -1)
    # Add x_bin to binned_data's obsm
    binned_data.obsm['X_bin'] = x_bin
    return adata


import torch
from scipy.sparse import issparse, csr_matrix

def get_x_bin_data_torch(adata, bin_size, batch_size=1000):
    """
    Perform binning of gene expression data from an AnnData object using GPU acceleration with PyTorch.
    Args:
        adata (AnnData): AnnData object containing gene expression data.
        bin_size (int): Number of genes per bin.
        batch_size (int): Number of rows to process in each batch.
    Returns:
        AnnData: The updated AnnData object with binned data added to `adata.uns['binned_data'].obsm`.
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
    # Create a GPU tensor to store the result
    x_bin = torch.zeros((n_rows, n_bins), device='cuda')

    # Process data in batches
    for i in range(0, n_rows, batch_size):
        batch_x = torch.tensor(x[i:i + batch_size].toarray(), device='cuda')

        # Compute the mean for each bin
        for j in range(n_bins):
            start_idx = j * bin_size
            end_idx = (j + 1) * bin_size
            x_bin[i:i + batch_size, j] = batch_x[:, start_idx:end_idx].mean(dim=1)

    # Move x_bin back to CPU and convert to a sparse matrix
    x_bin_cpu = x_bin.cpu().numpy()
    binned_data.obsm['X_bin'] = csr_matrix(x_bin_cpu)
    return adata


from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def construct_spatial_knn_graph(adata, n_neighbors):
    """
    Construct an interaction graph based on spatial coordinates using k-nearest neighbors.
    This optimized function avoids computing a full distance matrix.
    
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
    
    # Initialize sparse matrix for adjacency
    n_spots = positions.shape[0]
    rows = np.repeat(np.arange(n_spots), n_neighbors)
    cols = indices[:, 1:].flatten()  # Exclude self-neighbors
    data = np.ones(rows.shape[0])

    # Create a sparse adjacency matrix
    adjacency_matrix = csr_matrix((data, (rows, cols)), shape=(n_spots, n_spots))

    # Symmetrize the adjacency matrix
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    adjacency_matrix[adjacency_matrix > 1] = 1  # Ensure binary adjacency

    # Store the adjacency matrix in the AnnData object
    adata.obsm['graph_neigh'] = adjacency_matrix


from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix, issparse
import numpy as np
from scipy.special import softmax

def compute_edge_weights_and_probabilities(adata, use_norm_x=True, n_neighbors=5):
    """
    Compute edge weights and probabilities for a given graph using sklearn's NearestNeighbors.
    
    Args:
        adata (AnnData): AnnData object containing node embeddings in `obsm['norm_x']` or `obsm['feat']`.
        use_norm_x (bool): If True, use `norm_x` for node embeddings; otherwise, use `feat`.
        n_neighbors (int): Number of nearest neighbors to consider for each node.
    
    Effect:
        Updates `adata.obsm` with edge weights and probabilities in sparse format.
    """
    # Retrieve node embeddings and perform standardization and PCA
    node_emb = adata.obsm['norm_x'] if use_norm_x else adata.obsm['feat']
    scaler = MaxAbsScaler()
    
    # Check if node_emb is a sparse matrix, and convert it to a dense matrix if necessary
    if issparse(node_emb):
        embedding = scaler.fit_transform(node_emb.toarray())
    else:
        embedding = scaler.fit_transform(node_emb)
    
    # Use PCA for dimensionality reduction
    pca = PCA(n_components=32, random_state=42)
    embedding = pca.fit_transform(embedding)

    # Use sklearn's NearestNeighbors for k-NN search
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    # Initialize sparse matrices for edge weights and probabilities
    n_spots = embedding.shape[0]
    edge_weights = lil_matrix((n_spots, n_spots), dtype=float)
    edge_probabilities = lil_matrix((n_spots, n_spots), dtype=float)

    # Fill the edge_weights matrix
    for i in range(n_spots):
        neighbors = indices[i, 1:]  # Exclude self
        dist = distances[i, 1:]
        edge_weights[i, neighbors] = dist

    # Use edges from graph_neigh to compute softmax probabilities
    graph_neigh = adata.obsm['graph_neigh']
    for i in range(n_spots):
        neighbors = graph_neigh[i].nonzero()[1]  # Only use edges from graph_neigh
        if len(neighbors) > 0:
            non_zero_weights = edge_weights[i, neighbors].toarray().flatten()
            softmax_weights = softmax(non_zero_weights)
            edge_probabilities[i, neighbors] = softmax_weights

    # Store edge weights and probabilities in the AnnData object, and convert to csr format
    adata.obsm['edge_weights_norm_x' if use_norm_x else 'edge_weights'] = edge_weights.tocsr()
    adata.obsm['edge_probabilities_norm_x' if use_norm_x else 'edge_probabilities'] = edge_probabilities.tocsr()
