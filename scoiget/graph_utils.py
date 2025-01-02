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
    # 从adata中提取数据
    binned_data = adata.uns['binned_data']
    try:
        x = binned_data.X.todense()
    except AttributeError:
        x = binned_data.X
    # 确保列数是bin_size的整数倍
    n_vars = x.shape[1]
    remainder = n_vars % bin_size
    if remainder != 0:
        x = np.pad(x, ((0, 0), (0, bin_size - remainder)), mode='constant', constant_values=0)
    # 执行分箱和均值计算
    x_bin = x.reshape(-1, bin_size).copy().mean(axis=1).reshape(x.shape[0], -1)
    # 将x_bin添加到binned_data的obsm中
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
    # 创建一个存储结果的GPU张量
    x_bin = torch.zeros((n_rows, n_bins), device='cuda')

    # 分批次处理数据
    for i in range(0, n_rows, batch_size):
        batch_x = torch.tensor(x[i:i + batch_size].toarray(), device='cuda')

        # 计算每个bin的均值
        for j in range(n_bins):
            start_idx = j * bin_size
            end_idx = (j + 1) * bin_size
            x_bin[i:i + batch_size, j] = batch_x[:, start_idx:end_idx].mean(dim=1)

    # 将x_bin移动回CPU并转换为稀疏矩阵
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
    
    # 检查 node_emb 是否为稀疏矩阵，若是则转换为密集矩阵
    if issparse(node_emb):
        embedding = scaler.fit_transform(node_emb.toarray())
    else:
        embedding = scaler.fit_transform(node_emb)
    
    # 使用 PCA 降维
    pca = PCA(n_components=32, random_state=42)
    embedding = pca.fit_transform(embedding)

    # 使用 sklearn 的 NearestNeighbors 进行 k-NN 搜索
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    # 初始化稀疏矩阵用于边权重和边概率
    n_spots = embedding.shape[0]
    edge_weights = lil_matrix((n_spots, n_spots), dtype=float)
    edge_probabilities = lil_matrix((n_spots, n_spots), dtype=float)

    # 填充 edge_weights 矩阵
    for i in range(n_spots):
        neighbors = indices[i, 1:]  # 排除自己
        dist = distances[i, 1:]
        edge_weights[i, neighbors] = dist

    # 使用 graph_neigh 中的边来计算 softmax 概率
    graph_neigh = adata.obsm['graph_neigh']
    for i in range(n_spots):
        neighbors = graph_neigh[i].nonzero()[1]  # 仅使用 graph_neigh 中的边
        if len(neighbors) > 0:
            non_zero_weights = edge_weights[i, neighbors].toarray().flatten()
            softmax_weights = softmax(non_zero_weights)
            edge_probabilities[i, neighbors] = softmax_weights

    # 将边权重和概率存储在 AnnData 对象中，并转换为 csr 格式
    adata.obsm['edge_weights_norm_x' if use_norm_x else 'edge_weights'] = edge_weights.tocsr()
    adata.obsm['edge_probabilities_norm_x' if use_norm_x else 'edge_probabilities'] = edge_probabilities.tocsr()