import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from pyensembl import EnsemblRelease
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def add_genomic_locations(adata):
    """Add genomic location data to adata.var using pyensembl, and remove deprecated genes."""

    data = EnsemblRelease(98)  # Example: Human GRCh38

    # Initialize genomic location data
    adata.var['chromosome'] = "0"
    adata.var['start'] = 0
    adata.var['end'] = 0
    err_counter = 0

    # Ensure gene IDs column exists
    if 'gene_ids' not in adata.var.columns:
        raise ValueError("Gene IDs are missing in adata.var")

    # Filter valid genes (non-deprecated and non-NaN)
    valid_var_mask = (
        ~adata.var['gene_ids'].isna() & 
        ~adata.var['gene_ids'].str.startswith('DEPRECATED_')
    )

    # Remove invalid genes
    removed_genes = adata.var_names[~valid_var_mask]
    print(f"Removing {len(removed_genes)} deprecated or invalid genes.")
    adata._inplace_subset_var(valid_var_mask)

    # Iterate through valid genes and map genomic locations
    for idx in adata.var.index:
        gene_id = adata.var.at[idx, 'gene_ids']
        try:
            gene = data.gene_by_id(gene_id)
            adata.var.at[idx, "chromosome"] = f"chr{gene.contig}"
            adata.var.at[idx, "start"] = gene.start
            adata.var.at[idx, "end"] = gene.end
        except ValueError as e:
            print(f"Error mapping gene ID {gene_id}: {e}")
            err_counter += 1

    print(f"Finished adding genomic locations with {err_counter} errors.")

    # Remove any entries where chromosome is still "0" (unmapped genes)
    unmapped_genes = adata.var[adata.var['chromosome'] == "0"].index
    if len(unmapped_genes) > 0:
        print(f"Removing {len(unmapped_genes)} genes with unmapped genomic locations.")
        adata._inplace_subset_var(adata.var['chromosome'] != "0")

    return adata


def gene_binning_from_adata(adata, bin_size):
    """
    Perform gene binning for UMI counts using pre-existing genomic location data in adata.
    Args:
        adata (anndata object): Anndata object with UMI counts and genomic location data.
        bin_size (int): Number of genes per bin.

    Returns:
        adata (anndata object): Updated Anndata object with high expressed gene counts and additional information.
        chrom_list (list): List of chromosome boundary bins.
    """
    # Ensure that genomic location data exists
    if 'chromosome' not in adata.var.columns or 'start' not in adata.var.columns:
        raise ValueError("Genomic location data is missing in adata.var")
    # Normalize UMI counts
    sc.pp.filter_genes(adata, min_cells=1)
    # Filter out genes with missing chromosome or absolute position
    adata_clean = adata[:, adata.var['chromosome'].notna() & adata.var['start'].notna()].copy()
    adata_clean.var['abspos'] = adata_clean.var['start'].astype(int)
    # Sort genes by chromosome and absolute position
    adata_clean.var['chromosome'] = adata_clean.var['chromosome'].astype('category')
    adata_clean.var['chromosome'].cat.set_categories(
        [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"], 
        ordered=True, 
        inplace=True
    )
    adata_clean = adata_clean[:, list(adata_clean.var.sort_values(by=['chromosome', 'abspos']).index)].copy()
    if 'n_cells' not in adata.var.columns:
        raise ValueError("Cell counts are missing in adata.var")
    # Calculate the number of genes exceeded for each chromosome
    n_exceeded = adata_clean.var['chromosome'].value_counts() % bin_size
    ind_list = []
    for chrom in n_exceeded.index:
        n = n_exceeded[chrom]
        ind = adata_clean.var[adata_clean.var['chromosome'] == chrom].sort_values(by=['n_cells', 'abspos'])[:n].index.values
        ind_list.append(ind)
    # Remove exceeded genes
    data = adata_clean[:, ~adata_clean.var.index.isin(np.concatenate(ind_list))].copy()
    # Find chromosome boundary bins
    bin_number = adata_clean.var['chromosome'].value_counts() // bin_size
    chrom_bound = bin_number.cumsum()
    chrom_list = [(0, chrom_bound.iloc[0])]
    for i in range(1, len(chrom_bound)):
        start_p = chrom_bound.iloc[i - 1]
        end_p = chrom_bound.iloc[i]
        chrom_list.append((start_p, end_p))
    # Save results back to adata
    adata.uns['binned_data'] = data
    adata.uns['chrom_list'] = chrom_list
    return adata, chrom_list


### Automatically select the number of clusters
def perform_clustering(model, data, max_clusters=10):
    model.eval()
    with torch.no_grad():
        z_mean, z_var, z = model.z_encoder(data.x, data.edge_index)
        z = z.cpu().numpy() if z.is_cuda else z.numpy()

    # Initialize best parameters
    best_num_clusters = 2
    best_score = -1
    best_labels = None

    # Iterate through different cluster numbers
    for num_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(z)
        labels = kmeans.labels_
        score = silhouette_score(z, labels, metric="euclidean")
        if score > best_score:
            best_score = score
            best_num_clusters = num_clusters
            best_labels = labels

    return best_labels, z, best_num_clusters



def auto_corr(cluster_data):
    """
    Calculate the autocorrelation of cluster data using vectorized operations.
    
    Args:
        cluster_data (np.ndarray): Data array with shape (n_samples, n_features).
        
    Returns:
        autocorrelation: Autocorrelation value.
    """
    # Compute the dot product matrix (n_samples x n_samples)
    dot_product = np.dot(cluster_data, cluster_data.T)  # Matrix multiplication

    # Compute the average of the sum
    n_samples = cluster_data.shape[0]
    res = np.sum(dot_product) / (n_samples * (n_samples + 1) / 2)

    # Compute the average variance by row
    var_mean = np.var(cluster_data, axis=1).mean()

    # Compute autocorrelation
    autocorrelation = res / var_mean
    return autocorrelation


def find_normal_cluster(x_bin, pred_label, n_clusters=2):
    # Initialize an array to store autocorrelation values
    cluster_auto_corr = np.full(n_clusters, -np.inf)  # Initialize to a very small value
    for i in range(n_clusters):
        mask = (pred_label == i)
        cluster_data = x_bin[mask, :]  # Select data belonging to the current cluster
        if cluster_data.shape[0] == 0:  # If the current cluster is empty, skip
            continue
        cluster_mean = cluster_data.mean(axis=0)
        cluster_auto_corr[i] = auto_corr(cluster_data - cluster_mean)  # Autocorrelation calculation
    normal_index = np.argmax(cluster_auto_corr)  # Find the cluster with the highest autocorrelation
    return cluster_auto_corr, normal_index


def compute_pseudo_copy(x_bin, norm_mask):
    confident_norm_x = x_bin[norm_mask]
    print(np.shape(confident_norm_x))
    baseline = np.median(confident_norm_x, axis=0)
    baseline[baseline == 0] = 1
    pseudo_cp = x_bin / baseline * 2
    pseudo_cp[norm_mask] = 2.
    pseudo_cp = torch.tensor(pseudo_cp, dtype=torch.float)
    return pseudo_cp


### GPU acceleration

import cupy as cp
import torch

def perform_clustering_gpu(features, max_clusters=10, block_size=1000):
    """
    GPU-accelerated automatic clustering function that directly accepts feature tensors as input.
    
    Args:
    - features: Feature tensor for clustering, either a NumPy array or a Torch tensor.
    - max_clusters: Maximum number of clusters.
    - block_size: Block size for GPU batch computation.
    """
    # If it's a Torch tensor, convert it to a NumPy array
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    # Convert to a GPU array
    features_gpu = cp.asarray(features)

    best_num_clusters = 2
    best_score = -1
    best_labels = None

    # GPU implementation of KMeans
    for num_clusters in range(2, max_clusters + 1):
        labels = kmeans_gpu(features_gpu, num_clusters, block_size=block_size)
        score = silhouette_score_gpu(features_gpu, labels, block_size=block_size)
        if score > best_score:
            best_score = score
            best_num_clusters = num_clusters
            best_labels = labels

    return best_labels.get(), features, best_num_clusters


def find_normal_cluster_gpu(x_bin, pred_label, n_clusters=2):
    x_bin_gpu = cp.asarray(x_bin) if isinstance(x_bin, torch.Tensor) else x_bin
    pred_label_gpu = cp.asarray(pred_label)

    cluster_auto_corr = cp.full(n_clusters, -cp.inf)

    for i in range(n_clusters):
        mask = pred_label_gpu == i
        cluster_data = x_bin_gpu[mask, :]
        if cluster_data.shape[0] == 0:
            continue
        cluster_mean = cluster_data.mean(axis=0)
        cluster_auto_corr[i] = auto_corr_gpu(cluster_data - cluster_mean)

    normal_index = cp.argmax(cluster_auto_corr).get()
    return cluster_auto_corr.get(), normal_index

def auto_corr_gpu(x):
    cov_matrix = cp.cov(x, rowvar=False)
    diag = cp.diag(cov_matrix)
    return cp.mean(diag)

def kmeans_gpu(z, num_clusters, max_iter=100, tol=1e-4, block_size=1000):
    z_gpu = cp.asarray(z)

    # Initialize cluster centers
    indices = cp.random.choice(z_gpu.shape[0], num_clusters, replace=False)
    centers = z_gpu[indices]

    for _ in range(max_iter):
        # Compute distances in blocks
        labels = cp.zeros(z_gpu.shape[0], dtype=cp.int32)
        for i in range(0, z_gpu.shape[0], block_size):
            block = slice(i, min(i + block_size, z_gpu.shape[0]))
            distances = cp.linalg.norm(z_gpu[block, None, :] - centers[None, :, :], axis=2)
            labels[block] = cp.argmin(distances, axis=1)

        # Compute new centers
        new_centers = cp.array([z_gpu[labels == i].mean(axis=0) for i in range(num_clusters)])

        # Check for convergence
        if cp.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers

    return labels

def silhouette_score_gpu(z, labels, block_size=100):
    """
    GPU-optimized silhouette score computation to avoid memory overflow.
    """
    z_gpu = cp.asarray(z)
    labels_gpu = cp.asarray(labels)

    unique_labels = cp.unique(labels_gpu)
    n_samples = z_gpu.shape[0]

    scores = cp.zeros(n_samples)

    for i in range(n_samples):
        # Distance between the current point and other points
        current_point = z_gpu[i]
        distances = cp.linalg.norm(z_gpu - current_point, axis=1)
        
        # Compute a (intra-cluster distance)
        current_label = labels_gpu[i]
        intra_cluster_mask = labels_gpu == current_label
        if cp.sum(intra_cluster_mask) > 1:
            a = cp.mean(distances[intra_cluster_mask])
        else:
            a = 0

        # Compute b (nearest-cluster distance)
        inter_cluster_distances = []
        for label in unique_labels:
            if label != current_label:
                inter_cluster_mask = labels_gpu == label
                inter_cluster_distances.append(cp.mean(distances[inter_cluster_mask]))
        if inter_cluster_distances:
            b = cp.min(cp.array(inter_cluster_distances))
        else:
            b = 0

        # silhouette score
        scores[i] = (b - a) / cp.maximum(a, b)

    return cp.mean(scores)
