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
    """
    Add genomic location data to `adata.var` using the pyensembl library and filter out deprecated genes.
    
    Parameters:
    - adata: AnnData object with gene IDs in `adata.var['gene_ids']`.
    
    Returns:
    - Updated AnnData object with genomic location information added to `adata.var`.
    """
    data = EnsemblRelease(98)  # Example: Human GRCh38

    # Initialize genomic location columns
    adata.var['chromosome'] = "0"
    adata.var['start'] = 0
    adata.var['end'] = 0
    err_counter = 0

    # Check if gene IDs exist
    if 'gene_ids' not in adata.var.columns:
        raise ValueError("Gene IDs are missing in `adata.var`.")

    # Filter out invalid or deprecated genes
    valid_var_mask = (
        ~adata.var['gene_ids'].isna() & 
        ~adata.var['gene_ids'].str.startswith('DEPRECATED_')
    )
    removed_genes = adata.var_names[~valid_var_mask]
    print(f"Removing {len(removed_genes)} deprecated or invalid genes.")
    adata._inplace_subset_var(valid_var_mask)

    # Map genomic locations for valid genes
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

    # Remove genes with unmapped genomic locations
    unmapped_genes = adata.var[adata.var['chromosome'] == "0"].index
    if len(unmapped_genes) > 0:
        print(f"Removing {len(unmapped_genes)} genes with unmapped genomic locations.")
        adata._inplace_subset_var(adata.var['chromosome'] != "0")

    return adata


def gene_binning_from_adata(adata, bin_size):
    """
    Perform gene binning based on genomic locations in `adata`.

    Parameters:
    - adata: AnnData object with UMI counts and genomic location data in `adata.var`.
    - bin_size: Number of genes per bin.

    Returns:
    - Updated AnnData object with binned data saved in `adata.uns['binned_data']`.
    - List of chromosome boundary bins (`chrom_list`).
    """
    # Ensure genomic location data is present
    if 'chromosome' not in adata.var.columns or 'start' not in adata.var.columns:
        raise ValueError("Genomic location data is missing in `adata.var`.")

    # Normalize UMI counts and filter out invalid genes
    sc.pp.filter_genes(adata, min_cells=1)
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

    # Remove excess genes per chromosome to fit bin sizes
    n_exceeded = adata_clean.var['chromosome'].value_counts() % bin_size
    ind_list = []
    for chrom in n_exceeded.index:
        n = n_exceeded[chrom]
        ind = adata_clean.var[adata_clean.var['chromosome'] == chrom].sort_values(by=['n_cells', 'abspos'])[:n].index.values
        ind_list.append(ind)
    data = adata_clean[:, ~adata_clean.var.index.isin(np.concatenate(ind_list))].copy()

    # Compute chromosome boundary bins
    bin_number = adata_clean.var['chromosome'].value_counts() // bin_size
    chrom_bound = bin_number.cumsum()
    chrom_list = [(0, chrom_bound.iloc[0])]
    for i in range(1, len(chrom_bound)):
        chrom_list.append((chrom_bound.iloc[i - 1], chrom_bound.iloc[i]))

    # Save results
    adata.uns['binned_data'] = data
    adata.uns['chrom_list'] = chrom_list
    return adata, chrom_list


def perform_clustering(model, data, max_clusters=10):
    """
    Automatically determine the optimal number of clusters based on silhouette score.

    Parameters:
    - model: Model with a latent encoder (`z_encoder`).
    - data: Input data with `data.x` and `data.edge_index`.
    - max_clusters: Maximum number of clusters to evaluate.

    Returns:
    - Optimal clustering labels, latent features (`z`), and best number of clusters.
    """
    model.eval()
    with torch.no_grad():
        z_mean, z_var, z = model.z_encoder(data.x, data.edge_index)
        z = z.cpu().numpy() if z.is_cuda else z.numpy()

    best_num_clusters = 2
    best_score = -1
    best_labels = None

    # Iterate through possible cluster numbers
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
    Calculate autocorrelation for cluster data.

    Parameters:
    - cluster_data: 2D NumPy array of shape (n_samples, n_features).

    Returns:
    - Autocorrelation value.
    """
    dot_product = np.dot(cluster_data, cluster_data.T)  # Inner product matrix
    n_samples = cluster_data.shape[0]
    res = np.sum(dot_product) / (n_samples * (n_samples + 1) / 2)
    var_mean = np.var(cluster_data, axis=1).mean()
    autocorrelation = res / var_mean
    return autocorrelation


def find_normal_cluster(x_bin, pred_label, n_clusters=2):
    """
    Identify the cluster with the highest autocorrelation.

    Parameters:
    - x_bin: Gene expression data.
    - pred_label: Predicted cluster labels.
    - n_clusters: Total number of clusters.

    Returns:
    - Array of autocorrelation values and the index of the "normal" cluster.
    """
    cluster_auto_corr = np.full(n_clusters, -np.inf)
    for i in range(n_clusters):
        mask = (pred_label == i)
        cluster_data = x_bin[mask, :]
        if cluster_data.shape[0] == 0:
            continue
        cluster_mean = cluster_data.mean(axis=0)
        cluster_auto_corr[i] = auto_corr(cluster_data - cluster_mean)
    normal_index = np.argmax(cluster_auto_corr)
    return cluster_auto_corr, normal_index


def compute_pseudo_copy(x_bin, norm_mask):
    """
    Calculate pseudo copy number variations.

    Parameters:
    - x_bin: Gene expression data.
    - norm_mask: Mask for normal cells.

    Returns:
    - Pseudo copy number data as a Torch tensor.
    """
    confident_norm_x = x_bin[norm_mask]
    baseline = np.median(confident_norm_x, axis=0)
    baseline[baseline == 0] = 1
    pseudo_cp = x_bin / baseline * 2
    pseudo_cp[norm_mask] = 2.0
    pseudo_cp = torch.tensor(pseudo_cp, dtype=torch.float)
    return pseudo_cp