import os
import anndata
import pandas as pd
import numpy as np
import scanpy as sc
#import squidpy as sq
from skimage.io import imread
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import geopandas as gpd

from pathlib import Path
import scanpy as sc
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from skimage.io import imread
import json
import scipy


def generate_adata(sample_root, library_id, method="visium"):
    """
    Load spatial transcriptomics data using the specified method (e.g., Visium or Visium HD).
    Provides a structure for future extensions to other methods.

    Parameters:
    - sample_root (str): Root directory containing Visium or Visium HD data.
    - library_id (str): Library ID of the sample.
    - method (str): Method for loading data, either 'visium' or 'visiumhd'. More methods can be added in the future.

    Returns:
    - adata (AnnData): AnnData object containing spatial transcriptomics data.
    """
    sample_root = Path(sample_root)
    
    if method == "visiumhd":
        print(f"Loading Visium HD data from {sample_root}")
        
        # Load Visium HD count matrix
        raw_h5_file = sample_root / "filtered_feature_bc_matrix.h5"
        adata = sc.read_10x_h5(raw_h5_file)
        adata.var_names_make_unique()
        
        # Load spatial coordinates
        tissue_position_file = sample_root / "spatial/tissue_positions.parquet"
        df_tissue_positions = pd.read_parquet(tissue_position_file)
        df_tissue_positions = df_tissue_positions.set_index("barcode")
        
        # Merge coordinates into adata.obs
        adata.obs = pd.merge(adata.obs, df_tissue_positions, left_index=True, right_index=True)
        
        # Add spatial coordinates to obsm
        adata.obsm["spatial"] = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
        
        # Drop original coordinate columns
        adata.obs.drop(columns=["pxl_row_in_fullres", "pxl_col_in_fullres"], inplace=True)

        # Load scale factors and spatial images if available
        adata.uns["spatial"] = {library_id: {}}
        scalefactors_file = sample_root / "spatial/scalefactors_json.json"
        if scalefactors_file.exists():
            with open(scalefactors_file, 'r') as f:
                adata.uns["spatial"][library_id]["scalefactors"] = json.load(f)

        adata.uns["spatial"][library_id]["images"] = {}
        for res in ["hires", "lowres"]:
            image_path = sample_root / f"spatial/tissue_{res}_image.png"
            if image_path.exists():
                adata.uns["spatial"][library_id]["images"][res] = imread(str(image_path))
        
        print(f"Visium HD data loaded successfully with {adata.shape[0]} observations.")

    elif method == "visium":
        print(f"Loading standard Visium data from {sample_root}")
        
        # Load standard Visium data using Scanpy
        adata = sc.read_visium(
            path=sample_root,
            count_file='filtered_feature_bc_matrix.h5',
            library_id=library_id,
            load_images=True
        )
        print(f"Standard Visium data loaded successfully with {adata.shape[0]} observations.")

    else:
        raise ValueError(f"Method {method} not recognized. Please use 'visium' or 'visiumhd'.")

    return adata


def preprocess_adata(adata, cy_file_path: str, min_spot_counts: int = 20):
    """
    Preprocess the AnnData object by filtering cells and genes, performing normalization, and selecting features.

    Parameters:
    - adata (AnnData): Input AnnData object.
    - cy_file_path (str): Path to the file containing cell cycle genes.
    - min_spot_counts (int): Minimum count threshold for filtering spots.

    Returns:
    - adata (AnnData): Processed AnnData object.
    """
    # Ensure gene names are unique
    adata.var_names_make_unique()
    
    # Filter spots based on total counts
    print(f"---Filtering spots with counts less than {min_spot_counts}---")
    spot_counts = adata.X.sum(axis=1).A1 if isinstance(adata.X, scipy.sparse.spmatrix) else adata.X.sum(axis=1)
    spots_to_keep = spot_counts >= min_spot_counts
    print(f"Filtered out {(~spots_to_keep).sum()} spots with counts < {min_spot_counts}")
    adata = adata[spots_to_keep].copy()
    
    # Filter cells and genes with minimum thresholds
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)
    adata.var_names_make_unique()
    
    # Load cell cycle genes from file
    cy = pd.read_csv(cy_file_path, sep='\t')
    cy_genes = cy.values.ravel()
    cy_g = cy_genes[~pd.isnull(cy_genes)]
    
    # Filter unwanted genes: cell cycle, mitochondrial RNA, and HLA genes
    print("---Filtering unwanted genes---")
    adata.var['if_cycle'] = adata.var.index.isin(cy_g)
    adata.var['if_mt'] = adata.var.index.str.startswith('MT')
    adata.var['if_hla'] = adata.var.index.str.contains('HLA')
    unwanted_genes = adata.var[['if_cycle', 'if_mt', 'if_hla']].sum(axis=1)
    print(f"Filtered out {unwanted_genes.sum()} genes (cell cycle, mitochondrial, HLA)")
    adata = adata[:, unwanted_genes == 0].copy()
    
    # Preserve original data layer
    adata.layers["counts"] = adata.X.copy()
    
    # Select highly variable genes
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    
    # Normalize and log-transform data
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    return adata


def quality_control(adata, output_dir):
    """
    Perform quality control on spatial transcriptomics data and save histograms of QC metrics.

    Parameters:
    - adata (AnnData): AnnData object containing the data.
    - output_dir (str): Directory to save the QC metric plots.
    """
    # Compute QC metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=(10, 20, 50, 100), inplace=True)
    
    # Print basic summary statistics
    print(f"Total spots: {adata.n_obs}")
    print(f"Total genes: {adata.n_vars}")
    print(f"Total counts: {adata.obs['total_counts'].sum()}")
    print(f"Median counts per spot: {adata.obs['total_counts'].median()}")
    print(f"Median genes per spot: {adata.obs['n_genes_by_counts'].median()}")
    
    # Plot QC metrics
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0]).set_title("Total counts per spot")
    sns.histplot(adata.obs["n_genes_by_counts"], kde=False, ax=axs[1]).set_title("Number of genes per spot")
    sns.histplot(adata.obs["pct_counts_in_top_10_genes"], kde=False, ax=axs[2]).set_title("Percentage of counts in top 10 genes")
    if 'pct_counts_mt' in adata.obs.columns:
        sns.histplot(adata.obs["pct_counts_mt"], kde=False, ax=axs[3]).set_title("Mitochondrial gene percentage")
    else:
        axs[3].text(0.5, 0.5, 'No mitochondrial data', horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes)
    
    # Save QC plots
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "qc_metrics.png"))
    plt.close(fig)


def plot_adata_visualizations(adata, output_dir, library_id="spatial", wspace=0.4):
    """
    Generate and save UMAP and spatial scatter plots for the AnnData object.

    Parameters:
    - adata (AnnData): The input AnnData object.
    - output_dir (str): Directory to save the plots.
    - library_id (str): Library ID used for spatial plots.
    - wspace (float): Width spacing between subplots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform dimensionality reduction and clustering
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=1.0)

    # Save UMAP plot
    sc.pl.umap(adata, color=["total_counts", "leiden"], wspace=wspace, show=False)
    plt.savefig(os.path.join(output_dir, "UMAP_plot.png"))
    plt.close()

    # Determine spatial image resolution
    img_key = 'hires' if 'hires' in adata.uns["spatial"][library_id]["images"] else 'lowres'

    # Save spatial scatter plot
    if img_key:
        sc.pl.spatial(adata, img_key=img_key, color='leiden', size=1.6, show=False)
        plt.savefig(os.path.join(output_dir, f"Spatial_scatter_plot_{img_key}.png"))
        plt.close()
    else:
        print("No spatial image available for plotting.")