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
    根据用户指定的方法加载 Visium 或 Visium HD 数据，并为将来扩展其他方法预留接口。

    参数:
    - sample_root: str，Visium 或 Visium HD 数据的根目录。
    - library_id: str，样本的库 ID。
    - method: str，加载数据的方法，可选 'visium' 或 'visiumhd'，未来可以扩展更多方法。
    
    返回:
    - adata: AnnData 对象，包含空间转录组数据。
    """
    sample_root = Path(sample_root)
    
    if method == "visiumhd":
        print(f"Loading Visium HD data from {sample_root}")
        
        # Load Visium HD data
        raw_h5_file = sample_root / "filtered_feature_bc_matrix.h5"
        adata = sc.read_10x_h5(raw_h5_file)
        adata.var_names_make_unique()
        
        # Load the Spatial Coordinates
        tissue_position_file = sample_root / "spatial/tissue_positions.parquet"
        df_tissue_positions = pd.read_parquet(tissue_position_file)
        
        # Set the index of the dataframe to the barcodes
        df_tissue_positions = df_tissue_positions.set_index("barcode")
        
        # Merge coordinates into adata.obs
        adata.obs = pd.merge(adata.obs, df_tissue_positions, left_index=True, right_index=True)
        
        # Add spatial coordinates to obsm
        adata.obsm["spatial"] = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
        
        '''
        # Create a GeoDataFrame for spatial coordinates and add geometry information
        geometry = [
            Point(xy) for xy in zip(
                df_tissue_positions["pxl_col_in_fullres"],
                df_tissue_positions["pxl_row_in_fullres"]
            )
        ]
        gdf_coordinates = gpd.GeoDataFrame(df_tissue_positions, geometry=geometry)
        adata.obs['geometry'] = gdf_coordinates['geometry']
        '''
        
        # Drop original coordinate columns
        adata.obs.drop(columns=["pxl_row_in_fullres", "pxl_col_in_fullres"], inplace=True)

        # Load scale factors and images if available
        adata.uns["spatial"] = {library_id: {}}
        scalefactors_file = sample_root / "spatial/scalefactors_json.json"
        if scalefactors_file.exists():
            with open(scalefactors_file, 'r') as f:
                adata.uns["spatial"][library_id]["scalefactors"] = json.load(f)

        # Load images if available
        adata.uns["spatial"][library_id]["images"] = {}
        for res in ["hires", "lowres"]:
            image_path = sample_root / f"spatial/tissue_{res}_image.png"
            if image_path.exists():
                adata.uns["spatial"][library_id]["images"][res] = imread(str(image_path))
        
        print(f"Visium HD data loaded successfully with {adata.shape[0]} observations.")

    elif method == "visium":
        print(f"Loading standard Visium data from {sample_root}")
        
        # Use scanpy's built-in function for Visium
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
    Preprocess AnnData object by filtering cells and genes, and performing normalization.
    
    Parameters:
    - adata: AnnData object
    - cy_file_path: Path to the file containing cell cycle genes
    - min_spot_counts: Minimum count threshold for filtering spots
    
    Returns:
    - Processed AnnData object
    """
    # Make gene names unique
    adata.var_names_make_unique()
    
    # Filter spots based on total counts
    print(f"---Filtering spots with counts less than {min_spot_counts}---")
    spot_counts = adata.X.sum(axis=1).A1 if isinstance(adata.X, scipy.sparse.spmatrix) else adata.X.sum(axis=1)
    spots_to_keep = spot_counts >= min_spot_counts
    filtered_spots_count = (~spots_to_keep).sum()
    print(f"Filtered out {filtered_spots_count} spots with counts < {min_spot_counts}")
    adata = adata[spots_to_keep].copy()
    
    # Filter cells and genes
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)
    
    # Ensure gene names are unique again
    adata.var_names_make_unique()
    
    # Read cell cycle genes file
    cy = pd.read_csv(cy_file_path, sep='\t')
    cy_genes = cy.values.ravel()
    cy_g = cy_genes[~pd.isnull(cy_genes)]
    
    # Gene filtering: cell cycle genes, mitochondrial RNA (mtRNA), and HLA genes
    print("---Filtering cell cycle genes---")
    adata.var['if_cycle'] = adata.var.index.isin(cy_g)
    cycle_genes_count = adata.var['if_cycle'].sum()
    print(f"Filtered out {cycle_genes_count} cell cycle genes")
    
    print("---Filtering mitochondrial RNA (mtRNA) genes---")
    adata.var['if_mt'] = adata.var.index.str.startswith('MT')
    mt_genes_count = adata.var['if_mt'].sum()
    print(f"Filtered out {mt_genes_count} mitochondrial RNA (mtRNA) genes")
    
    print("---Filtering HLA genes---")
    adata.var['if_hla'] = adata.var.index.str.contains('HLA')
    hla_genes_count = adata.var['if_hla'].sum()
    print(f"Filtered out {hla_genes_count} HLA genes")
    
    adata = adata[:, ~(adata.var.if_cycle | adata.var.if_mt | adata.var.if_hla)].copy()
    
    # Preserve the original data layer
    adata.layers["counts"] = adata.X.copy()
    
    # Highly variable gene selection
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    
    # Data normalization and transformation
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    return adata



def quality_control(adata, output_dir):
    """
    Calculate quality control metrics and plot histograms for spatial transcriptome data.

    Parameters:
        adata (AnnData): The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes.
        output_dir (str): Directory where output QC metrics plots will be saved.
    """
    # Calculate quality control metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=(10, 20, 50, 100), inplace=True)
    # Print summary statistics
    print(f"Total observations (spots): {adata.n_obs}")
    print(f"Total variables (genes): {adata.n_vars}")
    print(f"Total counts: {adata.obs['total_counts'].sum()}")
    print(f"Median counts per spot: {adata.obs['total_counts'].median()}")
    print(f"Median genes per spot: {adata.obs['n_genes_by_counts'].median()}")
    # Plot quality control charts
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].set_title("Total counts per spot")
    sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
    axs[1].set_title("Number of genes per spot")
    sns.histplot(adata.obs["n_genes_by_counts"], kde=False, ax=axs[1])
    axs[2].set_title("Percentage of counts in top 10 genes")
    sns.histplot(adata.obs["pct_counts_in_top_10_genes"], kde=False, ax=axs[2])
    axs[3].set_title("Mitochondrial gene percentage")
    if 'pct_counts_mt' in adata.obs.columns:
        sns.histplot(adata.obs["pct_counts_mt"], kde=False, ax=axs[3])
    else:
        axs[3].text(0.5, 0.5, 'No mitochondrial data', horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes)
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Save quality control charts
    fig.savefig(os.path.join(output_dir, "qc_metrics.png"))
    plt.close(fig)



def plot_adata_visualizations(adata, output_dir, library_id="spatial", wspace=0.4):
    """
    Plots UMAP and spatial scatter plots for a given AnnData object and saves them to a specified directory.
    Parameters:
        adata (AnnData): The annotated data matrix.
        output_dir (str): The directory where the plots will be saved.
        library_id (str, optional): The library ID for spatial plots. Defaults to "spatial".
        wspace (float, optional): The amount of width reserved for space between subplots. Defaults to 0.4.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Dimensionality reduction and clustering (optional)
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=1.0)

    # Plot UMAP showing total counts and Leiden clusters
    sc.pl.umap(
        adata,
        color=["total_counts", "leiden"],
        wspace=wspace,
        show=False  # Do not display the plot, only save it
    )
    plt.savefig(os.path.join(output_dir, "UMAP_plot.png"))
    plt.show()
    plt.close()

    # Determine which resolution image to use
    img_key = None
    if 'hires' in adata.uns["spatial"][library_id]["images"]:
        img_key = 'hires'
    elif 'lowres' in adata.uns["spatial"][library_id]["images"]:
        img_key = 'lowres'

    # Plot spatial scatter showing Leiden clusters
    if img_key:
        sc.pl.spatial(
            adata, img_key=img_key,
            color='leiden',
            size=1.6,
            show=False)
        plt.savefig(os.path.join(output_dir, f"Spatial_scatter_plot_{img_key}.png"))
        plt.show()
        plt.close()
    else:
        print("No spatial image available for plotting.")
