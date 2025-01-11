import os
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def draw_spatial(adata, output_dir, library_id, method='leiden', has_ground_truth=False):
    """
    Draw and save spatial visualization of the data.

    Args:
        adata (AnnData): AnnData object containing spatial data.
        output_dir (str): Directory to save the output plot.
        library_id (str): Identifier for the library/sample.
        method (str): Clustering method used (e.g., 'leiden').
        has_ground_truth (bool): Whether ground truth labels are available for visualization.
    """
    # Set the save path
    save_path = os.path.join(output_dir, f'{library_id}_{method}_domain_spatial.png')
    colors = ["domain"]

    # Add ground truth to the plot if available
    if has_ground_truth:
        colors.insert(0, 'ground_truth')
    
    # Check for 'hires' image; fallback to 'lowres' if not available
    img_key = 'hires' if 'hires' in adata.uns['spatial'][library_id]['images'] else 'lowres'
    
    # Plot spatial visualization
    sc.pl.spatial(
        adata, img_key=img_key,
        size=1.6, color=colors,
        show=False
    )
    # Save the plot to the specified path
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()  # Display the plot
    plt.close()  # Close the plot


def draw_embedding(adata, output_dir, library_id, method='leiden'):
    """
    Draw and save the embedding visualization.

    Args:
        adata (AnnData): AnnData object containing embedding data.
        output_dir (str): Directory to save the output plot.
        library_id (str): Identifier for the library/sample.
        method (str): Clustering method used (e.g., 'leiden').
    """
    save_path = os.path.join(output_dir, f'{library_id}_{method}_embedding.png')
    sc.pl.embedding(
        adata, basis="spatial", color="domain", size=100,
        show=False
    )
    plt.savefig(save_path)
    plt.show()  # Display the plot
    plt.close()


def draw_umap(adata, output_dir, library_id, method='leiden', label=False, draw_batch=False, use_rep='latent'):
    """
    Draw and save UMAP visualizations for domain, batch, and ground truth.

    Args:
        adata (AnnData): AnnData object containing UMAP data.
        output_dir (str): Directory to save the output plots.
        library_id (str): Identifier for the library/sample.
        method (str): Clustering method used (e.g., 'leiden').
        label (bool): Whether to include ground truth labels in the plot.
        draw_batch (bool): Whether to include batch information in the plot.
        use_rep (str): Representation to use for UMAP computation (e.g., 'latent').
    """
    print('----Start UMAP----')
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)

    # Plot UMAP with domain coloring
    save_domain_path = os.path.join(output_dir, f'{library_id}_{method}_domain_umap.png')
    sc.pl.umap(adata, color='domain', show=False)
    plt.savefig(save_domain_path)
    plt.show()  # Display the plot
    plt.close()
    
    # Plot UMAP with batch coloring if requested
    if draw_batch:
        save_batch_path = os.path.join(output_dir, f'{library_id}_{method}_batch_umap.png')
        sc.pl.umap(adata, color='batch', show=False)
        plt.savefig(save_batch_path)
        plt.show()  # Display the plot
        plt.close()
    
    # Plot UMAP with ground truth labels if requested
    if label:
        save_label_path = os.path.join(output_dir, f'{library_id}_{method}_label_umap.png')
        sc.pl.umap(adata, color='ground_truth', show=False)
        plt.savefig(save_label_path)
        plt.show()  # Display the plot
        plt.close()


def draw_paga(adata, output_dir, library_id, method='leiden', use_rep='latent'):
    """
    Draw and save PAGA visualization for domain relationships.

    Args:
        adata (AnnData): AnnData object containing PAGA data.
        output_dir (str): Directory to save the output plot.
        library_id (str): Identifier for the library/sample.
        method (str): Clustering method used (e.g., 'leiden').
        use_rep (str): Representation to use for PAGA computation (e.g., 'latent').
    """
    print('----Start PAGA----')
    save_paga_path = os.path.join(output_dir, f'{library_id}_{method}_domain_paga.png')
    
    # Recompute neighbors and store under a custom key
    neighbor_key = 'paga_neighbors'
    sc.pp.neighbors(adata, use_rep=use_rep, key_added=neighbor_key)
    
    # Compute PAGA graph using the custom neighbor key
    sc.tl.paga(adata, groups='domain', neighbors_key=neighbor_key)
    
    # Ensure PAGA connectivities are stored in sparse format
    adata.uns['paga']['connectivities'] = csr_matrix(adata.uns['paga']['connectivities'])
    adata.uns['paga']['connectivities_tree'] = csr_matrix(adata.uns['paga']['connectivities_tree'])
    
    # Plot PAGA graph with a threshold
    sc.pl.paga(adata, color='domain', show=False, threshold=0.01)
    
    # Save the plot
    plt.savefig(save_paga_path)
    plt.show()
    plt.close()