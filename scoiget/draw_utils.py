import os
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def draw_spatial(adata, output_dir, library_id, method='leiden', has_ground_truth=False):
    # Set the save path
    save_path = os.path.join(output_dir, f'{library_id}_{method}_domain_spatial.png')
    colors = ["domain"]

    if has_ground_truth:
        colors.insert(0, 'ground_truth')
    
    # Check if 'hires' image exists, otherwise use 'lowres'
    img_key = 'hires' if 'hires' in adata.uns['spatial'][library_id]['images'] else 'lowres'
    # Draw the spatial plot
    sc.pl.spatial(
        adata, img_key=img_key,
        size=1.6, color=colors,
        show=False
    )
    # Save the image to the specified path
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()  # Display the image
    plt.close()  # Close the image

def draw_embedding(adata, output_dir, library_id, method='leiden'):
    save_path = os.path.join(output_dir, f'{library_id}_{method}_embedding.png')
    sc.pl.embedding(
        adata, basis="spatial", color="domain", size=100,
        show=False
    )
    plt.savefig(save_path)
    plt.show()  # Display the image
    plt.close()

def draw_umap(adata, output_dir, library_id, method='leiden', label=False, draw_batch=False, use_rep='latent'):
    print('----Start UMAP----')
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)
    # domain
    save_domain_path = os.path.join(output_dir, f'{library_id}_{method}_domain_umap.png')
    sc.pl.umap(adata, color='domain', show=False)
    plt.savefig(save_domain_path)
    plt.show()  # Display the image
    plt.close()
    
    # batch
    if draw_batch:
        save_batch_path = os.path.join(output_dir, f'{library_id}_{method}_batch_umap.png')
        sc.pl.umap(adata, color='batch', show=False)
        plt.savefig(save_batch_path)
        plt.show()  # Display the image
        plt.close()
    
    # label
    if label:
        save_label_path = os.path.join(output_dir, f'{library_id}_{method}_label_umap.png')
        sc.pl.umap(adata, color='ground_truth', show=False)
        plt.savefig(save_label_path)
        plt.show()  # Display the image
        plt.close()

def draw_paga(adata, output_dir, library_id, method='leiden', use_rep='latent'):
    print('----Start PAGA----')
    save_paga_path = os.path.join(output_dir, f'{library_id}_{method}_domain_paga.png')
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.paga(adata, groups='domain')
    sc.pl.paga(adata, color='domain', show=False)
    plt.savefig(save_paga_path)
    plt.show()  # Display the image
    plt.close()



def draw_paga(adata, output_dir, library_id, method='leiden', use_rep='latent'):
    print('----Start PAGA----')
    save_paga_path = os.path.join(output_dir, f'{library_id}_{method}_domain_paga.png')
    
    # Recompute neighbor relations and store them in a custom key
    neighbor_key = 'paga_neighbors'
    sc.pp.neighbors(adata, use_rep=use_rep, key_added=neighbor_key)
    
    # Compute PAGA graph using the custom neighbor information
    sc.tl.paga(adata, groups='domain', neighbors_key=neighbor_key)
    
    # Ensure the PAGA connectivity matrix is in sparse format
    adata.uns['paga']['connectivities'] = csr_matrix(adata.uns['paga']['connectivities'])
    adata.uns['paga']['connectivities_tree'] = csr_matrix(adata.uns['paga']['connectivities_tree'])
    
    # Draw the PAGA plot with a specified threshold
    sc.pl.paga(adata, color='domain', show=False, threshold=0.01)
    
    # Save the image
    plt.savefig(save_paga_path)
    plt.show()
    plt.close()
