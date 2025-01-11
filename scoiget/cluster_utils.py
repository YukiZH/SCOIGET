import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score
#from rpy2.robjects import robjects
#from rpy2.robjects.numpy2ri import activate

from tqdm import tqdm


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='latent', random_seed=2023):
    """
    Perform clustering using the mclust algorithm from the R package.
    
    Parameters:
    - adata: AnnData object containing data.
    - num_cluster: Number of clusters.
    - modelNames: Model type, default is 'EEE'.
    - used_obsm: Key to access the representation in adata.obsm.
    - random_seed: Random seed for reproducibility.
    
    Returns:
    - Updated AnnData object with mclust clustering results stored in adata.obs['mclust'].
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = mclust_res.astype('int').astype('category')
    return adata


def search_res(radius, adata, n_clusters, method='leiden', use_rep='latent', start=0.01, end=5.0, increment=0.01):
    """
    Search for the optimal resolution to achieve the target number of clusters.

    Parameters:
    - radius: Placeholder (not used in function logic).
    - adata: AnnData object containing data.
    - n_clusters: Target number of clusters.
    - method: Clustering method ('leiden' or 'louvain').
    - use_rep: Key to access the representation in adata.obsm.
    - start: Starting resolution value.
    - end: Ending resolution value.
    - increment: Step size for resolution adjustment.

    Returns:
    - Best resolution value.
    """
    print('Searching for the optimal resolution...')
    label_found = False
    best_resolution = None
    sc.pp.neighbors(adata, n_neighbors=20, use_rep=use_rep)
    sc.tl.leiden(adata, random_state=0, resolution=end)
    count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
    
    while count_unique > n_clusters + 2:
        print(f'Cluster count: {count_unique}, adjusting down')
        end -= 0.1
        sc.tl.leiden(adata, random_state=0, resolution=end)
        count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
    
    while count_unique < n_clusters + 2:
        print(f'Cluster count: {count_unique}, adjusting up')
        end += 0.1
        sc.tl.leiden(adata, random_state=0, resolution=end)
        count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
    
    for res in sorted(np.arange(start, end, increment), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print(f'resolution={res}, cluster count={count_unique}')
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print(f'resolution={res}, cluster count={count_unique}')
        if count_unique == n_clusters:
            label_found = True
            best_resolution = res
            print(f'Best resolution: {best_resolution}')
            break
    
    if not label_found:
        raise ValueError("Resolution not found. Try a larger range or smaller step size.")
    return best_resolution


def auto_choose_clusters(adata, max_clusters=10, method='leiden', use_rep='latent', start=0.01, end=5.0, increment=0.01):
    """
    Automatically select the optimal number of clusters based on the silhouette score.

    Parameters:
    - adata: AnnData object containing data.
    - max_clusters: Maximum number of clusters to test.
    - method: Clustering method ('leiden', 'louvain', or 'mclust').
    - use_rep: Key to access the representation in adata.obsm.
    - start: Starting resolution value.
    - end: Ending resolution value.
    - increment: Step size for resolution adjustment.

    Returns:
    - Optimal number of clusters.
    """
    best_n_clusters = None
    best_silhouette = -1
    
    for n_clusters in range(2, max_clusters + 1):
        try:
            if method != 'mclust':
                res = search_res(50, adata, n_clusters, method=method, use_rep=use_rep, start=start, end=end, increment=increment)
                if method == 'leiden':
                    sc.tl.leiden(adata, random_state=0, resolution=res)
                    labels = adata.obs['leiden'].to_numpy()
                elif method == 'louvain':
                    sc.tl.louvain(adata, random_state=0, resolution=res)
                    labels = adata.obs['louvain'].to_numpy()
            else:
                adata = mclust_R(adata, num_cluster=n_clusters, used_obsm=use_rep)
                labels = adata.obs['mclust'].to_numpy()
            
            silhouette_avg = silhouette_score(adata.obsm[use_rep], labels)
            print(f'Number of clusters: {n_clusters}, silhouette score: {silhouette_avg}')
            
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_n_clusters = n_clusters
        except ValueError as e:
            print(f"Skipping n_clusters={n_clusters}: {e}")
    
    if best_n_clusters is None:
        raise ValueError("Unable to find optimal clusters. Try a larger range or smaller step size.")
    
    return best_n_clusters


def refine_label(adata, radius=50, key='label'):
    """
    Refine cluster labels based on neighborhood smoothing.

    Parameters:
    - adata: AnnData object containing data.
    - radius: Number of neighbors for smoothing.
    - key: Key in adata.obs for cluster labels.

    Returns:
    - List of refined labels.
    """
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = [old_type[index[j]] for j in range(1, n_neigh + 1)]
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
    
    new_type = [str(i) for i in list(new_type)]
    return new_type


def clustering(adata, radius=50, method='leiden', start=0.01, end=5.0, increment=0.01, refinement=False, auto_choose=False, max_clusters=10, n_clusters=None, use_rep='latent'):
    """
    Perform spatial clustering based on the latent representation.

    Parameters:
    - adata: AnnData object containing data.
    - radius: Number of neighbors for clustering (used for refinement).
    - method: Clustering method ('leiden', 'louvain', or 'mclust').
    - start: Starting resolution for clustering.
    - end: Ending resolution for clustering.
    - increment: Step size for resolution adjustment.
    - refinement: Whether to apply label refinement.
    - auto_choose: Whether to automatically determine the number of clusters.
    - max_clusters: Maximum number of clusters for auto_choose.
    - n_clusters: Number of clusters (used if auto_choose is False).
    - use_rep: Key to access the representation in adata.obsm.

    Returns:
    - Updated AnnData object with clustering results.
    """
    if auto_choose:
        n_clusters = auto_choose_clusters(adata, max_clusters=max_clusters, method=method, use_rep=use_rep, start=start, end=end, increment=increment)
        print(f'Optimal number of clusters: {n_clusters}')
    elif n_clusters is None:
        raise ValueError("n_clusters must be specified when auto_choose is False")
    
    if method == 'mclust':
        adata = mclust_R(adata, num_cluster=n_clusters, used_obsm=use_rep)
        adata.obs['domain'] = adata.obs['mclust']
        print(f'Clustering method: mclust, Number of clusters: {n_clusters}')
    elif method in ['leiden', 'louvain']:
        res = search_res(radius, adata, n_clusters, use_rep=use_rep, method=method, start=start, end=end, increment=increment)
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            adata.obs['domain'] = adata.obs['leiden']
            print(f'Clustering method: leiden, Number of clusters: {n_clusters}, Resolution: {res}')
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            adata.obs['domain'] = adata.obs['louvain']
            print(f'Clustering method: louvain, Number of clusters: {n_clusters}, Resolution: {res}')
    
    if refinement:
        new_type = refine_label(adata, radius, key='domain')
        adata.obs['domain'] = new_type
        print('Refinement applied')
    
    return adata