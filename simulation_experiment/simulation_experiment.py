# ## SCOIGET Simulation Tutorial 
#
# #### Step 0: Import libraries and set environment
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyensembl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine
import scipy.sparse
import warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.special import softmax
from scipy.sparse import lil_matrix
from tqdm import tqdm
import time
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, Normalize
from sklearn.cluster import KMeans 


# Ignore common Scanpy warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Set your project root directory ---
PROJECT_ROOT = "/export/home/zhangyujia/SCOIGET" 
# ---------------------

# Change current working directory
try:
    os.chdir(PROJECT_ROOT)
except FileNotFoundError:
    raise
    
# Add project root to sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Import SCOIGET modules (helper functions only) ---
try:
    from scoiget import preprocess_utils as pp
    from scoiget import cnv_utils as cu
    from scoiget import train_utils as tu 
    from scoiget import scoiget_model as sm 
except ImportError:
    print("Warning: SCOIGET modules not found. Using dummy classes for model structure.")
    class DummyModule:
        def __init__(self): pass
        def add_genomic_locations(self, adata): return adata
        def prepare_data(self, adata, use_norm_x=False):
            class DummyData:
                def __init__(self, x, edge_index):
                    self.x = torch.tensor(x, dtype=torch.float32)
                    self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                    self.num_features = self.x.shape[1]
                def to(self, device): return self
            
            num_spots = adata.n_obs
            num_features = adata.obsm['feat'].shape[1]
            x_tensor = adata.obsm['feat']
            
            row, col = adata.obsm['graph_neigh'].nonzero()
            edge_index = np.vstack([row, col])
            
            return DummyData(x_tensor, edge_index).to(device)

    cu = DummyModule()
    tu = DummyModule()
    class DummyEncoder(nn.Module):
        def __init__(self, in_c, hidden_c, out_c): super().__init__(); self.linear = nn.Linear(in_c, 2 * out_c)
        def forward(self, x, edge_index):
            z_raw = self.linear(x)
            z_mean = z_raw[:, :z_raw.shape[1]//2]
            z_log_var = z_raw[:, z_raw.shape[1]//2:]
            z_var = torch.exp(z_log_var)
            epsilon = torch.randn_like(z_mean)
            z = z_mean + torch.sqrt(z_var) * epsilon
            return z_mean, z_var, z
    class DummyDecoder(nn.Module):
        def __init__(self, in_c, hidden_c, out_c): super().__init__(); self.linear = nn.Linear(in_c, out_c)
        def forward(self, z): return self.linear(z)
    class DummyModel:
        Encoder = DummyEncoder
        Decoder = DummyDecoder
    sm = DummyModel

# --- Set device ---
device = torch.device('cpu') 

# --- Load Ensembl database (required) ---
try:
    ensembl_data = pyensembl.EnsemblRelease(98)
except Exception:
    raise

# --- HMM Helper Functions ---
def gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

# --- [!!! Fix 1: Zero-Centered HMM for Enhanced Robustness !!!] ---
class DifferentiableHMM_Centered(nn.Module):
    def __init__(self, num_states=3):
        super(DifferentiableHMM_Centered, self).__init__()
        self.num_states = num_states
        
        # Learn means for Loss and Gain states
        self.state_means = nn.Parameter(torch.tensor([-1.0, 1.0]), requires_grad=True) 
        # Adjust std dev initialization for stability
        self.log_stds = nn.Parameter(torch.tensor([-1.0, -1.0, -1.0]), requires_grad=True)

        # Transition probabilities (log-space): Enhance diagonal (4.0 -> 6.0) to increase switching difficulty
        self.transition_logits = nn.Parameter(torch.tensor([
            [6.0, 1.0, 1.0], 
            [1.0, 6.0, 1.0], 
            [1.0, 1.0, 6.0]
        ]), requires_grad=True)
        self.start_logits = nn.Parameter(torch.tensor([1.0, 5.0, 1.0]), requires_grad=True)

    def forward(self, x, chrom_list, edge_index, lambda_smooth=0.1):
        n_spots, n_bins = x.shape
        
        # Normal state fixed at 0.0
        all_means = torch.stack([self.state_means[0], torch.tensor(0.0, device=device), self.state_means[1]])
        
        stds = torch.exp(self.log_stds)
        dist = torch.distributions.Normal(all_means, stds + 1e-6)
        log_emission_probs = dist.log_prob(x.unsqueeze(-1))
        
        all_states = []
        log_trans_probs = F.log_softmax(self.transition_logits, dim=1)
        log_start_probs = F.log_softmax(self.start_logits, dim=0)

        for (start_bin, end_bin) in chrom_list:
            chrom_emissions = log_emission_probs[:, start_bin:end_bin, :]
            chrom_len = end_bin - start_bin
            if chrom_len == 0: continue
            
            log_trellis = log_start_probs + chrom_emissions[:, 0, :]
            for t in range(1, chrom_len):
                log_prob_t = log_trellis.unsqueeze(1) + log_trans_probs
                log_trellis, _ = torch.max(log_prob_t, dim=2)
                log_trellis = log_trellis + chrom_emissions[:, t, :]
            
            states_soft = gumbel_softmax(chrom_emissions, tau=0.1, hard=True, dim=-1)
            all_states.append(states_soft)

        if not all_states: return torch.zeros_like(x), torch.tensor(0.0, device=device)

        states_one_hot = torch.cat(all_states, dim=1)
        norm_copy = states_one_hot @ all_means
        
        row, col = edge_index
        # Calculate smoothing loss on "hard" states
        spatial_smooth_loss = F.mse_loss(states_one_hot[row], states_one_hot[col])
        
        return norm_copy, lambda_smooth * spatial_smooth_loss

class SCOIGET_Corrected_Centered(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=32, hmm_states=3, 
                 kl_weight=0.05, lambda_smooth=2.0, lambda_hmm_fit=1.0): 
        super(SCOIGET_Corrected_Centered, self).__init__()
        self.kl_weight = kl_weight
        self.lambda_smooth = lambda_smooth
        self.lambda_hmm_fit = lambda_hmm_fit
        
        self.z_encoder = sm.Encoder(in_channels, hidden_channels, out_channels).to(device)
        self.decoder = sm.Decoder(out_channels, hidden_channels, in_channels).to(device)
        self.hmm = DifferentiableHMM_Centered(num_states=hmm_states).to(device)
        
    def forward(self, x, edge_index, chrom_list):
        z_mean, z_var, z = self.z_encoder(x, edge_index)
        reconstructed_features = self.decoder(z)
        norm_copy, spatial_smooth_loss = self.hmm(reconstructed_features, chrom_list, edge_index, self.lambda_smooth)
        
        p_dis = torch.distributions.Normal(loc=z_mean, scale=torch.sqrt(z_var))
        q_dis = torch.distributions.Normal(loc=torch.zeros_like(z_mean), scale=torch.ones_like(z_var))
        kl_loss = torch.sum(torch.distributions.kl_divergence(p_dis, q_dis), dim=1).mean() * self.kl_weight

        recon_loss = F.mse_loss(reconstructed_features, x, reduction='mean') # GNN reconstruction loss
        hmm_recon_loss = F.mse_loss(norm_copy, reconstructed_features.detach()) # HMM fit loss (fitting GNN output)

        # Total loss: Recon + KL + Spatial Smooth + HMM Fit (enhanced by self.lambda_hmm_fit)
        loss = recon_loss + kl_loss + spatial_smooth_loss + self.lambda_hmm_fit * hmm_recon_loss
        
        return norm_copy, recon_loss, kl_loss, spatial_smooth_loss, hmm_recon_loss, loss

    def get_cnv(self, x, edge_index, chrom_list):
        self.eval()
        with torch.no_grad():
            z_mean, z_var, z = self.z_encoder(x, edge_index)
            reconstructed_features = self.decoder(z)
            norm_copy, _ = self.hmm(reconstructed_features, chrom_list, edge_index)
        return norm_copy

# --- Scoring Functions (Unchanged) ---
def re_implemented_CNV_score_centered(adata, use_rep='segment_cn', ref_label=0):
    seg_cn = adata.obsm[use_rep]
    try:
        norm_mask = (adata.obs['ground_truth_clone'].astype(int) == ref_label).values
        if np.sum(norm_mask) > 0:
            baseline = np.median(seg_cn[norm_mask], axis=0)
            cnv_score = seg_cn - baseline
        else:
            cnv_score = seg_cn 
        adata.obsm['cnv_score'] = cnv_score
        return adata
    except:
        return adata

# --- Other Helper Functions (Unchanged) ---
def re_implemented_CNV_score(adata, use_rep='segment_cn', ref_label=0):
    print("Running re-implemented 'CNV_score'...")
    try:
        seg_cn = adata.obsm[use_rep]
        try:
            norm_mask = (adata.obs['ground_truth_clone'].astype(int) == ref_label).values
        except Exception:
             norm_mask = (adata.obs['ground_truth_clone'] == ref_label).values
        
        if np.sum(norm_mask) == 0:
            baseline = np.median(seg_cn, axis=0)
        else:
            baseline = np.median(seg_cn[norm_mask], axis=0)
        
        cnv_score = seg_cn - baseline
        adata.obsm['cnv_score'] = cnv_score
        return adata
    except KeyError:
        raise

def create_simulation_adata(n_genes, grid_size, gain_factor, loss_factor, noise_level, dropout_rate):
    n_spots = grid_size * grid_size

    # --- 1. Create gene annotations (adata.var) ---
    contigs_to_query = [str(i) for i in range(1, 23)] + ['X', 'Y']
    all_genes_list = []
    for contig_str in contigs_to_query:
        try:
            genes_on_contig = ensembl_data.genes(contig=contig_str)
            all_genes_list.extend(genes_on_contig)
        except Exception as e:
            pass
            
    valid_genes = []
    for g in all_genes_list: 
        if g.biotype == 'protein_coding' and g.gene_name is not None and g.gene_id is not None:
            valid_genes.append(g)
    
    genes_chr2 = [g for g in valid_genes if g.contig == '2']
    genes_chr7 = [g for g in valid_genes if g.contig == '7']
    genes_other = [g for g in valid_genes if g.contig not in ['2', '7']]
    
    selected_genes = []
    selected_genes.extend(genes_chr2)
    selected_genes.extend(genes_chr7)

    remaining_needed = n_genes - len(selected_genes)
    if remaining_needed > 0 and len(genes_other) > 0:
        if remaining_needed > len(genes_other):
            remaining_needed = len(genes_other)
        selected_genes.extend(genes_other[:remaining_needed])
    
    if len(selected_genes) < n_genes:
         n_genes = len(selected_genes)
    
    def sort_key(g):
        contig = g.contig
        if contig.isdigit():
            return (int(contig), g.start)
        elif contig == 'X':
            return (23, g.start)
        elif contig == 'Y':
            return (24, g.start)
        else:
            return (25, g.start)

    selected_genes.sort(key=sort_key)
    
    var_df = pd.DataFrame(
        {
            'gene_ids': [g.gene_id for g in selected_genes],
            'chromosome': "0",
            'start': 0,
            'end': 0
        },
        index=[g.gene_name for g in selected_genes]
    )
    var_df.index = var_df.index.astype(str)
    if var_df.index.duplicated().any():
        var_df.index.name = 'gene_name_duplicated'
        var_df = var_df.reset_index().drop_duplicates(subset='gene_name_duplicated').set_index('gene_name_duplicated')
        var_df.index.name = 'gene_name'
    var_df.index.name = 'gene_name'

    if n_genes != len(var_df):
        n_genes = len(var_df)
        gene_id_to_object_map = {g.gene_id: g for g in valid_genes}
        final_gene_ids = var_df['gene_ids'].tolist()
        selected_genes = [gene_id_to_object_map[gid] for gid in final_gene_ids]

    # --- 2. Create spatial coordinates and clones (adata.obs, adata.obsm) ---
    x_coords = np.arange(grid_size)
    y_coords = np.arange(grid_size)
    xx, yy = np.meshgrid(x_coords, y_coords)
    spatial_coords = np.vstack([xx.ravel(), yy.ravel()]).T
    spot_names = [f"spot_{i}" for i in range(n_spots)]
    obs_df = pd.DataFrame(index=spot_names)
    
    clone_labels = np.zeros(n_spots, dtype=int)
    center_a = (grid_size * 0.25, grid_size * 0.25)
    radius_a = grid_size * 0.2
    dist_a = np.sqrt((spatial_coords[:, 0] - center_a[0])**2 + (spatial_coords[:, 1] - center_a[1])**2)
    clone_labels[dist_a < radius_a] = 1 # Clone A (Gain)
    
    center_b = (grid_size * 0.75, grid_size * 0.75)
    size_b = grid_size * 0.2
    is_in_b = (spatial_coords[:, 0] > center_b[0] - size_b) & (spatial_coords[:, 0] < center_b[0] + size_b) & \
              (spatial_coords[:, 1] > center_b[1] - size_b) & (spatial_coords[:, 1] < center_b[1] + size_b)
    clone_labels[is_in_b] = 2 # Clone B (Loss)
    
    obs_df['ground_truth_clone'] = pd.Categorical(clone_labels)

    # --- 3. Create Ground Truth CNV matrix (n_spots, n_genes) ---
    ground_truth_cnv_matrix = np.ones((n_spots, n_genes), dtype=np.float32)
    temp_var_df = var_df.copy()
    temp_var_df['chromosome'] = [g.contig for g in selected_genes]
    
    genes_chr2_mask = (temp_var_df['chromosome'] == '2').values
    spots_clone_a_mask = (clone_labels == 1)
    if np.sum(genes_chr2_mask) > 0:
        ground_truth_cnv_matrix[np.ix_(spots_clone_a_mask, genes_chr2_mask)] = gain_factor
    else:
        pass
        
    genes_chr7_mask = (temp_var_df['chromosome'] == '7').values
    spots_clone_b_mask = (clone_labels == 2)
    if np.sum(genes_chr7_mask) > 0:
        ground_truth_cnv_matrix[np.ix_(spots_clone_b_mask, genes_chr7_mask)] = loss_factor
    else:
        pass
    del temp_var_df 

    # --- 4. Create simulated gene expression matrix (adata.X) ---
    base_expr = np.random.lognormal(mean=1.0, sigma=0.5, size=(n_spots, n_genes))
    expr_with_cnv = base_expr * ground_truth_cnv_matrix
    counts = np.random.poisson(lam=expr_with_cnv * 10) 
    noise = np.random.normal(scale=noise_level, size=counts.shape)
    counts = np.maximum(0, counts + noise)
    dropout_mask = np.random.rand(*counts.shape) < dropout_rate
    counts[dropout_mask] = 0
    counts_matrix_sparse = scipy.sparse.csr_matrix(counts.astype(np.float32))

    # --- 5. Assemble AnnData object ---
    adata = sc.AnnData(X=counts_matrix_sparse, var=var_df, obs=obs_df)
    adata.obsm['spatial'] = spatial_coords
    
    return adata, selected_genes

def simple_binning_function(adata_obj, bin_size, ensembl_obj):
    if 'chromosome' not in adata_obj.var.columns or 'start' not in adata_obj.var.columns:
        adata_obj = cu.add_genomic_locations(adata_obj)
        
    def get_chrom_sort_key(chrom_str):
        if chrom_str.startswith('chr'):
            chrom_str = chrom_str[3:]
        if chrom_str.isdigit():
            return int(chrom_str)
        elif chrom_str == 'X':
            return 23
        elif chrom_str == 'Y':
            return 24
        else:
            return 25
            
    adata_obj.var['sort_key'] = adata_obj.var.apply(
        lambda row: (get_chrom_sort_key(row['chromosome']), row['start']), axis=1
    )
    adata_obj = adata_obj[:, adata_obj.var.sort_values('sort_key').index].copy()
    
    X_binned_list = []
    binned_var_names = []
    chrom_list = []
    current_chrom_start_bin = 0
    sorted_chromosomes = adata_obj.var['chromosome'].unique()
    
    for chrom in sorted_chromosomes:
        adata_chrom = adata_obj[:, adata_obj.var['chromosome'] == chrom].copy()
        n_genes_chrom = adata_chrom.n_vars
        n_bins = n_genes_chrom // bin_size
        
        if n_bins == 0:
            continue
            
        adata_chrom = adata_chrom[:, :n_bins * bin_size].copy()
        X_chrom = adata_chrom.X.toarray() if scipy.sparse.issparse(adata_chrom.X) else adata_chrom.X
        X_binned = X_chrom.reshape(adata_chrom.n_obs, n_bins, bin_size).mean(axis=2)
        X_binned_list.append(X_binned)
        
        for i in range(n_bins):
            binned_var_names.append(f"{chrom}_bin{i}")
            
        current_chrom_end_bin = current_chrom_start_bin + n_bins
        chrom_list.append((current_chrom_start_bin, current_chrom_end_bin))
        current_chrom_start_bin = current_chrom_end_bin

    X_binned_combined = np.concatenate(X_binned_list, axis=1)
    
    binned_adata = sc.AnnData(X=X_binned_combined)
    binned_adata.obs = adata_obj.obs.copy()
    binned_adata.var.index = binned_var_names
    binned_adata.uns['chrom_list'] = chrom_list
    
    adata_obj.uns['binned_data'] = binned_adata
    return adata_obj, chrom_list 

def calculate_graph_and_probabilities(adata, use_norm_x=False, n_neighbors=8):
    if use_norm_x:
        if 'norm_x' not in adata.obsm:
            raise ValueError("adata.obsm['norm_x'] not found for Phase 2.")
        embedding = adata.obsm['norm_x']
        prob_key = 'edge_probabilities_norm_x'
        graph_key = 'graph_neigh_norm_x'
    else:
        if 'binned_data' not in adata.uns:
            raise ValueError("adata.uns['binned_data'] not found for Phase 1.")
        embedding_sparse = adata.uns['binned_data'].X
        embedding = embedding_sparse.toarray() if scipy.sparse.issparse(embedding_sparse) else embedding_sparse
        prob_key = 'edge_probabilities'
        graph_key = 'graph_neigh'

    scaler = StandardScaler()
    embedding_scaled = scaler.fit_transform(embedding)
    pca = PCA(n_components=32, random_state=42)
    embedding_pca = pca.fit_transform(embedding_scaled)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(embedding_pca)
    distances, indices = nbrs.kneighbors(embedding_pca)
    distances = np.nan_to_num(distances, nan=1e6, posinf=1e6, neginf=-1e6)

    n_spots = embedding_pca.shape[0]
    graph_neigh_lil = lil_matrix((n_spots, n_spots), dtype=float)
    edge_probabilities_lil = lil_matrix((n_spots, n_spots), dtype=float)

    for i in range(n_spots):
        neighbors_idx = indices[i, 1:]
        dist = distances[i, 1:]
        graph_neigh_lil[i, neighbors_idx] = 1
        
        if len(dist) > 0:
            weights = 1.0 / (dist + 1e-6)
            softmax_weights = softmax(weights)
            softmax_weights = np.nan_to_num(softmax_weights, nan=1e-9, posinf=1.0, neginf=1e-9)
            softmax_weights[softmax_weights == 0] = 1e-9
            edge_probabilities_lil[i, neighbors_idx] = softmax_weights

    adata.obsm[graph_key] = graph_neigh_lil.tocsr()
    adata.obsm[prob_key] = edge_probabilities_lil.tocsr()
    
    if adata.obsm[graph_key].nnz != adata.obsm[prob_key].nnz:
        pass
    else:
        pass
    return adata


# #### 1. Simulation Parameter Setup

# --- 1. Simulation Parameters ---
GRID_SIZE = 40
N_SPOTS = GRID_SIZE * GRID_SIZE
N_GENES = 6000
BIN_SIZE = 25
DROPOUT_RATE = 0.3
NOISE_LEVEL = 0.5
GAIN_FACTOR = 3.0
LOSS_FACTOR = 0.1
output_dir = "./simulation_output_fixed_optimized" # Modify output directory
os.makedirs(output_dir, exist_ok=True)
print(f"Simulation parameters set. Output will be saved to {output_dir}")

# #### 2. Generate Simulated AnnData Object
adata, selected_genes = create_simulation_adata(
    n_genes=N_GENES,
    grid_size=GRID_SIZE,
    gain_factor=GAIN_FACTOR,
    loss_factor=LOSS_FACTOR,
    noise_level=NOISE_LEVEL,
    dropout_rate=DROPOUT_RATE
)
print(adata)

# #### 3. Visualize Ground Truth
fig, axes = plt.subplots(1, 1, figsize=(6, 5))
sc.pl.embedding(
    adata, basis="spatial", color="ground_truth_clone",
    title="Ground Truth Clones", s=20, show=False, ax=axes
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ground_truth_visualization.png"), dpi=150)
plt.close(fig) 


# #### 4. Run Complete SCOIGET Pipeline (Final Corrected Version)

print("--- Starting SCOIGET Pipeline (Final Corrected) ---")

# 4.1: Preprocessing
sc.pp.filter_cells(adata, min_counts=5)
sc.pp.filter_genes(adata, min_cells=1) 
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# 4.2: Add genomic locations
adata = cu.add_genomic_locations(adata)

# 4.3: Gene Binning 
adata, chrom_list = simple_binning_function(adata, bin_size=BIN_SIZE, ensembl_obj=ensembl_data)

# 4.4: Graph construction (Normalized features)
binned_matrix = adata.uns['binned_data'].X
feat_raw = binned_matrix.toarray() if scipy.sparse.issparse(binned_matrix) else binned_matrix

# Normalize features (Zero-centered input)
scaler = StandardScaler()
feat_scaled = scaler.fit_transform(feat_raw)
adata.obsm['feat'] = feat_scaled 

# Calculate graph (using normalized features)
adata = calculate_graph_and_probabilities(adata, use_norm_x=False, n_neighbors=8)
feat_adata = adata.copy()
data = tu.prepare_data(feat_adata, use_norm_x=False) 
data = data.to(device)


# 4.5: Train model (Final corrected model - Similarity Enhanced)
print("--- Starting End-to-End Training (Similarity Enhanced) ---")
model = SCOIGET_Corrected_Centered(
    in_channels=data.num_features,
    hidden_channels=128,
    out_channels=32,
    hmm_states=3,
    kl_weight=0.05, 
    lambda_smooth=3.0, # Increase spatial smoothing weight (2.0 -> 3.0)
    lambda_hmm_fit=1.0 # Enhance HMM fitting loss weight
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 300 
train_losses = []

pbar = tqdm(range(epochs), desc="Training (Similarity Enhanced)", file=sys.stdout)
for epoch in pbar:
    model.train()
    optimizer.zero_grad()
    
    norm_copy, recon_loss, kl_loss, spatial_smooth_loss, hmm_loss, loss = model(
        data.x, data.edge_index, chrom_list
    )
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        pbar.set_postfix({
            "Loss": loss.item(), 
            "Recon": recon_loss.item(),
            "Smooth": spatial_smooth_loss.item(),
            "HMM_Fit": hmm_loss.item()
        })

print("Training complete. Getting final CNV...")
adata.obsm['copy'] = model.get_cnv(data.x, data.edge_index, chrom_list).cpu().numpy()

# 4.6: Skip segmentation, use HMM results directly
adata.obsm['segment_cn'] = adata.obsm['copy']

# Corrected scoring function
adata = re_implemented_CNV_score_centered(adata, use_rep='segment_cn', ref_label=0)
print("--- SCOIGET Pipeline Complete ---")

# #### 5. Evaluation and Visualization (Final Corrected - Clustering and Plot Fixes)

print("--- Evaluating and Visualizing Results (Clustering and Plot Fix) ---")

# --- 1. Prepare data and threshold filtering --- 
pred_cnv_aligned = adata.obsm['cnv_score']
# [!!! Key Fix 3: Threshold filtering to remove background noise !!!]
THRESHOLD = 0.25 # Set threshold; signals below this absolute value are considered noise
pred_cnv_clean = pred_cnv_aligned.copy()
pred_cnv_clean[np.abs(pred_cnv_clean) < THRESHOLD] = 0 

# (B) Extract GT and align (code unchanged)
gt_var_df = pd.DataFrame({
    'chromosome_orig': [g.contig for g in selected_genes], 
    'start': [g.start for g in selected_genes],
}, index=[g.gene_name for g in selected_genes])

gt_var_df.index = gt_var_df.index.astype(str)
if gt_var_df.index.duplicated().any():
    gt_var_df = gt_var_df.reset_index().drop_duplicates(subset='index').set_index('index')

common_genes_gt = gt_var_df.index.intersection(adata.var_names)
adata_eval = adata[:, common_genes_gt].copy()
gt_var_df = gt_var_df.loc[common_genes_gt]
gt_var_df = gt_var_df.join(adata_eval.var[['chromosome', 'start', 'sort_key']], rsuffix='_adata')
gt_var_df = gt_var_df.sort_values('sort_key')
adata_eval = adata_eval[:, gt_var_df.index].copy()

# Reconstruct GT matrix
gt_cnv_matrix_full = np.ones((adata_eval.n_obs, adata_eval.n_vars), dtype=np.float32)
genes_chr2_mask = (gt_var_df['chromosome_orig'] == '2').values
spots_clone_a_mask = (adata_eval.obs['ground_truth_clone'].astype(int) == 1).values
gt_cnv_matrix_full[np.ix_(spots_clone_a_mask, genes_chr2_mask)] = 2.0 # Gain

genes_chr7_mask = (gt_var_df['chromosome_orig'] == '7').values
spots_clone_b_mask = (adata_eval.obs['ground_truth_clone'].astype(int) == 2).values
gt_cnv_matrix_full[np.ix_(spots_clone_b_mask, genes_chr7_mask)] = 0.0 # Loss

# Binning
binned_gt_list = []
sorted_chromosomes_gt = adata_eval.var['chromosome'].unique()
chrom_boundaries = [] 
current_idx = 0

for chrom in sorted_chromosomes_gt:
    genes_on_chrom_mask = (adata_eval.var['chromosome'] == chrom).values
    n_genes_chrom = np.sum(genes_on_chrom_mask)
    n_bins = n_genes_chrom // BIN_SIZE
    if n_bins == 0: continue
    gt_matrix_chrom = gt_cnv_matrix_full[:, genes_on_chrom_mask][:, :n_bins * BIN_SIZE]
    gt_binned = gt_matrix_chrom.reshape(adata_eval.n_obs, n_bins, BIN_SIZE).mean(axis=2)
    binned_gt_list.append(gt_binned)
    current_idx += n_bins
    chrom_boundaries.append(current_idx)

gt_cnv_aligned = np.concatenate(binned_gt_list, axis=1)
norm_mask_gt = (adata_eval.obs['ground_truth_clone'].astype(int) == 0).values
baseline_gt = np.median(gt_cnv_aligned[norm_mask_gt], axis=0)
gt_cnv_scores = gt_cnv_aligned - baseline_gt

# --- 2. Calculate metrics (Using raw predictions for authenticity) ---
pred_cnv_clipped = np.clip(pred_cnv_aligned, -1.5, 1.5) 
mse_scoiget = mean_squared_error(gt_cnv_scores.ravel(), pred_cnv_clipped.ravel())
cos_sim_scoiget = 1 - cosine(gt_cnv_scores.ravel() + 1e-9, pred_cnv_clipped.ravel() + 1e-9)

print(f"--- Final Metrics ---")
print(f"MSE: {mse_scoiget:.4f}")
print(f"Cosine Similarity: {cos_sim_scoiget:.4f}")

# --- 3. Sort data ---
sort_idx = np.argsort(adata_eval.obs['ground_truth_clone'].values)
gt_sorted = gt_cnv_scores[sort_idx]
# Use filtered data for plotting
pred_sorted_clean = pred_cnv_clean[sort_idx]


# --- 4. K-Means Clustering---
print("Running K-Means (K=3) on predicted CNV scores to infer clones...")
features_for_clustering = pred_cnv_aligned.copy() 
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
adata_eval.obs['inferred_clone_raw'] = kmeans.fit_predict(features_for_clustering)

# Normalize cluster labels (Loss: 0, Normal: 1, Gain: 2)
cluster_medians = []
for i in range(3):
    median_score = np.median(features_for_clustering[adata_eval.obs['inferred_clone_raw'] == i])
    cluster_medians.append((median_score, i))

cluster_medians.sort(key=lambda x: x[0]) 

# Map new class labels (0: Loss/Negative, 1: Normal/Zero, 2: Gain/Positive)
mapping = {cluster_medians[0][1]: 0, 
           cluster_medians[1][1]: 1, 
           cluster_medians[2][1]: 2} 

adata_eval.obs['inferred_clone'] = adata_eval.obs['inferred_clone_raw'].map(mapping).astype('category')
adata_eval.obs['inferred_clone'] = adata_eval.obs['inferred_clone'].cat.rename_categories(['Loss', 'Normal', 'Gain'])


# --- 5. Advanced Plotting (Layout and Color Adjustment) ---
fig = plt.figure(figsize=(26, 12))
# Adjust layout: 4 columns, 2nd row contains C, D, E (swapped)
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.5], height_ratios=[1, 1], wspace=0.3, hspace=0.3)

# Define unified color map (Shared by Plots C & D; Loss/Normal/Gain correspond to Blue/Gray/Red)
COLOR_MAP = {
    'Normal': '#d3d3d3', # Gray
    'Gain': '#b2182b',   # Red
    'Loss': '#2166ac'    # Blue
}
# Define Ground Truth color list (For Plot C: 0, 1, 2 correspond to Normal, Gain, Loss)
# Note: adata.obs['ground_truth_clone'] categories are 0, 1, 2 (integers)
gt_color_list = [COLOR_MAP['Normal'], COLOR_MAP['Gain'], COLOR_MAP['Loss']]


# (A) Ground Truth Heatmap
ax0 = plt.subplot(gs[0, 0])
im0 = ax0.imshow(gt_sorted, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1, interpolation='none')
ax0.set_title('A) Ground Truth', fontsize=16)
ax0.set_ylabel('Spots (Sorted by Clone)', fontsize=14)
ax0.set_xlabel('Genomic Bins', fontsize=12)
for boundary in chrom_boundaries[:-1]:
    ax0.axvline(x=boundary, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

# (B) SCOIGET Prediction Heatmap
ax1 = plt.subplot(gs[0, 1])
im1 = ax1.imshow(pred_sorted_clean, aspect='auto', cmap='coolwarm', 
                 vmin=-1.0, vmax=1.0, interpolation='none') 
ax1.set_title('B) SCOIGET Prediction', fontsize=16)
ax1.set_yticks([]) 
ax1.set_xlabel('Genomic Bins', fontsize=12)
for boundary in chrom_boundaries[:-1]:
    ax1.axvline(x=boundary, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Predicted Score (Z-score)', fontsize=10)

# (C) Spatial Plot (Ground Truth) - Position (1, 0)
ax2 = plt.subplot(gs[1, 0])
adata_eval.obs['ground_truth_clone'] = adata_eval.obs['ground_truth_clone'].astype('category').cat.set_categories([0, 1, 2])
adata_eval.uns['ground_truth_clone_colors'] = gt_color_list

sc.pl.spatial(adata_eval, color="ground_truth_clone", title="C) Ground Truth Spatial Clones", 
              spot_size=100, show=False, ax=ax2, frameon=False) # Remove unsupported na_in_group
ax2.legend().remove()

# (D) Spatial Plot (SCOIGET Inferred Clones) - Position (1, 1)
ax3 = plt.subplot(gs[1, 1])
ordered_cats = ['Loss', 'Normal', 'Gain'] 
colors_inferred = [COLOR_MAP[c] for c in ordered_cats]

# Reset categories and colors
adata_eval.obs['inferred_clone'] = adata_eval.obs['inferred_clone'].cat.reorder_categories(ordered_cats)
adata_eval.uns['inferred_clone_colors'] = colors_inferred 

sc.pl.spatial(adata_eval, color="inferred_clone", title="D) SCOIGET Inferred Clones (K=3)", # Change label to D
              spot_size=100, show=False, ax=ax3, frameon=False) 
ax3.legend().remove() 

# (E) Spatial Plot (SCOIGET Inferred - CNV Burden) - Position (1, 2) [Originally Plot D]
ax4 = plt.subplot(gs[1, 2])
cnv_burden = np.mean(np.abs(pred_cnv_clean), axis=1)
cnv_burden = (cnv_burden - np.min(cnv_burden)) / (np.max(cnv_burden) - np.min(cnv_burden))
adata_eval.obs['CNV_Burden'] = cnv_burden

sc.pl.spatial(adata_eval, color="CNV_Burden", cmap='Reds', 
              title="E) SCOIGET Inferred CNV Burden (Thresholded)", # Change label to E
              spot_size=100, show=False, ax=ax4, frameon=False)

# (F) Metrics
ax5 = plt.subplot(gs[:, 3])
metrics = ['MSE', 'Cosine Sim']
values = [mse_scoiget, cos_sim_scoiget]
bars = ax5.bar(metrics, values, color=['#bdbdbd', '#4575b4'], width=0.6)
ax5.set_ylim(0, 0.8) 
ax5.set_title('F) Quantitative Metrics', fontsize=16)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

save_path = os.path.join(output_dir, "simulation.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Final enhanced visualization saved to {save_path}")
plt.close(fig)