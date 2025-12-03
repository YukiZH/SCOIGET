# SCOIGET Stability Test Script
# Based on: simulation_experiment.py
# Purpose: Run 5 independent trials with different random seeds to calculate Mean +/- STD of MSE and Cosine Similarity.

import os
import sys
import random
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- 1. Project Setup and Environment ---

# Adjust this path to your actual project location
PROJECT_ROOT = "/export/home/zhangyujia/SCOIGET"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Attempt to import necessary modules
try:
    import pyensembl
    from scoiget import preprocess_utils as pp
    from scoiget import cnv_utils as cu
    from scoiget import train_utils as tu
    from scoiget import scoiget_model as sm
    
    # Import simulation functions from your versioned script
    # Ensure 'simulation_experiment.py' is in the same directory or python path
    from simulation_experiment import (
        create_simulation_adata, 
        simple_binning_function, 
        calculate_graph_and_probabilities
    )
except ImportError as e:
    print(f"Warning: Could not import some custom modules. Error: {e}")
    print("Please ensure 'simulation_experiment.py' and the 'scoiget' package are in the path.")

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- 2. Core Utility Functions & Model Definitions ---

def set_seed(seed):
    """
    Sets all random seeds to ensure reproducibility for a single run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
    """
    Gumbel-Softmax sampling for differentiable discrete transitions.
    """
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

class DifferentiableHMM_Centered(nn.Module):
    """
    HMM Layer with centered initialization for CNV detection.
    """
    def __init__(self, num_states=3):
        super(DifferentiableHMM_Centered, self).__init__()
        self.num_states = num_states
        # Initialize means centered around 0 (e.g., loss, neutral, gain)
        self.state_means = nn.Parameter(torch.tensor([-1.0, 1.0]), requires_grad=True) 
        self.log_stds = nn.Parameter(torch.tensor([-1.0, -1.0, -1.0]), requires_grad=True)
        # Transition probabilities (favors staying in the same state)
        self.transition_logits = nn.Parameter(torch.tensor([
            [6.0, 1.0, 1.0], 
            [1.0, 6.0, 1.0], 
            [1.0, 1.0, 6.0]
        ]), requires_grad=True)
        self.start_logits = nn.Parameter(torch.tensor([1.0, 5.0, 1.0]), requires_grad=True)

    def forward(self, x, chrom_list, edge_index, lambda_smooth=0.1):
        n_spots, n_bins = x.shape
        # Construct full state means: [Loss, Neutral(0), Gain]
        all_means = torch.stack([self.state_means[0], torch.tensor(0.0, device=device), self.state_means[1]])
        stds = torch.exp(self.log_stds)
        
        # Emission probabilities
        dist = torch.distributions.Normal(all_means, stds + 1e-6)
        log_emission_probs = dist.log_prob(x.unsqueeze(-1))
        
        all_states = []
        
        # Process per chromosome
        for (start_bin, end_bin) in chrom_list:
            chrom_len = end_bin - start_bin
            if chrom_len == 0: continue
            chrom_emissions = log_emission_probs[:, start_bin:end_bin, :]
            
            # Use Gumbel Softmax for differentiable state selection
            states_soft = gumbel_softmax(chrom_emissions, tau=0.1, hard=True, dim=-1)
            all_states.append(states_soft)

        if not all_states: 
            return torch.zeros_like(x), torch.tensor(0.0, device=device)

        states_one_hot = torch.cat(all_states, dim=1)
        norm_copy = states_one_hot @ all_means
        
        # Spatial smoothness loss via graph edges
        row, col = edge_index
        spatial_smooth_loss = F.mse_loss(states_one_hot[row], states_one_hot[col])
        
        return norm_copy, lambda_smooth * spatial_smooth_loss

class SCOIGET_Corrected_Centered(nn.Module):
    """
    Main Model: Autoencoder + Differentiable HMM + Spatial Regularization.
    """
    def __init__(self, in_channels, hidden_channels=128, out_channels=32, hmm_states=3, 
                 kl_weight=0.05, lambda_smooth=3.0, lambda_hmm_fit=1.0): 
        super(SCOIGET_Corrected_Centered, self).__init__()
        self.kl_weight = kl_weight
        self.lambda_smooth = lambda_smooth
        self.lambda_hmm_fit = lambda_hmm_fit
        
        # Encoder/Decoder from scoiget_model (sm)
        self.z_encoder = sm.Encoder(in_channels, hidden_channels, out_channels).to(device)
        self.decoder = sm.Decoder(out_channels, hidden_channels, in_channels).to(device)
        self.hmm = DifferentiableHMM_Centered(num_states=hmm_states).to(device)
        
    def forward(self, x, edge_index, chrom_list):
        # VAE Pass
        z_mean, z_var, z = self.z_encoder(x, edge_index)
        reconstructed_features = self.decoder(z)
        
        # HMM Pass
        norm_copy, spatial_smooth_loss = self.hmm(reconstructed_features, chrom_list, edge_index, self.lambda_smooth)
        
        # Losses
        p_dis = torch.distributions.Normal(loc=z_mean, scale=torch.sqrt(z_var))
        q_dis = torch.distributions.Normal(loc=torch.zeros_like(z_mean), scale=torch.ones_like(z_var))
        kl_loss = torch.sum(torch.distributions.kl_divergence(p_dis, q_dis), dim=1).mean() * self.kl_weight
        
        recon_loss = F.mse_loss(reconstructed_features, x, reduction='mean') 
        hmm_recon_loss = F.mse_loss(norm_copy, reconstructed_features.detach())
        
        total_loss = recon_loss + kl_loss + spatial_smooth_loss + self.lambda_hmm_fit * hmm_recon_loss
        return norm_copy, total_loss

    def get_cnv(self, x, edge_index, chrom_list):
        self.eval()
        with torch.no_grad():
            z_mean, z_var, z = self.z_encoder(x, edge_index)
            reconstructed_features = self.decoder(z)
            norm_copy, _ = self.hmm(reconstructed_features, chrom_list, edge_index)
        return norm_copy


# --- 3. Single Experiment Execution Wrapper ---

def run_single_trial(seed, adata_base, chrom_list, gt_cnv_scores):
    """
    Executes a complete training and evaluation trial for a specific random seed.
    
    Args:
        seed (int): Random seed.
        adata_base (AnnData): The pre-generated dataset (raw/log1p).
        chrom_list (list): Chromosome boundaries.
        gt_cnv_scores (np.array): Ground truth for evaluation.
        
    Returns:
        tuple: (mse, cos_sim)
    """
    print(f"\n=== Starting Trial with Seed: {seed} ===")
    
    # 1. Set Seed (Controls PCA, Graph construction, and Neural Net Init)
    set_seed(seed)
    
    # 2. Data Preparation
    # We deep copy to ensure no tensors leak between trials
    adata = adata_base.copy() 
    
    # Note: To rigorously test stability, we re-run the stochastic parts of preprocessing
    # (PCA and Neighbors) under the specific seed.
    try:
        # If 'feat' doesn't exist or we want to re-scale
        binned_matrix = adata.uns['binned_data'].X
        feat_raw = binned_matrix.toarray() if scipy.sparse.issparse(binned_matrix) else binned_matrix
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_raw)
        adata.obsm['feat'] = feat_scaled
    except KeyError:
        print("Error: 'binned_data' not found in adata.uns. Check data generation.")
        return None, None

    # Prepare PyG Data object
    data = tu.prepare_data(adata, use_norm_x=False) 
    data = data.to(device)
    
    # 3. Initialize Model
    model = SCOIGET_Corrected_Centered(
        in_channels=data.num_features,
        hidden_channels=128,
        out_channels=32,
        hmm_states=3,
        kl_weight=0.05, 
        lambda_smooth=3.0, 
        lambda_hmm_fit=1.0
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 300
    
    # 4. Training Loop (Silent mode to avoid clutter)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        _, loss = model(data.x, data.edge_index, chrom_list)
        loss.backward()
        optimizer.step()
        
    # 5. Inference
    pred_cnv_raw = model.get_cnv(data.x, data.edge_index, chrom_list).cpu().numpy()
    
    # 6. Evaluation Logic
    # Center predictions based on normal cells (Clone 0)
    try:
        norm_mask = (adata.obs['ground_truth_clone'].astype(int) == 0).values
        if np.sum(norm_mask) > 0:
            baseline = np.median(pred_cnv_raw[norm_mask], axis=0)
            pred_cnv_score = pred_cnv_raw - baseline
        else:
            pred_cnv_score = pred_cnv_raw
    except KeyError:
        # Fallback if ground_truth_clone column is missing
        pred_cnv_score = pred_cnv_raw

    # Clip values to match common evaluation ranges (e.g., -1.5 to 1.5)
    pred_cnv_clipped = np.clip(pred_cnv_score, -1.5, 1.5)
    
    # Calculate Metrics
    mse = mean_squared_error(gt_cnv_scores.ravel(), pred_cnv_clipped.ravel())
    # Add epsilon to avoid division by zero in cosine similarity
    cos_sim = 1 - cosine(gt_cnv_scores.ravel() + 1e-9, pred_cnv_clipped.ravel() + 1e-9)
    
    print(f"-> Seed {seed} Result | MSE: {mse:.5f}, Cosine Sim: {cos_sim:.5f}")
    return mse, cos_sim


# --- 4. Main Execution Block ---

if __name__ == "__main__":
    print("--- SCOIGET Stability Test Initialization ---")
    
    # --- A. Data Generation (Fixed Step) ---
    print("1. Generating Simulation Data...")
    
    # Initialize Ensembl
    try:
        ensembl_data = pyensembl.EnsemblRelease(98)
    except Exception as e:
        print(f"Ensembl Error: {e}. Proceeding, but binning might fail if required.")

    # Generate Synthetic Data
    adata_base, selected_genes = create_simulation_adata(
        n_genes=6000, grid_size=40, gain_factor=3.0, loss_factor=0.1, 
        noise_level=0.5, dropout_rate=0.3
    )
    
    # Basic Preprocessing
    sc.pp.filter_cells(adata_base, min_counts=5)
    sc.pp.filter_genes(adata_base, min_cells=1) 
    sc.pp.normalize_total(adata_base)
    sc.pp.log1p(adata_base)
    
    # Add Genomic Locations & Binning
    adata_base = cu.add_genomic_locations(adata_base)
    adata_base, chrom_list = simple_binning_function(adata_base, bin_size=25, ensembl_obj=ensembl_data)
    
    # Initial Graph Construction (This structure is needed for GT alignment, 
    # though we re-calculate edges inside the loop for strict testing)
    adata_base = calculate_graph_and_probabilities(adata_base, use_norm_x=False, n_neighbors=8)
    
    
    # --- B. Ground Truth Reconstruction (Fixed Step) ---
    print("2. Reconstructing Ground Truth for Evaluation...")
    
    # Map genes to locations
    gt_var_df = pd.DataFrame({
        'chromosome_orig': [g.contig for g in selected_genes], 
        'start': [g.start for g in selected_genes],
    }, index=[g.gene_name for g in selected_genes])
    
    gt_var_df.index = gt_var_df.index.astype(str)
    # Remove duplicates
    if gt_var_df.index.duplicated().any():
        gt_var_df = gt_var_df.reset_index().drop_duplicates(subset='index').set_index('index')
        
    # Align with adata variables
    common_genes_gt = gt_var_df.index.intersection(adata_base.var_names)
    adata_eval = adata_base[:, common_genes_gt].copy()
    gt_var_df = gt_var_df.loc[common_genes_gt]
    gt_var_df = gt_var_df.join(adata_eval.var[['chromosome', 'start', 'sort_key']], rsuffix='_adata')
    gt_var_df = gt_var_df.sort_values('sort_key')
    adata_eval = adata_eval[:, gt_var_df.index].copy()
    
    # Construct Full CNV Matrix
    gt_cnv_matrix_full = np.ones((adata_eval.n_obs, adata_eval.n_vars), dtype=np.float32)
    
    # Apply synthetic CNVs (Chr2 Gain, Chr7 Loss)
    genes_chr2_mask = (gt_var_df['chromosome_orig'] == '2').values
    spots_clone_a_mask = (adata_eval.obs['ground_truth_clone'].astype(int) == 1).values
    gt_cnv_matrix_full[np.ix_(spots_clone_a_mask, genes_chr2_mask)] = 2.0 
    
    genes_chr7_mask = (gt_var_df['chromosome_orig'] == '7').values
    spots_clone_b_mask = (adata_eval.obs['ground_truth_clone'].astype(int) == 2).values
    gt_cnv_matrix_full[np.ix_(spots_clone_b_mask, genes_chr7_mask)] = 0.0 

    # Binning the Ground Truth
    binned_gt_list = []
    sorted_chromosomes_gt = adata_eval.var['chromosome'].unique()
    BIN_SIZE = 25
    
    for chrom in sorted_chromosomes_gt:
        genes_on_chrom_mask = (adata_eval.var['chromosome'] == chrom).values
        n_genes_chrom = np.sum(genes_on_chrom_mask)
        n_bins = n_genes_chrom // BIN_SIZE
        if n_bins == 0: continue
        
        gt_matrix_chrom = gt_cnv_matrix_full[:, genes_on_chrom_mask][:, :n_bins * BIN_SIZE]
        gt_binned = gt_matrix_chrom.reshape(adata_eval.n_obs, n_bins, BIN_SIZE).mean(axis=2)
        binned_gt_list.append(gt_binned)
        
    gt_cnv_aligned = np.concatenate(binned_gt_list, axis=1)
    
    # Normalize GT relative to normal cells
    norm_mask_gt = (adata_eval.obs['ground_truth_clone'].astype(int) == 0).values
    baseline_gt = np.median(gt_cnv_aligned[norm_mask_gt], axis=0)
    gt_cnv_scores = gt_cnv_aligned - baseline_gt

    print(f"Ground Truth Shape: {gt_cnv_scores.shape}")

    
    # --- C. Run Stability Test Loop ---
    SEEDS = [42, 123, 2024, 7, 99]
    results_mse = []
    results_cos = []
    
    print(f"\n--- Starting Stability Test across {len(SEEDS)} seeds ---")
    
    for seed in SEEDS:
        mse, cos = run_single_trial(seed, adata_base, chrom_list, gt_cnv_scores)
        if mse is not None:
            results_mse.append(mse)
            results_cos.append(cos)
        
    # --- D. Final Statistics ---
    mean_mse = np.mean(results_mse)
    std_mse = np.std(results_mse)
    mean_cos = np.mean(results_cos)
    std_cos = np.std(results_cos)
    
    print("\n" + "="*45)
    print("STABILITY TEST FINAL REPORT")
    print("="*45)
    print(f"Seeds tested: {SEEDS}")
    print("-" * 45)
    print(f"Metric             | Mean      | Std Dev")
    print("-" * 45)
    print(f"MSE                | {mean_mse:.5f}   | {std_mse:.5f}")
    print(f"Cosine Similarity  | {mean_cos:.5f}   | {std_cos:.5f}")
    print("="*45)