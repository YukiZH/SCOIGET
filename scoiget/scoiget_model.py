import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear
from hmmlearn import hmm
import numpy as np
from torch_scatter import scatter_mean


# Encoder Module
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.0):
        super(Encoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
        self.conv3 = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout)
        self.dense_mean = nn.Linear(out_channels, out_channels)
        self.dense_var = nn.Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        nn.init.xavier_uniform_(self.dense_mean.weight)
        nn.init.xavier_uniform_(self.dense_var.weight)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        z_mean = self.dense_mean(x)
        z_var = torch.exp(self.dense_var(x))
        z_var = torch.clamp(z_var, min=1e-8, max=1e2)

        return z_mean, z_var, x


# Decoder Module
class Decoder(nn.Module):
    def __init__(self, latent_dim, intermediate_dim, original_dim):
        super(Decoder, self).__init__()
        self.decoder_nn = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, original_dim)
        )

    def forward(self, x):
        return self.decoder_nn(x)


# CNEncoder Module (with edge_index as a parameter and spatial smoothing for HMM states)
import torch
import numpy as np
from torch_scatter import scatter_mean
from torch import nn

class CNEncoder(nn.Module):
    def __init__(self, original_dim, num_states=3, max_iters=20):
        super(CNEncoder, self).__init__()
        self.num_states = num_states
        self.max_iters = max_iters

    def forward(self, inputs, edge_index):
        norm_x, reconstructed_features = inputs
        device = reconstructed_features.device

        norm_x = norm_x.to(device)
        reconstructed_features = reconstructed_features.to(device)

        # Data preprocessing
        spot_mean = reconstructed_features.mean(dim=1, keepdim=True).detach().cpu().numpy()
        spot_mean = (spot_mean - spot_mean.mean()) / (spot_mean.std() + 1e-8)

        # Initialize MCMC parameters
        num_samples = spot_mean.shape[0]
        states = np.random.choice(self.num_states, size=num_samples)  # Initial states
        state_probs = np.zeros((num_samples, self.num_states))  # State posterior probabilities

        # Define prior probabilities and likelihood function
        state_prior = np.array([0.1, 0.8, 0.1])  # Initial state distribution
        transition_matrix = np.array([
            [0.98, 0.01, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98]
        ])

        def likelihood(state, mean):
            mean = mean.detach().cpu().numpy()  # Detach gradients
            diff = (spot_mean - mean[state]) ** 2
            return np.exp(-0.5 * diff)

        # MCMC sampling
        for iter_idx in range(self.max_iters):
            for i in range(num_samples):
                current_state = states[i]
                candidate_state = np.random.choice(self.num_states)  # New state

                # Compute acceptance probability
                prior_ratio = state_prior[candidate_state] / state_prior[current_state]
                likelihood_ratio = np.sum(likelihood(candidate_state, reconstructed_features.mean(dim=1))) / \
                                   np.sum(likelihood(current_state, reconstructed_features.mean(dim=1)))
                transition_ratio = transition_matrix[current_state, candidate_state] / \
                                   transition_matrix[candidate_state, current_state]

                acceptance_prob = prior_ratio * likelihood_ratio * transition_ratio

                # Ensure acceptance probability is a scalar
                if np.random.rand() < float(acceptance_prob):
                    states[i] = candidate_state  # Accept new state

                state_probs[i, states[i]] += 1  # Update state probability distribution

        # Smooth posterior probabilities
        row, col = edge_index
        state_probs_tensor = torch.tensor(state_probs, device=device, dtype=torch.float32)
        smoothed_probs = scatter_mean(state_probs_tensor[row], col, dim=0)
        final_states = smoothed_probs.argmax(dim=1).cpu().numpy()

        # State tensor
        states_tensor = torch.tensor(final_states + 1, dtype=torch.float32, device=device).reshape(-1, 1)

        # Adjust copy number with states
        state_adjusted_copy = reconstructed_features * states_tensor
        copy_sum = state_adjusted_copy.sum(dim=1, keepdim=True)

        # Normalization
        pseudo_sum = norm_x.sum(dim=1, keepdim=True)
        norm_copy = state_adjusted_copy / (copy_sum + 1e-8) * pseudo_sum

        # Dynamic range scaling
        range_min, range_max = norm_copy.min().item() * 0.8, norm_copy.max().item() * 1.2
        norm_copy = (norm_copy - norm_copy.min()) / (norm_copy.max() - norm_copy.min() + 1e-8)
        norm_copy = norm_copy * (range_max - range_min) + range_min

        # Adjust mean to 1
        current_mean = norm_copy.mean()
        norm_copy = norm_copy / current_mean

        # Regularization loss
        reg_loss = torch.sum(reconstructed_features ** 2) * 1e-4

        return norm_copy, reg_loss


# SCOIGET Model
class SCOIGET(nn.Module):
    def __init__(self, original_dim, intermediate_dim=128, latent_dim=32, max_cp=25, kl_weights=0.5, dropout=0.0, hmm_states=3, max_iters=20, gnn_heads=8, lambda_smooth=0.1, device='cpu'):
        super(SCOIGET, self).__init__()
        self.device = device
        self.max_cp = max_cp
        self.kl_weights = kl_weights
        self.lambda_smooth = lambda_smooth  # Spatial smoothing regularization coefficient

        self.z_encoder = Encoder(original_dim, intermediate_dim, latent_dim, heads=gnn_heads, dropout=dropout).to(self.device)
        self.decoder = Decoder(latent_dim, intermediate_dim, original_dim).to(self.device)
        self.encoder = CNEncoder(original_dim, num_states=hmm_states, max_iters=max_iters).to(self.device)

    def forward(self, inputs, edge_index, edge_attr):
        inputs = inputs.to(self.device)
        z_mean, z_var, z = self.z_encoder(inputs, edge_index)
        reconstructed_features = self.decoder(z)
        
        # Pass edge_index to CNEncoder for spatial smoothing of states
        norm_copy, reg_loss = self.encoder([inputs, reconstructed_features], edge_index)

        # KL divergence loss
        p_dis = torch.distributions.Normal(loc=z_mean, scale=torch.sqrt(z_var))
        q_dis = torch.distributions.Normal(loc=torch.zeros_like(z_mean), scale=torch.ones_like(z_var))
        kl_loss = torch.sum(torch.distributions.kl_divergence(p_dis, q_dis), dim=1) * self.kl_weights

        return norm_copy, reconstructed_features, kl_loss, reg_loss

    def train_step(self, data, edge_index, edge_attr, optimizer):
        self.train()
        optimizer.zero_grad()
        norm_copy, reconstructed_features, kl_loss, reg_loss = self(data, edge_index, edge_attr)
        recon_loss = F.mse_loss(reconstructed_features, data, reduction='mean')

        # Add spatial smoothing loss term
        row, col = edge_index
        spatial_smooth_loss = F.mse_loss(norm_copy[row], norm_copy[col])

        loss = recon_loss + kl_loss.mean() + reg_loss + self.lambda_smooth * spatial_smooth_loss
        loss.backward()
        optimizer.step()
        return loss.item()

    def validation_step(self, data, edge_index, edge_attr):
        self.eval()
        with torch.no_grad():
            norm_copy, reconstructed_features, kl_loss, reg_loss = self(data, edge_index, edge_attr)
            recon_loss = F.mse_loss(reconstructed_features, data, reduction='mean')
            
            # Compute spatial smoothing on validation set without backpropagation
            row, col = edge_index
            spatial_smooth_loss = F.mse_loss(norm_copy[row], norm_copy[col])
            
            loss = recon_loss + kl_loss.mean() + reg_loss + self.lambda_smooth * spatial_smooth_loss
        return loss.item()
