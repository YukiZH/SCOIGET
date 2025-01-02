import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear
from hmmlearn import hmm
import numpy as np
from torch_scatter import scatter_mean


# Encoder 模块
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


# Decoder 模块
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


# CNEncoder 模块（增加edge_index作为参数，并对HMM states进行空间平滑）
class CNEncoder(nn.Module):
    def __init__(self, original_dim, num_states=3, max_iters=20):
        super(CNEncoder, self).__init__()
        self.num_states = num_states
        self.max_iters = max_iters

    def forward(self, inputs, edge_index):
        # inputs: [norm_x, reconstructed_features]
        norm_x, reconstructed_features = inputs
        device = reconstructed_features.device

        norm_x = norm_x.to(device)
        reconstructed_features = reconstructed_features.to(device)

        # HMM建模
        spot_mean = reconstructed_features.mean(dim=1, keepdim=True).detach().cpu().numpy()
        spot_mean = spot_mean.astype(np.float64)
        spot_mean = (spot_mean - spot_mean.mean()) / (spot_mean.std() + 1e-8)

        hmm_model = hmm.GaussianHMM(
            n_components=self.num_states,
            covariance_type="diag",
            n_iter=500,
            init_params=""  # 禁用自动初始化
        )

        t = 0.01
        hmm_model.startprob_ = np.array([0.1, 0.8, 0.1], dtype=np.float64)
        hmm_model.transmat_ = np.array([
            [1 - 2 * t, t, t],
            [t, 1 - 2 * t, t],
            [t, t, 1 - 2 * t]
        ], dtype=np.float64)

        quantiles = np.quantile(spot_mean, [0.2, 0.5, 0.8])
        hmm_model.means_ = quantiles.reshape(-1, 1).astype(np.float64)
        epsilon = 1e-4
        hmm_model.covars_ = np.full((self.num_states, 1), np.var(spot_mean) + epsilon, dtype=np.float64)

        prev_log_likelihood = -np.inf
        for i in range(self.max_iters):
            try:
                hmm_model.fit(spot_mean)
                states = hmm_model.predict(spot_mean)
                log_likelihood = hmm_model.score(spot_mean)

                if np.abs(log_likelihood - prev_log_likelihood) < 1e-3:
                    print(f"Converged at iteration {i}")
                    break
                prev_log_likelihood = log_likelihood
            except Exception as e:
                print("HMM training encountered an error:", e)
                states = np.ones((spot_mean.shape[0],), dtype=int)

            # M 步骤更新HMM参数
            hmm_model.means_ = np.array([
                np.mean(spot_mean[states == s], axis=0) if np.any(states == s) else np.mean(spot_mean, axis=0)
                for s in range(self.num_states)
            ], dtype=np.float64).reshape(-1, 1)

            hmm_model.covars_ = np.array([
                np.var(spot_mean[states == s], axis=0) + epsilon if np.any(states == s) else np.var(spot_mean, axis=0) + epsilon
                for s in range(self.num_states)
            ], dtype=np.float64).reshape(-1, 1)

        # 状态张量
        states_tensor = torch.tensor(states + 1, dtype=torch.float32, device=device).reshape(-1, 1)

        # 对states进行空间平滑处理
        # edge_index: [2, E], row和col分别是边的起点和终点
        row, col = edge_index
        states_tensor_per_node = states_tensor.squeeze(-1)  # [N]

        # 对每个节点, 根据邻居节点的状态求平均
        neighbor_avg_states = scatter_mean(states_tensor_per_node[row], col, dim=0, dim_size=states_tensor.size(0))
        # 将节点状态与其邻居平均状态做混合(0.5:0.5比重)
        smoothed_states = 0.5 * states_tensor_per_node + 0.5 * neighbor_avg_states
        smoothed_states = smoothed_states.unsqueeze(-1)  # [N, 1]

        # 后续用平滑后的states计算拷贝数
        state_adjusted_copy = reconstructed_features * smoothed_states
        copy_sum = state_adjusted_copy.sum(dim=1, keepdim=True)

        # 归一化
        pseudo_sum = norm_x.sum(dim=1, keepdim=True)
        norm_copy = state_adjusted_copy / (copy_sum + 1e-8) * pseudo_sum

        # 动态范围缩放
        range_min, range_max = norm_copy.min().item() * 0.8, norm_copy.max().item() * 1.2
        norm_copy = (norm_copy - norm_copy.min()) / (norm_copy.max() - norm_copy.min() + 1e-8)
        norm_copy = norm_copy * (range_max - range_min) + range_min

        # 调整均值到 1
        current_mean = norm_copy.mean()
        norm_copy = norm_copy / current_mean
        
        # 正则化损失
        reg_loss = torch.sum(reconstructed_features ** 2) * 1e-4

        return norm_copy, reg_loss


# SCOIGET 模型
class SCOIGET(nn.Module):
    def __init__(self, original_dim, intermediate_dim=128, latent_dim=32, max_cp=25, kl_weights=0.5, dropout=0.0, hmm_states=3, max_iters=20, gnn_heads=8, lambda_smooth=0.1, device='cpu'):
        super(SCOIGET, self).__init__()
        self.device = device
        self.max_cp = max_cp
        self.kl_weights = kl_weights
        self.lambda_smooth = lambda_smooth  # 空间平滑正则项系数

        self.z_encoder = Encoder(original_dim, intermediate_dim, latent_dim, heads=gnn_heads, dropout=dropout).to(self.device)
        self.decoder = Decoder(latent_dim, intermediate_dim, original_dim).to(self.device)
        self.encoder = CNEncoder(original_dim, num_states=hmm_states, max_iters=max_iters).to(self.device)

    def forward(self, inputs, edge_index, edge_attr):
        inputs = inputs.to(self.device)
        z_mean, z_var, z = self.z_encoder(inputs, edge_index)
        reconstructed_features = self.decoder(z)
        
        # 传入edge_index进行CNEncoder的计算，以对states进行空间平滑
        norm_copy, reg_loss = self.encoder([inputs, reconstructed_features], edge_index)

        # KL 散度损失
        p_dis = torch.distributions.Normal(loc=z_mean, scale=torch.sqrt(z_var))
        q_dis = torch.distributions.Normal(loc=torch.zeros_like(z_mean), scale=torch.ones_like(z_var))
        kl_loss = torch.sum(torch.distributions.kl_divergence(p_dis, q_dis), dim=1) * self.kl_weights

        return norm_copy, reconstructed_features, kl_loss, reg_loss

    def train_step(self, data, edge_index, edge_attr, optimizer):
        self.train()
        optimizer.zero_grad()
        norm_copy, reconstructed_features, kl_loss, reg_loss = self(data, edge_index, edge_attr)
        recon_loss = F.mse_loss(reconstructed_features, data, reduction='mean')

        # 添加空间平滑损失项
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
            
            # 验证集上同样计算空间平滑但不反传
            row, col = edge_index
            spatial_smooth_loss = F.mse_loss(norm_copy[row], norm_copy[col])
            
            loss = recon_loss + kl_loss.mean() + reg_loss + self.lambda_smooth * spatial_smooth_loss
        return loss.item()