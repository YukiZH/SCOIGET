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

        # 数据预处理
        spot_mean = reconstructed_features.mean(dim=1, keepdim=True).detach().cpu().numpy()
        spot_mean = (spot_mean - spot_mean.mean()) / (spot_mean.std() + 1e-8)

        # 初始化 MCMC 参数
        num_samples = spot_mean.shape[0]
        states = np.random.choice(self.num_states, size=num_samples)  # 初始状态
        state_probs = np.zeros((num_samples, self.num_states))  # 状态后验概率

        # 定义先验概率和似然函数
        state_prior = np.array([0.1, 0.8, 0.1])  # 初始状态分布
        transition_matrix = np.array([
            [0.98, 0.01, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98]
        ])

        def likelihood(state, mean):
            mean = mean.detach().cpu().numpy()  # 使用 detach 分离梯度
            diff = (spot_mean - mean[state]) ** 2
            return np.exp(-0.5 * diff)

        # MCMC 采样
        for iter_idx in range(self.max_iters):
            for i in range(num_samples):
                current_state = states[i]
                candidate_state = np.random.choice(self.num_states)  # 新状态

                # 计算接受概率
                prior_ratio = state_prior[candidate_state] / state_prior[current_state]
                likelihood_ratio = np.sum(likelihood(candidate_state, reconstructed_features.mean(dim=1))) / \
                                   np.sum(likelihood(current_state, reconstructed_features.mean(dim=1)))
                transition_ratio = transition_matrix[current_state, candidate_state] / \
                                   transition_matrix[candidate_state, current_state]

                acceptance_prob = prior_ratio * likelihood_ratio * transition_ratio

                # 确保接受概率是标量
                if np.random.rand() < float(acceptance_prob):
                    states[i] = candidate_state  # 接受新状态

                state_probs[i, states[i]] += 1  # 更新状态概率分布

        # 平滑后验概率
        row, col = edge_index
        state_probs_tensor = torch.tensor(state_probs, device=device, dtype=torch.float32)
        smoothed_probs = scatter_mean(state_probs_tensor[row], col, dim=0)
        final_states = smoothed_probs.argmax(dim=1).cpu().numpy()

        # 状态张量
        states_tensor = torch.tensor(final_states + 1, dtype=torch.float32, device=device).reshape(-1, 1)

        # 用状态调整拷贝数
        state_adjusted_copy = reconstructed_features * states_tensor
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
