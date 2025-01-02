
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import seaborn as sns


def split_data_by_cluster(data, pred_label):
    """
    Split data by unique values in pred_label without excluding any clusters.
    Args:
        data (np.ndarray): Data array with shape (n_samples, n_features).
        pred_label (np.ndarray): Cluster labels for each sample.
    Returns:
        dict: Dictionary where keys are unique labels and values are corresponding subarrays.
    """
    unique_labels = np.unique(pred_label)
    split_data = {}
    for label in unique_labels:
        mask = (pred_label == label)
        split_data[label] = data[mask]
    return split_data


def pelt_multi(signal, penalty):
    """
    Perform segmentation using the PELT algorithm.
    Args:
        signal (np.ndarray or torch.Tensor): Input signal to be segmented.
        penalty (float): Penalty term for adding a new segment.
    Returns:
        torch.Tensor: Detected breakpoints.
    """
    n, m = signal.shape
    cost = torch.zeros(n + 1, device=signal.device)
    last_change = torch.zeros(n + 1, dtype=torch.int64, device=signal.device)
    seg_end = torch.zeros(n + 1, dtype=torch.int64, device=signal.device)
    for t in range(1, n + 1):
        min_cost = float('inf')
        for s in range(t):
            segment = signal[s:t]
            rss = torch.sum((segment - segment.mean(dim=0)) ** 2)
            new_cost = cost[s] + rss + penalty
            if new_cost < min_cost:
                min_cost = new_cost
                last_change[t] = s
                seg_end[t] = t
        cost[t] = min_cost
    breakpoints = []
    t = n
    while t > 0:
        breakpoints.append(t)
        t = last_change[t]
    breakpoints.reverse()
    return torch.tensor(breakpoints[:-1], device=signal.device)


def perform_segmentation(clone_cn, chrom_list, eta=1, use_gpu=True):
    start_time = time.time()
    if isinstance(clone_cn, np.ndarray):
        clone_cn = torch.tensor(clone_cn, dtype=torch.float32)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    clone_cn = clone_cn.to(device)
    print(f'bin_cp shape: {clone_cn.shape}')
    print(f'chrom_list: {chrom_list}')
    k = clone_cn.shape[0]
    n = clone_cn.shape[1]
    beta = eta * k * np.log(n)
    bp_arr = torch.tensor([], dtype=torch.int64, device=device)
    for idx, (start_bin, end_bin) in enumerate(tqdm(chrom_list, desc="Segmentation Progress")):
        if start_bin >= n:
            print(f'Skipping out of range: start_bin={start_bin}, end_bin={end_bin}, n={n}')
            continue
        if end_bin > n:
            end_bin = n
        chrom_bps = pelt_multi(clone_cn[:, start_bin:end_bin], beta)
        bps = chrom_bps + start_bin
        bp_arr = torch.cat((bp_arr, bps))
    bp_arr = bp_arr.cpu().numpy()
    print(f"{len(bp_arr)} breakpoints are detected.")
    seg_array = torch.zeros_like(clone_cn, device=device)
    for cell in range(k):
        st = 0
        end_p = None
        for I in range(len(bp_arr) - 1):
            start_p = int(bp_arr[I])
            end_p = int(bp_arr[I + 1])
            if start_p < 0 or end_p > clone_cn.shape[1]:
                print(f'Skipping invalid segment: start_p={start_p}, end_p={end_p}, max_index={clone_cn.shape[1]}')
                continue
            if end_p > start_p:
                seg_array[cell, start_p:end_p] = torch.median(clone_cn[cell, start_p:end_p])
        if end_p is not None and end_p < clone_cn.shape[1]:
            seg_array[cell, end_p:] = torch.median(clone_cn[cell, end_p:])
    segment_cn = torch.mean(seg_array, dim=0).cpu().numpy()
    seg_array = seg_array.cpu().numpy()
    total_time = time.time() - start_time
    print(f'Total segmentation time: {total_time:.2f} seconds')
    return segment_cn, bp_arr, seg_array



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_heatmap_and_segmentation(cluster_dict, chrom_list, output_dir, bin_size, library_id, segmentation_results, method):
    """
    Plot heatmap and segmentation results for all clusters and save the images.
    Args:
        cluster_dict (dict): Dictionary containing cluster data.
        chrom_list (list): List of chromosome boundary bins.
        output_dir (str): Output directory to save the images.
        bin_size (int): Size of each bin.
        library_id (str): Library ID for naming the files.
        segmentation_results (dict): Dictionary containing segmentation results.
        method (str): Clustering method used.
    """
    # 组合所有簇的分段拷贝数数据
    heatmap_data = []
    for label, (segment_cn, _, _) in segmentation_results.items():
        heatmap_data.append(segment_cn)
    heatmap_data = np.vstack(heatmap_data)
    
    # 绘制热图
    plt.figure(figsize=(20, 4))
    sns.heatmap(heatmap_data, cmap="coolwarm", cbar=True, cbar_kws={"label": "Copy Number Variation"})
    
    # 设置染色体标签
    chrom_labels = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    chrom_positions = [0] + [end for _, end in chrom_list]
    chrom_midpoints = [(chrom_positions[i] + chrom_positions[i+1]) // 2 for i in range(len(chrom_positions) - 1)]
    
    plt.xticks(chrom_midpoints, chrom_labels, rotation=90, fontsize=10)
    plt.xlabel("Chromosome", fontsize=14)
    plt.ylabel("Cluster", fontsize=14)
    
    # 添加染色体分割线（增强效果）
    for pos in chrom_positions:
        plt.axvline(x=pos, color='black', linestyle='-', linewidth=1.5)
    
    # 添加标题并优化布局
    plt.title(f"Copy Number Profile Heatmap by Cluster ({library_id})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存热图
    heatmap_path = os.path.join(output_dir, f"{library_id}_{method}_segmentation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.show()

    # 为每个簇绘制分段结果
    for label, (segment_cn, bp_arr, sp_cn) in segmentation_results.items():
        save_path = os.path.join(output_dir, f"{library_id}_{method}_segmentation_result_{label}.png")
        plot_segmentation(cluster_dict[label], segment_cn, bp_arr, chrom_list, save_path)

def plot_segmentation(clone_cn, segment_cn, bp_arr, chrom_list, save_path):
    """
    Plot and save the segmentation result.
    Args:
        clone_cn (torch.Tensor or np.ndarray): Original copy number data.
        segment_cn (np.ndarray): Segmented copy number data.
        bp_arr (np.ndarray): Detected breakpoints.
        chrom_list (list): List of chromosome boundary bins.
        save_path (str): Path to save the segmentation result image.
    """
    plt.figure(figsize=(20, 5))
    plt.plot(clone_cn[0], label='Original Signal', color="skyblue")  # Assuming clone_cn has at least one sample
    for bkp in bp_arr:
        plt.axvline(x=bkp, color='red', linestyle='--')
    plt.plot(segment_cn, label='Segmented Signal', linestyle='-', color="darkorange")
    
    # Draw chromosome boundaries
    chrom_labels = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    chrom_midpoints = [(start_bin + end_bin) // 2 for start_bin, end_bin in chrom_list]
    for j, (start_bin, end_bin) in enumerate(chrom_list):
        plt.axvline(x=start_bin, color='black', linestyle='-', linewidth=1.5)
        plt.text(chrom_midpoints[j], np.max(clone_cn[0]), chrom_labels[j], 
                 horizontalalignment='center', verticalalignment='bottom', fontsize=10, fontweight="bold")
    
    plt.title('Segmentation Result with Original Signal', fontsize=14)
    plt.xlabel('Genomic position (bins)', fontsize=12)
    plt.ylabel('Copy Number', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_with_original.png"), dpi=300)
    plt.show()

    # 再绘制一个不包含原始信号的分段结果图
    plt.figure(figsize=(20, 5))
    for bkp in bp_arr:
        plt.axvline(x=bkp, color='red', linestyle='--')
    plt.plot(segment_cn, label='Segmented Signal', linestyle='-', color="darkorange")
    
    # Draw chromosome boundaries
    for j, (start_bin, end_bin) in enumerate(chrom_list):
        plt.axvline(x=start_bin, color='black', linestyle='-', linewidth=1.5)
        plt.text(chrom_midpoints[j], np.max(segment_cn), chrom_labels[j], 
                 horizontalalignment='center', verticalalignment='bottom', fontsize=10, fontweight="bold")
    
    plt.title('Segmentation Result without Original Signal', fontsize=14)
    plt.xlabel('Genomic position (bins)', fontsize=12)
    plt.ylabel('Copy Number', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_without_original.png"), dpi=300)
    plt.show()