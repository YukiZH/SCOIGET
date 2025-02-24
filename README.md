# SCOIGET

<img width="200" alt="image" src="https://github.com/user-attachments/assets/6ca8a158-9127-44ce-bcee-1e35143fd6c6">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/5afb1e2c-344c-42d7-8c61-db972bcde820">

SCOIGET: A Tool for Predicting Spatial Tumor Evolution Pattern by Inferring Spatial Copy Number Variation Distributions


## Overview
**SCOIGET** (**S**patial **CO**py number **I**nference by **G**raph on **E**volution of **T**umor) is a groundbreaking framework designed to decode tumor heterogeneity by constructing comprehensive spatiotemporal maps through spatial omics data. Traditional methods, primarily developed for single-cell analysis, overlook critical spatial information, limiting their ability to capture tumor evolution dynamics. SCOIGET addresses this gap by integrating graph neural networks (GNNs) with graph attention layers to model spatial neighborhood features of gene expression and infer copy number variations (CNVs).

Key innovations include:
**· Spatiotemporal Integration**: Combines spatial multi-omics features to map tumor evolution with unprecedented resolution.
**· Enhanced Accuracy**: Achieves significant reductions in error metrics (e.g., mean squared error) and superior clustering performance (higher Silhouette Scores) compared to existing tools.
**· Generalizability**: Robust across diverse spatial omics technologies (e.g., Visium, VisiumHD) and cancer types, enabling versatile applications in downstream analyses.

By elucidating spatial and temporal changes within the tumor microenvironment, SCOIGET enhances the precision of tumor evolution modeling and accelerates insights into cancer progression. This tool empowers researchers to uncover actionable biomarkers and design personalized therapeutic strategies, bridging the gap between spatial omics innovation and clinical translation.


## System Requirements
### Hardware Requirements
· Memory: 16GB or higher for efficient processing.
· GPU: NVIDIA GPU (A100/3090) is highly recommended for accelerating training times.
### Operating System Requirements
· Linux: Ubuntu 16.04 or newer.


## Environment Setup
This section details the steps to set up the project environment using Anaconda.


### Prerequisites
```
Python 3.8.20
pytorch==2.2.2
```


### Cloning the Repository and Preparing the Environment
Actual installation time depends on network conditions and takes about 15 minutes.

#### 1. Clone the Repository:
```
git clone https://github.com/YukiZH/SCOIGET.git
cd SCOIGET
```
or download the code:
```
wget https://github.com/YukiZH/SCOIGET/archive/refs/heads/main.zip
unzip main.zip
cd /home/.../SCOIGET-main  ### your own path
```
#### 2. Create and Activate the Environment:
```
conda create -n scoiget_env python=3.8.20
conda activate scoiget_env

## Step1: Installing PyTorch 
# For GPU (CUDA 12.1)
pip install torch==2.2.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

## Step2: Installing Pyg
pip install torch-scatter==2.1.2+pt22cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch-sparse==0.6.18+pt22cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch-cluster==1.6.3+pt22cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.5.2
   
## Step3: Download other dependencies
pip install -r requirements.txt
```


## Usage
Note that we conducted experiments with the A100/3090 on Linux.

Before running, please download the compressed folder of the ```Dataset``` from [Google drive](https://drive.google.com/file/d/1gK1LXRvv3LXIGbHUNFPJJOyN0ZLVoS8R/view?usp=drive_link) and decompress it in ```./```, After decompression, the dir structure under ```./``` will be:
```
/home/.../SCOIGET
|-- Dataset
|   |-- PAT71397
|   |   |-- 6723_1
|   |   |   |-- 6723_KL_1_filtered_trimmed.h5ad
|   |   |   `-- 6723_1_pathology_annotation.csv
|   |   |-- 6723_2
|   |   |   |-- 6723_KL_2_filtered_trimmed.h5ad
|   |   |   `-- 6723_2_pathology_annotation.csv
|   |   |-- 6723_3
|   |   |   |-- 6723_KL_3_filtered_trimmed.h5ad
|   |   |   `-- 6723_3_pathology_annotation.csv
|   |   |-- 6723_4
|   |   |   |-- 6723_KL_4_filtered_trimmed.h5ad
|   |   |   `-- 6723_4_pathology_annotation.csv
|   |   `-- combined_processed_adata.h5ad

|-- merged_output
|   |-- chrom_list.npy
|   |-- copy.npy
|   |-- model_1st.pth
|   |-- model_2st.pth

|-- tutorial
|   |-- example_tutorial.ipynb

|-- requirements.txt

|-- scoiget
|   |-- __init__.py
|   |-- preprocess_utils.py
|   |-- cnv_utils.py
|   |-- graph_utils.py
|   |-- scoiget_model.py
|   |-- train_utils.py
|   |-- cluster_utils.py
|   |-- segment_utils.py
|   `-- draw_utils.py

`-- Macosko_cell_cycle_genes.txt
```


## License
This project is covered under the Apache 2.0 License.


