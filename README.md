# SCOIGET

<img width="200" alt="image" src="https://github.com/user-attachments/assets/6ca8a158-9127-44ce-bcee-1e35143fd6c6">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/5afb1e2c-344c-42d7-8c61-db972bcde820">

SCOIGET: A Tool for Predicting Spatial Tumor Evolution Pattern by Inferring Spatial Copy Number Variation Distributions


## Overview


## System Requirements
### Hardware Requirements
· Memory: 16GB or higher for efficient processing.
· GPU: NVIDIA GPU (A100/3090) is highly recommended for accelerating training times.
### Operating System Requirements
· Linux: Ubuntu 16.04 or newer.


## Environment Setup
This section details the steps to set up the project environment using Anaconda.

### Prerequisites
Python 3.8.20
pytorch==2.2.2

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

## License
This project is covered under the Apache 2.0 License.


