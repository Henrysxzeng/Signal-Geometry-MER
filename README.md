# Signal-Geometry Aware Micro-Expression Recognition

This repository contains the official PyTorch implementation for the paper:  
**"Signal-Geometry Aware Micro-Expression Recognition: A Dynamic Gradient Gain Approach"**.

Our framework integrates a Hierarchical Transformer (HTNet) with a Proxy Anchor Loss to enhance robustness in low-SNR micro-expression recognition tasks.

## 📋 Table of Contents
- [Environment Setup](#-environment-setup)
- [Data Preparation](#-data-preparation)
- [Usage](#-usage)
- [Files Description](#-files-description)
- [Acknowledgement](#-acknowledgement)

## 🛠️ Environment Setup

The code is tested on **Linux** with **Python 3.9** and **PyTorch 2.x (CUDA 11.8)**.

### 1. System Dependencies
Install necessary system tools:
```bash
sudo apt-get update
sudo apt-get install tmux unzip libgl1 -y
```

### 2. Python Environment

We recommend using Conda to manage the environment:

```bash
# Create and activate environment
conda create -n mer_env python=3.9 -y
conda activate mer_env

# Install PyTorch (CUDA 11.8 compatible)
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install other dependencies
pip install -r requirements.txt
```

## 📂 Data Preparation

1. **Preprocessing**:

   Run the preprocessing script to extract optical flow features. This script will automatically process the videos in `./datasets` and generate the necessary feature inputs.Bash

   ```bash
   python preprocess_local.py
   ```

## 🚀 Usage

We recommend using `tmux` for background training on remote servers.

### 1. Training (Ours: HTNet + Proxy Anchor)

To train the proposed model (Signal-Geometry Aware Framework):

```bash
# Start a new tmux session
tmux new -s train_ours

# Run training
python main.py --train True --seed 1
```

### 2. Training (Baseline: Cross-Entropy)

To reproduce the baseline HTNet results:

```bash
python main_HTNet_Baseline.py --train True --seed 1
```

### 3. Comparison with Angular Margin Losses

To train with ArcFace or CosFace for comparison:

```bash
# Example: Train with CosFace
python main_Arc+Cos.py --train True --loss_type CosFace --seed 42
```

### 4. Robustness & Visualization

To reproduce the analysis figures in the paper:

```bash
# Noise Robustness Analysis (Fig. 2)
python analysis_noise.py

# t-SNE Visualization (Fig. 3)
python analysis_tsne.py
```

## 📄 Files Description

- **Training Scripts**:
  - `main.py`: Main training script for the proposed method (Ours).
  - `main_HTNet_Baseline.py`: Training script for the Baseline (CE Loss).
  - `main_Arc+Cos.py`: Training script for SOTA angular margin losses.
- **Models & Logic**:
  - `Model_compare.py`: Architecture definition for HTNet with Proxy Anchor.
  - `Model_Baseline.py`: Architecture definition for Baseline HTNet.
  - `Compare_losses.py`: Implementation of Proxy Anchor, ArcFace, and CosFace losses.
  - `preprocess_local.py`: Data preprocessing (Optical Flow extraction).
- **Analysis**:
  - `analysis_noise.py`: Script for noise robustness evaluation.
  - `analysis_tsne.py`: Script for t-SNE visualization.

## 📢 Acknowledgement

This code repository is anonymously released for double-blind peer review. Strict measures have been taken to remove author-identifying information.