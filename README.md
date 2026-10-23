# Signal-Geometry Optimization for Micro-Expression Recognition

Research code accompanying **Signal-Geometry Optimization for Micro-Expression Recognition: A Dynamic Gradient Gain Perspective**, by Qi Zeng, Rujun Xu, and Cong Li, published in *IEEE Signal Processing Letters*, vol. 33, pp. 2066-2070, 2026.

**Paper:** https://doi.org/10.1109/LSP.2026.3693220

This project studies micro-expression recognition using a hierarchical Transformer backbone and hyperspherical proxy-based metric learning. It includes training, cross-entropy and angular-margin comparisons, and robustness/embedding visualizations.

## Repository Map

| File | Purpose |
| --- | --- |
| `main.py` | HTNet with Proxy Anchor loss |
| `main_HTNet_Baseline.py` | Cross-entropy baseline |
| `main_Arc+Cos.py` | ArcFace and CosFace comparisons |
| `Model_compare.py` / `Model_Baseline.py` | Model definitions |
| `Compare_losses.py` | Loss implementations |
| `preprocess_local.py` | Landmark-based patch preparation and tensor packing |
| `analysis_noise.py` | Noise-robustness analysis |
| `analysis_tsne.py` | Embedding visualization |
| `figures/` | Research figures |

The main training loop holds out one subject at a time and trains on the remaining subjects. See the paper for the benchmark definition and reported results. A separate desktop demonstration is not the research evaluation pipeline.

## Environment

The original development setup used Linux, Python 3.9, and CUDA 11.8. Dependencies are not fully version-pinned; identical results on other environments are not guaranteed.

```bash
conda create -n mer_env python=3.9 -y
conda activate mer_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data Preparation

Obtain datasets and permission to use them from their original providers. The code license does not grant redistribution rights to facial images, annotations, or derived dataset assets. A file's presence in this repository is not evidence of permission to redistribute it.

The preprocessing script expects these paths relative to the repository root:

```text
combined_3_class2_for_optical_flow.csv
datasets/
  combined_datasets_whole/
  STSNet_whole_norm_u_v_os/
  three_norm_u_v_os/
```

`preprocess_local.py` consumes existing optical-flow images; it is not a complete raw-video-to-optical-flow extraction pipeline. It detects landmarks, assembles local patches, and writes `processed_data.pt`.

```bash
python preprocess_local.py
```

Check sample identifiers, subject IDs, labels, and preprocessing outputs before training. Load only trusted tensor/checkpoint files.

## Training and Evaluation

Run from the repository root after preparing `processed_data.pt`:

```bash
# Proposed method
python main.py --train True --seed 1

# Cross-entropy baseline
python main_HTNet_Baseline.py --train True --seed 1

# Angular-margin comparisons
python main_Arc+Cos.py --train True --loss_type ArcFace --seed 1
python main_Arc+Cos.py --train True --loss_type CosFace --seed 1

# Evaluate the proposed method using existing subject checkpoints
python main.py --train False --seed 1
```

The scripts can reuse compatible existing subject checkpoints. Use a separate working copy or preserve and move checkpoints when a fresh training run is required. Do not mix methods, seeds, or preprocessing settings.

The proposed method writes `Anchor_weights_<seed>/`; the baseline writes `baseline_weights_<seed>/`; margin comparisons write `weights_<loss_type>_<seed>/`. Evaluation-only mode can skip subjects with missing checkpoints; verify subject coverage before reporting aggregate scores.

## Analysis

```bash
python analysis_noise.py --seed 1
python analysis_tsne.py
```

Before running:

- Align checkpoint directories with experiment outputs. The current `analysis_noise.py` uses `Aeights_<seed>` for the anchor directory; change its configuration to the intended checkpoint path.
- Set the seed and paths in `analysis_tsne.py`; its current seed is hardcoded.
- Verify model depth, checkpoint format, loss-head compatibility, and evaluation coverage across methods.

This documentation cleanup does not modify training code, checkpoints, or experimental results. Full end-to-end reproduction has not been performed as part of this update.

## Citation

```bibtex
@article{zeng2026signalgeometry,
  author  = {Zeng, Qi and Xu, Rujun and Li, Cong},
  title   = {Signal-Geometry Optimization for Micro-Expression Recognition: A Dynamic Gradient Gain Perspective},
  journal = {IEEE Signal Processing Letters},
  volume  = {33},
  pages   = {2066--2070},
  year    = {2026},
  doi     = {10.1109/LSP.2026.3693220}
}
```

## Acknowledgements and License

This research builds on HTNet and prior micro-expression recognition and metric-learning work. Cite the relevant original methods and datasets when using or extending the implementation.

See [LICENSE](LICENSE) for the repository code license. Third-party code and datasets remain subject to their respective terms.
