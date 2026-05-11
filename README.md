# TC-Pad&eacute;: Trajectory-Consistent Pad&eacute; Approximation for Diffusion Acceleration

**Official implementation of TC-Pad&eacute; (CVPR 2026)** 🎉

<p align="center">

<a href="https://arxiv.org/abs/2603.02943">
<img src="https://img.shields.io/badge/ArXiv-2602.23783-b31b1b.svg">

</p>


<p align="center">
  <img src="assets/teaser.png" width="100%" />
</p>


To address the computational inefficiency of diffusion models and the limitations of existing polynomial-based feature caching methods that suffer from error accumulation in practical low-step regimes, this paper proposes Trajectory-Consistent Padé approximation (TC-Padé). By modeling feature evolution through rational functions rather than Taylor series, TC-Padé captures complex asymptotic behaviors more accurately and incorporates adaptive coefficient modulation alongside step-aware prediction strategies to handle distinct denoising phases.

## 🔧 Installation

```bash
git clone https://github.com/Alibaba-Yufeng/TC_Pade.git
cd TC_Pade
pip install -r requirements.txt
```

**Requirements:** Python >= 3.9, PyTorch >= 2.6, CUDA-capable GPU.

## 🚀 Usage

### Baseline

```bash
python run.py --model_path /path/to/flux.1-dev --num_inference_steps 50
```

### TC-Pad&eacute; Accelerated Inference

```bash
python run.py \
    --model_path /path/to/flux.1-dev \
    --use_predict \
    --num_inference_steps 50 \
    --N 1.4 \
    --interval 8
```

### Argument List

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `path_to_flux.1-dev` | Path to the pretrained FLUX model |
| `--prompts_file` | `./example_prompts.json` | Path to the prompts JSON file |
| `--output_dir` | auto-generated | Output directory for generated images |
| `--num_inference_steps` | `50` | Number of denoising steps |
| `--seed` | `42` | Random seed for reproducibility |
| `--use_predict` | `False` | Enable TC-Pad&eacute; acceleration |
| `--start_step` | `4` | Step to begin prediction |
| `--interval` | `8` | Prediction interval |
| `--N` | `1.4` | Curvature threshold (larger = faster, more aggressive skipping) |
| `--predictor_order` | `3` | Pad&eacute; predictor order |
| `--predictor_history_size` | `6` | Residual history buffer size |

## 📊 Experimental Results

### Text-to-Image: FLUX.1-dev on DrawBench

| Method | Latency(s)↓ | Speed↑ | FLOPs(T)↓ | Speed↑ | CLIP↑ | Image Reward↑ | SSIM↑ | PSNR↑ | LPIPS↓ |
|--------|-------------|--------|-----------|--------|-------|---------------|-------|-------|--------|
| FLUX.1-dev (50 steps) | 30.46 | 1.00× | 3734.56 | 1.00× | 31.2627 | 0.9940 | 1.0000 | ∞ | 0.0000 |
| TC-Padé (Medium) | 8.74 | 3.49× | 760.08 | 4.91× | 31.2708 | 0.9402 | 0.6908 | 29.3693 | 0.3909 |
| TC-Padé (Fast) | 7.92 | 3.84× | 685.46 | 5.45× | 31.1870 | 0.8943 | 0.6707 | 29.2031 | 0.4182 |
| TC-Padé (Ultra) | 7.45 | 4.08× | 611.09 | 6.11× | 31.2176 | 0.8547 | 0.6518 | 29.1051 | 0.4513 |

### Text-to-Video: Wan2.1 on VBench2

| Method | Latency(s)↓ | Speed↑ | FLOPs(T)↓ | Speed↑ | SSIM↑ | PSNR↑ | LPIPS↓ |
|--------|-------------|--------|-----------|--------|-------|-------|--------|
| Wan2.1 (50 steps) | 142.32 | 1.00× | 8494.51 | 1.00× | 1.0000 | ∞ | 0.0000 |
| TC-Padé (Medium) | 40.54 | 3.51× | 1928.68 | 4.40× | 0.5968 | 28.6603 | 0.4895 |
| TC-Padé (Fast) | 33.37 | 4.26× | 1436.24 | 5.91× | 0.5664 | 28.4528 | 0.5627 |

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{cui2026tc,
  title={TC-Pad$\backslash$'e: Trajectory-Consistent Pad$\backslash$'e Approximation for Diffusion Acceleration},
  author={Cui, Benlei and He, Shaoxuan and Huang, Bukun and Ye, Zhizeng and Sun, Yunyun and Huang, Longtao and Xue, Hui and Yang, Yang and Tang, Jingqun and Zhao, Zhou and others},
  journal={arXiv preprint arXiv:2603.02943},
  year={2026}
}
```
