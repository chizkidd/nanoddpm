[![CI](https://github.com/chizkidd/nanoddpm/actions/workflows/ci.yml/badge.svg)](https://github.com/chizkidd/nanoddpm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=chizkidd.nanoddpm)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chizkidd/nanoddpm/blob/main/nanoddpm.ipynb)



# nanoddpm

From-scratch implementation of Denoising Diffusion Probabilistic Model (DDPM) on MNIST dataset in 200 lines. Educational and inspired by other minimal implementations like Andrej Karpathy's `microgpt`.

## Features
- Single-file PyTorch implementation (`nanoddpm.py`)
- Classic DDPM baseline with sinusoidal time embeddings
- Lightweight evaluation metrics (FID approximation, Sobel sharpness, histogram KL)
- Fully explicit forward and reverse diffusion equations
- CLI control for training and sampling

## Current Model Evolution
This repository now explores three stages of diffusion modeling:

- **DDPM baseline**
  - Discrete timestep conditioning
  - ε-prediction objective

- **Improved DDPM variants**
  - SNR-weighted losses
  - EMA stabilization
  - Better U-Net-style skip structure

## Quick Start
```bash
pip install -r requirements.txt
python nanoddpm.py --epochs 5 --steps 500 --batch_size 128
```

## Project Structure
```
nanoddpm/
├── .github/workflows/ci.yml  # CI: CPU smoke test
├── archive/                  # Legacy v1 implementations (preserved for reference)
│   ├── nanoddpm-v1.ipynb
├── nanoddpm.py               # Single-file diffusion implementation 
├── requirements.txt          # torch, torchvision, numpy, matplotlib, tqdm
├── blog.md                   # Mathematical walkthrough (forward, reverse, loss, metrics)
└── README.md
```

## Philosophy
- The readability of the code should translate to learnable mathematics.
- No black boxes. No heavy wrappers. Just raw PyTorch and explicit diffusion equations that fit in a single file.

## License
MIT
