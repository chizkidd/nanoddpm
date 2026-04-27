[![CI](https://github.com/chizkidd/nanoddpm/actions/workflows/ci.yml/badge.svg)](https://github.com/chizkidd/nanoddpm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=chizkidd.nanoddpm)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chizkidd/nanoddpm/blob/main/nanoddpm.ipynb)



# nanoddpm

From-scratch implementation of Denoising Diffusion Probabilistic Model (DDPM) on MNIST dataset in 200 lines. Educational and inspired by other minimal implementations like Andrej Karpathy's `microgpt`.

## Features
- Single-file implementation (`nanoddpm.py`)
- Sinusoidal time embeddings + time-conditioned CNN
- Forward/reverse process with explicit math comments
- Lightweight quality metrics (no InceptionV3 required)
- CLI args for epochs, batch size, diffusion steps, and device

## Quick Start
```bash
pip install -r requirements.txt
python nanoddpm.py --epochs 5 --steps 500 --batch_size 128
```

## Project Structure
```
nanoddpm/
├── .github/workflows/ci.yml  # CI: smoke test on CPU
├── nanoddpm.py               # Single-file implementation (<180 lines)
├── nanoddpm.ipynb            # Colab notebook with visualization
├── requirements.txt          # torch, torchvision, numpy, matplotlib, tqdm
├── blog.md                   # Math walkthrough (forward, reverse, loss, metrics)
└── README.md
```

## Philosophy
- The readability of the code should translate to learnable mathematics.
- No black boxes. No heavy wrappers. Just raw PyTorch and explicit diffusion equations that fit in a single file.

## License
MIT
