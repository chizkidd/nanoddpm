[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chizkidd/nanoddpm/blob/main/nanoddpm.ipynb)

# nanoddpm

From-scratch implementation of Denoising Diffusion Probabilistic Model (DDPM) on MNIST dataset in <200 lines. Educational and inspired by other minimal implemenations like Andrej Karpathy's `microgpt`.

## Features
- Single-file implementation (`nanoddpm.py`)
- Sinusoidal time embeddings + time-conditioned CNN
- Forward/reverse process with explicit math comments
- Lightweight quality metrics (no `InceptionV3` required)
- CLI args for epochs, batch size, diffusion steps, and device

## Quick Start
```bash
pip install -r requirements.txt
python nanoddpm.py --epochs 5 --steps 500 --batch_size 128
```

## Philosophy
- _The readability of the code should translate to learnable mathematics._
- _No black boxes. No heavy wrappers. Just raw PyTorch, and explicit diffusion equations that fits in a single file. Designed for learners who want to understand **how** diffusion works, not just how to call it._


## License
MIT