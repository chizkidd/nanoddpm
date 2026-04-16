# nanoddpm

From-scratch Denoising Diffusion Probabilistic Model (DDPM) for MNIST in <200 lines. Educational and designed to mirror other minimal implemenations like `microgpt`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chizkidd/nanoddpm/blob/main/nanoddpm.ipynb)

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

## Philosophy
_The readability of the code should translate to learnable mathematics._

## License
MIT