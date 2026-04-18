# nanoddpm

From-scratch implementation of Denoising Diffusion Probabilistic Model (DDPM) on MNIST dataset in <180 lines. Educational and inspired by other minimal implementations like Andrej Karpathy's `microgpt`.

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
- _The readability of the code should translate to learnable mathematics._
- _No black boxes. No heavy wrappers. Just raw PyTorch and explicit diffusion equations that fit in a single file._

## License
MIT