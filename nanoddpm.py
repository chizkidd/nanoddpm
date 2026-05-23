# nanoddpm.py: From-scratch DDPM for MNIST
# Educational build inspired by micrograd/minbpe.
# Run: python nanoddpm.py [--epochs 50] [--batch_size 128] [--device cuda]

import math, json, copy, argparse
from collections import namedtuple
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


# === 1. NOISE SCHEDULE & FORWARD PROCESS ===
NoiseSchedule = namedtuple('NoiseSchedule', [
    'beta', 'alpha', 'alpha_bar', 'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar', 'T'
])

def cosine_beta_schedule(T, device, s=0.008):
    x = torch.linspace(0, T, T + 1, device=device)
    alphas_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    beta = torch.clamp(1 - (alphas_bar[1:] / alphas_bar[:-1]), 1e-5, 0.999)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return NoiseSchedule(
        beta=beta, alpha=alpha, alpha_bar=alpha_bar,
        sqrt_alpha_bar=torch.sqrt(alpha_bar),
        sqrt_one_minus_alpha_bar=torch.sqrt(1 - alpha_bar),
        T=T,
    )

def forward_diffusion(x0, t, sched):
    """q(x_t | x_0) = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε"""
    sqrt_ab = sched.sqrt_alpha_bar[t][:, None, None, None]
    sqrt_1m = sched.sqrt_one_minus_alpha_bar[t][:, None, None, None]
    eps = torch.randn_like(x0)
    return sqrt_ab * x0 + sqrt_1m * eps, eps


# === 2. MODEL (Sinusoidal Time Embedding + U-Net style) ===
def sinusoidal_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    emb = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)

class TimeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))

    def forward(self, x, t_emb):
        t_proj = self.time_mlp(t_emb)[:, :, None, None]
        return F.silu(self.norm(self.conv(x) + t_proj))

class NanoDDPM(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        # Encoder
        self.down1 = TimeBlock(1,  32, time_dim)                        # 28x28
        self.pool1 = nn.Conv2d(32, 32, 4, stride=2, padding=1)          # 14x14
        self.down2 = TimeBlock(32, 64, time_dim)                        # 14x14
        self.pool2 = nn.Conv2d(64, 64, 4, stride=2, padding=1)          # 7x7
        # Bottleneck
        self.bottleneck = TimeBlock(64, 64, time_dim)
        # Decoder
        self.up2  = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)  # 14x14
        self.dec2 = TimeBlock(64+64, 32, time_dim)
        self.up1  = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)  # 28x28
        self.dec1 = TimeBlock(32+32, 32, time_dim)
        self.out  = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        s1 = self.down1(x, t_emb)                                        # [B, 32, 28, 28]
        s2 = self.down2(self.pool1(s1), t_emb)                           # [B, 64, 14, 14]
        x  = self.bottleneck(self.pool2(s2), t_emb)                      # [B, 64,  7,  7]
        x  = self.dec2(torch.cat([self.up2(x), s2], dim=1), t_emb)       # [B, 32, 14, 14]
        x  = self.dec1(torch.cat([self.up1(x), s1], dim=1), t_emb)       # [B, 32, 28, 28]
        return self.out(x)


# === 3. METRICS (From-scratch, pedagogical) ===
def approx_fid(real, gen, eps=1e-6):
    r = real.view(real.shape[0], -1).double()
    g = gen.view(gen.shape[0], -1).double()
    mu_r, mu_g = r.mean(0), g.mean(0)
    var_r, var_g = r.var(0) + eps, g.var(0) + eps
    return ((mu_r - mu_g)**2).sum().item() + (var_r + var_g - 2*torch.sqrt(var_r*var_g)).sum().item()

def sobel_grad(imgs):
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=imgs.device).view(1,1,3,3)
    sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=imgs.device).view(1,1,3,3)
    gx, gy = F.conv2d(imgs, sx, padding=1), F.conv2d(imgs, sy, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-8).mean().item()

def intensity_kl(real, gen, bins=50):
    r = real.cpu().view(-1).clamp(-1, 1).numpy()
    g = gen.cpu().view(-1).clamp(-1, 1).numpy()
    hr, _ = np.histogram(r, bins=bins, range=(-1, 1), density=True)
    hg, _ = np.histogram(g, bins=bins, range=(-1, 1), density=True)
    hr, hg = hr + 1e-8, hg + 1e-8
    hr /= hr.sum(); hg /= hg.sum()
    return float(np.sum(hg * np.log(hg / hr)))

def evaluate(model, sched, real_batch, n=256, steps=250):
    model.eval()
    device = sched.alpha_bar.device
    with torch.no_grad():
        x = torch.randn(n, 1, 28, 28, device=device)
        t_seq = torch.unique_consecutive(
            torch.flip(torch.linspace(0, sched.T - 1, steps, device=device), dims=[0]).long()
        )
        for i in range(len(t_seq) - 1):
            t, t_next = t_seq[i], t_seq[i + 1]
            eps = model(x, torch.full((n,), t, device=device, dtype=torch.long))
            x0  = (x - sched.sqrt_one_minus_alpha_bar[t] * eps) / sched.sqrt_alpha_bar[t]
            x   = sched.sqrt_alpha_bar[t_next] * x0 + sched.sqrt_one_minus_alpha_bar[t_next] * eps
        x = torch.clamp(x, -1.0, 1.0)
    return {
        'fid': approx_fid(real_batch[:n], x),
        'var': x.std().item(),
        'grad': sobel_grad(x),
        'kl':  intensity_kl(real_batch[:n], x),
        'samples': x,
    }

def update_ema(model, ema_model, ema_decay=0.999):
    with torch.no_grad():
        for p, ema_p in zip(model.parameters(), ema_model.parameters()):
            ema_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)


# === 4. VISUALIZATION ===
def plot_results(metrics_log):
    epochs = [m['epoch'] for m in metrics_log]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    axs[0].plot(epochs, [m['loss'] for m in metrics_log], marker='o')
    axs[0].set_title('Training Loss'); axs[0].grid(alpha=0.3)
    axs[1].plot(epochs, [m['fid']  for m in metrics_log], marker='s', color='orange')
    axs[1].set_title('Approx FID (↓ better)'); axs[1].grid(alpha=0.3)
    axs[2].plot(epochs, [m['grad'] for m in metrics_log], marker='^', color='green')
    axs[2].set_title('Sharpness (Sobel ↑)'); axs[2].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    final_grid = torchvision.utils.make_grid(
        metrics_log[-1]['samples'][:16], nrow=4, normalize=True, value_range=(-1, 1)
    )
    plt.figure(figsize=(4, 4))
    plt.imshow(final_grid.cpu().permute(1, 2, 0).numpy()); plt.axis('off'); plt.show()


# === 5. TRAINING ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--batch_size',    type=int,   default=128)
    parser.add_argument('--device',        type=str,   default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--steps',         type=int,   default=1000)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    cfg = parser.parse_args()

    device = torch.device(cfg.device)
    torch.manual_seed(42)
    print(f"▶ nanoddpm | Device: {device} | Steps: {cfg.steps} | Epochs: {cfg.epochs}")

    sched = cosine_beta_schedule(cfg.steps, device)

    transform  = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    dataset    = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader     = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    real_batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)))[0].to(device)

    model     = NanoDDPM(time_dim=128).to(device)
    ema_model = copy.deepcopy(model)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    print(f"▶ Params: {sum(p.numel() for p in model.parameters()):,}")

    metrics_log = []
    for epoch in trange(1, cfg.epochs + 1, desc="Training"):
        model.train()
        epoch_loss, count = 0.0, 0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            t    = torch.randint(0, cfg.steps, (imgs.shape[0],), device=device)
            xt, eps = forward_diffusion(imgs, t, sched)
            optimizer.zero_grad()
            loss = F.mse_loss(model(xt, t), eps, reduction="none").mean(dim=(1, 2, 3))
            snr  = sched.alpha_bar[t] / (1 - sched.alpha_bar[t] + 1e-8)
            loss = (loss * snr / (snr + 1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(model, ema_model)
            epoch_loss += loss.item() * imgs.shape[0]
            count      += imgs.shape[0]

        m = evaluate(ema_model, sched, real_batch)
        m['epoch'] = epoch
        m['loss']  = epoch_loss / count
        metrics_log.append(m)
        print(f"  Epoch {epoch:02d} | Loss: {m['loss']:.4f} | FID≈{m['fid']:.1f} | Var: {m['var']:.3f} | Grad: {m['grad']:.3f} | KL: {m['kl']:.4f}")

    serializable = [{k: v for k, v in m.items() if k != 'samples'} for m in metrics_log]
    with open('nanoddpm_metrics.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    print("Done. Metrics saved to nanoddpm_metrics.json")

    plot_results(metrics_log)


if __name__ == '__main__':
    main()
