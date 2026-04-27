# nanoddpm.py: From-scratch DDPM for MNIST (~260 lines)
# Educational build inspired by micrograd/minbpe. 
# Run: python nanoddpm.py [--epochs 3] [--batch_size 128] [--device cuda]

import argparse, torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt, numpy as np, math, json, copy
from tqdm import tqdm, trange
import torch.nn.functional as F

# === CONFIG ===
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--steps', type=int, default=1000) 
parser.add_argument('--learning_rate', type=float, default=5e-4)
args = parser.parse_args()
device = torch.device(args.device)
torch.manual_seed(42)
print(f"▶ nanoddpm | Device: {device} | Steps: {args.steps} | Epochs: {args.epochs}")

# === 1. NOISE SCHEDULE & FORWARD PROCESS ===
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps, device=device)
    alphas_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    beta = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(beta, 1e-5, 0.999)

T_steps = args.steps
beta = cosine_beta_schedule(T_steps)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
sqrt_alpha_bar, sqrt_one_minus_alpha_bar = torch.sqrt(alpha_bar), torch.sqrt(1 - alpha_bar)

def forward_diffusion(x0, t):
    """q(x_t | x_0) = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε"""
    sqrt_ab = sqrt_alpha_bar[t][:, None, None, None]
    sqrt_1m = sqrt_one_minus_alpha_bar[t][:, None, None, None]
    eps = torch.randn_like(x0)
    return sqrt_ab * x0 + sqrt_1m * eps, eps

# === 2. DATASET (MNIST → [-1, 1]) ===
transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
real_batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)))[0].to(device)

# === 3. MODEL (Sinusoidal Time Embedding + Time-Conditioned CNN) ===
def sinusoidal_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)

class TimeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
    def forward(self, x, t_emb):
        x = self.conv(x)
        t_proj = self.time_mlp(t_emb)[:, :, None, None]
        return nn.functional.silu(self.norm(x + t_proj))

class NanoDDPM(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()
        self.time_dim = time_dim

        # Encoder
        self.down1 = TimeBlock(1, 32, time_dim)                       # 28x28
        self.pool1 = nn.Conv2d(32, 32, 4, stride=2, padding=1)        # 14x14
        self.down2 = TimeBlock(32, 64, time_dim)                      # 14x14
        self.pool2 = nn.Conv2d(64, 64, 4, stride=2, padding=1)        # 7x7
        # Bottleneck
        self.bottleneck = TimeBlock(64, 64, time_dim)
        # Decoder
        self.up2 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)  # 14x14
        self.dec2 = TimeBlock(64 + 64, 32, time_dim)  # skip concat
        self.up1 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)  # 28x28
        self.dec1 = TimeBlock(32 + 32, 32, time_dim)  # skip concat
        # Output
        self.out = nn.Conv2d(32, 1, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))

        # Encoder
        s1 = self.down1(x, t_emb)        # [B, 32, 28, 28]
        x = self.pool1(s1)               # [B, 32, 14, 14]
        s2 = self.down2(x, t_emb)        # [B, 64, 14, 14]
        x = self.pool2(s2)               # [B, 64, 7, 7]

        # Bottleneck
        x = self.bottleneck(x, t_emb)    # [B, 64, 7, 7]

        # Decoder
        x = self.up2(x)                  # [B, 64, 14, 14]
        x = torch.cat([x, s2], dim=1)    # skip connection
        x = self.dec2(x, t_emb)          # [B, 32, 14, 14]
        x = self.up1(x)                  # [B, 32, 28, 28]
        x = torch.cat([x, s1], dim=1)    # skip connection
        x = self.dec1(x, t_emb)          # [B, 32, 28, 28]

        return self.out(x)

model = NanoDDPM(time_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
print(f"▶ Params: {sum(p.numel() for p in model.parameters()):,}")

# === 4. METRICS (From-scratch, pedagogical) ===
def approx_fid(real, gen, eps=1e-6):
    r, g = real.view(real.shape[0], -1).double(), gen.view(gen.shape[0], -1).double()
    mu_r, mu_g, var_r, var_g = r.mean(0), g.mean(0), r.var(0)+eps, g.var(0)+eps
    return ((mu_r - mu_g)**2).sum().item() + (var_r + var_g - 2*torch.sqrt(var_r*var_g)).sum().item()

def sobel_grad(imgs):
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=imgs.device).view(1,1,3,3)
    sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=imgs.device).view(1,1,3,3)
    gx, gy = F.conv2d(imgs, sx, padding=1), F.conv2d(imgs, sy, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-8).mean().item()

def intensity_kl(real, gen, bins=50):
    r, g = real.cpu().view(-1).clamp(-1,1).numpy(), gen.cpu().view(-1).clamp(-1,1).numpy()
    hr, _ = np.histogram(r, bins=bins, range=(-1,1), density=True)
    hg, _ = np.histogram(g, bins=bins, range=(-1,1), density=True)
    hr, hg = hr+1e-8, hg+1e-8
    hr /= hr.sum()
    hg /= hg.sum()
    return np.sum(hg * np.log(hg/hr)).item()

def evaluate(model, n=256, steps=250):
    model.eval()
    with torch.no_grad():
        x = torch.randn(n, 1, 28, 28, device=device)                                   
        t_seq = torch.linspace(0, T_steps - 1, steps, device=device)
        t_seq = torch.flip(t_seq, dims=[0]).long()
        t_seq = torch.unique_consecutive(t_seq)              
        for i in range(len(t_seq) - 1):
            t = t_seq[i]
            t_next = t_seq[i + 1]
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            eps = model(x, t_batch)
            ab, ab_next = alpha_bar[t], alpha_bar[t_next]
            x0 = (x - sqrt_one_minus_alpha_bar[t] * eps) / sqrt_alpha_bar[t]
            x = sqrt_alpha_bar[t_next] * x0 + sqrt_one_minus_alpha_bar[t_next] * eps
        x = torch.clamp(x, -1.0, 1.0)
        return {
            'fid': approx_fid(real_batch[:n], x),
            'var': x.std().item(),
            'grad': sobel_grad(x),
            'kl': intensity_kl(real_batch[:n], x),
            'samples': x
        }

def update_ema(model, ema_model, ema_decay=0.995):
    with torch.no_grad():
        for p, ema_p in zip(model.parameters(), ema_model.parameters()):
            ema_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

# === 5. TRAINING LOOP ===
ema_model = copy.deepcopy(model)
metrics_log = []
for epoch in trange(1, args.epochs+1, desc="Training"):
    model.train()
    epoch_loss, count = 0, 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        t = torch.randint(0, T_steps, (imgs.shape[0],), device=device)
        xt, eps = forward_diffusion(imgs, t)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(xt, t), eps, reduction="none")
        loss = loss.mean(dim=(1,2,3))
        snr = alpha_bar[t] / (1 - alpha_bar[t] + 1e-8)
        weight = snr / (snr + 1)
        loss = (loss * weight).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update_ema(model, ema_model, 0.999)
        epoch_loss += loss.item()*imgs.shape[0]
        count += imgs.shape[0]
    
    m = evaluate(ema_model)
    m['epoch'] = epoch
    m['loss'] = epoch_loss/count
    metrics_log.append(m)
    print(f"  Epoch {epoch:02d} | Loss: {m['loss']:.4f} | FID≈{m['fid']:.1f} | Var: {m['var']:.3f} | Grad: {m['grad']:.3f} | KL: {m['kl']:.4f}")


# === 6. JSON DUMP: Create a serializable version of metrics_log for JSON dumping ===
json_output_metrics = []
for metric_entry in metrics_log:
    serializable_entry = {k: v for k, v in metric_entry.items() if k != 'samples'}
    json_output_metrics.append(serializable_entry)

with open('nanoddpm_metrics.json', 'w') as f: 
    json.dump(json_output_metrics, f, indent=2)

# === 7. VISUALIZATION ===
def plot_results():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    axs[0].plot([m['epoch'] for m in metrics_log], [m['loss'] for m in metrics_log], marker='o')
    axs[0].set_title('Training Loss')
    axs[0].grid(alpha=0.3)
    axs[1].plot([m['epoch'] for m in metrics_log], [m['fid'] for m in metrics_log], marker='s', color='orange')
    axs[1].set_title('Approx FID (↓ better)')
    axs[1].grid(alpha=0.3)
    axs[2].plot([m['epoch'] for m in metrics_log], [m['grad'] for m in metrics_log], marker='^', color='green')
    axs[2].set_title('Sharpness (Sobel ↑)')
    axs[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nFinal samples:")
    final_grid = torchvision.utils.make_grid(metrics_log[-1]['samples'][:16], nrow=4, normalize=True, value_range=(-1,1))
    plt.figure(figsize=(4,4))
    plt.imshow(final_grid.cpu().permute(1,2,0).numpy())
    plt.axis('off')
    plt.show()

plot_results()
print("Done. Metrics saved to nanoddpm_metrics.json")