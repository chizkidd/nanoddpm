# nanoddpm.py: From-scratch DDPM for MNIST (~260 lines)
# Educational build inspired by micrograd/minbpe. 
# Run: python nanoddpm.py [--epochs 3] [--batch_size 128] [--device cuda]

import argparse, torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt, numpy as np, math, json
from tqdm import tqdm, trange
import torch.nn.functional as F

# === CONFIG ===
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--steps', type=int, default=1000) 
args = parser.parse_args()
device = torch.device(args.device)
torch.manual_seed(42)
print(f"▶ nanoddpm | Device: {device} | Steps: {args.steps} | Epochs: {args.epochs}")

# === 1. NOISE SCHEDULE & FORWARD PROCESS ===
T_steps = args.steps
beta = torch.linspace(1e-4, 0.02, T_steps, device=device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def forward_diffusion(x0, t):
    """q(x_t | x_0) = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε"""
    sqrt_ab = torch.sqrt(alpha_bar[t])[:, None, None, None]
    sqrt_1m = torch.sqrt(1.0 - alpha_bar[t])[:, None, None, None]
    eps = torch.randn_like(x0, device=device)
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
        self.b1 = TimeBlock(1, 16, time_dim)
        self.b2 = TimeBlock(16, 32, time_dim)
        self.out = nn.Conv2d(32, 1, 3, padding=1)
    def forward(self, x, t):
        t_emb = sinusoidal_embedding(t, self.time_dim)
        return self.out(self.b2(self.b1(x, t_emb), t_emb))

model = NanoDDPM(time_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-3)
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
    hr/=hr.sum()
    hg/=hg.sum()
    return np.sum(hg * np.log(hg/hr)).item()

def evaluate(model):
    model.eval()
    with torch.no_grad():
        x = torch.randn(256, 1, 28, 28, device=device)
        for t in reversed(range(T_steps)):
            t_t = torch.full((256,), t, device=device)
            eps_p = model(x, t_t)
            a, ab, b = alpha[t], alpha_bar[t], beta[t]
            x = (1/torch.sqrt(a))*(x - (b/torch.sqrt(1-ab))*eps_p)
            if t>0: 
                x += torch.sqrt(b)*torch.randn_like(x)
            x = torch.clip(x, -1.0, 1.0)
        return {
            'fid': approx_fid(real_batch[:256], x),
            'var': x.std().item(),
            'grad': sobel_grad(x),
            'kl': intensity_kl(real_batch[:256], x),
            'samples': x
        }

# === 5. TRAINING LOOP ===
metrics_log = []
for epoch in trange(1, args.epochs+1, desc="Training"):
    model.train()
    epoch_loss, count = 0, 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        t = torch.randint(0, T_steps, (imgs.shape[0],), device=device)
        xt, eps = forward_diffusion(imgs, t)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(xt, t), eps)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*imgs.shape[0]
        count += imgs.shape[0]
    
    m = evaluate(model)
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