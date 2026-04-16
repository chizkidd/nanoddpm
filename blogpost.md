# nanoddpm: Diffusion Models from Scratch in ~250 Lines

> Inspired by `micrograd`, `minbpe`, and `microGPT`. No black boxes, no high-level wrappers. Just the math, the code, and the gradients.

## The Core Idea
Diffusion models generate data by reversing a gradual noising process. Instead of learning to generate pixels directly, we train a network to **predict the noise** added at each step. Once trained, we start from pure noise and iteratively denoise it to produce realistic samples.

This post walks through the three pillars of DDPMs:
1. The forward noising process (math + code)
2. The reverse denoising process (math + code)
3. The simplified loss function (derivation + intuition)

Plus, we'll track generation quality with lightweight, from-scratch metrics.

---

## 1. Forward Process: Adding Noise
We define a fixed Markov chain that slowly adds Gaussian noise to data $x_0$ over $T$ steps:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

By the reparameterization trick, we can jump directly from $x_0$ to $x_t$:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$.

**In code:**
```python
def forward_diffusion(x0, t):
    sqrt_ab = torch.sqrt(alpha_bar[t])
    sqrt_1m = torch.sqrt(1.0 - alpha_bar[t])
    eps = torch.randn_like(x0)
    return sqrt_ab * x0 + sqrt_1m * eps, eps
```

---

## 2. Reverse Process: Removing Noise
The goal is to learn $p_\theta(x_{t-1} | x_t)$, the reverse transition. Ho et al. showed we can parameterize this as:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
$$

Where $\epsilon_\theta$ is our neural network predicting the noise, and $\sigma_t^2 = \beta_t$. At $t=0$, we drop the noise term $z$.

**In code:**
```python
for t in reversed(range(T_steps)):
    eps_pred = model(x, t)
    x = (1/sqrt_a)*(x - (b/sqrt_1m_ab)*eps_pred)
    if t > 0: x += sqrt_b * torch.randn_like(x)
    x = torch.clip(x, -1.0, 1.0)  # Keep in training range
```

---

## 3. The Loss Function
The variational lower bound (ELBO) for diffusion models simplifies dramatically. Instead of predicting $x_0$ or the mean $\mu_\theta$, we train the network to predict the **noise** $\epsilon$ added at step $t$:

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

Why? Because predicting noise is mathematically equivalent to predicting the mean of the reverse distribution, but removes scaling dependencies and stabilizes training.

**In code:**
```python
xt, eps = forward_diffusion(imgs, t)
eps_pred = model(xt, t)
loss = nn.functional.mse_loss(eps_pred, eps)
```

That's it. One line of MSE. The entire generative capability emerges from this simple signal.

---

## 4. Tracking Quality Without InceptionV3
Real FID requires heavy feature extractors. For educational builds, we track four lightweight proxies:

| Metric | Formula/Method | What it tells us |
|--------|----------------|------------------|
| **Approx-FID** | $\|\mu_r-\mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r\Sigma_g})$ | Distributional alignment (pixel-level) |
| **Sample Variance** | $\text{std}(x_{gen})$ | Detects mode collapse (low = boring) |
| **Sobel Gradient** | $\sqrt{(\nabla_x I)^2 + (\nabla_y I)^2}$ | Sharpness & edge definition |
| **Intensity KL** | $D_{KL}(P_{gen} \| P_{real})$ | Pixel histogram matching |

All run in <50 lines of pure PyTorch/NumPy. No external models.

---

## How to Run
```bash
git clone https://github.com/chizkidd/nanoddpm.git
cd nanoddpm
pip install -r requirements.txt
python nanoddpm.py --epochs 5 --steps 500 --batch_size 128
```

Or run directly in Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chizkidd/nanoddpm/blob/main/nanoddpm.ipynb)

---

## Next Steps
- Swap MNIST for CIFAR-10 (add downsampling + UNet)
- Implement classifier-free guidance
- Add DDIM sampling for 10x faster generation
- Replace pixel-FID with lightweight feature extractors (e.g., PCA on flattened patches)

Happy diffusing. 🌊


---

### 🔍 Similar Diffusion Models That Are Easily Implemented

| Model | Core Idea | Why It's Easy to Implement | How It Modifies `nanoddpm.py` | Colab/Library-PC Friendly |
|-------|-----------|----------------------------|-------------------------------|---------------------------|
| **DDIM** (Denoising Diffusion Implicit Models) | Deterministic reverse process; skips noise term for faster sampling | Same training objective as DDPM. Only changes the sampling loop (~15 lines) | Replace stochastic reverse step with deterministic ODE-like update | ✅ Excellent. Runs instantly on existing weights |
| **NCSN + Langevin Dynamics** (Score-Based Generative Modeling) | Train network to predict score $\nabla \log p(x)$, sample via annealed Langevin updates | Replaces MSE loss with denoising score matching. Sampling uses simple gradient descent + noise | Change loss to `mse(pred_score, -eps)` + add Langevin sampler | ✅ Very. No complex schedulers needed |
| **EDM** (Elucidating Diffusion Models, Karras et al.) | Clean design: better noise schedule, loss weighting, and preconditioning | Simplifies DDPM math into a unified framework. Often *fewer* lines than vanilla DDPM | Replace `alpha_bar` with EDM sigma schedule + add preconditioning wrappers | ✅ Excellent. More stable training |
| **DPM-Solver / ODE Solvers** | Treat reverse process as ODE, solve with adaptive step methods (1st/2nd/3rd order) | Math-heavy but code-light. Swap reverse loop for explicit RK1/RK2 steps | Replace reverse loop with `x_{t-1} = x_t + h * f(x_t, t)` using trained eps_pred | ✅ Good. CPU-friendly, dramatic speedup |
| **Cold Diffusion** (Bansal et al.) | Deterministic noising + deterministic denoising; uses non-Gaussian or fixed transforms | Removes stochasticity entirely. Sampling becomes a fixed iterative refinement | Change forward process to deterministic corruption + use gradient-based refinement | ✅ Moderate. Requires careful clipping but very fast |
| **v-Prediction Diffusion** (Stable Diffusion / Google) | Predict $v = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0$ instead of $\epsilon$ | Same architecture, just changes target & reverse formula. Better stability for long schedules | Change loss target to `v_target`, adjust reverse step algebra | ✅ Excellent. Drops in as a 10-line tweak |

