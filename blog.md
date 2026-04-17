# nanoddpm: Diffusion Models from Scratch in <200 Lines

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

