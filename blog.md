# nanoddpm: Diffusion Models from Scratch in <180 Lines

> Inspired by Karpathy's `microgpt`. No black boxes, no high-level wrappers. Just the math, the code, and the gradients.

## The Core Idea
Diffusion models generate data by reversing a gradual noising process. Instead of learning to generate pixels directly, we train a network to **predict the noise added at each step**. Once trained, we start from pure Gaussian noise and iteratively denoise it into realistic samples.

This post walks through the three core pillars of DDPMs:
1. Forward noising process (DDPM formulation)
2. Reverse denoising process (sampling dynamics)
3. Simplified noise-prediction loss

Plus lightweight metrics to track generative quality without external pretrained models.

---

## 1. Forward Process: Adding Noise
We define a fixed Markov chain that gradually corrupts data with Gaussian noise:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

Using the closed-form formulation, we sample directly from $x_0$:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Where 
- $\alpha_t = 1 - \beta_t$ 
- $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$.

### Code

```python
def forward_diffusion(x0, t):
    sqrt_ab = torch.sqrt(alpha_bar[t])
    sqrt_1m = torch.sqrt(1.0 - alpha_bar[t])
    eps = torch.randn_like(x0)
    return sqrt_ab * x0 + sqrt_1m * eps, eps
```

---

## 2. Reverse Process: Removing Noise (Denoising)
The goal is to learn $p_\theta(x_{t-1} | x_t)$, the reverse transition. Ho et al. showed we can parameterize this as:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
$$

At $t=0$, we drop the noise term $z$.

Where:

* $\epsilon_\theta$ is the learned noise predictor
* $z \sim \mathcal{N}(0, I)$
* $\sigma_t^2 = \beta_t$

At inference time, we iteratively apply this update from $t=T \to 0$.

### Code

```python
for t in reversed(range(T_steps)):
    eps_pred = model(x, t)
    x = (1 / sqrt_alpha[t]) * ( x - (beta[t] / sqrt_one_minus_alpha_bar[t]) * eps_pred )
    if t > 0:
        x += torch.sqrt(beta[t]) * torch.randn_like(x)
    x = torch.clamp(x, -1.0, 1.0)
```


---

## 3. The Loss Function
The variational lower bound (ELBO) for diffusion models simplifies dramatically to a simple regression objective. Instead of predicting $x_0$ or the mean $\mu_\theta$, we train the network to predict the **noise** $\epsilon$ added at step $t$:

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

We directly train the model to predict noise. This is equivalent to learning the reverse diffusion mean, but:

* stabilizes training
* removes scale dependencies
* simplifies optimization

### Code

```python
xt, eps = forward_diffusion(imgs, t)
eps_pred = model(xt, t)
loss = nn.functional.mse_loss(eps_pred, eps)
```

That is it. One line of MSE. The entire generative capability emerges from this simple signal.

---

## 4. Lightweight Quality Metrics (No Inception Network)
Real FID requires heavy feature extractors. To avoid heavy pretrained feature extractors for educational purposes, we track 4 simple proxy metrics:

| Metric | Formula/Method | What it tells us |
|--------|----------------|------------------|
| **Pixel-FID** | $\|\mu_r-\mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r\Sigma_g})$ | Distributional alignment (pixel-level proxy; true FID uses Inception features) |
| **Sample Variance** | $\text{std}(x_{gen})$ | Mode collapse detection (low = boring) |
| **Sobel Gradient** | $\sqrt{(\nabla_x I)^2 + (\nabla_y I)^2}$ | Sharpness and edge definition; low magnitude correlates with blurry samples |
| **Intensity KL** | $D_{KL}(P_{gen} \| P_{real})$ | Pixel histogram matching / intensity alignment |

These are not true FID metrics, but they are useful for debugging and iteration. No external models.

---

## 5. Model Architecture

### 5.1. Minimal Time-Conditioned CNN Model
The model is a sequential convolutional network that injects timestep information directly into the feature maps:

- **Input:** $x_t \in \mathbb{R}^{1 \times 28 \times 28}$ (MNIST)
- **Time embedding:** $\text{sinusoidal}(t) \in \mathbb{R}^{128}$, projected to channel dimension and broadcast spatially via `[:, :, None, None]`
- **2 conditional blocks** (`TimeBlock`): Each applies Conv2d, GroupNorm, adds the projected time embedding, and passes through SiLU
- **Output:** $\epsilon_\theta(x_t, t) \in \mathbb{R}^{1 \times 28 \times 28}$ via a final 3×3 convolution

Unlike U-Nets or ResNets, this architecture contains no skip connections, downsampling, or attention. It relies purely on the diffusion loss to learn hierarchical denoising. The entire network definition fits in ~40 lines, with the full training script under 180 lines.

---

### 5.2: Minimal U-Net Style Diffusion Model

The model is a compact **U-Net-inspired architecture** designed specifically for diffusion noise prediction.

It introduces a small but important upgrade over plain CNN stacks:

* hierarchical feature extraction (downsampling)
* structured reconstruction (upsampling)
* skip connections for information preservation
* time-conditioned residual blocks

Despite this, it remains fully minimal and readable.

#### Structure Overview

The network follows this pipeline:

```
x → down1 → pool → down2 → pool → bottleneck → up2 → up1 → output
      ↓                ↓                   ↑        ↑
     skip             skip              concat    concat
```

#### Key Components

* **Encoder path**

  * progressively reduces spatial resolution (28 → 14 → 7)
  * increases channel capacity (1 → 32 → 64)
  * stores intermediate features for skip connections

* **Bottleneck**

  * deepest representation (64 channels, 7×7)
  * captures global structure of the image

* **Decoder path**

  * upsamples back to original resolution
  * concatenates encoder features via skip connections
  * refines details progressively

* **Time conditioning**

  * sinusoidal embedding → MLP projection
  * injected into every `TimeBlock`

#### Why this design matters

This architecture is the first “real diffusion-grade upgrade” in the project:

* Skip connections preserve high-frequency detail lost during noising
* Downsampling expands receptive field for global structure modeling
* Bottleneck forces semantic compression of noisy representations
* Time conditioning is injected at every scale

Compared to the earlier flat CNN version in **section 5.1**, this model:

* improves convergence stability
* reduces blur in generated samples
* significantly improves FID slope during training
* better matches modern DDPM U-Net design principles

---


