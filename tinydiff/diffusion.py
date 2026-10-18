"""Forward (noising) process and the training objective."""
import math

import torch
import torch.nn.functional as F


def cosine_alphas_cumprod(T, s=0.008):
    """Nichol & Dhariwal (2021) cosine schedule; returns alpha_bar_t for t = 0..T-1."""
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    acp = f / f[0]
    betas = (1 - acp[1:] / acp[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0)


class GaussianDiffusion:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.alphas_cumprod = cosine_alphas_cumprod(T)  # float64, CPU; used by samplers
        self._acp = self.alphas_cumprod.float().to(device)

    def q_sample(self, x0, t, noise):
        a = self._acp[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def training_loss(self, model, x0, tokens, null_tokens, p_uncond=0.1, t=None, noise=None):
        """Epsilon-prediction MSE. Captions are dropped with prob p_uncond so the
        same network also learns the unconditional model needed for
        classifier-free guidance."""
        b = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.T, (b,), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        if p_uncond > 0:
            drop = torch.rand(b, device=x0.device) < p_uncond
            tokens = torch.where(drop[:, None], null_tokens.expand_as(tokens), tokens)
        eps = model(self.q_sample(x0, t, noise), t, tokens)
        return F.mse_loss(eps.float(), noise)
