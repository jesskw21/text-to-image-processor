"""Reverse-process samplers: DDPM (ancestral), DDIM, and DPM-Solver++(2M).

All three share the same trained epsilon-prediction network and the same
timestep grid, and spend exactly `steps` network evaluations (NFE). They differ
only in how they move from timestep t to the next timestep s < t.

Notation (discrete time, a_t = alpha_bar_t):
    x_t = sqrt(a_t) * x0 + sqrt(1 - a_t) * eps
    x0_hat = (x_t - sqrt(1 - a_t) * eps_hat) / sqrt(a_t)
"""
import math

import numpy as np
import torch

SAMPLERS = ("ddpm", "ddim", "dpmsolver")


def timestep_schedule(T, steps):
    """`steps` evenly spaced timesteps from T-1 down toward 0. After the last one
    the sampler jumps straight to the clean image (a = 1)."""
    if steps >= T:
        return list(range(T - 1, -1, -1))
    return np.round(np.linspace(T - 1, 0, steps + 1)[:-1]).astype(int).tolist()


class CFGDenoiser:
    """Classifier-free guidance: eps = eps_uncond + w * (eps_cond - eps_uncond).
    Conditional and unconditional passes run as one doubled batch, and the text
    encoder runs once per sampling call rather than once per step."""

    def __init__(self, model, tokens, null_tokens, scale):
        self.model, self.scale = model, scale
        both = torch.cat([tokens, null_tokens.expand_as(tokens)])
        self.text = model.encode_text(both)

    def __call__(self, x, t):
        tt = torch.full((2 * x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = self.model.denoise(torch.cat([x, x]), tt, self.text).float()
        cond, uncond = eps.chunk(2)
        return uncond + self.scale * (cond - uncond)


@torch.no_grad()
def sample(denoiser, alphas_cumprod, x_T, sampler="dpmsolver", steps=20, eta=0.0,
           clip_x0=True, generator=None):
    """denoiser(x, t:int) -> eps.  x_T: initial Gaussian noise."""
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}")
    acp = alphas_cumprod.tolist()
    ts = timestep_schedule(len(acp), steps)
    x = x_T
    prev_x0, prev_h = None, None

    def noise_like(x):
        return torch.randn(x.shape, generator=generator).to(x.device)

    for i, t in enumerate(ts):
        a_t = acp[t]
        last = i == len(ts) - 1
        a_s = 1.0 if last else acp[ts[i + 1]]

        eps = denoiser(x, t)
        x0 = (x - math.sqrt(1 - a_t) * eps) / math.sqrt(a_t)
        if clip_x0:
            x0 = x0.clamp(-1, 1)
        if last:
            # Every sampler's final step to a = 1 reduces to returning x0_hat.
            return x0

        if sampler == "ddpm":
            # Ancestral step using the posterior q(x_s | x_t, x0) of the
            # respaced chain, whose effective beta is 1 - a_t / a_s.
            beta = 1 - a_t / a_s
            mean = (math.sqrt(a_s) * beta / (1 - a_t)) * x0 \
                + (math.sqrt(a_t / a_s) * (1 - a_s) / (1 - a_t)) * x
            var = beta * (1 - a_s) / (1 - a_t)
            x = mean + math.sqrt(var) * noise_like(x)

        elif sampler == "ddim":
            # Song et al. (2021). eta=0 is deterministic; eta=1 matches DDPM's variance.
            eps = (x - math.sqrt(a_t) * x0) / math.sqrt(1 - a_t)  # consistent with clipped x0
            sigma = eta * math.sqrt((1 - a_s) / (1 - a_t) * (1 - a_t / a_s))
            x = math.sqrt(a_s) * x0 + math.sqrt(1 - a_s - sigma**2) * eps
            if sigma > 0:
                x = x + sigma * noise_like(x)

        else:
            # DPM-Solver++(2M), Lu et al. (2022): an exponential integrator of the
            # probability-flow ODE in log-SNR time lambda = log(alpha / sigma),
            # using the data prediction x0_hat. The multistep version reuses the
            # previous step's x0_hat for a second-order correction at no extra NFE.
            alpha_t, sigma_t = math.sqrt(a_t), math.sqrt(1 - a_t)
            alpha_s, sigma_s = math.sqrt(a_s), math.sqrt(1 - a_s)
            h = math.log(alpha_s / sigma_s) - math.log(alpha_t / sigma_t)
            if prev_x0 is None:
                d = x0  # first step: first order (equivalent to DDIM)
            else:
                r = prev_h / h
                d = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * prev_x0
            x = (sigma_s / sigma_t) * x - alpha_s * math.expm1(-h) * d
            prev_x0, prev_h = x0, h

    return x
