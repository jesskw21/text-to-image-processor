"""Sampler correctness against an analytically known denoiser.

If the data distribution is Gaussian N(mu, s^2), the optimal noise prediction
has a closed form, so every sampler can be checked without training a model:
    x_t ~ N(sqrt(a) mu, a s^2 + 1 - a)
    E[eps | x_t] = sqrt(1 - a) (x_t - sqrt(a) mu) / (a s^2 + 1 - a)
"""
import math

import torch

from tinydiff.diffusion import cosine_alphas_cumprod
from tinydiff.samplers import sample, timestep_schedule

MU, S = 0.3, 0.5
ACP = cosine_alphas_cumprod(1000)


def gaussian_denoiser(x, t):
    a = ACP[t].item()
    return math.sqrt(1 - a) * (x - math.sqrt(a) * MU) / (a * S**2 + 1 - a)


def draw(sampler, steps, n=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    x_T = torch.randn((n, 1, 1, 1), generator=g, dtype=torch.float64)
    x = sample(gaussian_denoiser, ACP, x_T, sampler, steps, clip_x0=False, generator=g)
    return x.mean().item(), x.std().item()


def test_timestep_schedule():
    for steps in (5, 20, 100, 1000):
        ts = timestep_schedule(1000, steps)
        assert len(ts) == steps
        assert ts[0] == 999
        assert all(a > b for a, b in zip(ts, ts[1:]))


def test_all_samplers_recover_distribution():
    for sampler, steps in [("ddpm", 1000), ("ddpm", 100), ("ddim", 100), ("dpmsolver", 20)]:
        mean, std = draw(sampler, steps)
        assert abs(mean - MU) < 0.02, (sampler, steps, mean)
        assert abs(std - S) < 0.02, (sampler, steps, std)


def test_dpmsolver_more_accurate_than_ddim_at_low_nfe():
    # Both integrate the same probability-flow ODE deterministically; the
    # second-order multistep solver should land closer at the same step count.
    for steps in (5, 10):
        _, std_ddim = draw("ddim", steps)
        _, std_dpm = draw("dpmsolver", steps)
        assert abs(std_dpm - S) < abs(std_ddim - S), (steps, std_ddim, std_dpm)


if __name__ == "__main__":
    test_timestep_schedule()
    test_all_samplers_recover_distribution()
    test_dpmsolver_more_accurate_than_ddim_at_low_nfe()
    for sampler, steps in [("ddpm", 100), ("ddpm", 20), ("ddim", 20), ("ddim", 10),
                           ("dpmsolver", 20), ("dpmsolver", 10), ("dpmsolver", 5)]:
        m, s = draw(sampler, steps)
        print(f"{sampler:10s} {steps:4d} steps  mean={m:.4f} std={s:.4f}  (target {MU}, {S})")
