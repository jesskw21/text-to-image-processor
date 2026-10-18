"""Generate images from a text prompt.

    python sample.py "a large red circle on the left" --sampler dpmsolver --steps 20
"""
import argparse
import time

import torch

from tinydiff.data import save_grid, to_uint8
from tinydiff.device import autocast, get_device, sync
from tinydiff.diffusion import GaussianDiffusion
from tinydiff.samplers import SAMPLERS, CFGDenoiser, sample
from tinydiff.training import load_for_inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--ckpt", default="runs/base/ckpt.pt")
    ap.add_argument("--sampler", default="dpmsolver", choices=SAMPLERS)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--out", default="sample.png")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    model, tok, _ = load_for_inference(args.ckpt, device)
    diffusion = GaussianDiffusion(device=device)
    tokens = torch.tensor([tok.encode(args.prompt)] * args.n, device=device)
    null = torch.tensor([tok.null()], device=device)
    g = torch.Generator().manual_seed(args.seed)
    x_T = torch.randn((args.n, 3, 32, 32), generator=g).to(device)

    t0 = time.perf_counter()
    with autocast(device, args.precision):
        den = CFGDenoiser(model, tokens, null, args.guidance)
        x = sample(den, diffusion.alphas_cumprod, x_T, args.sampler, args.steps, generator=g)
    sync(device)
    print(f"{args.n} images in {time.perf_counter() - t0:.2f}s -> {args.out}")
    save_grid(to_uint8(x.float()), args.out)


if __name__ == "__main__":
    main()
