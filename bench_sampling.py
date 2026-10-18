"""Benchmark sampler throughput vs. fidelity.

Every config generates the same prompts from the same initial noise, so the
only variable is the sampler and its step count.

    python bench_sampling.py --ckpt runs/base/ckpt.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tinydiff.data import caption_from_labels, random_labels, save_grid, to_uint8
from tinydiff.device import autocast, get_device, sync
from tinydiff.diffusion import GaussianDiffusion
from tinydiff.evaluator import (attribute_accuracy, feature_stats, frechet_distance,
                                load_evaluator, predict)
from tinydiff.samplers import CFGDenoiser, sample
from tinydiff.training import load_for_inference

DEFAULT_CONFIGS = "ddpm:100,ddpm:20,ddim:100,ddim:20,dpmsolver:20,dpmsolver:10"


def generate(model, diffusion, tokens, null, x_T, sampler, steps, guidance,
             batch, device, precision):
    out = []
    for i in range(0, len(tokens), batch):
        g = torch.Generator().manual_seed(1000 + i)  # same DDPM noise across runs
        with autocast(device, precision):
            den = CFGDenoiser(model, tokens[i:i + batch], null, guidance)
            out.append(sample(den, diffusion.alphas_cumprod, x_T[i:i + batch],
                              sampler, steps, generator=g).float())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/ckpt.pt")
    ap.add_argument("--evaluator", default="runs/evaluator.pt")
    ap.add_argument("--configs", default=DEFAULT_CONFIGS, help="sampler:steps,...")
    ap.add_argument("--baseline", default="ddpm:100")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--out", default="runs/bench_sampling")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok, ck = load_for_inference(args.ckpt, device)
    clf, ev = load_evaluator(args.evaluator, device)
    diffusion = GaussianDiffusion(device=device)

    labels = random_labels(args.n, np.random.default_rng(1234))
    tokens = torch.tensor([tok.encode(caption_from_labels(l)) for l in labels], device=device)
    null = torch.tensor([tok.null()], device=device)
    x_T = torch.randn((args.n, 3, 32, 32), generator=torch.Generator().manual_seed(0)).to(device)

    configs = [(c.split(":")[0], int(c.split(":")[1])) for c in args.configs.split(",")]
    if args.baseline not in args.configs.split(","):
        configs.insert(0, (args.baseline.split(":")[0], int(args.baseline.split(":")[1])))

    # Warm up kernels / allocator so the first config isn't penalized.
    generate(model, diffusion, tokens[:args.batch], null, x_T[:args.batch], "ddim", 5,
             args.guidance, args.batch, device, args.precision)

    results = []
    for sampler, steps in configs:
        sync(device)
        t0 = time.perf_counter()
        x = generate(model, diffusion, tokens, null, x_T, sampler, steps, args.guidance,
                     args.batch, device, args.precision)
        sync(device)
        dt = time.perf_counter() - t0

        pred, feats = predict(clf, x)
        mu, sigma = feature_stats(feats)
        acc = attribute_accuracy(pred, labels)
        r = {"sampler": sampler, "steps": steps, "nfe": steps, "seconds": dt,
             "img_per_s": args.n / dt, "fd": frechet_distance(mu, sigma, ev["mu"], ev["sigma"]),
             "prompt_acc": acc["all"], "attr_acc": acc}
        results.append(r)
        save_grid(to_uint8(x[:64]), out / f"{sampler}_{steps}.png")
        print(f"{sampler:10s} {steps:4d} steps  {r['img_per_s']:8.1f} img/s  "
              f"FD {r['fd']:7.3f}  prompt acc {r['prompt_acc']:.3f}")

    base = next(r for r in results if f"{r['sampler']}:{r['steps']}" == args.baseline)
    for r in results:
        r["speedup"] = r["img_per_s"] / base["img_per_s"]

    floor = ev["fd_floor"].get(args.n) or ev["fd_floor"].get(min(ev["fd_floor"], key=lambda k: abs(k - args.n)))
    lines = [
        f"Checkpoint step {ck['step']}, n={args.n} prompts, guidance {args.guidance}, "
        f"{args.precision}, device {device}",
        f"Real-vs-real Frechet distance (noise floor) at n~{args.n}: {floor:.3f}",
        "",
        "| sampler | steps (NFE) | img/s | speedup vs " + args.baseline + " | FD (lower=better) | prompt accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['sampler']} | {r['steps']} | {r['img_per_s']:.1f} | "
                     f"{r['speedup']:.2f}x (+{(r['speedup'] - 1) * 100:.0f}%) | "
                     f"{r['fd']:.3f} | {r['prompt_acc'] * 100:.1f}% |")
    table = "\n".join(lines)
    print("\n" + table)
    (out / "results.md").write_text(table + "\n")
    (out / "results.json").write_text(json.dumps(
        {"args": vars(args), "fd_floor": floor, "results": results}, indent=2))


if __name__ == "__main__":
    main()
