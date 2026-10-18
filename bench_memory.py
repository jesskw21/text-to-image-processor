"""Find the largest training micro-batch that fits a memory budget, FP32 vs FP16.

For each precision, measures peak memory of real training steps (forward,
backward, optimizer) at increasing batch sizes, then binary-searches the
largest batch whose peak stays under --budget-gb. Also reports training
throughput at that batch.

    python bench_memory.py --budget-gb 4

Peak memory is exact on CUDA. On Apple MPS it is sampled after each forward
pass (where activation memory peaks), so treat MPS numbers as approximate and
prefer a CUDA GPU (e.g. a free Colab T4) for the numbers you report.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tinydiff.data import DeviceDataset, Tokenizer, generate
from tinydiff.device import PeakMemory, free_memory, get_device, is_oom, sync
from tinydiff.diffusion import GaussianDiffusion
from tinydiff.model import ModelConfig, TextToImage
from tinydiff.training import Trainer


def trial(precision, batch, data, tok, device, cfg, steps=3):
    """Returns (peak_bytes, img_per_s) or (None, None) on OOM."""
    free_memory(device)
    model = trainer = None
    try:
        model = TextToImage(cfg, len(tok)).to(device)
        null = torch.tensor([tok.null()], device=device)
        trainer = Trainer(model, GaussianDiffusion(device=device), device, null, precision)
        peak = PeakMemory(device)
        get = lambda: data.batch(batch)
        trainer.step(get, on_forward=peak.mark)  # warm-up step allocates optimizer state
        sync(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            trainer.step(get, on_forward=peak.mark)
            peak.mark()
        sync(device)
        return peak.value(), batch * steps / (time.perf_counter() - t0)
    except (RuntimeError, torch.OutOfMemoryError) as e:
        if is_oom(e):
            return None, None
        raise
    finally:
        del model, trainer
        free_memory(device)


def max_batch(precision, budget, data, tok, device, cfg, start=16, limit=8192, granularity=8):
    measured = {}

    def fits(b):
        if b not in measured:
            measured[b] = trial(precision, b, data, tok, device, cfg)
            peak, ips = measured[b]
            print(f"  {precision} batch {b:5d}: "
                  + ("OOM" if peak is None else f"{peak / 2**30:6.2f} GB, {ips:7.1f} img/s"))
        peak = measured[b][0]
        return peak is not None and peak <= budget

    if not fits(start):
        return 0, measured
    lo, hi = start, start * 2
    while hi <= limit and fits(hi):
        lo, hi = hi, hi * 2
    while hi - lo > granularity:
        mid = (lo + hi) // 2 // granularity * granularity
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo, measured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, default=4.0)
    ap.add_argument("--precisions", default="fp32,fp16")
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--out", default="runs/bench_memory")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    budget = args.budget_gb * 2**30
    tok = Tokenizer()
    images, labels = generate(10000, seed=0)
    data = DeviceDataset(images, labels, tok, device)
    cfg = ModelConfig(base_ch=args.base_ch)
    exact = PeakMemory(device).exact
    print(f"device={device} budget={args.budget_gb} GB "
          f"({'exact' if exact else 'approximate'} peak memory)")

    results = {}
    for p in args.precisions.split(","):
        b, measured = max_batch(p, budget, data, tok, device, cfg)
        results[p] = {"max_batch": b, "img_per_s": measured.get(b, (None, None))[1],
                      "peak_gb": (measured.get(b, (None,))[0] or 0) / 2**30,
                      "measured": {k: [v[0], v[1]] for k, v in sorted(measured.items())}}

    lines = [f"Memory budget {args.budget_gb} GB on {device} "
             f"({'exact' if exact else 'approximate'} peak), base_ch={args.base_ch}", "",
             "| precision | max micro-batch | peak GB | train img/s |", "|---|---|---|---|"]
    for p, r in results.items():
        ips = f"{r['img_per_s']:.1f}" if r["img_per_s"] else "-"
        lines.append(f"| {p} | {r['max_batch']} | {r['peak_gb']:.2f} | {ips} |")
    if "fp32" in results and "fp16" in results and results["fp32"]["max_batch"]:
        gain = results["fp16"]["max_batch"] / results["fp32"]["max_batch"] - 1
        lines.append(f"\nFP16 fits a {gain * 100:.0f}% larger micro-batch than FP32 under the same budget.")
    table = "\n".join(lines)
    print("\n" + table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text(table + "\n")
    (out / "results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2))


if __name__ == "__main__":
    main()
