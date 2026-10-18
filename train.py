"""Train the text-to-image diffusion model.

    python train.py --steps 20000 --micro-batch 128 --accum 1 --precision fp16
"""
import argparse
import json
import time
from pathlib import Path

import torch

from tinydiff.data import DeviceDataset, Tokenizer, generate, save_grid, to_uint8
from tinydiff.device import autocast, get_device, sync
from tinydiff.diffusion import GaussianDiffusion
from tinydiff.model import ModelConfig, TextToImage
from tinydiff.samplers import CFGDenoiser, sample
from tinydiff.training import EMA, Trainer, save_checkpoint

PREVIEW_PROMPTS = [
    "a large red circle in the center",
    "a small blue square on the left",
    "a large yellow triangle at the top",
    "a small white circle at the bottom",
    "a large green square on the right",
    "a small purple triangle in the center",
    "a large blue triangle on the left",
    "a small red square at the top",
]


@torch.no_grad()
def preview(model, diffusion, tok, device, precision, path):
    tokens = torch.tensor([tok.encode(p) for p in PREVIEW_PROMPTS], device=device)
    null = torch.tensor([tok.null()], device=device)
    g = torch.Generator().manual_seed(0)
    x_T = torch.randn((len(tokens), 3, 32, 32), generator=g).to(device)
    with autocast(device, precision):
        den = CFGDenoiser(model, tokens, null, scale=3.0)
        x = sample(den, diffusion.alphas_cumprod, x_T, "dpmsolver", 20)
    save_grid(to_uint8(x), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--micro-batch", type=int, default=128)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--p-uncond", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} precision={args.precision} "
          f"effective_batch={args.micro_batch * args.accum} "
          f"({args.micro_batch} x {args.accum} accum)")

    tok = Tokenizer()
    images, labels = generate(args.n_train, seed=0)
    data = DeviceDataset(images, labels, tok, device)
    null = torch.tensor([tok.null()], device=device)

    cfg = ModelConfig(base_ch=args.base_ch)
    model = TextToImage(cfg, len(tok)).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    diffusion = GaussianDiffusion(device=device)
    trainer = Trainer(model, diffusion, device, null, args.precision, args.lr,
                      args.accum, args.p_uncond)
    ema = EMA(model, args.ema)

    log = open(out / "log.jsonl", "a")
    t0, seen = time.time(), 0
    for step in range(1, args.steps + 1):
        loss, gnorm = trainer.step(lambda: data.batch(args.micro_batch))
        ema.update(model)
        seen += args.micro_batch * args.accum

        if step % args.log_every == 0:
            sync(device)
            dt = time.time() - t0
            rec = {"step": step, "loss": loss.item(), "grad_norm": gnorm.item(),
                   "img_per_s": seen / dt, "loss_scale": trainer.scaler.get_scale()
                   if trainer.scaler.is_enabled() else 1.0}
            print(json.dumps(rec))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t0, seen = time.time(), 0

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(out / "ckpt.pt", model, ema, cfg, step, vars(args))
            preview(ema.model, diffusion, tok, device, args.precision,
                    out / f"preview_{step:06d}.png")
            t0, seen = time.time(), 0
    print(f"saved {out / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
