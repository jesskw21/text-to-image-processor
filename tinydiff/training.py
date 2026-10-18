"""Mixed-precision training step with gradient accumulation, plus EMA and checkpoints."""
import copy

import torch

from .data import Tokenizer
from .device import PRECISIONS, autocast
from .model import ModelConfig, TextToImage


class Trainer:
    """One optimizer step = `accum` micro-batches of size `micro_batch`.

    Mixed precision: the forward pass runs under autocast (convs/matmuls in
    fp16, reductions such as norms and the loss in fp32), while master weights
    and optimizer state stay fp32. With fp16 a GradScaler multiplies the loss
    before backward so small gradients don't underflow, unscales before
    clipping, and skips the step if any gradient overflowed to inf/NaN.

    Gradient accumulation: each micro-batch loss is divided by `accum` and
    backpropagated; gradients sum in .grad, so after `accum` micro-batches
    .grad equals the gradient of the mean loss over the full effective batch.
    """

    def __init__(self, model, diffusion, device, null_tokens, precision="fp16",
                 lr=2e-4, accum=1, p_uncond=0.1, grad_clip=1.0, warmup=500):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {list(PRECISIONS)}")
        self.model, self.diffusion, self.device = model, diffusion, device
        self.null_tokens = null_tokens
        self.precision, self.accum = precision, accum
        self.p_uncond, self.grad_clip = p_uncond, grad_clip
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / warmup)
        )
        self.scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")

    def step(self, get_batch, on_forward=None):
        """get_batch() -> (x0, tokens). on_forward() is called after each
        micro-batch forward (used by the memory benchmark). Returns
        (mean loss tensor, grad norm tensor)."""
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device=self.device)
        for _ in range(self.accum):
            x0, tokens = get_batch()
            with autocast(self.device, self.precision):
                loss = self.diffusion.training_loss(
                    self.model, x0, tokens, self.null_tokens, self.p_uncond
                ) / self.accum
            if on_forward:
                on_forward()
            self.scaler.scale(loss).backward()
            total += loss.detach()
        self.scaler.unscale_(self.opt)
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        return total, gnorm


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p, 1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def save_checkpoint(path, model, ema, cfg, step, args):
    torch.save({
        "model": model.state_dict(),
        "ema": ema.model.state_dict(),
        "config": cfg.to_dict(),
        "step": step,
        "args": args,
    }, path)


def load_for_inference(path, device, use_ema=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    d = ck["config"]
    d["ch_mults"], d["attn_res"] = tuple(d["ch_mults"]), tuple(d["attn_res"])
    cfg = ModelConfig(**d)
    tok = Tokenizer()
    model = TextToImage(cfg, len(tok))
    model.load_state_dict(ck["ema" if use_ema else "model"])
    return model.to(device).eval(), tok, ck
