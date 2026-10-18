"""Gradient accumulation must reproduce the full-batch gradient exactly."""
import copy

import torch

from tinydiff.data import Tokenizer
from tinydiff.diffusion import GaussianDiffusion
from tinydiff.model import ModelConfig, TextToImage
from tinydiff.training import Trainer

TINY = ModelConfig(img_size=16, base_ch=32, ch_mults=(1, 2), num_res=1,
                   attn_res=(8,), text_dim=32, text_depth=1, dropout=0.0)


class FixedNoiseDiffusion(GaussianDiffusion):
    """Feeds pre-drawn (t, noise) chunks so both runs see identical randomness."""

    def __init__(self, chunks):
        super().__init__()
        self.chunks = list(chunks)

    def training_loss(self, model, x0, tokens, null_tokens, p_uncond=0.1, t=None, noise=None):
        t, noise = self.chunks.pop(0)
        return super().training_loss(model, x0, tokens, null_tokens, 0.0, t, noise)


def grads_after_step(model, x, tokens, t, noise, accum):
    tok = Tokenizer()
    null = torch.tensor([tok.null()])
    micro = len(x) // accum
    sl = [slice(i * micro, (i + 1) * micro) for i in range(accum)]
    diffusion = FixedNoiseDiffusion([(t[s], noise[s]) for s in sl])
    batches = iter([(x[s], tokens[s]) for s in sl])
    trainer = Trainer(model, diffusion, torch.device("cpu"), null, precision="fp32",
                      accum=accum, grad_clip=1e9)
    trainer.step(lambda: next(batches))
    return [p.grad.clone() for p in model.parameters()]


def test_grad_accumulation_matches_full_batch():
    torch.manual_seed(0)
    tok = Tokenizer()
    base = TextToImage(TINY, len(tok))
    n = 8
    x = torch.randn(n, 3, 16, 16)
    tokens = torch.tensor([tok.encode("a small red circle on the left")] * n)
    t = torch.randint(0, 1000, (n,))
    noise = torch.randn_like(x)

    full = grads_after_step(copy.deepcopy(base), x, tokens, t, noise, accum=1)
    for accum in (2, 4):
        acc = grads_after_step(copy.deepcopy(base), x, tokens, t, noise, accum=accum)
        for g_full, g_acc in zip(full, acc):
            torch.testing.assert_close(g_acc, g_full, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    test_grad_accumulation_matches_full_batch()
    print("ok")
