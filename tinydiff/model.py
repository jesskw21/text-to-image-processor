"""Text encoder + text-conditioned UNet noise predictor."""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import MAX_LEN, PAD_ID


@dataclass
class ModelConfig:
    img_size: int = 32
    base_ch: int = 64
    ch_mults: tuple = (1, 2, 2)
    num_res: int = 2
    attn_res: tuple = (16, 8)
    text_dim: int = 128
    text_depth: int = 2
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, dim, depth, heads=4):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(dim)

    def forward(self, tokens):
        mask = tokens == PAD_ID  # True = ignore
        h = self.tok(tokens) + self.pos[:, : tokens.shape[1]]
        h = self.ln(self.enc(h, src_key_padding_mask=mask))
        keep = (~mask).unsqueeze(-1).to(h.dtype)
        pooled = (h * keep).sum(1) / keep.sum(1)
        return h, mask, pooled


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


def zero_init(m):
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    return m


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout)
        self.norm2 = nn.GroupNorm(32, cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = zero_init(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return self.skip(x) + h


class SpatialTransformer(nn.Module):
    """Self-attention over pixels, then cross-attention from pixels to text tokens."""

    def __init__(self, ch, ctx_dim, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.proj_in = nn.Conv2d(ch, ch, 1)
        self.ln1 = nn.LayerNorm(ch)
        self.self_attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(ch)
        self.cross_attn = nn.MultiheadAttention(
            ch, heads, kdim=ctx_dim, vdim=ctx_dim, batch_first=True
        )
        self.ln3 = nn.LayerNorm(ch)
        self.ff = nn.Sequential(nn.Linear(ch, 4 * ch), nn.GELU(), nn.Linear(4 * ch, ch))
        self.proj_out = zero_init(nn.Conv2d(ch, ch, 1))

    def forward(self, x, ctx, ctx_mask):
        b, c, hgt, wid = x.shape
        h = self.proj_in(self.norm(x)).flatten(2).transpose(1, 2)
        q = self.ln1(h)
        h = h + self.self_attn(q, q, q, need_weights=False)[0]
        q = self.ln2(h)
        h = h + self.cross_attn(q, ctx, ctx, key_padding_mask=ctx_mask, need_weights=False)[0]
        h = h + self.ff(self.ln3(h))
        return x + self.proj_out(h.transpose(1, 2).reshape(b, c, hgt, wid))


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Block(nn.ModuleList):
    """Runs a list of layers, passing each the inputs it needs."""

    def forward(self, x, emb, ctx, ctx_mask):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, ctx, ctx_mask)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        base, emb_dim = cfg.base_ch, cfg.base_ch * 4
        self.base = base
        self.time_mlp = nn.Sequential(
            nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.text_proj = nn.Linear(cfg.text_dim, emb_dim)
        self.inp = nn.Conv2d(3, base, 3, padding=1)

        def attn(ch, res):
            return [SpatialTransformer(ch, cfg.text_dim)] if res in cfg.attn_res else []

        self.down = nn.ModuleList()
        ch, res, skips = base, cfg.img_size, [base]
        for i, m in enumerate(cfg.ch_mults):
            out = base * m
            for _ in range(cfg.num_res):
                self.down.append(Block([ResBlock(ch, out, emb_dim, cfg.dropout), *attn(out, res)]))
                ch = out
                skips.append(ch)
            if i != len(cfg.ch_mults) - 1:
                self.down.append(Block([Downsample(ch)]))
                skips.append(ch)
                res //= 2

        self.mid = Block([
            ResBlock(ch, ch, emb_dim, cfg.dropout),
            SpatialTransformer(ch, cfg.text_dim),
            ResBlock(ch, ch, emb_dim, cfg.dropout),
        ])

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(cfg.ch_mults))):
            out = base * m
            for j in range(cfg.num_res + 1):
                layers = [ResBlock(ch + skips.pop(), out, emb_dim, cfg.dropout), *attn(out, res)]
                ch = out
                if i != 0 and j == cfg.num_res:
                    layers.append(Upsample(ch))
                    res *= 2
                self.up.append(Block(layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch), nn.SiLU(), zero_init(nn.Conv2d(ch, 3, 3, padding=1))
        )

    def forward(self, x, t, ctx, ctx_mask, pooled):
        emb = self.time_mlp(timestep_embedding(t, self.base)) + self.text_proj(pooled)
        h = self.inp(x)
        hs = [h]
        for block in self.down:
            h = block(h, emb, ctx, ctx_mask)
            hs.append(h)
        h = self.mid(h, emb, ctx, ctx_mask)
        for block in self.up:
            h = block(torch.cat([h, hs.pop()], dim=1), emb, ctx, ctx_mask)
        return self.out(h)


class TextToImage(nn.Module):
    """Predicts the noise eps added to x_t, conditioned on timestep and caption."""

    def __init__(self, cfg: ModelConfig, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.text = TextEncoder(vocab_size, cfg.text_dim, cfg.text_depth)
        self.unet = UNet(cfg)

    def encode_text(self, tokens):
        return self.text(tokens)

    def denoise(self, x, t, text):
        ctx, mask, pooled = text
        return self.unet(x, t, ctx, mask, pooled)

    def forward(self, x, t, tokens):
        return self.denoise(x, t, self.encode_text(tokens))
