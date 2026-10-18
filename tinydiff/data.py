"""Procedurally generated text-to-image dataset.

Every image is a single shape described by four attributes (shape, color,
position, size), rendered with supersampled anti-aliasing and small random
jitter. The caption is built from the attributes, so the attribute labels
double as ground truth for measuring how well generated images follow the
prompt.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle"]
COLORS = {
    "red": (230, 50, 50),
    "green": (50, 190, 70),
    "blue": (60, 100, 240),
    "yellow": (240, 220, 50),
    "purple": (160, 70, 210),
    "white": (235, 235, 235),
}
POSITIONS = {
    "left": (0.28, 0.50),
    "right": (0.72, 0.50),
    "top": (0.50, 0.28),
    "bottom": (0.50, 0.72),
    "center": (0.50, 0.50),
}
POS_PHRASE = {
    "left": "on the left",
    "right": "on the right",
    "top": "at the top",
    "bottom": "at the bottom",
    "center": "in the center",
}
SIZES = {"small": 0.13, "large": 0.22}  # radius as a fraction of image width

COLOR_NAMES = list(COLORS)
POS_NAMES = list(POSITIONS)
SIZE_NAMES = list(SIZES)
ATTRIBUTES = {
    "shape": SHAPES,
    "color": COLOR_NAMES,
    "position": POS_NAMES,
    "size": SIZE_NAMES,
}
BACKGROUND = (20, 20, 28)

PAD_ID, NULL_ID = 0, 1
MAX_LEN = 8


def caption_from_labels(label):
    """label: (shape, color, position, size) indices."""
    s, c, p, z = (int(v) for v in label)
    return f"a {SIZE_NAMES[z]} {COLOR_NAMES[c]} {SHAPES[s]} {POS_PHRASE[POS_NAMES[p]]}"


class Tokenizer:
    """Word-level tokenizer over the fixed caption vocabulary."""

    def __init__(self):
        words = {"a"}
        words.update(SHAPES, COLOR_NAMES, SIZE_NAMES)
        for phrase in POS_PHRASE.values():
            words.update(phrase.split())
        self.itos = ["<pad>", "<null>"] + sorted(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        words = text.lower().replace(",", " ").split()
        unknown = [w for w in words if w not in self.stoi]
        if unknown:
            raise ValueError(
                f"unknown words {unknown}; vocabulary is {self.itos[2:]}"
            )
        if len(words) > MAX_LEN:
            raise ValueError(f"prompt longer than {MAX_LEN} words")
        ids = [self.stoi[w] for w in words]
        return ids + [PAD_ID] * (MAX_LEN - len(ids))

    def null(self):
        # The unconditional "caption" for classifier-free guidance. It keeps one
        # unmasked token so attention never sees a fully masked key set.
        return [NULL_ID] + [PAD_ID] * (MAX_LEN - 1)


def render(label, rng, size=32, ss=4):
    s, c, p, z = (int(v) for v in label)
    S = size * ss
    bg = tuple(int(np.clip(v + rng.integers(-6, 7), 0, 255)) for v in BACKGROUND)
    img = Image.new("RGB", (S, S), bg)
    draw = ImageDraw.Draw(img)

    px, py = POSITIONS[POS_NAMES[p]]
    cx = (px + rng.uniform(-0.04, 0.04)) * S
    cy = (py + rng.uniform(-0.04, 0.04)) * S
    r = SIZES[SIZE_NAMES[z]] * rng.uniform(0.9, 1.1) * S
    color = tuple(
        int(np.clip(v + rng.integers(-15, 16), 0, 255)) for v in COLORS[COLOR_NAMES[c]]
    )

    shape = SHAPES[s]
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        h = r * 0.88
        draw.rectangle([cx - h, cy - h, cx + h, cy + h], fill=color)
    else:
        t = r * 1.15
        draw.polygon(
            [(cx, cy - t), (cx - 0.866 * t, cy + 0.5 * t), (cx + 0.866 * t, cy + 0.5 * t)],
            fill=color,
        )
    return np.asarray(img.resize((size, size), Image.LANCZOS))


def random_labels(n, rng):
    return np.stack(
        [rng.integers(0, len(v), n) for v in ATTRIBUTES.values()], axis=1
    ).astype(np.int64)


def generate(n, seed, size=32, cache_dir="data"):
    """Returns (images uint8 [n,H,W,3], labels int64 [n,4]); cached on disk."""
    path = Path(cache_dir) / f"shapes_n{n}_s{seed}_{size}px.npz"
    if path.exists():
        d = np.load(path)
        return d["images"], d["labels"]
    rng = np.random.default_rng(seed)
    labels = random_labels(n, rng)
    images = np.stack([render(l, rng, size) for l in labels])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=images, labels=labels)
    return images, labels


class DeviceDataset:
    """Holds the whole dataset on the accelerator; batches are random index draws.

    The dataset is small (50k x 3 x 32 x 32 uint8 = 150 MB), so this avoids
    DataLoader overhead entirely.
    """

    def __init__(self, images, labels, tokenizer, device):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(device)
        self.labels = torch.from_numpy(labels).to(device)
        tokens = [tokenizer.encode(caption_from_labels(l)) for l in labels]
        self.tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        self.device = device

    def __len__(self):
        return self.images.shape[0]

    def batch(self, b):
        idx = torch.randint(0, len(self), (b,), device=self.device)
        x = self.images[idx].float() / 127.5 - 1.0
        return x, self.tokens[idx]


def to_uint8(x):
    """[-1,1] float NCHW -> uint8 NHWC numpy."""
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def save_grid(images_uint8, path, nrow=8, scale=3):
    n, h, w, _ = images_uint8.shape
    ncol = min(nrow, n)
    nrows = (n + ncol - 1) // ncol
    grid = np.zeros((nrows * (h + 2) + 2, ncol * (w + 2) + 2, 3), dtype=np.uint8)
    for i, im in enumerate(images_uint8):
        r, c = divmod(i, ncol)
        grid[2 + r * (h + 2): 2 + r * (h + 2) + h, 2 + c * (w + 2): 2 + c * (w + 2) + w] = im
    img = Image.fromarray(grid)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
