"""Fidelity metrics for generated images.

An attribute classifier trained only on real images provides:
  * prompt accuracy - fraction of generated images whose predicted
    (shape, color, position, size) all match the prompt;
  * Frechet distance - FID computed in this classifier's 256-d feature space
    instead of Inception's, between generated images and held-out real ones.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy import linalg

from .data import ATTRIBUTES


class AttributeClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.conv = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU())
        self.heads = nn.ModuleDict({k: nn.Linear(256, len(v)) for k, v in ATTRIBUTES.items()})

    def features(self, x):
        return self.fc(self.conv(x))

    def forward(self, x):
        f = self.features(x)
        return {k: h(f) for k, h in self.heads.items()}, f


@torch.no_grad()
def predict(clf, x, batch=500):
    """x: float [-1,1] NCHW. Returns (pred labels [n,4], features [n,256]) on CPU."""
    preds, feats = [], []
    for i in range(0, len(x), batch):
        logits, f = clf(x[i:i + batch])
        preds.append(torch.stack([logits[k].argmax(1) for k in ATTRIBUTES], dim=1).cpu())
        feats.append(f.float().cpu())
    return torch.cat(preds), torch.cat(feats)


def feature_stats(feats):
    f = feats.double().numpy()
    return f.mean(0), np.cov(f, rowvar=False)


def frechet_distance(mu1, s1, mu2, s2):
    diff = mu1 - mu2
    # ReLU features leave some dimensions constant, so the covariances are
    # singular; a small ridge keeps sqrtm well defined.
    eps = np.eye(s1.shape[0]) * 1e-6
    covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps)).real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def attribute_accuracy(pred, target):
    """pred/target: [n,4] int. Returns dict of per-attribute and joint accuracy."""
    pred, target = torch.as_tensor(pred), torch.as_tensor(target)
    correct = pred == target
    out = {k: correct[:, i].float().mean().item() for i, k in enumerate(ATTRIBUTES)}
    out["all"] = correct.all(1).float().mean().item()
    return out


def load_evaluator(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    clf = AttributeClassifier()
    clf.load_state_dict(ck["model"])
    return clf.to(device).eval(), ck
