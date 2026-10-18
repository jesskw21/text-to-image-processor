"""Train the attribute classifier used to score generated images.

It is trained on real images only (different seed from the diffusion model's
training set), and it stores reference feature statistics from a further
held-out real set, so the generator is never scored on its own training data.

    python train_evaluator.py
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from tinydiff.data import ATTRIBUTES, generate
from tinydiff.device import get_device
from tinydiff.evaluator import (AttributeClassifier, attribute_accuracy,
                                feature_stats, frechet_distance, predict)


def to_tensor(images, device):
    return torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/evaluator.pt")
    ap.add_argument("--n-train", type=int, default=30000)
    ap.add_argument("--n-ref", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = get_device(args.device)
    torch.manual_seed(0)

    xtr, ytr = generate(args.n_train, seed=1)
    xte, yte = generate(5000, seed=2)
    xref, _ = generate(args.n_ref, seed=3)
    xtr, ytr = to_tensor(xtr, device), torch.from_numpy(ytr).to(device)
    xte = to_tensor(xte, device)

    clf = AttributeClassifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        clf.train()
        perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(xtr), 256):
            idx = perm[i:i + 256]
            logits, _ = clf(xtr[idx])
            loss = sum(F.cross_entropy(logits[k], ytr[idx, j]) for j, k in enumerate(ATTRIBUTES))
            opt.zero_grad()
            loss.backward()
            opt.step()
        clf.eval()
        pred, _ = predict(clf, xte)
        print(f"epoch {epoch + 1}: test accuracy {attribute_accuracy(pred, yte)}")

    # Reference stats from held-out real images, plus a real-vs-real Frechet
    # distance at benchmark sample size: the noise floor for "no fidelity loss".
    _, fref = predict(clf, to_tensor(xref, device))
    mu, sigma = feature_stats(fref)
    _, fte = predict(clf, xte)
    floor = {}
    for n in (1000, 2000, 5000):
        m, s = feature_stats(fte[:n])
        floor[n] = frechet_distance(m, s, mu, sigma)
    print(f"real-vs-real Frechet distance (noise floor) by sample size: {floor}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": clf.state_dict(), "mu": mu, "sigma": sigma,
                "fd_floor": floor, "test_acc": attribute_accuracy(pred, yte)}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
