"""Held-out proxy for the residual net, cheaper than an arena.

Trains one residual network on 90% of the relabelled quiet positions and reports, on the
other 10%, how well ``static + network`` predicts the depth-2 search score compared with
``static`` alone, the correlation of predicted and true residual, and the shrinkage factor
that minimises the error (1.0 means the raw residual is best). The trained weights are
saved next to the data for arena use.

    python training/proxy_eval.py [--data training/data/search_d2.npz] [--epochs 20]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parent
for path in (str(HERE), str(AGENT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import giraffe_eval as ge  # noqa: E402
from bootstrap import train  # noqa: E402
from model import GiraffeNet, cp_to_target  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=HERE / "data" / "search_d2.npz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    torch.set_num_threads(1)  # more threads spin-wait on a loaded machine
    snap = arguments.data.with_name(arguments.data.stem + "_snap.npz")
    shutil.copy(arguments.data, snap)  # a running relabel.py may rewrite the original
    data = np.load(snap)
    x = data["features"].astype(np.float32)
    residual = (data["labels"] - data["static"]).astype(np.float32)
    n = len(residual)
    print(
        f"{n} positions; residual mean {residual.mean():.1f} std {residual.std():.1f} "
        f"mean|r| {np.abs(residual).mean():.1f} p90|r| {np.percentile(np.abs(residual), 90):.0f}"
    )
    rng = np.random.default_rng(arguments.seed)
    order = rng.permutation(n)
    n_val = n // 10
    val, tr = order[:n_val], order[n_val:]

    model = GiraffeNet()
    started = time.time()
    train(
        model,
        x[tr],
        cp_to_target(residual[tr]),
        arguments.epochs,
        256,
        arguments.lr,
        arguments.weight_decay,
        arguments.seed,
        holdout=0.1,
    )
    model.eval()
    with torch.no_grad():
        pred = (
            np.arctanh(np.clip(model(torch.from_numpy(x[val])).numpy(), -0.999, 0.999))
            * ge.OUT_SCALE
        )
    r = residual[val]
    alphas = np.linspace(0, 1.5, 31)
    maes = [float(np.abs(r - a * pred).mean()) for a in alphas]
    best = int(np.argmin(maes))
    print(
        f"held-out MAE static-only {np.abs(r).mean():.1f} cp, "
        f"static+net {np.abs(r - pred).mean():.1f} cp, "
        f"corr {np.corrcoef(r, pred)[0, 1]:.3f}, "
        f"best alpha {alphas[best]:.2f} -> {maes[best]:.1f} cp  "
        f"[{time.time() - started:.0f}s]"
    )
    out = arguments.data.with_name(arguments.data.stem + "_proxy_net.npz")
    np.savez(out, weights=model.to_flat())
    print(f"weights saved to {out}")


if __name__ == "__main__":
    main()
