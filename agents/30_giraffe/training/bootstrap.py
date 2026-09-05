"""Supervised pretraining of the Giraffe network (stage 2).

    python training/bootstrap.py --data training/data/search_d2.npz --epochs 30 --out models/giraffe.npz

The network predicts the residual ``label - static`` where ``static`` is the handcrafted
score the evaluator adds back at play time; a data file without a ``static`` array is
treated as absolute labels (the legacy bootstrap set). Loss is mean squared error in the
network's tanh output space (targets are ``tanh(cp / OUT_SCALE)``). Ten percent of the data is held out for validation; the
checkpoint with the best validation loss is exported as the flat weight file the numba
evaluator loads.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model import GiraffeNet, cp_to_target, load_weights, save_weights  # noqa: E402

import giraffe_eval as ge  # noqa: E402


def train(
    model: GiraffeNet,
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    log_prefix: str = "",
    out: Path | None = None,
    holdout: float = 0.1,
) -> tuple[float, float]:
    """Train in place; returns (final train loss, best validation loss)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(targets)
    order = rng.permutation(n)
    n_val = int(n * holdout)
    val_idx, train_idx = order[:n_val], order[n_val:]
    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(targets.astype(np.float32))
    x_val, y_val = x[val_idx], y[val_idx]
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, epochs))
    best_val = float("inf")
    train_loss = float("nan")
    for epoch in range(epochs):
        model.train()
        perm = train_idx[rng.permutation(len(train_idx))]
        total = 0.0
        count = 0
        started = time.time()
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = x[idx], y[idx]
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        scheduler.step()
        train_loss = total / max(1, count)
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val) if n_val else torch.zeros(0)
            val_loss = float(torch.mean((val_pred - y_val) ** 2)) if n_val else train_loss
            # error in centipawns after undoing the tanh, on the held-out set
            cp_err = (
                float(torch.mean(torch.abs(torch.atanh(val_pred.clamp(-0.999, 0.999)) - torch.atanh(y_val.clamp(-0.999, 0.999))))) * ge.OUT_SCALE
                if n_val
                else float("nan")
            )
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            if out is not None:
                save_weights(model, out)
        print(
            f"{log_prefix}epoch {epoch + 1}/{epochs} train {train_loss:.5f} val {val_loss:.5f} "
            f"(~{cp_err:.0f} cp mean abs err) {'*' if improved else ''} {time.time() - started:.0f}s",
            flush=True,
        )
    return train_loss, best_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=HERE / "data" / "bootstrap.npz")
    parser.add_argument("--out", type=Path, default=HERE.parent / "models" / "giraffe.npz")
    parser.add_argument("--init", type=Path, default=None, help="flat weights to start from")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=1, help="torch threads; more than one spin-waits on a loaded machine")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    torch.set_num_threads(arguments.threads)
    data = np.load(arguments.data)
    features = data["features"]
    labels = data["labels"].astype(np.float32)
    if "static" in data:
        labels = labels - data["static"].astype(np.float32)
        print(f"{len(labels)} positions, residual (label - static) mean {labels.mean():.0f} std {labels.std():.0f} cp")
    else:
        print(f"{len(labels)} positions, absolute label std {labels.std():.0f} cp")
    model = GiraffeNet()
    if arguments.init is not None:
        model.load_flat(load_weights(arguments.init))
    train(
        model,
        features,
        cp_to_target(labels),
        arguments.epochs,
        arguments.batch_size,
        arguments.lr,
        arguments.weight_decay,
        arguments.seed,
        out=arguments.out,
    )
    print(f"best checkpoint written to {arguments.out}")


if __name__ == "__main__":
    main()
