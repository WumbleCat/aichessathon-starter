"""Train the Chessformer from random initialisation on shards written by gen_data.py.

Losses: cross-entropy of the policy logits against the teacher's best move (over all 4192
indices, so the net also learns which moves are legal), plus MSE of the tanh value against a
blend of the teacher score and the game result.

    python training/train.py --data training/data --out models/chessformer.pt \
        --dim 64 --layers 2 --heads 4 --smol-hidden 32 --epochs 6

Checkpoints hold {"config", "state_dict", "meta"}; cf_infer loads them with torch.load.
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cf_encode import expand  # noqa: E402
from cf_model import Chessformer, Config, count_params  # noqa: E402


def load_shards(pattern: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no shards match {pattern}")
    parts = {"x": [], "move": [], "score": [], "depth": [], "result": []}
    for f in files:
        z = np.load(f)
        for k in parts:
            parts[k].append(z[k])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    print(f"loaded {len(files)} shards, {len(data['x'])} positions")
    return data


def value_target(score: np.ndarray, result: np.ndarray, result_weight: float) -> np.ndarray:
    v = np.tanh(score.astype(np.float32) / 500.0)
    return (1.0 - result_weight) * v + result_weight * result.astype(np.float32)


def evaluate(model: Chessformer, x: np.ndarray, move: np.ndarray, vt: np.ndarray, bs: int) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    top3 = 0
    ploss = 0.0
    vloss = 0.0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(expand(x[i : i + bs]))
            mb = torch.from_numpy(move[i : i + bs].astype(np.int64))
            vb = torch.from_numpy(vt[i : i + bs])
            logits, value = model(xb)
            ploss += F.cross_entropy(logits, mb, reduction="sum").item()
            vloss += F.mse_loss(value, vb, reduction="sum").item()
            pred = logits.argmax(dim=1)
            correct += (pred == mb).sum().item()
            t3 = logits.topk(3, dim=1).indices
            top3 += (t3 == mb[:, None]).any(dim=1).sum().item()
            total += len(mb)
    model.train()
    return {
        "policy_loss": ploss / total,
        "value_mse": vloss / total,
        "top1": correct / total,
        "top3": top3 / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(HERE, "data"))
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--out", default=os.path.join(ROOT, "models", "chessformer.pt"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--smol-dim", type=int, default=16)
    parser.add_argument("--smol-hidden", type=int, default=32)
    parser.add_argument("--policy-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--result-weight", type=float, default=0.3)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-positions", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = load_shards(os.path.join(args.data, args.pattern))
    n = len(data["x"])
    perm = rng.permutation(n)
    if args.max_positions:
        perm = perm[: args.max_positions]
        n = len(perm)
    n_val = max(1000, int(n * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    vt = value_target(data["score"], data["result"], args.result_weight)
    x_val, m_val, v_val = data["x"][val_idx], data["move"][val_idx], vt[val_idx]

    cfg = Config(
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        smol_dim=args.smol_dim,
        smol_hidden=args.smol_hidden,
        policy_dim=args.policy_dim,
    )
    model = Chessformer(cfg)
    if args.resume:
        blob = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(blob["state_dict"])
        print(f"resumed from {args.resume}")
    print(f"config {cfg.as_dict()} params {count_params(model) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(train_idx) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(500, total_steps // 10)

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))

    history = []
    step = 0
    t0 = time.time()
    model.train()
    for epoch in range(args.epochs):
        order = rng.permutation(train_idx)
        run_p = run_v = 0.0
        seen = 0
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            xb = torch.from_numpy(expand(data["x"][idx]))
            mb = torch.from_numpy(data["move"][idx].astype(np.int64))
            vb = torch.from_numpy(vt[idx])
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            logits, value = model(xb)
            p_loss = F.cross_entropy(logits, mb)
            v_loss = F.mse_loss(value, vb)
            loss = p_loss + args.value_weight * v_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            run_p += p_loss.item() * len(idx)
            run_v += v_loss.item() * len(idx)
            seen += len(idx)
            if step % 200 == 0:
                print(
                    f"epoch {epoch + 1} step {step}/{total_steps} policy {run_p / seen:.3f} "
                    f"value {run_v / seen:.4f} lr {lr_at(step):.2e} {time.time() - t0:.0f}s",
                    flush=True,
                )
        val = evaluate(model, x_val, m_val, v_val, 512)
        record = {
            "epoch": epoch + 1,
            "train_policy": run_p / max(1, seen),
            "train_value": run_v / max(1, seen),
            **val,
            "elapsed_s": time.time() - t0,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save(
            {
                "config": cfg.as_dict(),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "meta": {
                    "positions": int(n),
                    "epochs": epoch + 1,
                    "history": history,
                    "args": vars(args),
                    "data_shards": sorted(os.path.basename(f) for f in glob.glob(os.path.join(args.data, args.pattern))),
                },
            },
            args.out,
        )
        print(f"saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
