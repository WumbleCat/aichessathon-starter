"""Train the policy network on teacher-labelled shards and export weights.

Target for a position: p(a) = softmax(score(a) / T) over the labelled (legal) moves.
Loss = soft cross-entropy over legal moves (illegal moves masked out)
       + value_weight * MSE(value_head, tanh(best_score / 400)).

    python training/train.py --data training/data --channels 64 --blocks 5 --epochs 8

Outputs models/policy.npz (numpy weights for the engine), models/train_log.txt and torch
states in training/checkpoints/ (git-ignored, outside the shipped models/ directory).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.nn import functional as fn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from model import PolicyNet, count_params, export_numpy  # noqa: E402
from pn_encoding import NUM_ACTIONS, codes_to_planes  # noqa: E402


class Data:
    """All shards concatenated; labels stay sparse (CSR-like) until a batch is built."""

    def __init__(self, files: list[str], min_depth: int = 1) -> None:
        codes, meta, n_moves, best, value, result, ply = [], [], [], [], [], [], []
        l_action, l_score = [], []
        for f in files:
            d = np.load(f)
            keep = d["depth"] >= min_depth
            if not keep.any():
                continue
            # label rows belong to kept positions only
            pos_keep = keep[d["label_pos"]]
            codes.append(d["codes"][keep])
            meta.append(d["meta"][keep])
            n_moves.append(d["n_moves"][keep])
            best.append(d["best"][keep])
            value.append(d["value"][keep])
            result.append(d["result"][keep])
            ply.append(d["ply"][keep])
            l_action.append(d["label_action"][pos_keep])
            l_score.append(d["label_score"][pos_keep])
        self.codes = np.concatenate(codes)
        self.meta = np.concatenate(meta).astype(np.int64)
        self.n_moves = np.concatenate(n_moves).astype(np.int64)
        self.best = np.concatenate(best).astype(np.int64)
        self.value = np.concatenate(value).astype(np.float32)
        self.result = np.concatenate(result).astype(np.float32)
        self.ply = np.concatenate(ply).astype(np.int64)
        self.l_action = np.concatenate(l_action).astype(np.int64)
        self.l_score = np.concatenate(l_score).astype(np.float32)
        self.offsets = np.zeros(len(self.n_moves) + 1, dtype=np.int64)
        np.cumsum(self.n_moves, out=self.offsets[1:])
        assert self.offsets[-1] == len(self.l_action)

    def __len__(self) -> int:
        return len(self.meta)

    def batch(self, idx: np.ndarray, temperature: float) -> tuple[torch.Tensor, ...]:
        planes = codes_to_planes(self.codes[idx], self.meta[idx])
        n = len(idx)
        starts = self.offsets[idx]
        counts = self.n_moves[idx]
        rows = np.repeat(np.arange(n), counts)
        flat = np.concatenate([np.arange(s, s + c) for s, c in zip(starts, counts, strict=True)])
        actions = self.l_action[flat]
        scores = self.l_score[flat]
        # per-row softmax of scores / T
        best = np.full(n, -1e9, dtype=np.float32)
        np.maximum.at(best, rows, scores)
        w = np.exp(np.maximum((scores - best[rows]) / temperature, -40.0))
        z = np.zeros(n, dtype=np.float32)
        np.add.at(z, rows, w)
        w = w / z[rows]
        target = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
        target[rows, actions] = w
        legal = np.zeros((n, NUM_ACTIONS), dtype=bool)
        legal[rows, actions] = True
        value = np.tanh(self.value[idx] / 400.0)
        return (
            torch.from_numpy(planes),
            torch.from_numpy(target),
            torch.from_numpy(legal),
            torch.from_numpy(self.best[idx]),
            torch.from_numpy(value.astype(np.float32)),
        )


def evaluate(model: PolicyNet, data: Data, idx: np.ndarray, temperature: float, batch: int) -> dict:
    model.eval()
    tot_loss = tot_top1 = tot_top3 = tot_vloss = 0.0
    n = 0
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            planes, target, legal, best, value = data.batch(idx[s : s + batch], temperature)
            logits, v = model(planes)
            logits = logits.masked_fill(~legal, -1e9)
            logp = fn.log_softmax(logits, dim=1)
            loss = -(target * logp).sum(1)
            top = logits.topk(3, dim=1).indices
            tot_loss += loss.sum().item()
            tot_top1 += (top[:, 0] == best).float().sum().item()
            tot_top3 += (top == best[:, None]).any(1).float().sum().item()
            tot_vloss += fn.mse_loss(v, value, reduction="sum").item()
            n += len(best)
    model.train()
    return {
        "loss": tot_loss / n,
        "top1": tot_top1 / n,
        "top3": tot_top3 / n,
        "vloss": tot_vloss / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(HERE, "data"))
    parser.add_argument("--out", default=os.path.join(ROOT, "models"))
    parser.add_argument("--ckpt-dir", default=os.path.join(HERE, "checkpoints"))
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--time-limit-min", type=float, default=0.0, help="stop after this many minutes"
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-depth", type=int, default=1, help="drop positions whose teacher depth is lower"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    ckpt_dir = args.ckpt_dir
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data, "shard_*.npz")))
    if not files:
        raise SystemExit("no shards found in " + args.data)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * args.val_fraction))
    val_files, train_files = files[:n_val], files[n_val:]
    if not train_files:  # smoke runs with a single shard
        train_files = val_files
    print(f"shards: train {len(train_files)} val {len(val_files)}")
    train = Data(train_files, args.min_depth)
    val = Data(val_files, args.min_depth)
    print(f"positions: train {len(train):,} val {len(val):,}  labels {len(train.l_action):,}")

    model = PolicyNet(args.channels, args.blocks)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print("resumed from", args.resume)
    print(f"model: channels {args.channels} blocks {args.blocks} params {count_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train) // args.batch
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1, anneal_strategy="cos"
    )
    log_path = os.path.join(args.out, "train_log.txt")
    log = open(log_path, "a")  # noqa: SIM115  (open for the whole run, closed at the end)
    log.write(
        json.dumps(
            {
                "args": vars(args),
                "params": count_params(model),
                "train": len(train),
                "val": len(val),
            }
        )
        + "\n"
    )
    started = time.time()
    step = 0
    val_idx = np.arange(len(val))
    best_val = math.inf
    stop = False
    for epoch in range(args.epochs):
        perm = np.random.permutation(len(train))
        model.train()
        run_loss = run_top1 = 0.0
        run_n = 0
        for s in range(0, steps_per_epoch * args.batch, args.batch):
            idx = perm[s : s + args.batch]
            planes, target, legal, best, value = train.batch(idx, args.temperature)
            logits, v = model(planes)
            logits = logits.masked_fill(~legal, -1e9)
            logp = fn.log_softmax(logits, dim=1)
            ploss = -(target * logp).sum(1).mean()
            vloss = fn.mse_loss(v, value)
            loss = ploss + args.value_weight * vloss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += ploss.item() * len(idx)
            run_top1 += (logits.argmax(1) == best).float().sum().item()
            run_n += len(idx)
            if step % 200 == 0:
                el = (time.time() - started) / 60
                print(
                    f"ep {epoch} step {step}/{total_steps} loss {run_loss / run_n:.4f} "
                    f"top1 {run_top1 / run_n:.3f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {el:.1f} min",
                    flush=True,
                )
                run_loss = run_top1 = 0.0
                run_n = 0
            if args.time_limit_min and (time.time() - started) / 60 > args.time_limit_min:
                stop = True
                break
        metrics = evaluate(model, val, val_idx, args.temperature, 512)
        metrics.update({"epoch": epoch, "step": step, "minutes": (time.time() - started) / 60})
        print("VAL", json.dumps(metrics), flush=True)
        log.write(json.dumps(metrics) + "\n")
        log.flush()
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "policy_last.pt"))
        if metrics["loss"] < best_val:
            best_val = metrics["loss"]
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "policy.pt"))
            export_numpy(model, os.path.join(args.out, "policy.npz"))
            print("exported models/policy.npz", flush=True)
        if stop:
            break
    log.close()


if __name__ == "__main__":
    main()
