"""Train the (768 -> H) x 2 -> 1 NNUE from random initialisation on teacher-labelled positions.

    python training/train.py --hidden 256 --epochs 12 --out models/nnue_h256.pt

Data: every ``training/data/positions_*.txt`` line ``fen,stm_cp,stm_result,ply``.
Features are encoded once into ``training/data/encoded_<hash>.npz`` (int16 index lists).
Validation is split by game (consecutive lines sharing the game-length field), not by
position, so adjacent plies of one game cannot leak across the split.
Target: sigmoid(cp / SCALE) blended with the game result; loss is MSE in that space.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from nnue import NUM_FEATURES, QA, QB, SCALE  # noqa: E402

PAD = NUM_FEATURES  # padding index (zero row)
MAX_PIECES = 32


def encode_fen(fen: str, white_idx: np.ndarray, black_idx: np.ndarray) -> None:
    """Fill two length-32 index arrays (padded with PAD) for the white/black perspectives."""
    board_part = fen.split(" ", 1)[0]
    white_idx[:] = PAD
    black_idx[:] = PAD
    k = 0
    sq = 56  # a8
    for ch in board_part:
        if ch == "/":
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            colour = 0 if ch.isupper() else 1
            t = "pnbrqk".index(ch.lower())
            white_idx[k] = (colour * 6 + t) * 64 + sq
            black_idx[k] = ((colour ^ 1) * 6 + t) * 64 + (sq ^ 56)
            k += 1
            sq += 1


def load_dataset(pattern: str, limit: int | None) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no data files match {pattern}")
    sizes = [os.path.getsize(f) for f in files]
    tag = hashlib.md5((pattern + json.dumps([files, sizes, limit, "v2"])).encode()).hexdigest()[:10]
    cache = os.path.join(HERE, "data", f"encoded_{tag}.npz")
    if os.path.exists(cache):
        print(f"loading cached features {cache}")
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    lines: list[str] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            lines.extend(line.rstrip("\n") for line in fh if line.count(",") >= 3)
    if limit:
        lines = lines[:limit]
    n = len(lines)
    print(f"encoding {n} positions from {len(files)} files")
    stm_idx = np.full((n, MAX_PIECES), PAD, dtype=np.int16)
    nstm_idx = np.full((n, MAX_PIECES), PAD, dtype=np.int16)
    cp = np.zeros(n, dtype=np.float32)
    res = np.zeros(n, dtype=np.float32)
    game = np.zeros(n, dtype=np.int32)
    prev_len = None
    game_id = -1
    w = np.zeros(MAX_PIECES, dtype=np.int64)
    b = np.zeros(MAX_PIECES, dtype=np.int64)
    t0 = time.time()
    for i, line in enumerate(lines):
        fen, cp_s, res_s, len_s = line.rsplit(",", 3)
        if len_s != prev_len:
            game_id += 1
            prev_len = len_s
        game[i] = game_id
        encode_fen(fen, w, b)
        white_to_move = fen.split(" ")[1] == "w"
        if white_to_move:
            stm_idx[i] = w
            nstm_idx[i] = b
        else:
            stm_idx[i] = b
            nstm_idx[i] = w
        cp[i] = float(cp_s)
        res[i] = float(res_s)
        if i % 200000 == 0 and i:
            print(f"  {i}/{n} ({time.time() - t0:.0f}s)")
    data: dict[str, np.ndarray] = {
        "stm": stm_idx,
        "nstm": nstm_idx,
        "cp": cp,
        "res": res,
        "game": game,
    }
    np.savez(cache, stm=stm_idx, nstm=nstm_idx, cp=cp, res=res, game=game)
    return data


class NNUE(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.ft = nn.Embedding(NUM_FEATURES + 1, hidden, padding_idx=PAD)
        self.ft_bias = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(2 * hidden, 1)
        nn.init.normal_(self.ft.weight, std=0.05)
        with torch.no_grad():
            self.ft.weight[PAD].zero_()

    def forward(self, stm: torch.Tensor, nstm: torch.Tensor) -> torch.Tensor:
        a = torch.clamp(self.ft(stm).sum(1) + self.ft_bias, 0.0, 1.0)
        b = torch.clamp(self.ft(nstm).sum(1) + self.ft_bias, 0.0, 1.0)
        out: torch.Tensor = self.out(torch.cat([a, b], dim=1)).squeeze(1)
        return out

    def clip(self) -> None:
        with torch.no_grad():
            lim1 = 32767 / QA / 40  # keep the int16 accumulator far from overflow
            self.ft.weight.clamp_(-lim1, lim1)
            self.ft.weight[PAD].zero_()
            self.out.weight.clamp_(-32767 / QB, 32767 / QB)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(HERE, "data", "positions_*.txt"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lambda-result", type=float, default=0.1, help="weight of game result in target"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "models", "nnue.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    data = load_dataset(args.data, args.limit)
    n = len(data["cp"])
    if "game" in data:
        games = np.unique(data["game"])
        val_games = np.random.permutation(games)[: max(1, len(games) // 20)]
        is_val = np.isin(data["game"], val_games)
        val_ix = np.flatnonzero(is_val)
        train_ix = np.flatnonzero(~is_val)
        print(f"{len(games)} games; validation holds out {len(val_games)} whole games")
    else:  # old feature cache without game ids
        perm = np.random.permutation(n)
        n_val = min(50_000, n // 20)
        val_ix, train_ix = perm[:n_val], perm[n_val:]
    n_val = len(val_ix)
    stm = torch.from_numpy(data["stm"].astype(np.int64))
    nstm = torch.from_numpy(data["nstm"].astype(np.int64))
    cp = torch.from_numpy(data["cp"])
    res = torch.from_numpy(data["res"])
    target = (1 - args.lambda_result) * torch.sigmoid(cp / SCALE) + args.lambda_result * (
        res + 1
    ) / 2

    model = NNUE(args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.3)
    print(
        f"train {len(train_ix)} val {n_val} hidden {args.hidden} "
        f"params {sum(p.numel() for p in model.parameters())}"
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        np.random.shuffle(train_ix)
        total = 0.0
        batches = 0
        for start in range(0, len(train_ix), args.batch):
            ix = torch.from_numpy(train_ix[start : start + args.batch])
            pred = torch.sigmoid(model(stm[ix], nstm[ix]))
            loss = ((pred - target[ix]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.clip()
            total += loss.item()
            batches += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            vix = torch.from_numpy(val_ix)
            out = model(stm[vix], nstm[vix])
            vloss = ((torch.sigmoid(out) - target[vix]) ** 2).mean().item()
            mae_cp = (out * SCALE - cp[vix]).abs().clamp(max=2000).mean().item()
            sign_acc = ((out.sign() == cp[vix].sign()) | (cp[vix].abs() < 30)).float().mean().item()
        rec = {
            "epoch": epoch + 1,
            "train_loss": total / batches,
            "val_loss": vloss,
            "val_mae_cp": mae_cp,
            "val_sign_acc": sign_acc,
            "seconds": time.time() - t0,
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save(
            {
                "hidden": args.hidden,
                "state": model.state_dict(),
                "history": history,
                "args": vars(args),
                "positions": n,
            },
            args.out,
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
