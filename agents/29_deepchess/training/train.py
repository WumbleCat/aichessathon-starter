"""Train the DeepChess-style pairwise position evaluator.

Model (shared position encoder, scalar head):

    773 binary features -> 256 -> 32 -> 32 -> 1        (clipped ReLU between layers)

Objective (DeepChess pairwise preference, README option B):

    logit = (V_w(A) - V_w(B)) / tau
    loss  = BCEWithLogits(logit, [A better than B])       pairs drawn inside each batch
          + value_weight * SmoothL1(V, teacher_cp / 100)  auxiliary regression

``V_w`` is the value from White's point of view (the encoder sees every position from the
side to move, so V_w = V when White moves and -V otherwise). Labels come from Stockfish
searches produced by ``gen_data.py``; the network itself starts from random weights.

Output: ``models/deepchess.npz`` holding float32 weights with the output layer pre-scaled
so that the inference kernel's ``out * 100`` is centipawns.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

PAD = 773  # padding feature index; its embedding row is zero
SLOTS = 37  # 32 pieces + 4 castling + 1 en passant


# ------------------------------------------------------------------------------ features


def encode_features(boards: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """(N,64) piece codes + (N,5) meta -> (N,37) int16 feature indices, padded with PAD.

    Mirrors ``agent.features_to_indices``: side-to-move perspective, board flipped
    vertically when Black is to move, feature = (piece_type*2 + is_theirs)*64 + square.
    """
    n = boards.shape[0]
    turn = meta[:, 0].astype(np.int64)  # 1 white
    codes = boards.astype(np.int64)
    occupied = codes > 0
    pt = (codes - 1) % 6
    white = (codes >= 1) & (codes <= 6)
    ours = white == (turn[:, None] == 1)
    sq = np.arange(64, dtype=np.int64)[None, :]
    flip = np.where(turn == 1, 0, 56)[:, None]
    feat = (pt * 2 + np.where(ours, 0, 1)) * 64 + (sq ^ flip)
    feat = np.where(occupied, feat, PAD)
    feat.sort(axis=1)  # valid indices first, PAD (773) last
    pieces = feat[:, :32]
    castling = meta[:, 1].astype(np.int64)
    wk, wq, bk, bq = castling & 1, (castling >> 1) & 1, (castling >> 2) & 1, (castling >> 3) & 1
    ok = np.where(turn == 1, wk, bk)
    oq = np.where(turn == 1, wq, bq)
    tk = np.where(turn == 1, bk, wk)
    tq = np.where(turn == 1, bq, wq)
    extra = np.stack([
        np.where(ok == 1, 768, PAD), np.where(oq == 1, 769, PAD),
        np.where(tk == 1, 770, PAD), np.where(tq == 1, 771, PAD),
        np.where(meta[:, 2] == 1, 772, PAD),
    ], axis=1)
    out = np.concatenate([pieces, extra], axis=1).astype(np.int16)
    assert out.shape == (n, SLOTS)
    return out


def load_dataset(pattern: str, exclude_check: bool, exclude_capture: bool,
                 max_abs_cp: int) -> tuple[np.ndarray, ...]:
    feats, scores, turns, games, results = [], [], [], [], []
    game_offset = 0
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no data files match {pattern}")
    for f in files:
        with np.load(f) as d:
            boards, meta, sc, g = d["boards"], d["meta"], d["scores"], d["games"]
            res = d["results"]
        keep = np.abs(sc.astype(np.int64)) <= max_abs_cp
        if exclude_check:
            keep &= meta[:, 3] == 0
        if exclude_capture:
            keep &= meta[:, 4] == 0
        boards, meta, sc, g, res = boards[keep], meta[keep], sc[keep], g[keep], res[keep]
        feats.append(encode_features(boards, meta))
        scores.append(sc.astype(np.int16))
        turns.append(meta[:, 0].astype(np.int8))
        games.append(g.astype(np.int64) + game_offset)
        results.append(res.astype(np.int8))
        game_offset += int(g.max()) + 1 if len(g) else 0
        print(f"loaded {f}: kept {keep.sum()}/{len(keep)}", flush=True)
    return (np.concatenate(feats), np.concatenate(scores), np.concatenate(turns),
            np.concatenate(games), np.concatenate(results))


def rebalance(scores: np.ndarray, cap_cp: int, share: float, seed: int) -> np.ndarray:
    """Indices that keep every near-equal position and only ``share`` of the decided ones.

    Generated games spend most of their plies in positions that are already decided: in this
    data only about a third of positions are inside +-150 cp, and the error analysis
    (training/analyse.py) shows the network spending its capacity on telling "won" from "very
    won" while getting the sign wrong in 1 balanced position in 7. The NNUE dataset study
    recommends at least half the set inside +-100 cp for the same reason.
    """
    rng = np.random.default_rng(seed)
    near = np.abs(scores.astype(np.int64)) < cap_cp
    decided = np.nonzero(~near)[0]
    keep_decided = decided[rng.random(decided.shape[0]) < share]
    idx = np.concatenate([np.nonzero(near)[0], keep_decided])
    idx.sort()
    return idx


# --------------------------------------------------------------------------------- model


class DeepChessNet(nn.Module):
    def __init__(self, hidden1: int = 256, hidden2: int = 32, hidden3: int = 32) -> None:
        super().__init__()
        self.embed = nn.EmbeddingBag(PAD + 1, hidden1, mode="sum", padding_idx=PAD)
        self.b1 = nn.Parameter(torch.zeros(hidden1))
        self.l2 = nn.Linear(hidden1, hidden2)
        self.l3 = nn.Linear(hidden2, hidden3)
        self.l4 = nn.Linear(hidden3, 1)
        self.log_tau = nn.Parameter(torch.zeros(()))
        nn.init.uniform_(self.embed.weight, -0.05, 0.05)
        with torch.no_grad():
            self.embed.weight[PAD].zero_()

    def latent(self, idx: torch.Tensor) -> torch.Tensor:
        x = (self.embed(idx) + self.b1).clamp(0.0, 1.0)
        x = self.l2(x).clamp(0.0, 1.0)
        x = self.l3(x).clamp(0.0, 1.0)
        return x

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.l4(self.latent(idx)).squeeze(-1)


def pairwise_loss(v_white: torch.Tensor, cp_white: torch.Tensor, tau: torch.Tensor,
                  margin: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pair every position in the batch with a random other one."""
    n = v_white.shape[0]
    perm = torch.randperm(n, device=v_white.device)
    diff_cp = cp_white - cp_white[perm]
    valid = diff_cp.abs() >= margin
    if valid.sum() == 0:
        zero = v_white.sum() * 0.0
        return zero, zero, 0
    logit = ((v_white - v_white[perm]) / tau)[valid]
    target = (diff_cp[valid] > 0).float()
    loss = F.binary_cross_entropy_with_logits(logit, target)
    acc = ((logit > 0).float() == target).float().mean()
    return loss, acc, int(valid.sum())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum()))


def evaluate_model(model: DeepChessNet, feats: torch.Tensor, cp_stm: torch.Tensor,
                   turn: torch.Tensor, margin: float, batch: int = 16384) -> dict[str, float]:
    model.eval()
    vs = []
    with torch.no_grad():
        for i in range(0, feats.shape[0], batch):
            vs.append(model(feats[i:i + batch].long()))
    v = torch.cat(vs)
    sign = torch.where(turn == 1, 1.0, -1.0)
    v_w, cp_w = v * sign, cp_stm * sign
    tau = model.log_tau.exp()
    with torch.no_grad():
        pl, pacc, npairs = pairwise_loss(v_w, cp_w, tau, margin)
    v_np, cp_np = v.detach().cpu().numpy(), cp_stm.detach().cpu().numpy()
    # linear calibration V -> cp on this set (used to report MAE in centipawns)
    a, b = np.polyfit(v_np, cp_np, 1)
    mae = float(np.abs(a * v_np + b - cp_np).mean())
    model.train()
    return {
        "pair_loss": float(pl), "pair_acc": float(pacc), "pairs": npairs,
        "spearman": spearman(v_np, cp_np), "mae_cp": mae, "scale": float(a),
        "offset": float(b), "tau": float(tau),
    }


def export(model: DeepChessNet, scale: float, path: Path) -> None:
    """Write the numpy weights the agent kernel reads; fold the cp calibration in."""
    w1 = model.embed.weight.detach()[:PAD].cpu().numpy().astype(np.float32)
    factor = scale / 100.0  # kernel multiplies the output by 100 to get centipawns
    np.savez(
        path,
        w1=w1, b1=model.b1.detach().cpu().numpy().astype(np.float32),
        w2=model.l2.weight.detach().t().contiguous().cpu().numpy().astype(np.float32),
        b2=model.l2.bias.detach().cpu().numpy().astype(np.float32),
        w3=model.l3.weight.detach().t().contiguous().cpu().numpy().astype(np.float32),
        b3=model.l3.bias.detach().cpu().numpy().astype(np.float32),
        w4=(model.l4.weight.detach()[0] * factor).cpu().numpy().astype(np.float32),
        b4=(model.l4.bias.detach() * factor).cpu().numpy().astype(np.float32),
    )


# ---------------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA / "chunk_*.npz"))
    parser.add_argument("--out", default=str(MODELS / "deepchess.npz"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=30.0, help="cp margin for pairs")
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--max-abs-cp", type=int, default=2000)
    parser.add_argument("--keep-check", action="store_true")
    parser.add_argument("--keep-capture", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--init",
        default=None,
        help="a .pt checkpoint to fine-tune from (default: random initialisation)",
    )
    parser.add_argument(
        "--wdl",
        action="store_true",
        help="train the value head in win-probability space instead of centipawns",
    )
    parser.add_argument("--wdl-scale", type=float, default=400.0, help="cp per logit unit")
    parser.add_argument(
        "--result-weight",
        type=float,
        default=0.3,
        help="how much of the WDL target is the game's actual outcome rather than the "
        "teacher's score (0 = score only)",
    )
    parser.add_argument("--balance-cp", type=int, default=150)
    parser.add_argument(
        "--balance-share",
        type=float,
        default=1.0,
        help="keep only this share of positions outside +-balance-cp (1.0 = keep everything)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (cuda when the installed torch can see a GPU), cpu, or cuda. The shipped "
        "weights are a device-independent .npz, and the agent always runs on CPU.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    MODELS.mkdir(exist_ok=True)

    feats, scores, turns, games, results = load_dataset(
        args.data, not args.keep_check, not args.keep_capture, args.max_abs_cp)
    if args.balance_share < 1.0:
        idx = rebalance(scores, args.balance_cp, args.balance_share, args.seed)
        before = feats.shape[0]
        feats, scores, turns = feats[idx], scores[idx], turns[idx]
        games, results = games[idx], results[idx]
        near = float((np.abs(scores.astype(np.int64)) < args.balance_cp).mean())
        print(f"rebalanced {before} -> {feats.shape[0]} positions, "
              f"{near:.0%} inside +-{args.balance_cp} cp", flush=True)
    n = feats.shape[0]
    # split by game so that neighbouring positions never straddle train/val
    rng = np.random.default_rng(args.seed)
    n_games = int(games.max()) + 1
    val_games = rng.random(n_games) < args.val_fraction
    is_val = val_games[games]
    print(f"{n} positions, {n_games} games, {is_val.sum()} validation positions", flush=True)

    feats_t = torch.from_numpy(feats.astype(np.int64))
    cp_t = torch.from_numpy(scores.astype(np.float32))
    turn_t = torch.from_numpy(turns.astype(np.float32))
    result_t = torch.from_numpy(results.astype(np.float32))  # +1/0/-1 from White
    tr = torch.from_numpy(np.nonzero(~is_val)[0])
    va = torch.from_numpy(np.nonzero(is_val)[0])

    model = DeepChessNet(hidden1=args.hidden)
    if args.init:
        # fine-tune: start from a trained checkpoint instead of random weights, so a run on a
        # smaller, targeted dataset sharpens the model rather than relearning it from nothing
        state = torch.load(args.init, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"initialised from {args.init}", flush=True)
    n_params = sum(p.numel() for p in model.parameters()) - model.embed.weight[PAD].numel()
    print(f"parameters: {n_params}", flush=True)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        # the whole dataset is a few hundred MB of int16/float32 and fits in 6 GB, so it is
        # moved once rather than per batch; the model is tiny, so the copy is the only cost
        print(f"device: {torch.cuda.get_device_name(device)}", flush=True)
        torch.backends.cudnn.benchmark = True
    else:
        print(f"device: cpu ({torch.__version__})", flush=True)
    model = model.to(device)
    feats_t = feats_t.to(device)
    cp_t = cp_t.to(device)
    turn_t = turn_t.to(device)
    result_t = result_t.to(device)
    tr = tr.to(device)
    va = va.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = max(1, len(tr) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05, anneal_strategy="cos")

    history = []
    step = 0
    started = time.time()
    for epoch in range(args.epochs):
        perm = tr[torch.randperm(len(tr), device=tr.device)]
        sums = {"pair": 0.0, "value": 0.0, "acc": 0.0}
        count = 0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch:(i + 1) * args.batch]
            x = feats_t[idx]
            cp = cp_t[idx]
            sign = torch.where(turn_t[idx] == 1, 1.0, -1.0)
            v = model(x)
            tau = model.log_tau.exp()
            ploss, pacc, _ = pairwise_loss(v * sign, cp * sign, tau, args.margin)
            if args.wdl:
                # Train in win-probability space rather than centipawns. A +1500 and a +3000
                # position are both simply "won", so the network stops spending capacity
                # telling them apart (its worst bucket by far: 2034 cp of error) and sharpens
                # where games are actually decided. The target mixes the teacher's score with
                # how the game really ended, which is how Stockfish and Lc0 train.
                scale = args.wdl_scale / 100.0  # v is in units of 100 cp
                target = torch.sigmoid(cp / args.wdl_scale)
                if args.result_weight > 0.0:
                    outcome = (result_t[idx] * sign + 1.0) / 2.0
                    target = (1.0 - args.result_weight) * target + args.result_weight * outcome
                vloss = F.binary_cross_entropy_with_logits(v / scale, target)
            else:
                vloss = F.smooth_l1_loss(v, cp / 100.0)
            loss = ploss + args.value_weight * vloss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                model.embed.weight[PAD].zero_()
            sums["pair"] += float(ploss.detach())
            sums["value"] += float(vloss.detach())
            sums["acc"] += float(pacc.detach())
            count += 1
            step += 1
        val = evaluate_model(model, feats_t[va], cp_t[va], turn_t[va], args.margin)
        record = {
            "epoch": epoch + 1,
            "train_pair_loss": sums["pair"] / count,
            "train_value_loss": sums["value"] / count,
            "train_pair_acc": sums["acc"] / count,
            **{f"val_{k}": v for k, v in val.items()},
            "elapsed_s": time.time() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        export(model, val["scale"], Path(args.out))
        torch.save(model.state_dict(), Path(args.out).with_suffix(".pt"))
        with open(Path(args.out).with_suffix(".json"), "w") as fh:
            json.dump({"args": vars(args), "parameters": n_params, "positions": n,
                       "history": history}, fh, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
