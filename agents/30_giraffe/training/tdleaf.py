"""TD-Leaf(lambda) self-play refinement of the Giraffe network (Giraffe stage 3).

Each iteration samples positions from the bootstrap set, applies one random legal move
(Lai's trick for diversity), and lets the engine play ``--plies`` plies against itself
with a fixed-depth search driven by the current network. For every ply the search score
``V_t`` and the principal-variation leaf are recorded; the leaf's features are then trained
towards ``V_t + sum_k lambda^k (V_{t+k+1} - V_{t+k})`` (terminal values come from the rules).
A replay buffer of the last few iterations plus a slice of the bootstrap data keeps the
updates from oscillating. Every ``--gate-every`` iterations the candidate plays an
in-process arena against the last accepted checkpoint and is kept only if it does not lose.

    python training/tdleaf.py --init models/giraffe.npz --iterations 40 --workers 6
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import time
from collections import deque
from pathlib import Path

import chess
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parent
for path in (str(HERE), str(AGENT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import giraffe_eval as ge  # noqa: E402
import selfplay_arena  # noqa: E402
from bootstrap import train  # noqa: E402
from giraffe_search import MATE_BOUND, Searcher  # noqa: E402
from model import GiraffeNet, cp_to_target, load_weights, save_weights  # noqa: E402

VALUE_CLAMP = 1500.0
LEAF_MISMATCH_CP = 80  # skip leaves whose static value disagrees with the search score


def trajectory(
    evaluator: ge.Evaluator, fen: str, plies: int, depth: int, lam: float
) -> tuple[list[np.ndarray], list[float]]:
    board = chess.Board(fen)
    searcher = Searcher(evaluator)
    values: list[float] = []  # V_t from white's point of view
    leaves: list[tuple[np.ndarray, bool, int] | None] = []  # features, side to move, static score
    for _ in range(plies):
        if board.is_game_over(claim_draw=True):
            break
        searcher.remember(board)
        move, score = searcher.search(board, 1e9, depth)
        pv = searcher.principal_variation(board, depth + 6)
        leaf = board.copy(stack=False)
        for pv_move in pv:
            leaf.push(pv_move)
        leaf_score = score if leaf.turn == board.turn else -score
        usable = abs(score) < MATE_BOUND and not leaf.is_check() and abs(evaluator(leaf) - leaf_score) <= LEAF_MISMATCH_CP
        values.append(float(max(-VALUE_CLAMP, min(VALUE_CLAMP, score if board.turn == chess.WHITE else -score))))
        leaves.append((ge.board_features(leaf), leaf.turn, ge.hce_eval(leaf)) if usable else None)
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        terminal = 0.0 if outcome.winner is None else (VALUE_CLAMP if outcome.winner == chess.WHITE else -VALUE_CLAMP)
        values.append(terminal)

    xs: list[np.ndarray] = []
    ys: list[float] = []
    for t, leaf_info in enumerate(leaves):
        if leaf_info is None:
            continue
        target = values[t]
        weight = 1.0
        for k in range(t, len(values) - 1):
            target += weight * (values[k + 1] - values[k])
            weight *= lam
        x, leaf_turn, static = leaf_info
        xs.append(x)
        # the network fits the residual over the static score, from the leaf's side to move
        ys.append((target if leaf_turn == chess.WHITE else -target) - static)
    return xs, ys


def worker(args: tuple[np.ndarray, list[str], int, int, float, int]) -> tuple[np.ndarray, np.ndarray]:
    weights, fens, plies, depth, lam, seed = args
    rng = random.Random(seed)
    evaluator = ge.NetEvaluator(weights)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for fen in fens:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        if not moves:
            continue
        board.push(rng.choice(moves))  # one random move for diversity
        if board.is_game_over():
            continue
        x, y = trajectory(evaluator, board.fen(), plies, depth, lam)
        xs.extend(x)
        ys.extend(y)
    if not xs:
        return np.zeros((0, ge.N_INPUT), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, default=AGENT_DIR / "models" / "giraffe.npz")
    parser.add_argument("--out", type=Path, default=AGENT_DIR / "models" / "giraffe_tdleaf.npz")
    parser.add_argument("--data", type=Path, default=HERE / "data" / "search_d2.npz")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--positions", type=int, default=256, help="start positions per iteration")
    parser.add_argument("--plies", type=int, default=12)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay", type=int, default=4, help="iterations kept in the replay buffer")
    parser.add_argument("--anchor", type=float, default=0.5, help="bootstrap samples per TD sample")
    parser.add_argument("--gate-every", type=int, default=5)
    parser.add_argument("--gate-pairs", type=int, default=12)
    parser.add_argument("--gate-budget", type=float, default=0.15, help="seconds per move in the gate arena")
    parser.add_argument("--gate-depth", type=int, default=0, help="fixed depth for the gate arena (0 = use --gate-budget)")
    parser.add_argument("--checkpoints", type=Path, default=HERE / "data" / "tdleaf", help="per-iteration weight dumps")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=1, help="torch threads; more than one spin-waits on a loaded machine")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    torch.set_num_threads(arguments.threads)
    rng = random.Random(arguments.seed)
    data = np.load(arguments.data)
    fens = data["fens"]
    boot_x = data["features"]
    if "static" not in data:
        raise SystemExit("tdleaf needs a residual dataset with a 'static' array (training/relabel.py)")
    boot_y = cp_to_target(data["labels"].astype(np.float32) - data["static"].astype(np.float32))

    accepted = load_weights(arguments.init)
    weights = accepted.copy()
    replay: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=arguments.replay)
    model = GiraffeNet()
    log = open(AGENT_DIR / "results" / "tdleaf.log", "a", encoding="utf-8")  # noqa: SIM115

    def note(message: str) -> None:
        print(message, flush=True)
        log.write(message + "\n")
        log.flush()

    note(f"tdleaf start init={arguments.init} iterations={arguments.iterations} depth={arguments.depth} plies={arguments.plies} lam={arguments.lam}")
    with mp.Pool(arguments.workers) as pool:
        for iteration in range(1, arguments.iterations + 1):
            started = time.time()
            picks = [str(fens[rng.randrange(len(fens))]) for _ in range(arguments.positions)]
            chunk = max(1, len(picks) // arguments.workers)
            jobs = [
                (weights, picks[i : i + chunk], arguments.plies, arguments.depth, arguments.lam, arguments.seed * 100_000 + iteration * 100 + j)
                for j, i in enumerate(range(0, len(picks), chunk))
            ]
            parts = pool.map(worker, jobs)
            x_td = np.concatenate([p[0] for p in parts])
            y_td = np.concatenate([p[1] for p in parts])
            selfplay_s = time.time() - started
            if len(y_td) == 0:
                note(f"iter {iteration}: no samples")
                continue
            replay.append((x_td, y_td))
            x_all = np.concatenate([r[0] for r in replay])
            y_all = np.concatenate([r[1] for r in replay])
            n_anchor = int(len(y_all) * arguments.anchor)
            idx = np.random.default_rng(iteration).choice(len(boot_y), size=n_anchor, replace=False)
            x_train = np.concatenate([x_all, boot_x[idx].astype(np.float32)])
            y_train = np.concatenate([cp_to_target(y_all), boot_y[idx]])

            model.load_flat(weights)
            with torch.no_grad():
                before = model(torch.from_numpy(x_td)).numpy()
            td_err = float(np.mean(np.abs(before - cp_to_target(y_td)))) * ge.OUT_SCALE
            train(
                model,
                x_train,
                y_train,
                arguments.epochs,
                arguments.batch_size,
                arguments.lr,
                0.0,
                arguments.seed + iteration,
                log_prefix=f"  iter {iteration} ",
                holdout=0.0,
            )
            weights = model.to_flat()
            save_weights(model, arguments.checkpoints / f"tdleaf_iter{iteration:03d}.npz")
            note(
                f"iter {iteration}: {len(y_td)} td samples (replay {len(y_all)}, anchor {n_anchor}) "
                f"mean |td error| {td_err:.0f} cp, self-play {selfplay_s:.0f}s, total {time.time() - started:.0f}s"
            )

            if iteration % arguments.gate_every == 0:
                gate_budget = 1e9 if arguments.gate_depth else arguments.gate_budget
                gate_depth = arguments.gate_depth or 64
                wins, draws, losses = selfplay_arena.run(
                    weights, accepted, arguments.gate_pairs, gate_budget, gate_depth, arguments.workers, iteration, pool=pool
                )
                score = (wins + draws / 2) / max(1, wins + draws + losses)
                verdict = "accepted" if score >= 0.5 else "rejected"
                note(f"gate iter {iteration}: candidate vs accepted +{wins} ={draws} -{losses} ({score:.1%}) -> {verdict}")
                if score >= 0.5:
                    accepted = weights.copy()
                    np.savez(arguments.out, weights=accepted)
                else:
                    weights = accepted.copy()
    np.savez(arguments.out, weights=accepted)
    note(f"tdleaf done; accepted weights at {arguments.out}")


if __name__ == "__main__":
    main()
