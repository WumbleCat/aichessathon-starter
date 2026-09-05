"""Measure what the network's move ordering buys: effective nodes to reach a fixed depth.

For a sample of positions the engine searches to a fixed depth once without the network and
once per --min-depth setting with it. Nodes are compared with every network call charged its
measured node-equivalent (latency x engine nodes/s), so a ratio below 1.0 means the ordering
saves more search than the network costs. CPU time is reported as well. Load-independent, no
games needed.

    python training/bench_ordering.py --model models/tiny_v1.npz --depth 5 --positions 40
"""

import argparse
import os
import sys
import time

import chess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from cf_encode import COMPACT_SIZE  # noqa: E402
from cf_infer import PolicyModel  # noqa: E402
from cf_search import Searcher  # noqa: E402

PIECES = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}


def compact_to_board(row: np.ndarray) -> chess.Board:
    """Inverse of cf_encode.compact for white to move (the mirror does not matter here)."""
    board = chess.Board(None)
    for sq in range(64):
        p = int(row[sq])
        if p == 0:
            continue
        colour = chess.WHITE if p <= 6 else chess.BLACK
        board.set_piece_at(sq, chess.Piece((p - 1) % 6 + 1, colour))
    board.turn = chess.WHITE
    bits = int(row[64])
    rights = ""
    if bits & 1:
        rights += "K"
    if bits & 2:
        rights += "Q"
    if bits & 4:
        rights += "k"
    if bits & 8:
        rights += "q"
    board.set_castling_fen(rights or "-")
    if row[65] != 255:
        board.ep_square = int(row[65])
    return board


def sample_positions(shard: str, n: int, seed: int) -> list[chess.Board]:
    z = np.load(shard)
    rng = np.random.default_rng(seed)
    boards = []
    for i in rng.permutation(len(z["x"])):
        row = z["x"][i]
        assert len(row) == COMPACT_SIZE
        try:
            b = compact_to_board(row)
        except (ValueError, AssertionError):
            continue
        if b.is_valid() and not b.is_game_over() and len(list(b.legal_moves)) > 8:
            boards.append(b)
        if len(boards) >= n:
            break
    return boards


def run(
    engine: Searcher, boards: list[chess.Board], depth: int
) -> tuple[int, float, int, list[str]]:
    nodes = 0
    cpu = 0.0
    calls = 0
    moves = []
    for b in boards:
        engine.tt.clear()
        engine.game_history.clear()
        c0 = time.process_time()
        res = engine.search(b.copy(), budget_s=3600.0, max_depth=depth)
        cpu += time.process_time() - c0
        nodes += res.nodes
        calls += engine.policy_calls
        moves.append(res.move.uci() if res.move else "0000")
    return nodes, cpu, calls, moves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(ROOT, "models", "chessformer.npz"))
    parser.add_argument("--shard", default=os.path.join(HERE, "data", "shard_d3_s2000.npz"))
    parser.add_argument("--positions", type=int, default=40)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--min-depths", default="3,4,5")
    parser.add_argument(
        "--pv-only", action="store_true", help="consult the network at PV nodes only"
    )
    parser.add_argument(
        "--rel-depth", type=int, default=64, help="network only within this many plies of the root"
    )
    parser.add_argument(
        "--policy-cost", type=int, default=-1, help="nodes charged per network call (-1: measure)"
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    boards = sample_positions(args.shard, args.positions, args.seed)
    print(f"{len(boards)} positions, depth {args.depth}", flush=True)
    base = Searcher()
    n0, cpu0, _, moves0 = run(base, boards, args.depth)
    nps = n0 / max(1e-6, cpu0)
    model = PolicyModel(args.model)
    model.warm_up()
    c0 = time.process_time()
    for _ in range(30):
        model.priors(boards[0])
    latency = (time.process_time() - c0) / 30
    # CPU-time measurements on the shared box swing several-fold with the load (hyper-thread
    # and cache contention), so the charge can be pinned for comparable runs
    cost = round(latency * nps) if args.policy_cost < 0 else args.policy_cost
    lines = [
        f"model {args.model} depth {args.depth} positions {len(boards)}",
        f"no model: nodes {n0} cpu {cpu0:.1f}s ({nps:.0f} nodes/s); "
        f"network {latency * 1000:.1f} ms; charging {cost} nodes per call",
    ]
    print(lines[-1], flush=True)
    for md in (int(x) for x in args.min_depths.split(",")):
        eng = Searcher(policy_fn=model.priors, policy_min_depth=md)
        eng.policy_node_cost = cost
        eng.policy_pv_only = args.pv_only
        eng.policy_rel_depth = args.rel_depth
        n1, cpu1, calls, moves1 = run(eng, boards, args.depth)
        raw = n1 - calls * cost
        agree = sum(a == b for a, b in zip(moves0, moves1, strict=True))
        tag = f"min_depth {md} rel_depth {args.rel_depth}{' pv-only' if args.pv_only else ''}"
        line = (
            f"{tag}: raw nodes {raw} ({raw / n0:.2f}x), charged {n1} ({n1 / n0:.2f}x), "
            f"cpu {cpu1:.1f}s ({cpu1 / cpu0:.2f}x), calls {calls} ({calls / len(boards):.1f}/pos), "
            f"same best move {agree}/{len(boards)}"
        )
        print(line, flush=True)
        lines.append(line)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    main()
