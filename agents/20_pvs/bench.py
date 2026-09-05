"""Fixed-position benchmark and feature A/B for 20_pvs.

    python agents/20_pvs/bench.py                 # depth-limited run, prints nodes / nps / depth
    python agents/20_pvs/bench.py --time 2        # time-limited run
    python agents/20_pvs/bench.py --ab P_NULL     # same run with one feature disabled

CPU time is reported next to wall time because this machine is often oversubscribed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import pvs_search as ps  # noqa: E402
from pvs_board import Position, move_to_uci  # noqa: E402

POSITIONS = [
    ("start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("middlegame", "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("rook endgame", "8/8/4k3/8/2R5/8/3K4/6r1 w - - 0 1"),
    ("tactics", "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 8"),
]


def run(params: np.ndarray, depth: int | None, seconds: float | None, verbose: bool) -> dict:
    totals = {"nodes": 0, "qnodes": 0, "cpu": 0.0, "wall": 0.0, "depth": 0, "seldepth": 0,
              "tt_hits": 0, "beta_cuts": 0, "first_cuts": 0, "null_cuts": 0, "lmr_research": 0}
    for name, fen in POSITIONS:
        searcher = ps.Searcher(params.copy())
        pos = Position(fen)
        c0, w0 = time.process_time(), time.perf_counter()
        move, score, d, info = searcher.search(
            pos, time_budget=seconds, max_depth=depth or 64, verbose=verbose
        )
        cpu, wall = time.process_time() - c0, time.perf_counter() - w0
        print(
            f"{name:12s} depth {d:2d}/{info['seldepth']:2d} score {score:6d} "
            f"move {move_to_uci(move):6s} nodes {info['nodes']:9d} q% {100 * info['qnodes'] // max(1, info['nodes']):3d} "
            f"cpu {cpu:6.2f}s ({info['nodes'] / max(cpu, 1e-6) / 1000:6.0f} knps) wall {wall:6.2f}s "
            f"tt {info['tt_hits']} cut1 {100 * info['first_cuts'] // max(1, info['beta_cuts'])}% "
            f"null {info['null_cuts']} lmr_re {info['lmr_research']}"
        )
        for k in ("nodes", "qnodes", "tt_hits", "beta_cuts", "first_cuts", "null_cuts",
                  "lmr_research"):
            totals[k] += info[k]
        totals["depth"] += d
        totals["seldepth"] += info["seldepth"]
        totals["cpu"] += cpu
        totals["wall"] += wall
    n = len(POSITIONS)
    print(
        f"TOTAL nodes {totals['nodes']} q% {100 * totals['qnodes'] // max(1, totals['nodes'])} "
        f"cpu {totals['cpu']:.2f}s knps(cpu) {totals['nodes'] / max(totals['cpu'], 1e-6) / 1000:.0f} "
        f"avg depth {totals['depth'] / n:.1f} avg seldepth {totals['seldepth'] / n:.1f} "
        f"cut1 {100 * totals['first_cuts'] // max(1, totals['beta_cuts'])}%"
    )
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--ab", action="append", default=[], help="P_* name to disable")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    params = ps.default_params()
    for name in args.ab:
        params[getattr(ps, name)] = 0
        print(f"disabled {name}")
    c0 = time.process_time()
    import agent  # noqa: F401  (triggers compile + warm-up)

    print(f"import + warm-up cpu {time.process_time() - c0:.1f}s")
    run(params, None if args.time else args.depth, args.time, args.verbose)


if __name__ == "__main__":
    main()
