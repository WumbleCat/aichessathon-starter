"""Benchmark move ordering: fixed-depth search with ordering on and off.

    uv run python my-agents/06_move_ordering/bench.py [depth]

Prints, per position, the chosen move, score, nodes, beta cutoffs, first-move cutoff
rate and elapsed time for both configurations. Scores must match; nodes should drop.
Positions flagged as too tactical are only searched with ordering on: without it the
capture search alone runs into hundreds of thousands of nodes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent

# name -> (fen, also run without ordering)
POSITIONS: dict[str, tuple[str, bool]] = {
    "start": (chess.STARTING_FEN, True),
    "kiwipete": ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", False),
    "italian": ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", True),
    "sicilian": ("rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5", True),
    "endgame": ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", True),
    "tactics": ("r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9", True),
}


def run(fen: str, depth: int, order: bool) -> tuple[str, int, agent.SearchStats, float]:
    searcher = agent.Searcher(order=order)
    started = time.perf_counter()
    move, score = searcher.search_root(chess.Board(fen), depth)
    return move.uci(), score, searcher.stats, time.perf_counter() - started


def main() -> None:
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"depth {depth}\n")
    print(
        f"{'position':10} {'order':6} {'move':6} {'score':>7} {'nodes':>9} "
        f"{'cutoffs':>8} {'first':>6} {'time':>7}"
    )
    totals = {True: 0, False: 0}
    for name, (fen, compare) in POSITIONS.items():
        results: dict[bool, int] = {}
        for order in (False, True):
            if not order and not compare:
                continue
            move, score, stats, elapsed = run(fen, depth, order)
            results[order] = score
            if compare:
                totals[order] += stats.nodes
            print(
                f"{name:10} {'on' if order else 'off':6} {move:6} {score:7d} {stats.nodes:9d} "
                f"{stats.beta_cutoffs:8d} {stats.first_move_cutoff_rate:6.0%} {elapsed:6.2f}s"
            )
        if compare and results[True] != results[False]:
            print("   ^^^ SCORE MISMATCH")
    ratio = totals[True] / totals[False] if totals[False] else 0.0
    print(f"\ncompared positions: nodes off {totals[False]}  on {totals[True]}  ratio {ratio:.2f}")


if __name__ == "__main__":
    main()
