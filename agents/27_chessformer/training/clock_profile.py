"""Play the agent against itself at the contest clock and report move-time percentiles.

    python training/clock_profile.py --base-ms 120000 --increment-ms 500 --plies 60

Each side keeps its own clock exactly as the harness does (time_left - took + increment); the
report shows median / p95 / p99 / max of the wall time per move and the minimum clock reached.
"""

import argparse
import os
import sys
import time

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ms", type=int, default=120_000)
    parser.add_argument("--increment-ms", type=int, default=500)
    parser.add_argument("--plies", type=int, default=60)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    import agent

    init_s = time.perf_counter() - t0
    board = chess.Board()
    clocks = {chess.WHITE: float(args.base_ms), chess.BLACK: float(args.base_ms)}
    took_ms: list[float] = []
    for _ in range(args.plies):
        if board.is_game_over():
            break
        side = board.turn
        t = time.perf_counter()
        uci = agent.get_move(board.fen(), int(clocks[side]))
        took = (time.perf_counter() - t) * 1000
        took_ms.append(took)
        clocks[side] += args.increment_ms - took
        board.push_uci(uci)
    took_ms.sort()
    n = len(took_ms)

    def pct(p: float) -> float:
        return took_ms[min(n - 1, int(p * n))]

    report = (
        f"import {init_s:.1f}s; {n} moves at {args.base_ms}+{args.increment_ms} ms: "
        f"median {pct(0.5):.0f} ms, p95 {pct(0.95):.0f} ms, p99 {pct(0.99):.0f} ms, "
        f"max {took_ms[-1]:.0f} ms; "
        f"clocks left W {clocks[chess.WHITE]:.0f} B {clocks[chess.BLACK]:.0f}"
    )
    print(report)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
