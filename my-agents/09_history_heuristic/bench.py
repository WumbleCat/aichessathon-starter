"""Node-count benchmark: iterative deepening to a fixed depth, history on vs off.

Run from the repo root:

    uv run python my-agents/09_history_heuristic/bench.py [depth]
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import chess

_AGENT_PATH = Path(__file__).with_name("agent.py")
_SPEC = importlib.util.spec_from_file_location("history_agent", _AGENT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
agent = importlib.util.module_from_spec(_SPEC)
sys.modules["history_agent"] = agent
_SPEC.loader.exec_module(agent)

POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "2r3k1/pp3ppp/2n5/3p4/3P4/2N2N2/PP3PPP/2R3K1 b - - 0 20",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/5pk1/6p1/8/3K4/8/5PPP/8 w - - 0 40",
]


def deepen(fen: str, depth: int, use_history: bool) -> tuple[int, float, str]:
    searcher = agent.Searcher(use_history=use_history)
    searcher.new_search(float("inf"))
    board = chess.Board(fen)
    started = time.monotonic()
    move = None
    for d in range(1, depth + 1):
        move, _ = searcher.search_root(board, d)
    assert move is not None
    return searcher.nodes, time.monotonic() - started, move.uci()


def main() -> None:
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    total_on = total_off = 0
    print(f"{'position':<12} {'history':>10} {'no history':>11} {'saved':>7}  moves")
    for i, fen in enumerate(POSITIONS):
        on, t_on, m_on = deepen(fen, depth, True)
        off, t_off, m_off = deepen(fen, depth, False)
        total_on += on
        total_off += off
        saved = 100.0 * (off - on) / off
        print(f"{i:<12} {on:>10} {off:>11} {saved:>6.1f}%  {m_on} {m_off}  "
              f"({t_on:.1f}s vs {t_off:.1f}s)")
    saved = 100.0 * (total_off - total_on) / total_off
    print(f"{'total':<12} {total_on:>10} {total_off:>11} {saved:>6.1f}%   depth {depth}")


if __name__ == "__main__":
    main()
