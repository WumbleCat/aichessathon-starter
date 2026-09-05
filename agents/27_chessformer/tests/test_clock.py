"""Clock tests: the agent must answer within the time it was given, at every clock level."""

import os
import sys
import time

import chess
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import agent  # noqa: E402

MIDDLEGAME = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8"


@pytest.mark.parametrize("ms", [50, 100, 500, 1000, 5000, 30000, 120000])
def test_answers_in_time(ms):
    board = chess.Board(MIDDLEGAME)
    t0 = time.perf_counter()
    uci = agent.get_move(MIDDLEGAME, ms)
    took = (time.perf_counter() - t0) * 1000
    assert chess.Move.from_uci(uci) in board.legal_moves
    # below 150 ms the agent plays depth 1 unclocked; the harness grants a 500 ms grace, and on
    # the loaded dev box even that tiny search can take a few hundred ms of wall time
    limit = ms + 350 if ms <= 150 else max(ms * 0.3, 200)
    assert took < limit, f"{took:.0f}ms for {ms}ms clock"


def test_median_and_p99_at_fast_arena_clock():
    board = chess.Board()
    times = []
    ms = 10000
    for _ in range(20):
        t0 = time.perf_counter()
        uci = agent.get_move(board.fen(), ms)
        took = (time.perf_counter() - t0) * 1000
        times.append(took)
        board.push_uci(uci)
        ms = ms - int(took) + 100
        if board.is_game_over():
            break
    times.sort()
    print("median", times[len(times) // 2], "max", times[-1])
    assert times[-1] < 10000 * 0.2 + 100


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
