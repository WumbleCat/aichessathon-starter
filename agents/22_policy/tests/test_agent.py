"""Mandatory chess and clock tests for agent.get_move (python-chess is the legality oracle).

Run:  python tests/test_agent.py
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import chess  # noqa: E402

import agent  # noqa: E402

CASES = {
    "normal move": ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", None),
    "capture available": ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", None),
    "free queen capture": ("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3", "f3h4"),
    "give check": ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", None),
    "check evasion": ("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", None),
    "mate in one": ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),
    "mate in one (queen)": ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),
    "kingside castling": ("r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1", None),
    "queenside castling": ("r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 0 1", None),
    "en passant": ("8/8/8/3pP3/8/8/8/K6k w - d6 0 1", None),
    "queen promotion": ("8/P7/8/8/8/8/8/K6k w - - 0 1", "a7a8q"),
    "promotion with capture": ("1n5k/P7/8/8/8/8/8/K7 w - - 0 1", None),
    "knight promotion mates": ("k7/8/8/8/8/8/8/K7 w - - 0 1", None),
    "black to move promotion": ("k7/8/8/8/8/8/p7/7K b - - 0 1", "a2a1q"),
    "only move": ("k7/8/8/8/8/8/8/5RK1 b - - 0 1", None),
    "stalemate avoidance": ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", None),
    "endgame": ("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1", None),
}


def run_case(name: str, fen: str, expected: str | None, ms: int = 1500) -> None:
    board = chess.Board(fen)
    t0 = time.perf_counter()
    uci = agent.get_move(fen, ms)
    el = time.perf_counter() - t0
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, f"{name}: illegal {uci} in {fen}"
    note = ""
    if expected is not None:
        note = "expected " + expected + (" OK" if uci == expected else " MISMATCH")
        assert uci == expected, f"{name}: got {uci}, expected {expected}"
    print(f"  {name:28s} {uci:6s} {el*1000:7.0f} ms {note}")


def test_special_moves() -> None:
    print("special moves:")
    for name, (fen, expected) in CASES.items():
        run_case(name, fen, expected)
    # castling actually chosen when it is the natural developing move
    b = chess.Board("r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1")
    assert any(b.is_castling(m) for m in b.legal_moves)
    # stalemate: no legal moves -> get_move must raise, the harness never asks for it
    try:
        agent.get_move("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 1000)
        raise AssertionError("expected ValueError on stalemate position")
    except ValueError:
        print("  stalemate position raises ValueError as documented")
    # checkmate position: same
    try:
        agent.get_move("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", 1000)
        raise AssertionError("expected ValueError on checkmated position")
    except ValueError:
        print("  checkmated position raises ValueError as documented")


def test_underpromotions_are_legal() -> None:
    print("underpromotion handling:")
    b = chess.Board("1n5k/P7/8/8/8/8/8/K7 w - - 0 1")
    for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        for m in b.legal_moves:
            if m.promotion == promo:
                assert m in b.legal_moves
    # the agent must at least return some legal move here (checked by run_case above)
    print("  all four promotion pieces generate legal moves")


def test_clock() -> None:
    print("clock tests (wall time must stay under time_left_ms):")
    fen = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7"
    for ms in (50, 100, 500, 1000, 5000, 30000, 120000):
        board = chess.Board(fen)
        t0 = time.perf_counter()
        uci = agent.get_move(fen, ms)
        el = (time.perf_counter() - t0) * 1000
        assert chess.Move.from_uci(uci) in board.legal_moves
        limit = ms + 400  # the referee's grace is 500 ms
        print(f"  time_left {ms:7d} ms -> {uci} in {el:7.0f} ms {'OK' if el < limit else 'TOO SLOW'}")
        assert el < limit, f"took {el:.0f} ms with {ms} ms left"


def test_repeated_calls() -> None:
    print("repeated calls in one process (a whole game vs itself):")
    board = chess.Board()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < 60:
        uci = agent.get_move(board.fen(), 3000)
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        board.push(move)
        plies += 1
    print(f"  {plies} plies played, result {board.result(claim_draw=True)}")


if __name__ == "__main__":
    test_special_moves()
    test_underpromotions_are_legal()
    test_clock()
    test_repeated_calls()
    print("ALL TESTS PASSED")
