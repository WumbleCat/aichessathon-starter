"""Tests for the SEE agent.

Run from the repo root:

    uv run python my-agents/13_static_exchange_evaluation/test_see.py

Also works under pytest. The hand-built positions check exact centipawn values; the
random-position sweep compares SEE against a brute-force capture-only search on the
same square, which handles pins and legality exactly, and reports the agreement rate.
"""

import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent

P, N, B, R, Q = 100, 320, 330, 500, 900

# (description, fen, uci, expected SEE)
CASES: list[tuple[str, str, str, int]] = [
    ("pawn takes undefended queen", "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", "e4d5", Q),
    ("queen takes pawn defended by pawn", "4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1", "d1d5", P - Q),
    ("equal pawn exchange", "4k3/8/4p3/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 0),
    ("rook exchange with x-ray rook behind", "3r3k/8/8/3r4/8/8/3R4/3R3K w - - 0 1", "d2d5", R),
    ("rook takes defended rook, no x-ray", "3r3k/8/8/3r4/8/8/3R4/7K w - - 0 1", "d2d5", 0),
    ("bishop takes knight, queen x-rays behind", "4k3/8/3p4/4n3/8/8/1B6/Q3K3 w - - 0 1",
     "b2e5", N - B + P),
    ("defender is pinned to its king", "4k3/4n3/8/3p4/2P5/8/8/4R2K w - - 0 1", "c4d5", P),
    ("defender pinned by the capturer itself may still take", "k7/1b6/2p5/3B4/8/8/8/7K w - - 0 1",
     "d5c6", P - B),
    ("en passant, undefended", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", P),
    ("en passant, recaptured by pawn", "4k3/2p5/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 0),
    ("capture with promotion, king recaptures", "2kr4/4P3/8/8/8/8/8/4K3 w - - 0 1",
     "e7d8q", R + (Q - P) - Q),
    ("capture with promotion, undefended", "3r4/4P3/8/8/8/8/8/k3K3 w - - 0 1", "e7d8q",
     R + (Q - P)),
    ("king cannot recapture while the square is attacked", "8/8/4k3/3p4/8/8/3Q4/3RK3 w - - 0 1",
     "d2d5", P),
    ("king may recapture when the square is safe", "8/8/4k3/3p4/8/8/3Q4/4K3 w - - 0 1",
     "d2d5", P - Q),
    ("knight takes pawn, defender declines a losing recapture",
     "4k3/8/2b5/3p4/5N2/8/3R4/3RK3 w - - 0 1", "f4d5", P),
    ("knight takes pawn, x-ray rook covers the recapture",
     "4k3/8/2b5/3p4/5N2/8/3R4/3RK3 w - - 0 1", "d2d5", P - R + B),
    ("queen takes rook defended by bishop", "4k3/8/8/8/8/8/4K1b1/Q6r w - - 0 1", "a1h1", R - Q),
    ("quiet move onto an attacked square hangs the piece", "4k3/8/8/3p4/8/8/8/4K1N1 w - - 0 1",
     "g1e2", 0),
    ("quiet move into a pawn attack hangs the knight", "4k3/8/8/3p4/8/2N5/8/4K3 w - - 0 1",
     "c3e4", -N),
]


def check_hand_cases() -> int:
    failures = 0
    for description, fen, uci, expected in CASES:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, f"{description}: {uci} is not legal in {fen}"
        before = board.fen()
        got = agent.see(board, move)
        assert board.fen() == before, f"{description}: SEE modified the board"
        status = "ok  " if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print(f"{status} {description:55s} {uci}  expected {expected:5d}  got {got:5d}")
    return failures


def _victim_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return P
    piece = board.piece_type_at(move.to_square)
    return agent.SEE_VALUE[piece] if piece else 0


def _best_recapture(board: chess.Board, target: chess.Square) -> int:
    """Best material the side to move can win by capturing on `target`, or 0 to stop."""
    best = 0
    for move in board.legal_moves:
        if move.to_square != target or not board.is_capture(move):
            continue
        gain = _victim_value(board, move)
        if move.promotion:
            gain += agent.SEE_VALUE[move.promotion] - P
        board.push(move)
        gain -= _best_recapture(board, target)
        board.pop()
        best = max(best, gain)
    return best


def see_reference(board: chess.Board, move: chess.Move) -> int:
    gain = _victim_value(board, move)
    if move.promotion:
        gain += agent.SEE_VALUE[move.promotion] - P
    board.push(move)
    gain -= _best_recapture(board, move.to_square)
    board.pop()
    return gain


def random_sweep(positions: int = 400, seed: int = 13) -> tuple[int, int]:
    """Compare SEE with the brute-force reference over random middlegame positions."""
    rng = random.Random(seed)
    agree = total = 0
    shown = 0
    for _ in range(positions):
        board = chess.Board()
        for _ply in range(rng.randint(6, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over() or board.is_check():
            continue
        for move in board.legal_moves:
            if not board.is_capture(move):
                continue
            total += 1
            got = agent.see(board, move)
            want = see_reference(board, move)
            if got == want:
                agree += 1
            elif shown < 5:
                shown += 1
                print(f"     mismatch {board.fen()} {move.uci()} see {got} reference {want}")
    return agree, total


def check_ordering() -> None:
    """Good captures come first and losing captures last."""
    board = chess.Board("4k3/8/4p3/3p4/4P3/8/8/3QK3 w - - 0 1")
    moves = agent.ordered_moves(board)
    assert moves[0] == chess.Move.from_uci("e4d5"), moves[0]
    assert moves[-1] == chess.Move.from_uci("d1d5"), moves[-1]
    print("ok   ordering: pawn takes pawn first, queen takes defended pawn last")


def check_play() -> None:
    """The agent returns legal moves and does not fall for a poisoned pawn."""
    for fen in (
        chess.STARTING_FEN,
        "4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "8/8/8/8/8/5k2/8/4K2q w - - 0 1",
    ):
        board = chess.Board(fen)
        uci = agent.get_move(fen, 2_000)
        assert chess.Move.from_uci(uci) in board.legal_moves, (fen, uci)
    uci = agent.get_move("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1", 3_000)
    assert uci != "d1d5", "took the poisoned pawn"
    print("ok   play: legal moves returned, poisoned pawn declined")


def test_hand_cases() -> None:
    assert check_hand_cases() == 0


def test_random_sweep() -> None:
    agree, total = random_sweep(150)
    assert agree >= 0.97 * total, (agree, total)


def test_ordering() -> None:
    check_ordering()


def test_play() -> None:
    check_play()


if __name__ == "__main__":
    failed = check_hand_cases()
    check_ordering()
    agree, total = random_sweep()
    print(f"random sweep: {agree}/{total} captures agree with brute force ({agree / total:.1%})")
    check_play()
    if failed:
        raise SystemExit(f"{failed} hand-built case(s) failed")
    print("all SEE tests passed")
