"""Tests for the move-ordering agent.

Run from the repo root:

    uv run python my-agents/06_move_ordering/test_move_ordering.py

Also collectable by pytest if it is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent

START = chess.STARTING_FEN
# Kiwipete: many captures, castling, promotions nearby. Standard perft position.
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
# White pawn on e7 can promote on e8 or capture the rook on d8; the black king is on g8.
PROMO = "3r2k1/4P3/8/1q6/8/8/8/4K3 w - - 0 1"
# Assorted middlegame positions for the equivalence / node-count tests.
POSITIONS = [
    START,
    KIWIPETE,
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]
# Positions where the unordered search is still tractable at depth 3. Kiwipete without
# ordering explodes in quiescence (over 100k nodes at depth 1), which is the point.
COMPARE_POSITIONS = [fen for fen in POSITIONS if fen != KIWIPETE]
COMPARE_DEPTH = 3


def scores(board: chess.Board, **hints: object) -> dict[chess.Move, int]:
    return {m: agent.move_order_score(board, m, **hints) for m in board.legal_moves}  # type: ignore[arg-type]


def test_every_legal_move_exactly_once() -> None:
    for fen in POSITIONS:
        board = chess.Board(fen)
        ordered = agent.ordered_moves(board)
        legal = list(board.legal_moves)
        assert len(ordered) == len(legal)
        assert set(ordered) == set(legal)
        assert len(set(ordered)) == len(ordered)


def test_no_illegal_moves() -> None:
    for fen in POSITIONS:
        board = chess.Board(fen)
        for move in agent.ordered_moves(board):
            assert board.is_legal(move), move


def test_tt_move_ranks_first() -> None:
    board = chess.Board(KIWIPETE)
    # Pick a boring quiet move as the "TT move" and check it still comes first.
    quiet = chess.Move.from_uci("a2a3")
    assert quiet in board.legal_moves
    ordered = agent.ordered_moves(board, tt_move=quiet)
    assert ordered[0] == quiet
    assert agent.ordered_moves(board)[0] != quiet  # it would not be first on its own


def test_queen_capture_above_pawn_capture() -> None:
    # The c4 pawn can take the queen on b5 or the pawn on d5.
    board = chess.Board("4k3/8/8/1q1p4/2P5/8/8/4K3 w - - 0 1")
    take_queen = chess.Move.from_uci("c4b5")
    take_pawn = chess.Move.from_uci("c4d5")
    s = scores(board)
    assert s[take_queen] > s[take_pawn]


def test_mvv_lva_examples() -> None:
    # pawn x queen > knight x rook > queen x pawn, all on one board.
    board = chess.Board("4k3/8/8/2q1r1p1/1P6/3N4/8/6QK w - - 0 1")
    assert not board.is_check()
    pxq = chess.Move.from_uci("b4c5")
    nxr = chess.Move.from_uci("d3e5")
    qxp = chess.Move.from_uci("g1g5")
    for m in (pxq, nxr, qxp):
        assert m in board.legal_moves, m
    s = scores(board)
    assert s[pxq] > s[nxr] > s[qxp]


def test_promotion_high_priority() -> None:
    board = chess.Board(PROMO)
    s = scores(board)
    queen_promo = chess.Move.from_uci("e7e8q")
    knight_promo = chess.Move.from_uci("e7e8n")
    quiet = chess.Move.from_uci("e1f2")
    assert quiet in board.legal_moves
    assert s[queen_promo] > s[knight_promo] > s[quiet]
    # Underpromotions are still present.
    for promo in "qrbn":
        assert chess.Move.from_uci(f"e7e8{promo}") in agent.ordered_moves(board)
    # Queen promotion outranks every quiet move.
    for move, score in s.items():
        if not move.promotion and not board.is_capture(move):
            assert s[queen_promo] > score


def test_losing_capture_ranks_below_quiet() -> None:
    # Queen takes a defended pawn: obviously losing, goes to the bottom tier.
    board = chess.Board("4k3/8/3p4/4p3/8/8/8/Q3K3 w - - 0 1")
    qxp = chess.Move.from_uci("a1e5")
    assert board.is_capture(qxp) and agent.capture_looks_losing(board, qxp)
    s = scores(board)
    quiet = chess.Move.from_uci("e1d1")
    assert s[qxp] < s[quiet]


def test_killer_above_quiet_below_capture() -> None:
    board = chess.Board(KIWIPETE)
    killer = chess.Move.from_uci("a2a3")
    s = scores(board, killers=[killer])
    capture = chess.Move.from_uci("e2a6")  # bishop takes bishop
    other_quiet = chess.Move.from_uci("a2a4")
    assert s[capture] > s[killer] > s[other_quiet]


def test_check_bonus_is_moderate() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    check = chess.Move.from_uci("a1a8")
    quiet = chess.Move.from_uci("a1b1")
    s = scores(board)
    assert s[check] > s[quiet]
    assert s[check] - s[quiet] < agent.SCORE_PROMOTION - agent.SCORE_QUIET


def test_same_score_as_unordered_alpha_beta() -> None:
    for fen in COMPARE_POSITIONS:
        ordered = agent.Searcher(order=True)
        plain = agent.Searcher(order=False)
        _, s1 = ordered.search_root(chess.Board(fen), COMPARE_DEPTH)
        _, s2 = plain.search_root(chess.Board(fen), COMPARE_DEPTH)
        assert s1 == s2, (fen, s1, s2)


def test_ordering_reduces_nodes() -> None:
    for fen in COMPARE_POSITIONS:
        ordered = agent.Searcher(order=True)
        plain = agent.Searcher(order=False)
        ordered.search_root(chess.Board(fen), COMPARE_DEPTH)
        plain.search_root(chess.Board(fen), COMPARE_DEPTH)
        assert ordered.stats.nodes < plain.stats.nodes, (fen, ordered.stats, plain.stats)
        assert ordered.stats.first_move_cutoff_rate >= plain.stats.first_move_cutoff_rate


def test_instrumentation_counts() -> None:
    searcher = agent.Searcher(order=True)
    searcher.search_root(chess.Board(KIWIPETE), 3)
    st = searcher.stats
    assert st.nodes > 0 and st.beta_cutoffs > 0
    assert 0 < st.first_move_cutoffs <= st.beta_cutoffs


def test_get_move_returns_legal_uci() -> None:
    for fen in POSITIONS:
        board = chess.Board(fen)
        uci = agent.get_move(fen, 2_000)
        assert chess.Move.from_uci(uci) in board.legal_moves, (fen, uci)


def test_finds_mate_in_one() -> None:
    fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
    assert agent.get_move(fen, 5_000) == "a1a8"


def test_low_time_still_answers() -> None:
    uci = agent.get_move(KIWIPETE, 100)
    assert chess.Move.from_uci(uci) in chess.Board(KIWIPETE).legal_moves


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
