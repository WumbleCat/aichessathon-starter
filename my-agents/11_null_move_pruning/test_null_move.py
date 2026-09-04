"""Tests for the null-move pruning agent, mirroring the checklist in the spec.

Run from this directory with the repo interpreter:

    python test_null_move.py

Every test is a plain function using assert, so pytest picks them up too.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent
from agent import (
    MATE_BOUND,
    SearchConfig,
    Searcher,
    can_try_null_move,
    get_move,
    non_pawn_material,
)

# Kiwipete: a busy, tactical middlegame.
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
MIDDLEGAMES = [
    KIWIPETE,
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 8",
    "2rq1rk1/pp1bppbp/3p1np1/8/2PNP3/2N1B3/PP2BPPP/R2Q1RK1 b - - 0 12",
]
# King-and-pawn endings: pure zugzwang territory, null move must stay off.
PAWN_ENDINGS = [
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    "8/8/8/8/8/3k4/4p3/4K3 w - - 0 1",
    "8/5k2/8/8/8/8/4PK2/8 b - - 0 1",
]
# Well-known zugzwang test positions (from null-move test suites). Each has a
# best move that a careless null-move search overlooks.
ZUGZWANG = [
    ("8/8/p1p5/1p5p/1P5p/8/PPP2K1p/4R1rk w - - 0 1", "e1f1"),
    ("1q1k4/2Rr4/8/2Q3K1/8/8/8/8 w - - 0 1", "g5h6"),
    ("7k/5K2/5P1p/3p4/6P1/3p4/8/8 w - - 0 1", "g4g5"),
]


def _fixed_depth(fen: str, depth: int, config: SearchConfig) -> tuple[chess.Move, int, Searcher]:
    """Search to a fixed depth with a fresh table so runs are comparable."""
    board = chess.Board(fen)
    searcher = Searcher(config, {})
    move = None
    for d in range(1, depth + 1):
        move, _ = searcher.search_root(board, d, move)
    assert move is not None
    return move, searcher.nodes, searcher


def test_no_null_move_while_in_check() -> None:
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_check()
    cfg = SearchConfig()
    assert not can_try_null_move(board, 6, 0, True, board.is_check(), cfg)
    # The same node is fine once the check is gone.
    quiet = chess.Board(KIWIPETE)
    assert can_try_null_move(quiet, 6, -10_000, True, quiet.is_check(), cfg)


def test_no_null_move_in_pawn_endings() -> None:
    cfg = SearchConfig()
    for fen in PAWN_ENDINGS:
        board = chess.Board(fen)
        assert non_pawn_material(board, board.turn) == 0
        assert not can_try_null_move(board, 8, -10_000, True, False, cfg)
        # Whole-tree check: no promotion is reachable within 4 plies, so every node
        # of the search is a king-and-pawn node and the counter must stay at zero.
        _, _, searcher = _fixed_depth(fen, 4, cfg)
        assert searcher.null_tries == 0, fen


def test_other_safeguards() -> None:
    board = chess.Board(KIWIPETE)
    cfg = SearchConfig()
    assert not can_try_null_move(board, 1, -10_000, True, False, cfg)  # depth too small
    strict = SearchConfig(null_move_min_depth=3)
    assert not can_try_null_move(board, 2, -10_000, True, False, strict)
    assert not can_try_null_move(board, 6, -10_000, False, False, cfg)  # parent passed
    assert not can_try_null_move(board, 6, MATE_BOUND, True, False, cfg)  # mate window
    assert not can_try_null_move(board, 6, 10_000, True, False, cfg)  # eval below beta
    assert not can_try_null_move(board, 6, -10_000, True, False, SearchConfig(null_move=False))
    loose = SearchConfig(null_move_needs_eval_above_beta=False)
    assert can_try_null_move(board, 6, 10_000, True, False, loose)


def test_board_restored_after_null_move() -> None:
    # Direct push/pop, including a position with an en-passant square set.
    for fen in [KIWIPETE, "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"]:
        board = chess.Board(fen)
        board.push(chess.Move.null())
        assert board.turn != chess.Board(fen).turn
        board.pop()
        assert board.fen() == fen
    # Through a whole search that performs many null moves.
    for fen in MIDDLEGAMES:
        board = chess.Board(fen)
        searcher = Searcher(SearchConfig(), {})
        for d in range(1, 5):
            searcher.search_root(board, d)
        assert searcher.null_tries > 0
        assert board.fen() == fen
        assert len(board.move_stack) == 0


def test_feature_can_be_toggled() -> None:
    on = SearchConfig(null_move=True)
    off = SearchConfig(null_move=False)
    _, _, s_on = _fixed_depth(KIWIPETE, 4, on)
    _, _, s_off = _fixed_depth(KIWIPETE, 4, off)
    assert s_on.null_tries > 0
    assert s_off.null_tries == 0


def test_middlegame_node_count_decreases() -> None:
    total_on = total_off = 0
    for fen in MIDDLEGAMES:
        _, nodes_on, s_on = _fixed_depth(fen, 5, SearchConfig(null_move=True))
        _, nodes_off, _ = _fixed_depth(fen, 5, SearchConfig(null_move=False))
        print(
            f"  depth 5  {fen[:30]:30s}  nodes off {nodes_off:8d}  on {nodes_on:8d}  "
            f"({nodes_on / nodes_off:.0%})  cutoffs {s_on.null_cutoffs}/{s_on.null_tries}"
        )
        total_on += nodes_on
        total_off += nodes_off
    assert total_on < 0.8 * total_off, (total_on, total_off)


def test_zugzwang_positions_not_mispruned() -> None:
    """Pruning on must agree with pruning off, so a null move never changes the answer.

    The first two positions have enough material for null moves to fire and the
    engine finds the book move at depth 6. The third is a king-and-pawn ending: the
    material guard keeps null moves off entirely (and the book move g4g5 is beyond
    this engine's horizon, with or without pruning, so only agreement is checked).
    """
    for fen, expected in ZUGZWANG:
        move_on, _, searcher = _fixed_depth(fen, 6, SearchConfig(null_move=True))
        move_off, _, _ = _fixed_depth(fen, 6, SearchConfig(null_move=False))
        print(f"  zugzwang {fen[:30]:30s}  on {move_on.uci()}  off {move_off.uci()}  "
              f"book {expected}  null {searcher.null_cutoffs}/{searcher.null_tries}")
        assert move_on == move_off, (fen, move_on.uci(), move_off.uci())
        board = chess.Board(fen)
        if non_pawn_material(board, board.turn) == 0:
            assert searcher.null_tries == 0
        else:
            assert searcher.null_tries > 0
            assert move_on.uci() == expected, (fen, move_on.uci())


def test_finds_forced_mate_with_pruning_on() -> None:
    # Nf6+ gxf6 Bxf7# (mate in 2).
    fen = "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 0"
    move, _, searcher = _fixed_depth(fen, 4, SearchConfig())
    board = chess.Board(fen)
    _, score = searcher.search_root(board, 4, move)
    assert move.uci() == "d5f6", move.uci()
    assert score >= MATE_BOUND


def test_get_move_is_legal_and_fast() -> None:
    fens = [chess.STARTING_FEN, *MIDDLEGAMES, *PAWN_ENDINGS, *[fen for fen, _ in ZUGZWANG]]
    for fen in fens:
        board = chess.Board(fen)
        started = time.monotonic()
        uci = get_move(fen, 3_000)
        elapsed = time.monotonic() - started
        assert chess.Move.from_uci(uci) in board.legal_moves, (fen, uci)
        assert elapsed < 1.0, elapsed
    # Almost no time left: still legal, still quick.
    uci = get_move(KIWIPETE, 120)
    assert chess.Move.from_uci(uci) in chess.Board(KIWIPETE).legal_moves


def main() -> None:
    agent.CONFIG.null_move = True
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = 0
    for name, fn in tests:
        started = time.monotonic()
        try:
            fn()
        except AssertionError as error:
            failures += 1
            print(f"FAIL {name}: {error}")
        else:
            print(f"ok   {name}  ({time.monotonic() - started:.1f}s)")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("all tests passed")


if __name__ == "__main__":
    main()
