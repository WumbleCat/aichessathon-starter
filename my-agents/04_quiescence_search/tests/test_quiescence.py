"""Tests for the quiescence search, following my-agents-readmes/04_quiescence_search.md.

Run from the repository root:

    uv run python my-agents/04_quiescence_search/tests/test_quiescence.py

No pytest in the platform stack, so this is a tiny self-contained runner. Every
function named test_* is executed and a failed assert reports the test name.
"""

from __future__ import annotations

import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent

INF = agent.INFINITY

# Positions ------------------------------------------------------------------

# White queen on d1 can take the black rook on d5, but the rook is defended by the
# e6 pawn. Depth-1 static evaluation thinks Qxd5 wins a rook; quiescence sees exd5.
QUEEN_TAKES_DEFENDED_ROOK = "4k3/8/4p3/3r4/8/8/8/3QK3 w - - 0 1"

# Knight on e3 can take the pawn on d5, defended by the pawn on e6. The capture
# "wins" a pawn and loses the knight one ply later.
KNIGHT_TAKES_DEFENDED_PAWN = "4k3/8/4p3/3p4/8/4N3/8/4K3 w - - 0 1"

# Doubled white rooks on the d-file against a rook on d8 and a pawn on d5.
# Rxd5 Rxd5 Rxd5 is a forced sequence that nets White exactly one pawn.
DOUBLED_ROOKS_RECAPTURE = "3rk3/8/8/3p4/8/8/3R4/3RK3 w - - 0 1"

# Back-rank mate: White to move, in check, no evasion. Static evaluation would say
# "three pawns against a rook, not too bad"; quiescence must report mate.
BACK_RANK_MATE = "6k1/8/8/8/8/8/5PPP/r5K1 w - - 0 1"

# White is in check by the rook on a1 and is materially far ahead (queen, bishop,
# two pawns against a rook). The only evasion is Qe1, after which Rxe1 is mate.
# Stand pat would be wildly positive; searching evasions finds the mate.
CHECK_ONLY_EVASION_LOSES = "k7/8/4Q3/8/8/8/5PPB/r5K1 w - - 0 1"

# White is in check with several evasions. The engine must still pick one and
# return a finite score.
CHECK_WITH_EVASIONS = "4k3/8/8/8/8/8/3Q4/r3K3 w - - 0 1"

# Kiwipete, a dense middlegame with many mutual captures. Used for termination.
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

# Scholar's mate position: Qxf7# is mate in one for White.
MATE_IN_ONE = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"

# Pawn about to promote, with a capture-promotion available too.
PROMOTION = "1n2k3/2P5/8/8/8/8/8/4K3 w - - 0 1"


def _root(
    fen: str, depth: int, use_quiescence: bool = True
) -> tuple[chess.Move, int, agent.Searcher]:
    board = chess.Board(fen)
    searcher = agent.Searcher(None, use_quiescence=use_quiescence)
    move, score = searcher.search_root(board, depth)
    assert board.fen() == fen, "search_root left the board changed"
    assert not board.move_stack, "search_root left moves on the stack"
    return move, score, searcher


def _quiescence(fen: str) -> tuple[int, agent.Searcher]:
    board = chess.Board(fen)
    searcher = agent.Searcher(None)
    score = searcher.quiescence(board, -INF, INF)
    assert board.fen() == fen, "quiescence left the board changed"
    assert not board.move_stack, "quiescence left moves on the stack"
    return score, searcher


# Tests ----------------------------------------------------------------------


def test_queen_does_not_take_defended_rook() -> None:
    move, score, _ = _root(QUEEN_TAKES_DEFENDED_ROOK, 1)
    assert move.uci() != "d1d5", f"engine played Qxd5 into exd5, score {score}"
    # Queen versus rook and pawn: White is still well ahead after a sane move.
    assert 150 < score < 600, score

    # The same position with quiescence switched off shows the horizon effect.
    blind_move, blind_score, _ = _root(QUEEN_TAKES_DEFENDED_ROOK, 1, use_quiescence=False)
    assert blind_move.uci() == "d1d5", blind_move
    assert blind_score > 600, blind_score


def test_apparent_winning_capture_loses_material() -> None:
    move, _, _ = _root(KNIGHT_TAKES_DEFENDED_PAWN, 1)
    assert move.uci() != "e3d5", "engine played Nxd5 into exd5"

    # Quiescence after Nxd5 (Black to move) sees the recapture: Black is ahead by
    # roughly a knight minus a pawn.
    board = chess.Board(KNIGHT_TAKES_DEFENDED_PAWN)
    board.push_uci("e3d5")
    score, _ = _quiescence(board.fen())
    assert 100 < score < 350, score


def test_forced_recapture_sequence_is_resolved() -> None:
    move, score, _ = _root(DOUBLED_ROOKS_RECAPTURE, 1)
    assert move.uci() == "d2d5", move
    # White starts two rooks against rook and pawn (+400). Rxd5 Rxd5 Rxd5 ends with
    # a lone rook against a bare king: +500 absolute, a pawn better than standing still.
    assert 400 < score < 600, score
    assert score > agent.evaluate(chess.Board(DOUBLED_ROOKS_RECAPTURE)) + 50

    # From Black's side after the first capture the exchange is a rook down.
    board = chess.Board(DOUBLED_ROOKS_RECAPTURE)
    board.push_uci("d2d5")
    black_score, _ = _quiescence(board.fen())
    assert -600 < black_score < -400, black_score

    # A full-width search deep enough to see the whole exchange agrees with the
    # depth-1 + quiescence answer: quiescence added nothing false.
    deep_move, deep_score, _ = _root(DOUBLED_ROOKS_RECAPTURE, 4, use_quiescence=False)
    assert deep_move.uci() == "d2d5", deep_move
    assert abs(deep_score - score) <= 60, (deep_score, score)


def test_in_check_with_no_evasion_is_mate() -> None:
    board = chess.Board(BACK_RANK_MATE)
    assert board.is_check() and board.is_checkmate()
    score, _ = _quiescence(BACK_RANK_MATE)
    assert score <= -agent.MATE_BOUND, score


def test_in_check_must_search_evasions_not_stand_pat() -> None:
    board = chess.Board(CHECK_ONLY_EVASION_LOSES)
    assert board.is_check()
    assert agent.evaluate(board) > 500, "stand pat should look great for White here"
    score, searcher = _quiescence(CHECK_ONLY_EVASION_LOSES)
    assert score <= -agent.MATE_BOUND, f"expected a mate score, got {score}"
    assert score == -(agent.MATE_SCORE - 2), score
    assert searcher.qnodes >= 3, searcher.qnodes


def test_in_check_with_evasions_returns_sane_score() -> None:
    board = chess.Board(CHECK_WITH_EVASIONS)
    assert board.is_check()
    score, _ = _quiescence(CHECK_WITH_EVASIONS)
    # Queen against rook after a king move: about +400.
    assert 250 < score < 600, score
    move, _, _ = _root(CHECK_WITH_EVASIONS, 1)
    assert move in board.legal_moves


def test_quiescence_terminates_in_a_wild_position() -> None:
    started = time.monotonic()
    score, searcher = _quiescence(KIWIPETE)
    elapsed = time.monotonic() - started
    assert isinstance(score, int)
    assert searcher.max_qply <= agent.QS_MAX_PLY, searcher.max_qply
    assert searcher.qnodes < 200_000, searcher.qnodes
    assert elapsed < 20.0, elapsed


def test_qnodes_are_tracked_separately() -> None:
    _, _, searcher = _root(KIWIPETE, 2)
    assert searcher.nodes > 0
    assert searcher.qnodes > 0
    assert searcher.qnodes != searcher.nodes


def test_board_state_unchanged_after_search() -> None:
    for fen in (
        QUEEN_TAKES_DEFENDED_ROOK,
        KNIGHT_TAKES_DEFENDED_PAWN,
        DOUBLED_ROOKS_RECAPTURE,
        CHECK_WITH_EVASIONS,
        KIWIPETE,
        PROMOTION,
    ):
        _quiescence(fen)
        _root(fen, 2)


def test_finds_mate_in_one() -> None:
    move, score, _ = _root(MATE_IN_ONE, 2)
    assert move.uci() == "h5f7", move
    assert score >= agent.MATE_BOUND, score
    # The mated side is one ply from the root, so the score is exactly MATE - 1.
    assert score == agent.MATE_SCORE - 1, score


def test_quiescence_mate_distance_is_measured_from_the_root() -> None:
    # The same forced mate found three plies into the tree must score as a mate that
    # is three plies further away, so a real mate in one at the root always wins.
    board = chess.Board(CHECK_ONLY_EVASION_LOSES)
    searcher = agent.Searcher(None)
    at_root = searcher.quiescence(board, -INF, INF, 0)
    deeper = searcher.quiescence(board, -INF, INF, 3)
    assert at_root == -(agent.MATE_SCORE - 2), at_root
    assert deeper == -(agent.MATE_SCORE - 5), deeper
    assert board.fen() == CHECK_ONLY_EVASION_LOSES


def test_promotions_are_tactical_moves() -> None:
    board = chess.Board(PROMOTION)
    tactical = {move.uci() for move in agent.tactical_moves(board)}
    assert "c7c8q" in tactical, tactical
    assert "c7b8q" in tactical, tactical
    assert "c7c8n" not in tactical, "underpromotions are left out of quiescence"
    score, _ = _quiescence(PROMOTION)
    assert score > 700, score


def test_get_move_returns_legal_uci() -> None:
    for fen in (
        chess.STARTING_FEN,
        QUEEN_TAKES_DEFENDED_ROOK,
        CHECK_WITH_EVASIONS,
        CHECK_ONLY_EVASION_LOSES,
        KIWIPETE,
        PROMOTION,
    ):
        board = chess.Board(fen)
        uci = agent.get_move(fen, 2_000)
        assert chess.Move.from_uci(uci) in board.legal_moves, (fen, uci)


def test_search_respects_deadline() -> None:
    board = chess.Board(KIWIPETE)
    started = time.monotonic()
    uci = agent.get_move(board.fen(), 3_000)  # budget is 3000 / 30 = 100 ms
    elapsed = time.monotonic() - started
    assert chess.Move.from_uci(uci) in board.legal_moves, uci
    assert elapsed < 1.5, elapsed


def test_root_avoids_revisiting_positions_when_ahead() -> None:
    fen = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"  # rook and king against a bare king
    board = chess.Board(fen)
    searcher = agent.Searcher(None)
    preferred, score = searcher.search_root(board, 2)
    assert score > 0

    # Pretend the game already passed through the position the preferred move leads
    # to. The engine, being ahead, must now pick something else.
    board.push(preferred)
    visited = frozenset({agent.position_key(board)})
    board.pop()
    alternative, alt_score = agent.Searcher(None).search_root(board, 2, visited)
    assert alternative != preferred, alternative
    assert alt_score > 0
    assert board.fen() == fen

    # get_move records both the position it was given and the one it moves into.
    agent.reset_history()
    uci = agent.get_move(fen, 2_000)
    board.push_uci(uci)
    assert agent.position_key(chess.Board(fen)) in agent._visited
    assert agent.position_key(board) in agent._visited
    agent.reset_history()


def test_board_restored_after_deadline_abort() -> None:
    board = chess.Board(KIWIPETE)
    move = agent.choose_move(board, 1_500)  # budget 50 ms, aborts mid-iteration
    assert board.fen() == KIWIPETE, board.fen()
    assert not board.move_stack
    assert move in board.legal_moves


# Runner ---------------------------------------------------------------------


def main() -> int:
    tests: list[tuple[str, Callable[[], None]]] = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for name, test in tests:
        try:
            test()
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
