"""Tests for the handcrafted evaluation and the agent around it.

Run directly (no pytest needed):

    uv run python my-agents/03_handcrafted_evaluation/tests/test_evaluation.py

or with pytest if it is installed. Every check is one of the cases listed under
"Tests" in my-agents-readmes/03_handcrafted_evaluation.md, plus a few for the agent.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluation
from evaluation import evaluate, evaluate_white

import agent


def _without(board: chess.Board, square: chess.Square) -> chess.Board:
    """A copy of `board` with the piece on `square` removed."""
    copy = board.copy()
    copy.remove_piece_at(square)
    return copy


def _white_diff(fen: str, square: str) -> int:
    """How much the piece on `square` is worth to White in this position."""
    board = chess.Board(fen)
    return evaluate_white(board) - evaluate_white(_without(board, chess.parse_square(square)))


def test_extra_queen_is_worth_about_900() -> None:
    # Kings plus one white queen. Mobility and placement add a little on top.
    diff = _white_diff("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", "d1")
    assert 850 <= diff <= 1050, diff


def test_extra_pawn_is_worth_about_100() -> None:
    # Symmetric pawn chains so structure terms stay small, then one extra white pawn.
    fen = "4k3/pppp4/8/8/8/8/PPPP3P/4K3 w - - 0 1"
    diff = _white_diff(fen, "h2")
    assert 60 <= diff <= 160, diff


def test_centralized_knight_beats_corner_knight() -> None:
    centre = evaluate_white(chess.Board("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1"))
    corner = evaluate_white(chess.Board("4k3/8/8/8/8/8/8/N3K3 w - - 0 1"))
    assert centre > corner, (centre, corner)


def _white_structure(own: int, enemy: int) -> tuple[int, int]:
    """White's own (middlegame, endgame) pawn-structure score against `enemy` pawns."""
    return evaluation._pawn_structure_one_side(own, enemy, chess.WHITE)


def test_passed_pawn_receives_bonus() -> None:
    # A lone white e-pawn: blocked by a black e-pawn, or passed once the black pawn is
    # on the g-file instead.
    blocked = _white_structure(chess.BB_E2, chess.BB_E7)
    passed = _white_structure(chess.BB_E2, chess.BB_G7)
    assert passed[0] > blocked[0] and passed[1] > blocked[1], (blocked, passed)
    # The bonus scales with rank: the same passed pawn further up the board is worth more.
    far = _white_structure(chess.BB_E5, chess.BB_G7)
    assert far[0] > passed[0] and far[1] > passed[1], (passed, far)
    # The whole evaluation agrees. White d4/e5 are held by a black pawn on e6; move that
    # pawn to g6 (still blocked by h2) and both white pawns become connected passers.
    blocked_board = chess.Board("4k3/8/4p2p/4P3/3P4/8/7P/4K3 w - - 0 1")
    passed_board = chess.Board("4k3/8/6pp/4P3/3P4/8/7P/4K3 w - - 0 1")
    assert evaluate_white(passed_board) > evaluate_white(blocked_board) + 50


def test_connected_passed_pawns_receive_bonus() -> None:
    lone = _white_structure(chess.BB_E5, chess.BB_A7)
    connected = _white_structure(chess.BB_E5 | chess.BB_D5, chess.BB_A7)
    # Two passed pawns are worth more than twice one only if the connection bonus fires.
    assert connected[0] > 2 * lone[0] and connected[1] > 2 * lone[1], (lone, connected)


def test_doubled_pawns_receive_penalty() -> None:
    # Two pawns side by side versus the same two pawns stacked on one file.
    side_by_side = _white_structure(chess.BB_D2 | chess.BB_E2, chess.BB_D7 | chess.BB_E7)
    stacked = _white_structure(chess.BB_E2 | chess.BB_E3, chess.BB_D7 | chess.BB_E7)
    assert stacked[0] < side_by_side[0] and stacked[1] < side_by_side[1]


def test_isolated_pawn_receives_penalty() -> None:
    supported = _white_structure(chess.BB_D2 | chess.BB_E2, chess.BB_D7 | chess.BB_E7)
    isolated = _white_structure(chess.BB_D2 | chess.BB_H2, chess.BB_D7 | chess.BB_H7)
    assert isolated[0] < supported[0] and isolated[1] < supported[1]


def test_backward_pawn_receives_penalty() -> None:
    # White pawns d4 and e3, black pawn d5: e3 cannot advance because d5 x e4, and no
    # white pawn can ever support it from behind. Compare with the pawn already on e4,
    # side by side with d4, which is not backward.
    backward = _white_structure(chess.BB_D4 | chess.BB_E3, chess.BB_D5 | chess.BB_F7)
    healthy = _white_structure(chess.BB_D4 | chess.BB_E4, chess.BB_D5 | chess.BB_F7)
    assert backward[0] < healthy[0] and backward[1] < healthy[1]


def test_bishop_pair_receives_bonus() -> None:
    # A bishop plus a knight versus two bishops, so material stays close. The pair
    # bonus must be visible on top of the value difference.
    pair = evaluate_white(chess.Board("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1"))
    no_pair = evaluate_white(chess.Board("4k3/8/8/8/8/8/8/2B1KN2 w - - 0 1"))
    material_gap = evaluation.BISHOP_VALUE - evaluation.KNIGHT_VALUE
    assert pair - no_pair >= material_gap + min(evaluation.BISHOP_PAIR)


def test_mirrored_positions_have_opposite_scores() -> None:
    """The evaluation is colour-symmetric: mirroring the board negates the score."""
    fens = [
        chess.STARTING_FEN,
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r3k2r/pp3ppp/2n1b3/3p4/3P4/2N5/PP2BPPP/R3K2R w KQkq - 0 12",
        "8/5pk1/6p1/8/2P5/1P4P1/5PK1/8 w - - 0 40",
        "2r3k1/5ppp/8/3P4/8/8/5PPP/2R3K1 w - - 0 30",
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1",
        "4k3/pppp4/8/8/8/8/PPPP3P/4K3 w - - 0 1",
    ]
    for fen in fens:
        board = chess.Board(fen)
        mirrored = board.mirror()
        assert evaluate_white(mirrored) == -evaluate_white(board), fen
        # Side-to-move perspective: the mirror also flips the turn, so `evaluate`
        # returns the same number for both.
        assert evaluate(mirrored) == evaluate(board), fen


def test_symmetry_on_random_positions() -> None:
    rng = random.Random(3)
    for _ in range(200):
        board = chess.Board()
        for _ply in range(rng.randint(0, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        mirrored = board.mirror()
        assert evaluate_white(mirrored) == -evaluate_white(board), board.fen()


def test_material_dominates_positional_terms() -> None:
    # A whole minor piece outweighs every positional nicety the other side can have.
    # Black: perfect centre knight, castled king with full shield, bishop pair, rook on
    # an open file. White: same, but with an extra knight tucked in the corner.
    board = chess.Board("2r2rk1/pp3ppp/8/8/3n4/8/PP3PPP/N1B1KB1R w K - 0 1")
    plus = evaluate_white(board)
    minus = evaluate_white(_without(board, chess.A1))
    assert plus - minus > 150, (plus, minus)


def test_evaluation_is_deterministic() -> None:
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    first = evaluate(board)
    evaluation._pawn_cache.clear()
    assert all(evaluate(board) == first for _ in range(5))


def test_evaluation_does_not_touch_the_board() -> None:
    board = chess.Board("r3k2r/pp3ppp/2n1b3/3p4/3P4/2N5/PP2BPPP/R3K2R b KQkq - 0 12")
    before = board.fen()
    evaluate(board)
    assert board.fen() == before


def test_evaluation_is_fast_enough_for_leaf_nodes() -> None:
    board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2Q1RK1 w - - 0 9")
    count = 2000
    started = time.perf_counter()
    for _ in range(count):
        evaluate(board)
    per_call_us = (time.perf_counter() - started) / count * 1e6
    print(f"evaluate: {per_call_us:.0f} us per call")
    assert per_call_us < 500, per_call_us


def test_agent_returns_legal_moves() -> None:
    rng = random.Random(7)
    for _ in range(12):
        board = chess.Board()
        for _ply in range(rng.randint(0, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        uci = agent.get_move(board.fen(), 3000)
        assert chess.Move.from_uci(uci) in board.legal_moves, (board.fen(), uci)


def test_agent_finds_mate_in_one() -> None:
    uci = agent.get_move("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 5000)
    assert uci == "a1a8", uci


def test_agent_takes_the_hanging_queen() -> None:
    uci = agent.get_move("rnb1kbnr/pppp1ppp/8/4p3/4P2q/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3", 5000)
    # The queen on h4 is undefended; anything that wins it is fine. No white piece
    # attacks h4 here, so instead check the agent at least keeps its own material.
    assert chess.Move.from_uci(uci) in chess.Board(
        "rnb1kbnr/pppp1ppp/8/4p3/4P2q/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3"
    ).legal_moves
    uci = agent.get_move("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", 5000)
    assert uci == "f3h4", uci


def test_search_restores_the_board() -> None:
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    before = board.fen()
    searcher = agent.Searcher(time.monotonic() + 60)
    searcher.search_root(board, 3)
    assert board.fen() == before
    assert searcher.beta_cutoffs > 0


def test_alpha_beta_matches_plain_negamax() -> None:
    """Pruning must never change the root score."""

    def plain(board: chess.Board, depth: int) -> float:
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0.0
        moves = list(board.legal_moves)
        if depth == 0:
            # Same leaf rule as the agent: only a mate is recognised at the horizon.
            if board.is_check() and not moves:
                return -float(agent.MATE_SCORE)
            return float(evaluate(board))
        if not moves:
            return -(agent.MATE_SCORE + depth) if board.is_check() else 0.0
        best = -agent.INFINITY
        for move in moves:
            board.push(move)
            best = max(best, -plain(board, depth - 1))
            board.pop()
        return best

    for fen in [
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/5pk1/6p1/8/2P5/1P4P1/5PK1/8 b - - 0 40",
    ]:
        board = chess.Board(fen)
        for depth in (1, 2, 3):
            searcher = agent.Searcher(time.monotonic() + 120)
            _, pruned = searcher.search_root(board, depth)
            assert pruned == plain(board, depth), (fen, depth)


def main() -> None:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as failure:
            failures += 1
            print(f"FAIL  {name}: {failure!r}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
