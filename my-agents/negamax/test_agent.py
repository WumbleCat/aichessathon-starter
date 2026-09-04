"""Tests for the negamax agent.

Run from the repository root:

    uv run python my-agents/negamax/test_agent.py
"""

import math
import sys
import time
import unittest
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent


def plain_negamax(board: chess.Board, depth: int, ply: int = 0) -> float:
    """The readme's basic negamax (no pruning), with the same leaf handling as the agent."""
    if ply > 0 and (board.halfmove_clock >= 100 or board.is_repetition(2)):
        return 0
    if depth == 0:
        return agent.Searcher().quiescence(board, -agent.INFINITY, agent.INFINITY, ply)
    moves = list(board.legal_moves)
    if not moves:
        return -(agent.MATE_SCORE - ply) if board.is_check() else 0
    best = -math.inf
    for move in moves:
        board.push(move)
        best = max(best, -plain_negamax(board, depth - 1, ply + 1))
        board.pop()
    return best


def plain_minimax(board: chess.Board, depth: int, maximiser: chess.Color, ply: int = 0) -> float:
    """Classic minimax with separate MAX and MIN cases, scored for `maximiser`."""
    if ply > 0 and (board.halfmove_clock >= 100 or board.is_repetition(2)):
        return 0
    if depth == 0:
        leaf = agent.Searcher().quiescence(board, -agent.INFINITY, agent.INFINITY, ply)
        return leaf if board.turn == maximiser else -leaf
    moves = list(board.legal_moves)
    if not moves:
        if not board.is_check():
            return 0
        mate = agent.MATE_SCORE - ply
        return -mate if board.turn == maximiser else mate
    best = -math.inf if board.turn == maximiser else math.inf
    for move in moves:
        board.push(move)
        score = plain_minimax(board, depth - 1, maximiser, ply + 1)
        board.pop()
        best = max(best, score) if board.turn == maximiser else min(best, score)
    return best


# Kiwipete (index 2) is capture-heavy; the unpruned reference searches skip it.
POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
]
QUIET_POSITIONS = [fen for index, fen in enumerate(POSITIONS) if index != 2]


def forces_mate_in_two(board: chess.Board, move: chess.Move) -> bool:
    """Brute force: after `move`, does every reply allow mate on the next move?"""
    board.push(move)
    try:
        replies = list(board.legal_moves)
        if not replies:
            return board.is_checkmate()
        for reply in replies:
            board.push(reply)
            try:
                if not any(_mates(board, m) for m in board.legal_moves):
                    return False
            finally:
                board.pop()
        return True
    finally:
        board.pop()


def _mates(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    mate = board.is_checkmate()
    board.pop()
    return mate


class EvaluationTests(unittest.TestCase):
    def test_start_position_is_balanced(self) -> None:
        self.assertEqual(agent.evaluate(chess.Board()), 0)

    def test_score_is_from_movers_perspective(self) -> None:
        board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
        white_view = agent.evaluate(board)
        board.turn = chess.BLACK
        black_view = agent.evaluate(board)
        self.assertGreater(white_view, 0)
        self.assertEqual(white_view, -black_view)

    def test_piece_square_tables_are_symmetric(self) -> None:
        for phase in agent.PST.values():
            for piece_type, white in phase[chess.WHITE].items():
                black = phase[chess.BLACK][piece_type]
                self.assertEqual(white[chess.E4], black[chess.E5])
                self.assertEqual(white[chess.A1], black[chess.A8])


class SearchTests(unittest.TestCase):
    def test_negamax_matches_minimax(self) -> None:
        for fen in QUIET_POSITIONS:
            board = chess.Board(fen)
            expected = plain_minimax(board, 2, board.turn)
            got = agent.Searcher().negamax(board, 2, -agent.INFINITY, agent.INFINITY, 0)
            self.assertEqual(got, expected, fen)

    def test_alpha_beta_matches_plain_negamax(self) -> None:
        for fen in QUIET_POSITIONS:
            board = chess.Board(fen)
            expected = plain_negamax(board, 2)
            got = agent.Searcher().negamax(board, 2, -agent.INFINITY, agent.INFINITY, 0)
            self.assertEqual(got, expected, fen)

    def test_board_is_restored_after_search(self) -> None:
        board = chess.Board(POSITIONS[2])
        before = board.fen()
        agent.Searcher().search_root(board, 3)
        self.assertEqual(board.fen(), before)
        self.assertEqual(len(board.move_stack), 0)

    def test_board_is_restored_after_an_aborted_search(self) -> None:
        board = chess.Board(POSITIONS[2])
        before = board.fen()
        searcher = agent.Searcher(deadline=time.monotonic() + 0.05)
        with self.assertRaises(agent.OutOfTime):
            while True:
                searcher.search_root(board, 6)
        self.assertGreater(len(board.move_stack), 0)  # the abort leaves moves pushed ...
        move = agent.choose_move(chess.Board(POSITIONS[2]), 200)  # ... choose_move must undo
        self.assertIn(move, chess.Board(POSITIONS[2]).legal_moves)
        board = chess.Board(POSITIONS[2])
        agent.choose_move(board, 200)
        self.assertEqual(board.fen(), before)

    def test_partial_iteration_result_is_used(self) -> None:
        # With a hopelessly short deadline, depth 1 aborts. The first root move is the
        # static pick; if quiescence shows it loses material and a later move finished,
        # the finished move must win. Kiwipete: static likes Qxf6?? (a defended knight).
        board = chess.Board(POSITIONS[2])
        self.assertEqual(agent.static_best(board).uci(), "f3f6")
        for _ in range(5):
            move = agent.choose_move(chess.Board(POSITIONS[2]), 40)
            self.assertNotEqual(move.uci(), "f3f6")

    def test_finds_mate_in_one(self) -> None:
        board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
        move, score = agent.Searcher().search_root(board, 2)
        self.assertEqual(move.uci(), "a1a8")
        self.assertGreaterEqual(score, agent.MATE_SCORE - 10)

    def test_finds_mate_in_two(self) -> None:
        # Two rooks ladder the king: 1.Ra7 Kg8 2.Rb8# (or the mirror image).
        board = chess.Board("7k/8/8/8/8/8/8/RR4K1 w - - 0 1")
        move, score = agent.Searcher().search_root(board, 3)
        self.assertTrue(forces_mate_in_two(board, move), move.uci())
        self.assertGreaterEqual(score, agent.MATE_SCORE - 10)

    def test_prefers_the_faster_mate(self) -> None:
        board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
        move = agent.choose_move(board, 5_000, max_depth=4)
        self.assertEqual(move.uci(), "a1a8")

    def test_quiescence_does_not_grab_a_defended_piece(self) -> None:
        # White's queen can take a pawn on e5 that a knight guards. Depth 1 with a
        # naive evaluation would love it; quiescence sees the recapture.
        board = chess.Board("4k3/8/3n4/4p3/8/8/8/4K2Q w - - 0 1")
        move, _ = agent.Searcher().search_root(board, 1)
        self.assertNotEqual(move.uci(), "h1e5")


class InterfaceTests(unittest.TestCase):
    def test_always_returns_a_legal_move(self) -> None:
        for fen in POSITIONS:
            board = chess.Board(fen)
            uci = agent.get_move(fen, 2_000)
            self.assertIn(chess.Move.from_uci(uci), board.legal_moves, fen)

    def test_promotion_is_encoded(self) -> None:
        # a8=Q is mate, so the promotion piece must be spelled out in the UCI string.
        board = chess.Board("7k/P5pp/8/8/8/8/8/K7 w - - 0 1")
        uci = agent.get_move(board.fen(), 2_000)
        self.assertEqual(uci, "a7a8q")
        move, _ = agent.Searcher().search_root(chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1"), 1)
        self.assertEqual(move.uci(), "a7a8q")

    def test_respects_a_tiny_clock(self) -> None:
        started = time.monotonic()
        uci = agent.get_move(POSITIONS[2], 300)
        elapsed_ms = (time.monotonic() - started) * 1000
        self.assertIn(chess.Move.from_uci(uci), chess.Board(POSITIONS[2]).legal_moves)
        self.assertLess(elapsed_ms, 250)

    def test_budget_never_exceeds_half_the_clock(self) -> None:
        for left in (50, 300, 1_000, 10_000, 120_000):
            self.assertLessEqual(agent.move_budget_ms(left), max(agent.MIN_BUDGET_MS, left // 2))
        self.assertEqual(agent.move_budget_ms(120_000), 4_000)
        self.assertEqual(agent.move_budget_ms(2_000), 300)
        self.assertEqual(agent.move_budget_ms(400), 100)

    def test_remembers_the_position_after_its_own_move(self) -> None:
        agent._seen_positions.clear()
        fen = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
        uci = agent.get_move(fen, 300)
        board = chess.Board(fen)
        board.push_uci(uci)
        self.assertIn(board.epd(), agent._seen_positions)

    def test_avoids_repeating_when_ahead(self) -> None:
        # White (a rook up) has shuffled Ke1-e2, Ke2-e1; the same position comes back.
        # Ke1-e2 would repeat it, so the engine must pick something else.
        agent._seen_positions.clear()
        fen = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
        agent.get_move(fen, 500)
        agent.get_move(fen, 500)
        board = chess.Board(fen)
        board.push_uci("e1e2")
        after_ke2 = board.epd()
        agent._seen_positions[after_ke2] = 1
        repeating = agent._repeating_moves(chess.Board(fen))
        self.assertIn(chess.Move.from_uci("e1e2"), repeating)
        move = agent.choose_move(chess.Board(fen), 500, repeating, max_depth=2)
        self.assertNotEqual(move.uci(), "e1e2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
