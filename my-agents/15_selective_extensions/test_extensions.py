"""Tests for the selective-extensions agent.

Run from the repository root:

    uv run python -m unittest my-agents/15_selective_extensions/test_extensions.py -v
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent
from agent import SearchConfig, Searcher, see

NO_EXTENSIONS = SearchConfig(
    check_extension=False,
    recapture_extension=False,
    passed_pawn_extension=False,
    singular_extension=False,
)

# White to move: 1.Ra8+ Kh7 2.Qh8# (or 1...Kg7 2.Qh8#). A pure checking line.
MATE_IN_TWO = "6k1/5p2/6p1/8/8/8/1Q6/R5K1 w - - 0 1"
# White queen and rook against a bare king: checks everywhere.
CHECK_HEAVY = "7k/8/8/8/8/8/R7/1Q4K1 w - - 0 1"
# Kings and pawns only, far apart: no move can give check within a few plies.
QUIET = "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1"


def searched(fen: str, depth: int, config: SearchConfig) -> Searcher:
    searcher = Searcher(config)
    searcher.search_root(chess.Board(fen), depth)
    return searcher


class ExtensionRuleTests(unittest.TestCase):
    """Each rule fires on the situation it is named after, and only there."""

    def _extension_for(
        self, fen: str, uci: str, config: SearchConfig, last_capture: tuple[int, int] | None = None
    ) -> tuple[int, Searcher]:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        self.assertIn(move, board.legal_moves)
        searcher = Searcher(config)
        is_capture = board.is_capture(move)
        board.push(move)
        gives_check = board.is_check()
        ext = searcher.extension(board, move, 3, gives_check, is_capture, last_capture, 0)
        board.pop()
        return ext, searcher

    def test_check_extends(self) -> None:
        ext, s = self._extension_for(CHECK_HEAVY, "a2a8", SearchConfig())
        self.assertEqual(ext, 1)
        self.assertEqual(s.extensions["check"], 1)

    def test_hanging_check_does_not_extend(self) -> None:
        # Qh7+ can simply be taken by the king: not a sound check.
        fen = "7k/8/8/8/8/8/8/1Q4K1 w - - 0 1"
        ext, _ = self._extension_for(fen, "b1h7", SearchConfig())
        self.assertEqual(ext, 0)

    def test_quiet_move_does_not_extend(self) -> None:
        ext, s = self._extension_for(QUIET, "e2e4", SearchConfig())
        self.assertEqual(ext, 0)
        self.assertEqual(sum(s.extensions.values()), 0)

    def test_recapture_extends(self) -> None:
        # After ...Nxe4 (a knight captured on e4), Nxe4 recaptures a piece of similar value.
        fen = "rnbqkb1r/pppppppp/8/8/4n3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 3"
        ext, s = self._extension_for(
            fen, "c3e4", SearchConfig(check_extension=False), last_capture=(chess.E4, 100)
        )
        self.assertEqual(ext, 0, "a knight for a pawn is not an even recapture")
        ext, s = self._extension_for(
            fen, "c3e4", SearchConfig(check_extension=False), last_capture=(chess.E4, 320)
        )
        self.assertEqual(ext, 1)
        self.assertEqual(s.extensions["recapture"], 1)

    def test_passed_pawn_extends(self) -> None:
        # The a-pawn steps to a7 with nothing in its way.
        fen = "7k/8/P7/8/8/8/8/7K w - - 0 1"
        ext, s = self._extension_for(fen, "a6a7", SearchConfig(check_extension=False))
        self.assertEqual(ext, 1)
        self.assertEqual(s.extensions["passed_pawn"], 1)
        # Only the step onto the seventh rank counts, not earlier advances.
        fen = "7k/8/8/P7/8/8/8/7K w - - 0 1"
        ext, _ = self._extension_for(fen, "a5a6", SearchConfig(check_extension=False))
        self.assertEqual(ext, 0)
        # A rook on g7 takes the pawn for free on a7: not worth a ply.
        fen = "7k/6r1/P7/8/8/8/8/7K w - - 0 1"
        ext, _ = self._extension_for(fen, "a6a7", SearchConfig(check_extension=False))
        self.assertEqual(ext, 0)

    def test_each_rule_toggles_independently(self) -> None:
        ext, _ = self._extension_for(CHECK_HEAVY, "a2a8", SearchConfig(check_extension=False))
        self.assertEqual(ext, 0)
        fen = "7k/8/P7/8/8/8/8/7K w - - 0 1"
        ext, _ = self._extension_for(fen, "a6a7", SearchConfig(passed_pawn_extension=False))
        self.assertEqual(ext, 0)
        fen = "rnbqkb1r/pppppppp/8/8/4n3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 3"
        ext, _ = self._extension_for(
            fen, "c3e4", SearchConfig(recapture_extension=False), last_capture=(chess.E4, 320)
        )
        self.assertEqual(ext, 0)

    def test_budget_denies_extension(self) -> None:
        board = chess.Board(CHECK_HEAVY)
        move = chess.Move.from_uci("a2a8")
        searcher = Searcher(SearchConfig(max_extensions=2))
        board.push(move)
        self.assertEqual(searcher.extension(board, move, 3, True, False, None, 1), 1)
        self.assertEqual(searcher.extension(board, move, 3, True, False, None, 2), 0)
        self.assertEqual(searcher.extensions["budget_denied"], 1)


class SearchDepthTests(unittest.TestCase):
    """Extensions change how deep the main search goes, within the budget."""

    def test_forcing_line_gets_extra_depth(self) -> None:
        depth = 3
        with_ext = searched(CHECK_HEAVY, depth, SearchConfig())
        without = searched(CHECK_HEAVY, depth, NO_EXTENSIONS)
        self.assertGreater(with_ext.extensions["check"], 0)
        self.assertGreater(with_ext.max_main_ply, depth)
        self.assertEqual(without.max_main_ply, depth)

    def test_quiet_position_gets_no_extra_depth(self) -> None:
        depth = 3
        searcher = searched(QUIET, depth, SearchConfig())
        self.assertEqual(sum(searcher.extensions.values()), 0)
        self.assertEqual(searcher.max_main_ply, depth)

    def test_budget_bounds_growth(self) -> None:
        depth = 3
        for budget in (0, 1, 2, 3):
            searcher = searched(CHECK_HEAVY, depth, SearchConfig(max_extensions=budget))
            self.assertLessEqual(searcher.max_main_ply, depth + budget, f"budget {budget}")
        unbounded = searched(CHECK_HEAVY, depth, SearchConfig(max_extensions=2))
        self.assertGreater(unbounded.extensions["budget_denied"], 0)

    def test_extensions_nodes_stay_manageable(self) -> None:
        depth = 4
        with_ext = searched(CHECK_HEAVY, depth, SearchConfig(max_extensions=2))
        without = searched(CHECK_HEAVY, depth, NO_EXTENSIONS)
        self.assertLess(with_ext.nodes, 12 * max(without.nodes, 1))


class TacticsTests(unittest.TestCase):
    def test_mate_found_sooner_with_extensions(self) -> None:
        board = chess.Board(MATE_IN_TWO)
        with_ext = Searcher(SearchConfig())
        move, score = with_ext.search_root(board, 2)
        self.assertEqual(move.uci(), "a1a8")
        self.assertGreaterEqual(score, agent.MATE_BOUND)
        without = Searcher(NO_EXTENSIONS)
        _, score = without.search_root(board, 2)
        self.assertLess(score, agent.MATE_BOUND)

    def test_mate_in_one_is_played(self) -> None:
        fen = "6k1/5ppp/8/8/8/8/1Q3PPP/6K1 w - - 0 1"
        self.assertEqual(agent.get_move(fen, 1_000), "b2b8")

    def test_does_not_hang_the_queen(self) -> None:
        # Qxf7+?? Kxf7. Without the bishop on c4 the queen is simply lost.
        fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5Q2/PPPP1PPP/RNB1KBNR w KQkq - 0 3"
        move = agent.get_move(fen, 4_000)
        self.assertNotEqual(move, "f3f7")


class SeeTests(unittest.TestCase):
    def test_winning_and_losing_captures(self) -> None:
        board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
        self.assertEqual(see(board, chess.Move.from_uci("e4d5")), 100)
        board = chess.Board("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1")
        self.assertEqual(see(board, chess.Move.from_uci("e4d5")), 0)
        board = chess.Board("4k3/8/2p5/3p4/8/8/8/4KQ2 w - - 0 1")
        self.assertEqual(see(board, chess.Move.from_uci("f1d3")), 0)
        board = chess.Board("4k3/8/2p5/3p4/8/8/8/4K2Q w - - 0 1")
        self.assertEqual(see(board, chess.Move.from_uci("h1d5")), 100 - 900)

    def test_quiet_move_into_attack_is_negative(self) -> None:
        board = chess.Board("4k3/8/8/3p4/8/8/8/4KQ2 w - - 0 1")
        self.assertLess(see(board, chess.Move.from_uci("f1c4")), 0)


class InterfaceTests(unittest.TestCase):
    def test_legal_moves_from_many_positions(self) -> None:
        fens = [
            chess.STARTING_FEN,
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "8/P7/8/8/8/8/8/k6K w - - 0 1",
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "8/8/8/8/8/8/6k1/4K2q w - - 0 1",
        ]
        for fen in fens:
            board = chess.Board(fen)
            move = chess.Move.from_uci(agent.get_move(fen, 800))
            self.assertIn(move, board.legal_moves, fen)

    def test_terminates_within_budget(self) -> None:
        time_left = 2_000
        budget = agent.move_budget_ms(time_left) / 1000.0
        started = time.monotonic()
        agent.get_move(CHECK_HEAVY, time_left)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, budget + 0.25)

    def test_low_clock_still_answers(self) -> None:
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        started = time.monotonic()
        move = agent.get_move(fen, 120)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(chess.Move.from_uci(move))

    def test_board_is_restored_after_search(self) -> None:
        board = chess.Board(MATE_IN_TWO)
        before = board.fen()
        Searcher(SearchConfig()).search_root(board, 4)
        self.assertEqual(board.fen(), before)
        self.assertEqual(len(board.move_stack), 0)


if __name__ == "__main__":
    unittest.main()
