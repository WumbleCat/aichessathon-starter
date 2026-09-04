"""Tests for the futility pruning / reverse futility / razoring agent.

Run from the repo root:

    uv run python my-agents/14_futility_pruning_and_razoring/test_pruning.py
"""

from __future__ import annotations

import math
import sys
import time
import unittest
from pathlib import Path
from typing import ClassVar

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent
from agent import DEFAULT_CONFIG, NO_PRUNING, PruningConfig, Searcher

FAR_FUTURE = time.monotonic() + 3600.0

MIDDLEGAME_FENS = [
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 8",
    "rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
    "2r3k1/pp3ppp/2n5/3p4/3P4/2P2N2/P4PPP/4R1K1 b - - 0 20",
    "8/5pk1/6p1/8/3P4/2P3P1/5PK1/8 w - - 0 40",
]


def reference_negamax(searcher: Searcher, board: chess.Board, depth: int, ply: int) -> float:
    """Plain full-width negamax using the agent's own quiescence at the horizon.

    No alpha-beta and no pruning, so the score is the exact minimax value of the tree
    the searcher is supposed to be approximating.
    """
    if ply > 0 and (board.is_repetition(2) or board.halfmove_clock >= 100):
        return 0.0
    if depth == 0:
        return searcher.quiescence(board, -math.inf, math.inf, ply)
    moves = list(board.legal_moves)
    if not moves:
        return -(agent.MATE_SCORE - ply) if board.is_check() else 0.0
    best = -math.inf
    for move in moves:
        board.push(move)
        best = max(best, -reference_negamax(searcher, board, depth - 1, ply + 1))
        board.pop()
    return best


def root_score(
    board: chess.Board, depth: int, config: PruningConfig
) -> tuple[chess.Move, float, Searcher]:
    searcher = Searcher(FAR_FUTURE, config)
    move, score = searcher.search_root(board, depth)
    return move, score, searcher


class ConfigTests(unittest.TestCase):
    def test_all_pruning_can_be_toggled(self) -> None:
        self.assertTrue(DEFAULT_CONFIG.futility)
        self.assertTrue(DEFAULT_CONFIG.reverse_futility)
        self.assertTrue(DEFAULT_CONFIG.razoring)
        self.assertTrue(DEFAULT_CONFIG.delta)
        self.assertFalse(NO_PRUNING.futility)
        self.assertFalse(NO_PRUNING.reverse_futility)
        self.assertFalse(NO_PRUNING.razoring)
        self.assertFalse(NO_PRUNING.delta)
        self.assertEqual(NO_PRUNING.max_prune_depth, 0)
        self.assertEqual(DEFAULT_CONFIG.max_prune_depth, 3)

    def test_margins_must_cover_depth(self) -> None:
        with self.assertRaises(ValueError):
            PruningConfig(futility_depth=3, futility_margins=(0, 200, 350))
        with self.assertRaises(ValueError):
            PruningConfig(razor_depth=3, razor_margins=(0, 300))
        # Disabled features do not need margins.
        PruningConfig(futility=False, futility_depth=5, futility_margins=())


class BaselineEquivalenceTests(unittest.TestCase):
    def test_feature_off_matches_reference_search(self) -> None:
        """With every feature off, the root score is the exact minimax value."""
        for fen in MIDDLEGAME_FENS:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                reference = reference_negamax(Searcher(FAR_FUTURE, NO_PRUNING), board, 2, 0)
                _, score, _ = root_score(board, 2, NO_PRUNING)
                self.assertEqual(score, reference)

    def test_feature_off_never_prunes(self) -> None:
        board = chess.Board(MIDDLEGAME_FENS[1])
        _, _, searcher = root_score(board, 4, NO_PRUNING)
        self.assertEqual(searcher.stats.futility_pruned, 0)
        self.assertEqual(searcher.stats.reverse_futility_cutoffs, 0)
        self.assertEqual(searcher.stats.razor_drops, 0)
        self.assertEqual(searcher.stats.delta_pruned, 0)


class QuietMoveTests(unittest.TestCase):
    def test_checks_captures_and_promotions_are_not_quiet(self) -> None:
        board = chess.Board("4k3/1P6/8/8/8/8/8/R3K3 w - - 0 1")
        self.assertFalse(agent.is_quiet(board, chess.Move.from_uci("b7b8q")))  # promotion
        self.assertFalse(agent.is_quiet(board, chess.Move.from_uci("a1a8")))  # check
        self.assertTrue(agent.is_quiet(board, chess.Move.from_uci("a1a2")))  # quiet
        board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
        self.assertFalse(agent.is_quiet(board, chess.Move.from_uci("e4d5")))  # capture
        self.assertTrue(agent.is_quiet(board, chess.Move.from_uci("e4e5")))

    def test_pruned_moves_are_never_checks_captures_or_promotions(self) -> None:
        """Record every move the search actually skipped and inspect it."""
        skipped: list[tuple[str, str]] = []
        original = agent.is_quiet

        def recording_is_quiet(board: chess.Board, move: chess.Move) -> bool:
            quiet = original(board, move)
            if quiet:
                skipped.append((board.fen(), move.uci()))
            return quiet

        agent.is_quiet = recording_is_quiet
        try:
            for fen in MIDDLEGAME_FENS[:3]:
                root_score(chess.Board(fen), 4, DEFAULT_CONFIG)
        finally:
            agent.is_quiet = original

        self.assertGreater(len(skipped), 0, "expected futility pruning to skip some moves")
        for fen, uci in skipped:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            self.assertIn(move, board.legal_moves)
            self.assertIsNone(move.promotion)
            self.assertFalse(board.is_capture(move))
            self.assertFalse(board.gives_check(move))
            self.assertFalse(board.is_check())


class PruningActivityTests(unittest.TestCase):
    def test_quiet_hopeless_branches_prune(self) -> None:
        totals = agent.SearchStats()
        for fen in MIDDLEGAME_FENS[:3]:
            _, _, searcher = root_score(chess.Board(fen), 4, DEFAULT_CONFIG)
            totals.futility_pruned += searcher.stats.futility_pruned
            totals.reverse_futility_cutoffs += searcher.stats.reverse_futility_cutoffs
            totals.razor_drops += searcher.stats.razor_drops
            totals.delta_pruned += searcher.stats.delta_pruned
        self.assertGreater(totals.futility_pruned, 0)
        self.assertGreater(totals.reverse_futility_cutoffs, 0)
        self.assertGreater(totals.razor_drops, 0)
        self.assertGreater(totals.delta_pruned, 0)

    def test_node_count_decreases(self) -> None:
        for fen in MIDDLEGAME_FENS[:2]:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                _, _, pruned = root_score(board, 4, DEFAULT_CONFIG)
                _, _, full = root_score(board, 4, NO_PRUNING)
                self.assertLess(pruned.stats.nodes, full.stats.nodes)

    def test_each_feature_reduces_nodes_alone(self) -> None:
        board = chess.Board(MIDDLEGAME_FENS[1])
        _, _, full = root_score(board, 4, NO_PRUNING)
        singles = {
            "futility": PruningConfig(reverse_futility=False, razoring=False, delta=False),
            "reverse_futility": PruningConfig(futility=False, razoring=False, delta=False),
            "razoring": PruningConfig(futility=False, reverse_futility=False, delta=False),
            "delta": PruningConfig(futility=False, reverse_futility=False, razoring=False),
        }
        for name, config in singles.items():
            with self.subTest(feature=name):
                _, _, only = root_score(board, 4, config)
                self.assertLess(only.stats.nodes, full.stats.nodes)


class TacticalTests(unittest.TestCase):
    # (fen, expected best move, depth) - every one must survive the pruning.
    PUZZLES: ClassVar[list[tuple[str, str, int]]] = [
        # Scholar's mate in one.
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7", 2),
        # Nf6+ gxf6 Bxf7#: mate in two.
        ("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1", "d5f6", 4),
        # Back-rank mate in one for Black.
        ("6k1/5ppp/8/8/8/8/2r2PPP/6K1 b - - 0 1", "c2c1", 2),
        # Free queen on h4.
        ("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f3h4", 3),
        # Knight fork wins the rook.
        ("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1", "b5c7", 4),
        # Promotion is the only good move (odd depth so it cannot be postponed a move).
        ("8/1P4k1/8/8/8/8/8/4K3 w - - 0 1", "b7b8q", 3),
    ]

    def test_tactics_survive_pruning(self) -> None:
        for fen, expected, depth in self.PUZZLES:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                move, _, _ = root_score(board, depth, DEFAULT_CONFIG)
                self.assertEqual(move.uci(), expected)
                baseline, _, _ = root_score(board, depth, NO_PRUNING)
                self.assertEqual(baseline.uci(), expected)

    def test_mate_scores_are_found(self) -> None:
        board = chess.Board("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1")
        _, score, _ = root_score(board, 4, DEFAULT_CONFIG)
        self.assertTrue(agent.is_mate_score(score))
        self.assertGreater(score, 0)

    def test_in_check_positions_are_not_pruned(self) -> None:
        # 1.e4 e5 2.f3 Qh4+: White is in check; g3 is the only reply that does not
        # drop the e-pawn, and the in-check root must be searched in full to see that.
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5P2/PPPP2PP/RNBQKBNR w KQkq - 1 3")
        self.assertTrue(board.is_check())
        move, _, _ = root_score(board, 3, DEFAULT_CONFIG)
        self.assertIn(move, board.legal_moves)
        self.assertEqual(move.uci(), "g2g3")


class InterfaceTests(unittest.TestCase):
    FENS: ClassVar[list[str]] = [
        chess.STARTING_FEN,
        *MIDDLEGAME_FENS,
        "rnb1kbnr/pppp1ppp/8/4p3/4P2q/5P2/PPPP2PP/RNBQKBNR w KQkq - 1 3",  # in check
        "7k/8/8/8/8/8/6q1/7K w - - 0 1",  # only move
        "8/1P4k1/8/8/8/8/8/4K3 w - - 0 1",  # promotion
        "8/8/8/8/8/2k5/8/K7 w - - 0 1",  # bare king endgame
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",  # en passant available
    ]

    def test_get_move_returns_legal_uci(self) -> None:
        for fen in self.FENS:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                uci = agent.get_move(fen, 1_000)
                self.assertIn(chess.Move.from_uci(uci), board.legal_moves)

    def test_respects_time_budget(self) -> None:
        started = time.monotonic()
        agent.get_move(MIDDLEGAME_FENS[1], 3_000)  # budget is 100 ms
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 1.5)

    def test_board_is_restored_after_timeout(self) -> None:
        """OutOfTime unwinds through pushed moves; the caller's board must be intact."""
        board = chess.Board(MIDDLEGAME_FENS[1])
        move = agent.choose_move(board, 3_000)
        self.assertEqual(board.fen(), MIDDLEGAME_FENS[1])
        self.assertEqual(len(board.move_stack), 0)
        self.assertIn(move, board.legal_moves)

    def test_game_memory_resets_on_new_game(self) -> None:
        memory = agent.GameMemory()
        late = chess.Board(MIDDLEGAME_FENS[3])
        memory.record(late)
        self.assertEqual(len(memory.seen), 1)
        memory.record(chess.Board())  # fewer plies: a new game has started
        self.assertEqual(len(memory.seen), 1)
        self.assertIn(chess.Board()._transposition_key(), memory.seen)

    def test_repetition_memory_scores_known_positions_as_draws(self) -> None:
        board = chess.Board("7k/8/8/8/8/8/8/R6K w - - 0 1")
        # Pretend the position after Ra1-a2 Kh8-g8 Ra2-a1 Kg8-h8 has already occurred.
        history = frozenset({board._transposition_key()})
        searcher = Searcher(FAR_FUTURE, DEFAULT_CONFIG, history)
        for uci in ("a1a2", "h8g8", "a2a1", "g8h8"):
            board.push_uci(uci)
        self.assertEqual(searcher.negamax(board, 2, -math.inf, math.inf, ply=4), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
