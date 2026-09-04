"""Tests for the Late Move Reductions agent.

Run from the repository root:

    python -m pytest my-agents/12_late_move_reductions/test_lmr.py -q

or, without pytest, as a plain script (every test_* function is called in turn):

    python my-agents/12_late_move_reductions/test_lmr.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent
from agent import INFINITY, LmrPolicy, SearchConfig, Searcher

NO_DEADLINE = float("inf")
MATE_IN_ONE = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
Result = tuple[chess.Move, int, Searcher]

QUIET_CONTEXT = {
    "is_quiet": True,
    "in_check": False,
    "gives_check": False,
    "is_tt_move": False,
    "is_killer": False,
    "near_mate": False,
}


def _search(fen: str, depth: int, config: SearchConfig | None = None) -> Result:
    agent.TRANSPOSITION_TABLE.clear()
    searcher = Searcher(NO_DEADLINE, config)
    move, score = searcher.search_root(chess.Board(fen), depth)
    return move, score, searcher


def _no_lmr() -> SearchConfig:
    return SearchConfig(lmr=LmrPolicy(enabled=False))


# -- policy: the predicate and the amount -------------------------------------


def test_early_moves_are_full_depth() -> None:
    policy = LmrPolicy()
    for index in range(policy.full_depth_moves):
        assert not policy.should_reduce(move_index=index, depth=8, **QUIET_CONTEXT)


def test_late_quiet_moves_receive_reduction() -> None:
    policy = LmrPolicy()
    assert policy.should_reduce(move_index=4, depth=3, **QUIET_CONTEXT)
    assert policy.reduction(4, 3) == 1
    # A very late move deep in the tree loses a second ply.
    assert policy.reduction(policy.extra_late_moves, policy.extra_min_depth) == 2
    # ... but never more than max_reduction, and never below one real ply of search.
    assert policy.reduction(40, 3) == 1
    assert policy.reduction(40, 2) == 0


def test_shallow_nodes_are_not_reduced() -> None:
    policy = LmrPolicy()
    assert not policy.should_reduce(move_index=10, depth=policy.min_depth - 1, **QUIET_CONTEXT)


def test_tactical_moves_are_not_reduced() -> None:
    policy = LmrPolicy()
    base = dict(QUIET_CONTEXT)
    assert not policy.should_reduce(move_index=9, depth=6, **{**base, "is_quiet": False})
    assert not policy.should_reduce(move_index=9, depth=6, **{**base, "gives_check": True})
    assert not policy.should_reduce(move_index=9, depth=6, **{**base, "in_check": True})
    assert not policy.should_reduce(move_index=9, depth=6, **{**base, "is_tt_move": True})
    assert not policy.should_reduce(move_index=9, depth=6, **{**base, "is_killer": True})
    assert not policy.should_reduce(move_index=9, depth=6, **{**base, "near_mate": True})


def test_policy_can_be_disabled() -> None:
    policy = LmrPolicy(enabled=False)
    assert not policy.should_reduce(move_index=20, depth=10, **QUIET_CONTEXT)


# -- search: reductions really happen, captures really are exempt -------------


class _DepthRecorder(Searcher):
    """Records the (ply, move, depth) of every child search so tests can see the tree."""

    def __init__(self, config: SearchConfig | None = None) -> None:
        super().__init__(NO_DEADLINE, config)
        self.calls: list[tuple[int, chess.Move | None, int]] = []
        self.root_order: list[chess.Move] = []

    def _ordered_moves(
        self, board: chess.Board, tt_move: chess.Move | None, ply: int
    ) -> list[tuple[int, chess.Move]]:
        ordered = super()._ordered_moves(board, tt_move, ply)
        if ply == 0:
            self.root_order = [move for _, move in ordered]
        return ordered

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int = 0,
        allow_null: bool = True,
    ) -> int:
        last = board.peek() if board.move_stack else None
        self.calls.append((ply, last, depth))
        return super().negamax(board, depth, alpha, beta, ply, allow_null)


def test_search_reduces_late_quiet_moves_and_not_captures() -> None:
    # A calm middlegame position with plenty of quiet moves and a few captures on offer.
    fen = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7"
    agent.TRANSPOSITION_TABLE.clear()
    searcher = _DepthRecorder(SearchConfig(use_null_move=False))
    searcher.search_root(chess.Board(fen), 3)
    assert searcher.stats.reductions > 0

    root = chess.Board(fen)
    root_moves = searcher.root_order
    depth_by_move: dict[chess.Move, set[int]] = {}
    for ply, move, depth in searcher.calls:
        if ply == 1 and move is not None:
            depth_by_move.setdefault(move, set()).add(depth)

    # The first four root moves were searched only at full depth (2 plies).
    for move in root_moves[:4]:
        assert depth_by_move[move] == {2}, (move, depth_by_move[move])
    # At least one late quiet move was searched at depth 1.
    late_quiet = [m for m in root_moves[4:] if not root.is_capture(m) and not m.promotion]
    assert any(1 in depth_by_move[m] for m in late_quiet)
    # No capture was ever searched shallower than full depth.
    for move in root_moves:
        if root.is_capture(move):
            assert 1 not in depth_by_move[move], move


def test_promising_reduced_move_is_researched() -> None:
    """A reduced move that beats alpha must be searched again at full depth."""

    class _Optimist(Searcher):
        """Every reduced child search returns a huge score, forcing a re-search."""

        def __init__(self) -> None:
            super().__init__(NO_DEADLINE, SearchConfig(use_null_move=False))
            self.reduced_calls: list[chess.Move] = []
            self.full_calls: list[chess.Move] = []

        def negamax(
            self,
            board: chess.Board,
            depth: int,
            alpha: int,
            beta: int,
            ply: int = 0,
            allow_null: bool = True,
        ) -> int:
            if ply == 1 and depth == 1:  # the reduced search for a root move at depth 3
                self.reduced_calls.append(board.peek())
                return -10_000  # negated by the parent: looks like +10000 for the mover
            if ply == 1 and depth == 2:
                self.full_calls.append(board.peek())
            return super().negamax(board, depth, alpha, beta, ply, allow_null)

    agent.TRANSPOSITION_TABLE.clear()
    searcher = _Optimist()
    searcher.search_root(chess.Board(), 3)
    assert searcher.reduced_calls, "no root move was reduced"
    assert searcher.stats.researches == len(searcher.reduced_calls)
    for move in searcher.reduced_calls:
        assert move in searcher.full_calls, f"{move} beat alpha when reduced; no re-search"


def test_no_research_when_reduced_search_fails_low() -> None:
    fen = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7"
    _, _, searcher = _search(fen, 4)
    # Well-ordered positions re-search only a small fraction of reduced moves.
    assert searcher.stats.reductions > 0
    assert searcher.stats.researches < searcher.stats.reductions


# -- strength and correctness ---------------------------------------------------

def test_best_move_stable_on_tactical_suite() -> None:
    """LMR must not change the answer on forced tactics: compare with LMR off."""
    suite = [
        MATE_IN_ONE,
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",  # Ra8#
        "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq d3 0 3",  # exd4 wins pawn
        "rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq d6 0 3",  # Nxe5 wins pawn
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "2kr4/ppp5/8/8/8/8/PPP5/2KR4 w - - 0 1",
    ]
    for fen in suite:
        with_lmr, score_lmr, _ = _search(fen, 4)
        without_lmr, score_plain, _ = _search(fen, 4, _no_lmr())
        assert with_lmr == without_lmr, (fen, with_lmr, without_lmr)
        assert abs(score_lmr - score_plain) <= 150, (fen, score_lmr, score_plain)


def test_mate_in_one_found_with_lmr() -> None:
    move, score, _ = _search(MATE_IN_ONE, 3)
    assert move.uci() == "h5f7"
    assert score >= agent.MATE_BOUND


def test_mate_in_two_found_with_lmr() -> None:
    # 1.Nf6+ gxf6 2.Bxf7#. Three plies plus one to see the mated side has no moves.
    fen = "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 10"
    move, score, _ = _search(fen, 4)
    assert move.uci() == "d5f6", move
    assert score >= agent.MATE_BOUND, score
    # Same answer with LMR off: the reduction did not change a forced line.
    plain, plain_score, _ = _search(fen, 4, _no_lmr())
    assert plain == move
    assert plain_score == score


def test_reduced_node_count() -> None:
    """The whole point: at equal depth, LMR searches fewer nodes."""
    fens = [
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
        "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
        "r2q1rk1/pp1bppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/2KR1B1R w - - 0 9",
    ]
    total_with = total_without = 0
    for fen in fens:
        _, _, with_lmr = _search(fen, 5)
        _, _, without_lmr = _search(fen, 5, _no_lmr())
        total_with += with_lmr.stats.nodes
        total_without += without_lmr.stats.nodes
    print(f"nodes at depth 5: lmr {total_with}, no lmr {total_without}")
    assert total_with < total_without


def test_returns_legal_moves_everywhere() -> None:
    fens = [
        chess.STARTING_FEN,
        "8/P7/8/8/8/8/k6K/8 w - - 0 1",  # promotion
        "8/8/8/3pP3/8/8/8/k6K w - d6 0 1",  # en passant
        "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",  # in check
        "rnbqk1nr/pppp1ppp/8/4p3/1b1PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3",  # in check, many replies
        "7k/8/8/8/8/8/8/K6R w - - 0 1",
        "k7/8/8/8/8/8/8/6RK b - - 0 1",  # black, few moves
        "8/8/8/8/8/2k5/8/K7 w - - 0 1",  # bare kings, only legal moves matter
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",  # mate in 1
    ]
    for fen in fens:
        board = chess.Board(fen)
        uci = agent.get_move(fen, 2_000)
        assert chess.Move.from_uci(uci) in board.legal_moves, (fen, uci)


def test_board_restored_after_search() -> None:
    board = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7")
    before = board.fen()
    agent.TRANSPOSITION_TABLE.clear()
    Searcher(NO_DEADLINE).search_root(board, 4)
    assert board.fen() == before
    assert not board.move_stack


def test_search_respects_deadline() -> None:
    import time

    board = chess.Board()
    agent.TRANSPOSITION_TABLE.clear()
    searcher = Searcher(time.monotonic() + 0.05)
    try:
        searcher.negamax(board, 20, -INFINITY, INFINITY, 0)
    except agent.OutOfTime:
        return
    raise AssertionError("a depth-20 search finished inside 50 ms; deadline not honoured")


def test_time_budget_is_a_slice_of_the_clock() -> None:
    assert agent.move_budget_ms(120_000) == 4_000
    assert agent.move_budget_ms(600) <= 600 - agent.SAFETY_MS
    assert agent.move_budget_ms(10) >= 1


def test_config_toggles_are_independent() -> None:
    fen = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7"
    base = SearchConfig()
    variants = [
        replace(base, use_pvs=False),
        replace(base, use_null_move=False),
        replace(base, use_tt=False),
        replace(base, use_killers=False, use_history=False),
    ]
    for config in variants:
        move, _, _ = _search(fen, 3, config)
        assert move in chess.Board(fen).legal_moves


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"ok    {name}")
            except Exception as error:
                failures += 1
                print(f"FAIL  {name}: {error!r}")
    raise SystemExit(1 if failures else 0)
