"""Tests for the killer heuristic, following the checklist in 08_killer_heuristic.md.

Run from the repo root:

    uv run python my-agents/08_killer_heuristic/test_killers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess

import agent

START = chess.STARTING_FEN
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
MIDGAME = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8"
ENDGAME = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
MATE_IN_TWO = "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1"
PROMOTION_ONLY = "8/P6k/8/8/8/8/8/K7 w - - 0 1"


class RecordingSearcher(agent.Searcher):
    """Records every cutoff with the facts the killer store decision depends on."""

    def __init__(self) -> None:
        super().__init__()
        self.stored: list[tuple[int, chess.Move, bool]] = []  # (ply, move, was_quiet)

    def _on_cutoff(self, board: chess.Board, move: chess.Move, ply: int, index: int) -> None:
        assert self.killers is not None
        before = list(self.killers.get(ply))
        super()._on_cutoff(board, move, ply, index)
        after = list(self.killers.get(ply))
        quiet = agent.is_quiet(board, move)
        if after != before:
            self.stored.append((ply, move, quiet))
        else:
            # Nothing changed: either the move was not quiet, or it already was killer 0.
            assert not quiet or before[0] == move, (ply, move, before, after)


def test_quiet_cutoff_is_stored() -> None:
    searcher = RecordingSearcher()
    searcher.search_fixed_depth(chess.Board(MIDGAME), 3)
    assert searcher.stats.cutoffs > 0
    assert searcher.stored, "no killer was ever stored"
    assert all(quiet for _, _, quiet in searcher.stored)
    assert searcher.stats.killers_stored == len(searcher.stored)


def test_capture_cutoff_is_not_stored() -> None:
    """Every move that changed the killer table was quiet; captures never got in."""
    searcher = RecordingSearcher()
    searcher.search_fixed_depth(chess.Board(KIWIPETE), 3)
    assert searcher.stats.cutoffs > searcher.stats.killers_stored, (
        "expected at least one capture cutoff that was refused by the killer table"
    )
    assert all(quiet for _, _, quiet in searcher.stored)


def test_killer_table_store_and_replace() -> None:
    table = agent.KillerTable()
    a, b, c = (
        chess.Move.from_uci("g1f3"),
        chess.Move.from_uci("b1c3"),
        chess.Move.from_uci("e2e4"),
    )
    assert table.get(5) == [None, None]
    table.store(5, a)
    assert table.get(5) == [a, None]
    table.store(5, b)
    assert table.get(5) == [b, a], "new killer goes to slot 0, old one shifts to slot 1"
    table.store(5, b)
    assert table.get(5) == [b, a], "re-storing killer 0 must not duplicate it into slot 1"
    table.store(5, c)
    assert table.get(5) == [c, b]
    table.store(5, b)
    assert table.get(5) == [b, c], "promoting the secondary killer swaps, never duplicates"
    assert table.rank(5, b) == 0 and table.rank(5, c) == 1 and table.rank(5, a) == -1
    assert table.get(4) == [None, None], "killers are per ply"
    table.store(agent.MAX_PLY + 10, a)  # out of range is ignored, not an error
    table.clear()
    assert table.get(5) == [None, None]


def test_no_duplicates_during_search() -> None:
    searcher = agent.Searcher()
    searcher.search_fixed_depth(chess.Board(KIWIPETE), 4)
    assert searcher.killers is not None
    for ply, entry in enumerate(searcher.killers.table):
        k0, k1 = entry
        assert k0 is None or k0 != k1, f"ply {ply} has the same move in both slots"


def test_killer_receives_ordering_bonus() -> None:
    board = chess.Board(MIDGAME)
    quiet_moves = [m for m in board.legal_moves if agent.is_quiet(board, m)]
    captures = [m for m in board.legal_moves if board.is_capture(m)]
    assert captures, "test position needs a capture"
    killer0, killer1 = quiet_moves[-1], quiet_moves[-2]

    table = agent.KillerTable()
    table.store(3, killer1)
    table.store(3, killer0)

    plain = agent.ordered_moves(board)
    with_killers = agent.ordered_moves(board, None, table, 3)
    assert sorted(m.uci() for m in plain) == sorted(m.uci() for m in with_killers), (
        "ordering must not add or drop moves"
    )

    position = {m: i for i, m in enumerate(with_killers)}
    assert position[killer0] < position[killer1]
    for capture in captures:
        assert position[capture] < position[killer0], "captures still come before killers"
    for move in quiet_moves:
        if move not in (killer0, killer1):
            assert position[killer1] < position[move], "killers come before other quiet moves"

    # At a different ply the killers are just ordinary quiet moves again.
    other_ply = agent.ordered_moves(board, None, table, 4)
    assert other_ply == plain

    # Score-level checks, independent of sort stability.
    s0 = agent.move_order_score(board, killer0, None, table, 3)
    s1 = agent.move_order_score(board, killer1, None, table, 3)
    sc = min(agent.move_order_score(board, c, None, table, 3) for c in captures)
    assert sc > s0 > s1 > 0


def test_result_unchanged_with_and_without_killers() -> None:
    """Killers reorder moves only; the alpha-beta value at a fixed depth is identical."""
    for fen, depth in [(START, 4), (KIWIPETE, 3), (MIDGAME, 3), (ENDGAME, 4)]:
        on = agent.Searcher(use_killers=True, use_hash_move=False)
        off = agent.Searcher(use_killers=False, use_hash_move=False)
        _, score_on = on.search_fixed_depth(chess.Board(fen), depth)
        _, score_off = off.search_fixed_depth(chess.Board(fen), depth)
        assert score_on == score_off, (fen, depth, score_on, score_off)
        assert off.stats.killers_stored == 0
        print(
            f"  {fen[:24]:24s} depth {depth}  score {score_on:6.0f}  "
            f"nodes on {on.stats.nodes:7d}  off {off.stats.nodes:7d}"
        )


def test_finds_mate_in_two() -> None:
    board = chess.Board(MATE_IN_TWO)
    searcher = agent.Searcher()
    move, score = searcher.search_fixed_depth(board, 4)
    assert score >= agent.MATE_SCORE - agent.MAX_PLY
    board.push(move)
    # Every reply must lose to a mating move.
    for reply in list(board.legal_moves):
        board.push(reply)
        mates = [m for m in board.legal_moves if board.gives_check(m)]
        found = False
        for m in mates:
            board.push(m)
            found = board.is_checkmate()
            board.pop()
            if found:
                break
        board.pop()
        assert found, f"{move.uci()} does not force mate after {reply.uci()}"
    board.pop()


def test_get_move_is_legal() -> None:
    for fen in [START, KIWIPETE, MIDGAME, ENDGAME, MATE_IN_TWO, PROMOTION_ONLY]:
        board = chess.Board(fen)
        uci = agent.get_move(fen, 3_000)
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, (fen, uci)
    assert agent.get_move(PROMOTION_ONLY, 3_000) == "a7a8q"


def test_feature_can_be_disabled() -> None:
    off = agent.Searcher(use_killers=False)
    assert off.killers is None
    off.search_fixed_depth(chess.Board(MIDGAME), 3)
    assert off.stats.killers_stored == 0 and off.stats.killer_cutoffs == 0


TESTS = [
    test_killer_table_store_and_replace,
    test_quiet_cutoff_is_stored,
    test_capture_cutoff_is_not_stored,
    test_no_duplicates_during_search,
    test_killer_receives_ordering_bonus,
    test_result_unchanged_with_and_without_killers,
    test_finds_mate_in_two,
    test_get_move_is_legal,
    test_feature_can_be_disabled,
]


def main() -> None:
    failures = 0
    for test in TESTS:
        print(f"{test.__name__} ...", flush=True)
        try:
            test()
        except Exception as error:
            failures += 1
            print(f"  FAIL: {error!r}")
        else:
            print("  ok")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print(f"all {len(TESTS)} tests passed")


if __name__ == "__main__":
    main()
