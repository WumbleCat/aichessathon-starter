"""Tests and benchmark for the PVS agent.

Run from the repository root:

    uv run python my-agents/10_principal_variation_search/test_pvs.py

Works under pytest too. Each `test_*` function checks one item from the spec's
"Tests" list; `benchmark()` prints node counts with PVS on and off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent
from agent import Config, Searcher, search_fixed_depth

# A mix of quiet middlegames, tactics and endgames; deterministic and cheap at depth 3-4.
POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkb1r/pp2pppp/3p1n2/2p5/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq c6 0 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9",
]

# Positions with a single forced answer, so ordering cannot change the move.
FORCED = [
    # back-rank mate in one
    ("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "e1e8"),
    # scholar's mate
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),
    # win the hanging queen
    ("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f3h4"),
]

NO_TT = Config(use_pvs=True, use_tt=False)
AB_NO_TT = Config(use_pvs=False, use_tt=False)
PLAIN_AB = Config(use_pvs=False)


def test_pvs_matches_alpha_beta_score_and_move() -> None:
    """Without the TT both searches are deterministic and must agree exactly.

    Node counts are compared in aggregate: on a badly ordered node (all quiet moves,
    no history yet) a re-search costs a few nodes more than plain alpha-beta, and the
    spec only promises equal or fewer nodes on well-ordered positions.
    """
    pvs_nodes = ab_nodes = 0
    for fen in POSITIONS:
        board = chess.Board(fen)
        for depth in (1, 2, 3):
            pvs_move, pvs_score, pvs = search_fixed_depth(board, depth, NO_TT)
            ab_move, ab_score, ab = search_fixed_depth(board, depth, AB_NO_TT)
            assert pvs_score == ab_score, (fen, depth, pvs_score, ab_score)
            assert pvs_move == ab_move, (fen, depth, pvs_move, ab_move)
            if depth >= 2:
                pvs_nodes += pvs.nodes
                ab_nodes += ab.nodes
    assert pvs_nodes <= ab_nodes, (pvs_nodes, ab_nodes)


def test_pvs_finds_forced_moves_with_everything_on() -> None:
    for fen, expected in FORCED:
        board = chess.Board(fen)
        move, _, _ = search_fixed_depth(board, 4)
        assert move.uci() == expected, (fen, move.uci(), expected)
        ab_move, _, _ = search_fixed_depth(board, 4, PLAIN_AB)
        assert ab_move.uci() == expected, (fen, ab_move.uci(), expected)


def test_research_happens_when_later_move_beats_alpha() -> None:
    """Generation order puts a bad move first, so a later move must fail high and be re-searched."""
    board = chess.Board(FORCED[0][0])
    config = Config(use_tt=False, use_ordering=False)
    move, score, searcher = search_fixed_depth(board, 3, config)
    assert move.uci() == "e1e8"
    assert score >= agent.MATE_BOUND
    assert searcher.null_window_searches > 0
    assert searcher.researches > 0, "expected a re-search when a later move exceeded alpha"


def test_no_research_when_null_window_fails_low() -> None:
    """With the TT move (the mate) searched first, every other root move fails low."""
    for fen, expected in FORCED:
        board = chess.Board(fen)
        first = Searcher()
        first.search_root(board, 2)
        # Same TT, fresh counters: the stored best move is now searched first.
        second = Searcher(tt=first.tt)
        move, score = second.search_root(board, 2)
        assert move.uci() == expected
        assert score >= agent.MATE_BOUND or fen == FORCED[2][0]
        assert second.null_window_searches == len(list(board.legal_moves)) - 1
        assert second.researches == 0, (fen, second.researches)


def test_no_null_window_searches_when_pvs_disabled() -> None:
    board = chess.Board(POSITIONS[1])
    _, _, searcher = search_fixed_depth(board, 3, AB_NO_TT)
    assert searcher.null_window_searches == 0
    assert searcher.researches == 0


def test_board_restored() -> None:
    for fen in POSITIONS:
        board = chess.Board(fen)
        before = board.fen()
        stack = len(board.move_stack)
        search_fixed_depth(board, 3)
        assert board.fen() == before
        assert len(board.move_stack) == stack


def test_tt_move_is_searched_first() -> None:
    board = chess.Board(POSITIONS[1])
    searcher = Searcher()
    searcher.search_root(board, 2)
    entry = searcher.tt.probe(chess.polyglot.zobrist_hash(board))
    assert entry is not None and entry.move is not None
    ordered = searcher.ordered_moves(board, entry.move, 0)
    assert ordered[0] == entry.move


def test_principal_variation_is_legal() -> None:
    board = chess.Board(POSITIONS[4])
    searcher = Searcher()
    searcher.search_root(board, 3)
    assert searcher.pv
    probe = board.copy()
    for move in searcher.pv:
        assert move in probe.legal_moves
        probe.push(move)


def test_get_move_returns_legal_uci() -> None:
    for fen in POSITIONS + [f for f, _ in FORCED]:
        board = chess.Board(fen)
        uci = agent.get_move(fen, 2_000)
        assert chess.Move.from_uci(uci) in board.legal_moves, (fen, uci)


def test_mate_score_normalisation_round_trips() -> None:
    for ply in (0, 1, 7, 40):
        for score in (agent.MATE_SCORE - 3, -(agent.MATE_SCORE - 5), 120, -40):
            assert agent.score_from_tt(agent.score_to_tt(score, ply), ply) == score


def benchmark(depth: int = 4) -> None:
    """Node counts with iterative deepening, PVS on vs off, everything else equal."""
    total_pvs = total_ab = 0
    print(f"{'position':<12} {'ab nodes':>10} {'pvs nodes':>10} {'nw':>8} {'re-search':>9}")
    for index, fen in enumerate(POSITIONS):
        results = {}
        for name, config in (("ab", PLAIN_AB), ("pvs", Config())):
            searcher = Searcher(config=config)
            board = chess.Board(fen)
            for d in range(1, depth + 1):
                move, _ = searcher.search_root(board, d)
            results[name] = (move, searcher)
        ab = results["ab"][1]
        pvs = results["pvs"][1]
        total_ab += ab.nodes
        total_pvs += pvs.nodes
        print(
            f"{index:<12} {ab.nodes:>10} {pvs.nodes:>10} {pvs.null_window_searches:>8} "
            f"{pvs.researches:>9}   ab {results['ab'][0].uci()}  pvs {results['pvs'][0].uci()}"
        )
    print(f"{'total':<12} {total_ab:>10} {total_pvs:>10}   pvs/ab = {total_pvs / total_ab:.2f}")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print()
    benchmark()
