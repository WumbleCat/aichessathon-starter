"""Tests for the alpha-beta agent. Run with

    uv run python my-agents/02_alpha_beta_pruning/test_agent.py

or under pytest. Follows the Tests section of 02_alpha_beta_pruning.md, plus the negamax
tests from 01 that it builds on.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent

START = chess.STARTING_FEN
# Scholar's mate is on: Qh5xf7#.
MATE_IN_ONE = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4"
# Black queen on d5 attacked by the c4 bishop, nothing else going on.
HANGING_QUEEN = "rnb1kbnr/pppp1ppp/8/3qp3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 3"
STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
CHECKMATED = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"  # black to move, mated
# A middlegame with lots of pieces: many legal moves, lots to prune.
MIDDLEGAME = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8"
# Mate in two for white: 1.Kg6 Kg8 2.Ra8#. There is no mate in one.
MATE_IN_TWO = "7k/5K2/8/8/8/8/R6P/8 w - - 0 1"
# Small endgames where even plain negamax at depth 4 is quick.
ENDGAMES = [
    "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
    "8/5k2/8/8/8/8/1R6/K7 w - - 0 1",
    "8/8/3k4/8/8/3K4/3P4/8 b - - 0 1",
    "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
]


Comparison = tuple[chess.Move | None, float, int, int, int]
_compared: dict[tuple[str, int], Comparison] = {}


def _compare(fen: str, depth: int) -> Comparison:
    """Return (move, score, plain nodes, alpha-beta nodes, cutoffs) at one depth.

    Plain negamax is slow, so each (fen, depth) pair is only searched once per run.
    """
    key = (fen, depth)
    if key not in _compared:
        _compared[key] = _compare_uncached(fen, depth)
    return _compared[key]


def _compare_uncached(fen: str, depth: int) -> Comparison:
    board = chess.Board(fen)
    plain = agent.Searcher()
    plain_move, plain_score = plain.search_root(board, depth, pruning=False)
    plain_nodes = plain.stats.nodes_searched
    assert board.fen() == fen and not board.move_stack

    pruned = agent.Searcher()
    ab_move, ab_score = pruned.search_root(board, depth, pruning=True)
    assert board.fen() == fen and not board.move_stack

    assert ab_score == plain_score, f"{fen} depth {depth}: {ab_score} != {plain_score}"
    assert ab_move == plain_move, f"{fen} depth {depth}: {ab_move} != {plain_move}"
    assert pruned.stats.nodes_searched <= plain_nodes
    return ab_move, ab_score, plain_nodes, pruned.stats.nodes_searched, pruned.stats.beta_cutoffs


def test_alpha_beta_matches_plain_negamax() -> None:
    """Same move and score as plain Negamax at depths 1-4."""
    for fen in [START, MIDDLEGAME, HANGING_QUEEN, MATE_IN_ONE, MATE_IN_TWO]:
        for depth in (1, 2, 3):
            _compare(fen, depth)
    for fen in ENDGAMES:
        for depth in (1, 2, 3, 4):
            _compare(fen, depth)


def test_pruning_searches_fewer_nodes() -> None:
    """Cutoffs happen and node counts drop substantially on busy positions."""
    for fen in [START, MIDDLEGAME, HANGING_QUEEN]:
        _, _, plain_nodes, ab_nodes, cutoffs = _compare(fen, 3)
        assert cutoffs > 0, f"no beta cutoffs at depth 3 on {fen}"
        assert ab_nodes * 2 < plain_nodes, f"{fen}: {ab_nodes} vs {plain_nodes} plain"
        print(f"  depth 3 {fen.split()[0][:20]:<20} plain {plain_nodes:>8} "
              f"alpha-beta {ab_nodes:>7} cutoffs {cutoffs:>6}")
    # At depth 1 there is nothing below the root to prune, so nothing is pruned.
    _, _, plain_nodes, ab_nodes, cutoffs = _compare(START, 1)
    assert ab_nodes == plain_nodes and cutoffs == 0


def test_counters_reset_between_root_searches() -> None:
    searcher = agent.Searcher()
    searcher.search_root(chess.Board(MIDDLEGAME), 3)
    assert searcher.stats.nodes_searched > 0 and searcher.stats.beta_cutoffs > 0
    searcher.search_root(chess.Board(START), 1)
    assert searcher.stats.nodes_searched == 20  # the 20 replies; the root is not counted
    assert searcher.stats.beta_cutoffs == 0


def test_board_unchanged_after_cutoffs() -> None:
    """A cutoff breaks out of the loop after pop(): the board must be pristine."""
    for fen in [MIDDLEGAME, MATE_IN_TWO]:
        board = chess.Board(fen)
        before = (board.fen(), list(board.move_stack), board.castling_rights, board.ep_square)
        searcher = agent.Searcher()
        searcher.negamax(board, 3, -agent.INFINITY, agent.INFINITY)
        assert searcher.stats.beta_cutoffs > 0
        after = (board.fen(), list(board.move_stack), board.castling_rights, board.ep_square)
        assert before == after
        # Narrow windows cause cutoffs at the very first move too.
        searcher.negamax(board, 3, -50, 50)
        assert board.fen() == fen and not board.move_stack


def test_finds_mate_in_one() -> None:
    board = chess.Board(MATE_IN_ONE)
    move, score = agent.Searcher().search_root(board, 2)
    assert move == chess.Move.from_uci("h5f7")
    assert score == agent.MATE_SCORE - 1
    assert agent.get_move(MATE_IN_ONE, 60_000) == "h5f7"


def test_finds_mate_in_two_and_prefers_faster_mate() -> None:
    board = chess.Board(MATE_IN_TWO)
    move, score = agent.Searcher().search_root(board, 3)
    assert move == chess.Move.from_uci("f7g6"), f"played {move}"
    # The engine reports mate: MATE_SCORE minus the ply at which mate lands.
    assert score == agent.MATE_SCORE - 3
    # Ask for depth 5: the score must not get "better" by delaying the mate.
    _, deeper = agent.Searcher().search_root(board, 5)
    assert deeper == agent.MATE_SCORE - 3


def test_checkmate_scores_propagate() -> None:
    # Side to move is mated: large negative from its own point of view.
    _, score = agent.Searcher().search_root(chess.Board(CHECKMATED), 3)
    assert score == -agent.MATE_SCORE
    # One ply earlier (white to move, can mate in one) it is large positive.
    board = chess.Board(CHECKMATED)
    board.turn = chess.WHITE
    board.set_piece_at(chess.G7, None)
    board.set_piece_at(chess.F7, chess.Piece(chess.QUEEN, chess.WHITE))
    _, score = agent.Searcher().search_root(board, 3)
    assert score == agent.MATE_SCORE - 1
    # Deep inside a tree the value still reaches the root through negations.
    searcher = agent.Searcher()
    value = searcher.negamax(chess.Board(MATE_IN_TWO), 4, -agent.INFINITY, agent.INFINITY)
    assert value == agent.MATE_SCORE - 3


def test_stalemate_is_zero() -> None:
    move, score = agent.Searcher().search_root(chess.Board(STALEMATE), 3)
    assert move is None and score == 0.0
    # And a search that runs into stalemate one ply down sees 0 for that line.
    board = chess.Board("7k/8/6K1/8/8/8/8/5Q2 w - - 0 1")
    searcher = agent.Searcher()
    board.push(chess.Move.from_uci("f1f7"))  # stalemates black
    assert searcher.negamax(board, 2, -agent.INFINITY, agent.INFINITY, 1) == 0.0
    board.pop()
    # The engine, being a queen up, avoids the stalemating move.
    move, score = searcher.search_root(board, 3)
    assert move != chess.Move.from_uci("f1f7") and score > 0


def test_captures_hanging_queen() -> None:
    # Both the c4 bishop and the e4 pawn can take the queen.
    captures = {chess.Move.from_uci("c4d5"), chess.Move.from_uci("e4d5")}
    for depth in (1, 2, 3):
        move, _ = agent.Searcher().search_root(chess.Board(HANGING_QUEEN), depth)
        assert move in captures, f"depth {depth} played {move}"
    assert chess.Move.from_uci(agent.get_move(HANGING_QUEEN, 60_000)) in captures


def test_get_move_is_legal_and_deterministic() -> None:
    # A full clock gives a 4 s budget for a depth-3 search that takes well under 1 s,
    # so the deadline never fires and the answer depends only on the position.
    for fen in [START, MIDDLEGAME, *ENDGAMES, MATE_IN_ONE, HANGING_QUEEN]:
        board = chess.Board(fen)
        agent.new_game()
        first = agent.get_move(fen, 120_000)
        assert chess.Move.from_uci(first) in board.legal_moves
        agent.new_game()
        assert agent.get_move(fen, 120_000) == first


def test_avoids_repeating_a_position_when_ahead() -> None:
    fen = "8/8/8/8/8/4k3/8/R3K3 w - - 4 10"  # rook up, no captures in the air
    board = chess.Board(fen)
    agent.new_game()
    agent.remember(board)
    first, first_score = agent.Searcher().search_root(board, 2)
    assert first is not None and first_score > 0
    # Pretend the game already went through the position that move leads to.
    board.push(first)
    agent.remember(board)
    assert agent.terminal_score(board, list(board.legal_moves), 1) == 0.0
    board.pop()
    second, second_score = agent.Searcher().search_root(board, 2)
    assert second != first and second_score > 0
    assert board.fen() == fen
    # A new game forgets everything and the original choice is back.
    agent.new_game()
    assert agent.Searcher().search_root(board, 2)[0] == first
    # Game history survives get_move calls and is keyed on the position, not the FEN
    # counters, so a repeat reached by a different route is still recognised.
    agent.remember(chess.Board(fen))
    board.push(first)
    agent.remember(board)
    board.pop()
    assert agent.get_move(fen, 120_000) != first.uci()
    agent.new_game()


def test_low_clock_still_answers_quickly() -> None:
    for time_left in (5, 200, 1_000):
        started = time.monotonic()
        uci = agent.get_move(MIDDLEGAME, time_left)
        elapsed_ms = (time.monotonic() - started) * 1000
        assert chess.Move.from_uci(uci) in chess.Board(MIDDLEGAME).legal_moves
        assert elapsed_ms < 1_500, f"{elapsed_ms:.0f} ms with {time_left} ms left"


def test_deadline_abort_restores_the_board() -> None:
    board = chess.Board(MIDDLEGAME)
    searcher = agent.Searcher(deadline=time.monotonic() - 1.0)  # already expired
    move = agent.search_root_guarded(searcher, board, 4)
    assert move is None, "an expired deadline must abort the search"
    assert 0 < searcher.stats.nodes_searched <= 256
    # The abort fired deep in the tree; every pushed move must have been popped.
    assert board.fen() == MIDDLEGAME and not board.move_stack
    # And get_move still answers with a legal move when the target depth is too slow.
    uci = agent.get_move(MIDDLEGAME, 120_000)
    assert chess.Move.from_uci(uci) in board.legal_moves


def main() -> None:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        started = time.monotonic()
        fn()
        print(f"ok  {name} ({time.monotonic() - started:.1f}s)")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
