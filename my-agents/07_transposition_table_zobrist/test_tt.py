"""Tests for the Zobrist hash and transposition table in agent.py.

Run from the repository root:

    uv run python my-agents/07_transposition_table_zobrist/test_tt.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent

FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


def test_identical_position_same_hash() -> None:
    a = chess.Board(FENS[2])
    b = chess.Board(FENS[2])
    check(agent.zobrist_key(a) == agent.zobrist_key(b), "identical position gives the same hash")


def test_push_pop_restores_hash() -> None:
    board = chess.Board(FENS[2])
    key = agent.zobrist_key(board)
    for move in board.legal_moves:
        agent.zobrist_push(board, key, move)
        board.pop()
        check_silent(agent.zobrist_key(board) == key)
    check(True, "move + undo restores the previous hash for every legal move")


def check_silent(condition: bool) -> None:
    if not condition:
        raise AssertionError("silent check failed")


def test_side_to_move_changes_hash() -> None:
    board = chess.Board(FENS[3])
    key_white = agent.zobrist_key(board)
    board.turn = chess.BLACK
    key_black = agent.zobrist_key(board)
    check(key_white != key_black, "different side to move changes the hash")


def test_castling_rights_change_hash() -> None:
    board = chess.Board(FENS[2])
    full = agent.zobrist_key(board)
    board.castling_rights &= ~chess.BB_H1  # drop white kingside
    without_k = agent.zobrist_key(board)
    board.castling_rights = chess.BB_EMPTY
    none = agent.zobrist_key(board)
    check(len({full, without_k, none}) == 3, "castling-right change changes the hash")


def test_en_passant_changes_hash() -> None:
    # After 1. e4 (ep square e3 but no black pawn can capture) vs. a position where a
    # legal en-passant capture exists.
    with_ep = chess.Board("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3")
    no_ep = chess.Board("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3")
    check(with_ep.has_legal_en_passant(), "fixture: en-passant capture is legal")
    check(
        agent.zobrist_key(with_ep) != agent.zobrist_key(no_ep),
        "legal en-passant state changes the hash",
    )
    # A meaningless ep marker (no pawn can capture) must not split the same position.
    marker = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    plain = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    check(
        agent.zobrist_key(marker) == agent.zobrist_key(plain),
        "an en-passant marker with no legal capture does not change the hash",
    )


def test_incremental_matches_full_over_random_games() -> None:
    """Play random games and confirm the incremental key equals a from-scratch key."""
    rng = random.Random(7)
    positions = 0
    castles = promotions = en_passants = 0
    for game in range(60):
        board = chess.Board(FENS[game % len(FENS)])
        key = agent.zobrist_key(board)
        stack: list[int] = []
        for _ in range(120):
            moves = list(board.legal_moves)
            if not moves:
                break
            # Bias towards the special moves the incremental update has to get right.
            special = [
                m
                for m in moves
                if board.is_castling(m) or m.promotion or board.is_en_passant(m)
            ]
            move = rng.choice(special) if special and rng.random() < 0.8 else rng.choice(moves)
            castles += board.is_castling(move)
            promotions += bool(move.promotion)
            en_passants += board.is_en_passant(move)
            stack.append(key)
            key = agent.zobrist_push(board, key, move)
            check_silent(key == agent.zobrist_key(board))
            positions += 1
        # Unwind and confirm the stack of keys matches on the way back.
        while stack:
            board.pop()
            key = stack.pop()
            check_silent(key == agent.zobrist_key(board))
    check(
        positions > 1000 and castles > 20 and promotions > 20 and en_passants > 5,
        f"incremental key == full key over {positions} positions "
        f"({castles} castles, {promotions} promotions, {en_passants} en passants)",
    )


def test_mate_score_normalisation() -> None:
    for ply in (0, 1, 7, 30):
        for score in (agent.MATE_SCORE - 3, -(agent.MATE_SCORE - 5), 120.0, -40.0):
            check_silent(agent.score_from_tt(agent.score_to_tt(score, ply), ply) == score)
    # Mate in 3 found at ply 2 (score MATE-5) stored, then probed at ply 4: it is still
    # "mate in 3 from here" so the root-relative score becomes MATE-7.
    stored = agent.score_to_tt(agent.MATE_SCORE - 5, 2)
    check(
        agent.score_from_tt(stored, 4) == agent.MATE_SCORE - 7,
        "mate scores are normalised to the node on store and to the root on probe",
    )


def fixed_depth(fen: str, depth: int, use_tt: bool, table: agent.TranspositionTable):
    board = chess.Board(fen)
    searcher = agent.Searcher(table, deadline=time.monotonic() + 600, use_tt=use_tt)
    move, score = searcher.search_root(board, depth)
    return move, score, searcher.nodes


def test_tt_reduces_nodes_on_repeated_search() -> None:
    table = agent.TranspositionTable(bits=16)
    table.new_search()
    _, score1, nodes1 = fixed_depth(FENS[1], 4, True, table)
    table.new_search()
    _, score2, nodes2 = fixed_depth(FENS[1], 4, True, table)
    check(
        nodes2 < nodes1 // 4 and score1 == score2,
        f"repeating a depth-4 search hits the table: {nodes1} -> {nodes2} nodes, same score",
    )


def test_cached_result_matches_uncached() -> None:
    for fen in FENS[:5]:
        table = agent.TranspositionTable(bits=16)
        table.new_search()
        move_tt, score_tt, nodes_tt = fixed_depth(fen, 4, True, table)
        move_plain, score_plain, nodes_plain = fixed_depth(fen, 4, False, table)
        check(
            score_tt == score_plain and move_tt == move_plain,
            f"{fen[:24]:24s} depth 4: with TT {move_tt.uci()} {score_tt:.0f} ({nodes_tt} n) "
            f"== without {move_plain.uci()} {score_plain:.0f} ({nodes_plain} n)",
        )


def test_tt_move_orders_first() -> None:
    board = chess.Board(FENS[2])
    quiet = next(m for m in board.legal_moves if not board.is_capture(m) and not m.promotion)
    ordered = agent.ordered_moves(board, quiet)
    check(ordered[0] == quiet, "the TT move is ordered first")
    check(
        sorted(m.uci() for m in ordered) == sorted(m.uci() for m in board.legal_moves),
        "ordering keeps every legal move exactly once",
    )


def test_bounds_are_used() -> None:
    """A stored LOWERBOUND raises alpha; UPPERBOUND lowers beta; EXACT returns."""
    table = agent.TranspositionTable(bits=8)
    table.new_search()
    board = chess.Board(FENS[1])
    key = agent.zobrist_key(board)
    searcher = agent.Searcher(table, deadline=time.monotonic() + 60)
    table.store(key, 9, 12345.0, agent.EXACT, None)
    check(searcher.negamax(board, key, 3, 1, -agent.INFINITY, agent.INFINITY) == 12345.0,
          "an EXACT entry of sufficient depth is returned without searching")
    table.store(key, 9, 12345.0, agent.LOWERBOUND, None)
    check(searcher.negamax(board, key, 3, 1, -agent.INFINITY, 100.0) == 12345.0,
          "a LOWERBOUND above beta cuts off")
    table.store(key, 9, -12345.0, agent.UPPERBOUND, None)
    check(searcher.negamax(board, key, 3, 1, -100.0, agent.INFINITY) == -12345.0,
          "an UPPERBOUND below alpha cuts off")
    table.store(key, 2, 12345.0, agent.EXACT, None)
    nodes_before = searcher.nodes
    searcher.negamax(board, key, 3, 1, -agent.INFINITY, agent.INFINITY)
    check(searcher.nodes > nodes_before + 10, "an entry that is too shallow is not trusted")


def test_finds_mates_and_returns_legal_moves() -> None:
    # Mate in 1 and mate in 2 positions.
    cases = [
        ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", {"a1a8"}),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", {"h5f7"}),
        # Mate in 2: Ra6! bxa6 b7#, and Legal-style Nf6+ gxf6 Bxf7#. Both verified by
        # brute force. Mate in 2 is three plies, so a fixed depth-3 search must find it
        # (fixed depth keeps this independent of how loaded the machine is).
        ("kbK5/pp6/1P6/8/8/8/8/R7 w - - 0 1", {"a1a6"}),
        ("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 0", {"d5f6"}),
    ]
    for fen, expected in cases:
        board = chess.Board(fen)
        table = agent.TranspositionTable(bits=16)
        table.new_search()
        move, score, _ = fixed_depth(fen, 3, True, table)
        check(move.uci() in expected, f"depth 3 finds the mating move {move.uci()}")
        check(score >= agent.MATE_BOUND, f"and scores it as mate ({score:.0f})")
        uci = agent.get_move(fen, 30_000)
        check(chess.Move.from_uci(uci) in board.legal_moves, f"{uci} is legal in {fen[:20]}")
    for fen in FENS:
        board = chess.Board(fen)
        uci = agent.get_move(fen, 1_500)
        check_silent(chess.Move.from_uci(uci) in board.legal_moves)
    check(True, "get_move returns a legal move on every fixture")


def main() -> None:
    tests = [
        test_identical_position_same_hash,
        test_push_pop_restores_hash,
        test_side_to_move_changes_hash,
        test_castling_rights_change_hash,
        test_en_passant_changes_hash,
        test_incremental_matches_full_over_random_games,
        test_mate_score_normalisation,
        test_tt_move_orders_first,
        test_bounds_are_used,
        test_tt_reduces_nodes_on_repeated_search,
        test_cached_result_matches_uncached,
        test_finds_mates_and_returns_legal_moves,
    ]
    for test in tests:
        print(test.__name__)
        test()
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    main()
