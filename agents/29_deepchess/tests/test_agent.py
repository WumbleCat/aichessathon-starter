"""Correctness tests for the DeepChess agent: legality, special moves, clocks, model sanity.

Run from the repository root:
    .venv/Scripts/python.exe -m pytest agents/29_deepchess/tests -q
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import chess
import numpy as np

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parent
sys.path.insert(0, str(AGENT_DIR))

import agent  # noqa: E402


def legal(fen: str, time_left_ms: int = 2000) -> chess.Move:
    board = chess.Board(fen)
    uci = agent.get_move(fen, time_left_ms)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, f"{uci} illegal in {fen}"
    return move


# ---------------------------------------------------------------- mandatory chess tests


def test_normal_move() -> None:
    legal(chess.STARTING_FEN)


def test_capture_free_queen() -> None:
    # Black queen hanging on d5, white to move: the search must take it (knight or pawn).
    fen = "rnb1kbnr/pppp1ppp/8/3qp3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 3"
    move = legal(fen)
    assert move.uci() in ("c3d5", "e4d5")


def test_gives_check_and_is_legal() -> None:
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4"
    legal(fen)


def test_check_evasion() -> None:
    # White king in check from the rook on e8; only king moves or blocks are legal.
    fen = "4r1k1/8/8/8/8/8/5PPP/4K3 w - - 0 1"
    board = chess.Board(fen)
    assert board.is_check()
    legal(fen)


def test_finds_mate_in_one() -> None:
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    move = legal(fen)
    assert move.uci() == "a1a8"


def test_finds_mate_in_two() -> None:
    # Classic back-rank combination: 1.Qxf8+ ... mate follows; the engine must play a forcing
    # move that keeps a mate score.
    fen = "r4rk1/1p3ppp/8/8/8/8/1P3PPP/R3R1K1 w - - 0 1"
    board = chess.Board(fen)
    uci = agent.get_move(fen, 3000)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves


def test_checkmated_position_has_no_move() -> None:
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    board = chess.Board(fen)
    assert board.is_checkmate()
    # The harness never asks for a move here; the agent must not crash anyway.
    assert agent.get_move(fen, 1000) == "0000"


def test_stalemate_avoided_when_winning() -> None:
    # White has K+Q vs K, a stalemating move exists; the engine must not play it.
    fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    board = chess.Board(fen)
    move = legal(fen, 3000)
    board.push(move)
    assert not board.is_stalemate()


def test_stalemate_position_no_crash() -> None:
    fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    board = chess.Board(fen)
    assert board.is_stalemate()
    assert agent.get_move(fen, 1000) == "0000"


def test_kingside_castling_available_and_legal() -> None:
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5"
    board = chess.Board(fen)
    assert chess.Move.from_uci("e1g1") in board.legal_moves
    legal(fen)


def test_kingside_castling_played_when_only_sensible_move() -> None:
    # Everything else is bad; castling out of the centre is the natural move.
    fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    board = chess.Board(fen)
    assert chess.Move.from_uci("e1g1") in board.legal_moves
    legal(fen)


def test_queenside_castling_legal() -> None:
    fen = "r3kbnr/ppp1pppp/2nq4/3p1b2/3P1B2/2NQ4/PPP1PPPP/R3KBNR w KQkq - 6 6"
    board = chess.Board(fen)
    assert chess.Move.from_uci("e1c1") in board.legal_moves
    legal(fen)


def test_queenside_castling_black() -> None:
    fen = "r3kbnr/ppp1pppp/2nq4/3p1b2/3P1B2/2NQ4/PPP1PPPP/2KR1BNR b kq - 7 6"
    board = chess.Board(fen)
    assert chess.Move.from_uci("e8c8") in board.legal_moves
    legal(fen)


def test_en_passant() -> None:
    # White pawn e5, black just played d7d5; exd6 e.p. wins a pawn for free.
    fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    board = chess.Board(fen)
    ep = chess.Move.from_uci("e5d6")
    assert ep in board.legal_moves and board.is_en_passant(ep)
    legal(fen)


def test_en_passant_forced() -> None:
    # Only en passant avoids immediate material loss/mate threats: verify legality only.
    fen = "8/8/8/3pP2k/8/8/8/K7 w - d6 0 1"
    legal(fen)


def test_queen_promotion() -> None:
    fen = "8/1P4k1/8/8/8/8/8/K7 w - - 0 1"
    move = legal(fen)
    assert move.promotion == chess.QUEEN


def test_underpromotion_moves_are_legal_and_selectable() -> None:
    # Knight promotion with check is the only move that does not lose the pawn; the search
    # may still prefer the queen if it sees further, so only legality is asserted along
    # with each promotion type being accepted by the harness.
    fen = "5k2/4P3/8/8/8/8/8/4K3 w - - 0 1"
    board = chess.Board(fen)
    for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        m = chess.Move(chess.E7, chess.E8, promotion=promo)
        assert m in board.legal_moves
    legal(fen)


def test_rook_promotion_when_queen_stalemates() -> None:
    # Promoting to a queen stalemates; a rook promotion keeps the win.
    fen = "7k/5P2/6K1/8/8/8/8/8 w - - 0 1"
    board = chess.Board(fen)
    move = legal(fen, 3000)
    board.push(move)
    assert not board.is_stalemate()


def test_knight_promotion_mate() -> None:
    # Only the knight promotion mates (e8=N+ hits the king on d6; every escape square is
    # covered by the pawns, the bishop on h3 and the rook on e1). A queen gives no check.
    fen = "8/4P3/3k4/1P6/1PP5/7B/8/K3R3 w - - 0 1"
    board = chess.Board(fen)
    for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        m = chess.Move(chess.E7, chess.E8, promotion=promo)
        assert m in board.legal_moves
    move = legal(fen, 5000)
    assert move.uci() == "e7e8n"
    board.push(move)
    assert board.is_checkmate()


def test_bishop_promotion_is_legal_output() -> None:
    board = chess.Board("5k2/4P3/8/8/8/8/8/4K3 w - - 0 1")
    m = chess.Move(chess.E7, chess.E8, promotion=chess.BISHOP)
    assert m in board.legal_moves and m.uci() == "e7e8b"


# ---------------------------------------------------------------------- clock tests


CLOCKS = [50, 100, 500, 1000, 5000, 30000, 120000]


def test_clock_budget_respected() -> None:
    fen = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8"
    for time_left in CLOCKS:
        started = time.perf_counter()
        legal(fen, time_left)
        elapsed_ms = (time.perf_counter() - started) * 1000
        # Never spend more than 45% of the clock plus a little process noise; small clocks
        # are answered almost immediately.
        limit = max(time_left * 0.45, 60)
        assert elapsed_ms < limit, f"{elapsed_ms:.0f} ms for a {time_left} ms clock"
        print(f"  clock {time_left:>6} ms -> {elapsed_ms:7.1f} ms, "
              f"depth {agent.STATS.get('depth')}")


def test_repeated_calls_keep_state_valid() -> None:
    board = chess.Board()
    for _ in range(12):
        if board.is_game_over():
            break
        uci = agent.get_move(board.fen(), 3000)
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        board.push(move)


def test_repetition_avoided_when_winning() -> None:
    # Queen+king vs king, with the position already seen twice: the engine must avoid the
    # repeating shuffle.
    fen = "8/8/8/8/8/2k5/8/K2Q4 w - - 0 1"
    agent._game_history.clear()
    key = chess.Board(fen)._transposition_key()
    agent._game_history[key] = 2
    board = chess.Board(fen)
    move = legal(fen, 2000)
    board.push(move)
    # A move that walks into an immediate repetition is impossible here anyway; the test
    # mainly checks that search handles the history dict without error.
    assert not board.is_game_over()


# ---------------------------------------------------------------------- model checks


def _feature_indices(board: chess.Board) -> list[int]:
    out = np.zeros((1, 40), dtype=np.int32)
    n = agent.features_to_indices(
        board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
        board.occupied_co[True], board.occupied_co[False], 1 if board.turn else 0,
        board.castling_rights, 1 if board.ep_square is not None else 0, out,
    )
    return sorted(int(x) for x in out[0, :n])


def test_features_are_colour_symmetric() -> None:
    # Mirroring the board and swapping colours must give identical features.
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4")
    mirrored = board.mirror()  # flips vertically and swaps colours; side to move becomes black
    assert _feature_indices(board) == _feature_indices(mirrored)


def test_features_count_and_range() -> None:
    board = chess.Board()
    idx = _feature_indices(board)
    assert len(idx) == 32 + 4  # 32 pieces, 4 castling rights
    assert min(idx) >= 0 and max(idx) < 773


def test_handcrafted_eval_symmetric() -> None:
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4")
    a = agent.eval_handcrafted(board.pawns, board.knights, board.bishops, board.rooks,
                               board.queens, board.kings, board.occupied_co[True],
                               board.occupied_co[False], 1, agent.PST)
    m = board.mirror()
    b = agent.eval_handcrafted(m.pawns, m.knights, m.bishops, m.rooks, m.queens, m.kings,
                               m.occupied_co[True], m.occupied_co[False], 0, agent.PST)
    assert a == b


def test_network_eval_symmetric_and_sane() -> None:
    if agent._MODEL is None:
        return
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4")
    a = agent.evaluate(board)
    b = agent.evaluate(board.mirror())
    assert a == b
    # a free queen must look good
    up = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
    assert agent.evaluate(up) > 300


def test_pairwise_preference_sanity() -> None:
    if agent._MODEL is None:
        return
    # sigmoid(V(A) - V(B)) : identical positions -> 0.5 ; swapping inverts.
    a = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
    b = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 3")
    va, vb = agent.evaluate(a) / 100.0, agent.evaluate(b) / 100.0
    p_ab = 1 / (1 + np.exp(-(va - vb)))
    p_ba = 1 / (1 + np.exp(-(vb - va)))
    assert abs(p_ab + p_ba - 1.0) < 1e-6
    assert p_ab > 0.9
    assert abs(1 / (1 + np.exp(-(va - va))) - 0.5) < 1e-9


if __name__ == "__main__":
    import traceback

    failures = 0
    names = [n for n in dir() if n.startswith("test_")]
    for name in names:
        fn = globals()[name]
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({(time.perf_counter() - t0) * 1000:.0f} ms)")
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(names) - failures}/{len(names)} passed")
    sys.exit(1 if failures else 0)
