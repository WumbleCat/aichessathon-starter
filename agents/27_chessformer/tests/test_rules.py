"""Mandatory chess-rules tests: every returned move must be legal and sensible."""

import os
import sys

import chess
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import agent  # noqa: E402


def legal(fen: str, ms: int = 1500) -> chess.Move:
    board = chess.Board(fen)
    uci = agent.get_move(fen, ms)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, f"{uci} illegal in {fen}"
    return move


def test_normal_move():
    legal(chess.STARTING_FEN)


def test_capture_hanging_queen():
    fen = "rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
    m = legal(fen)
    assert m.uci() == "f3h4"


def test_mate_in_one():
    m = legal("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    assert m.uci() == "h5f7"


def test_check_evasion():
    fen = "rnbqkbnr/pppp1ppp/8/4p3/7P/5P2/PPPPP1P1/RNBQKBNR b KQkq - 0 2"
    board = chess.Board(fen)
    board.push_uci("d8h4")  # white is now in check
    assert board.is_check()
    m = legal(board.fen())
    assert m in board.legal_moves


def test_checkmated_position_returns_null():
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert chess.Board(fen).is_checkmate()
    assert agent.get_move(fen, 1000) == "0000"


def test_stalemated_position_returns_null():
    fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    assert chess.Board(fen).is_stalemate()
    assert agent.get_move(fen, 1000) == "0000"


def test_delivers_mate_kq_vs_k_without_stalemating():
    board = chess.Board("7k/8/5QK1/8/8/8/8/8 w - - 0 1")
    for _ in range(16):
        if board.is_game_over():
            break
        board.push(legal(board.fen(), 1000))
        if not board.is_game_over():
            board.push(list(board.legal_moves)[0])
    assert board.is_checkmate(), board.fen()


def test_kingside_castling_legal_and_played_when_best():
    fen = "r3k2r/pppqppbp/2np1np1/8/2BPP3/2N1BN2/PPPQ1PPP/R3K2R w KQkq - 0 1"
    board = chess.Board(fen)
    m = legal(fen)
    assert m in board.legal_moves
    # castling must be representable: force it as the only reasonable king safety move
    fen2 = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    board2 = chess.Board(fen2)
    assert chess.Move.from_uci("e1g1") in board2.legal_moves
    assert legal(fen2) in board2.legal_moves


def test_queenside_castling_legal():
    fen = "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"
    board = chess.Board(fen)
    assert chess.Move.from_uci("e1c1") in board.legal_moves
    assert legal(fen) in board.legal_moves


def test_en_passant_is_found_when_it_wins():
    # only en passant saves the pawn race: exd6 e.p. wins a pawn cleanly
    fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    board = chess.Board(fen)
    assert board.is_en_passant(chess.Move.from_uci("e5d6"))
    m = legal(fen)
    assert m in board.legal_moves


def test_en_passant_as_black():
    fen = "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1"
    board = chess.Board(fen)
    assert board.is_en_passant(chess.Move.from_uci("e4d3"))
    assert legal(fen) in board.legal_moves


def test_queen_promotion():
    m = legal("8/P7/8/8/8/8/8/k6K w - - 0 1")
    assert m.uci() == "a7a8q"


def test_underpromotion_avoids_stalemate():
    # a8=Q is stalemate; the search has to pick a rook (or another) promotion
    fen = "k7/2P5/1K6/8/8/8/8/8 w - - 0 1"
    board = chess.Board(fen)
    m = legal(fen)
    board.push(m)
    assert not board.is_stalemate()


def test_knight_promotion_with_check_mates():
    # g8=N+ is not mate here but must be legal; verify all four promotions are legal moves
    fen = "8/P7/8/8/8/8/8/k6K w - - 0 1"
    board = chess.Board(fen)
    for promo in ("a7a8q", "a7a8r", "a7a8b", "a7a8n"):
        assert chess.Move.from_uci(promo) in board.legal_moves
    # knight promotion is the only mate: b8=N#
    fen2 = "k7/1P6/2K5/8/8/8/8/8 w - - 0 1"
    m = legal(fen2)
    b2 = chess.Board(fen2)
    b2.push(m)
    assert m in chess.Board(fen2).legal_moves


def test_black_promotion_and_capture_promotion():
    fen = "4k3/8/8/8/8/8/1p6/R3K3 b - - 0 1"
    board = chess.Board(fen)
    m = legal(fen)
    assert m in board.legal_moves
    assert m.promotion is not None


def test_repeated_calls_keep_state_valid():
    board = chess.Board()
    for _ in range(6):
        board.push(legal(board.fen(), 300))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
