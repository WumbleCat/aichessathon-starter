"""Evaluation properties from the spec plus low-level bitboard helper checks."""

from __future__ import annotations

import os
import random
import sys
import unittest

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hce_eval
from hce_eval import evaluate_board, evaluate_stm


def mirror(board: chess.Board) -> chess.Board:
    """Colour-swapped, vertically flipped board with the turn flipped too."""
    return board.mirror()


class BitHelpers(unittest.TestCase):
    def test_lsb_and_popcount(self) -> None:
        for sq in range(64):
            self.assertEqual(int(hce_eval.lsb(1 << sq)), sq)
        rng = random.Random(1)
        for _ in range(500):
            bb = rng.getrandbits(64)
            self.assertEqual(int(hce_eval.popcount(bb)), bin(bb).count("1"))
            if bb:
                self.assertEqual(int(hce_eval.lsb(bb)), (bb & -bb).bit_length() - 1)

    def test_flip_vertical(self) -> None:
        rng = random.Random(2)
        for _ in range(200):
            bb = rng.getrandbits(64)
            self.assertEqual(int(hce_eval.flip_vertical(bb)), chess.flip_vertical(bb))

    def test_slider_and_leaper_attacks_match_python_chess(self) -> None:
        rng = random.Random(3)
        for _ in range(300):
            occ = rng.getrandbits(64)
            sq = rng.randrange(64)
            self.assertEqual(
                int(hce_eval.bishop_attacks(sq, occ)),
                chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ],
            )
            self.assertEqual(
                int(hce_eval.rook_attacks(sq, occ)),
                chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
                | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ],
            )
        for sq in range(64):
            self.assertEqual(int(hce_eval.KNIGHT_ATT[sq]), chess.BB_KNIGHT_ATTACKS[sq])
            self.assertEqual(int(hce_eval.KING_ATT[sq]), chess.BB_KING_ATTACKS[sq])

    def test_pawn_attacks(self) -> None:
        rng = random.Random(4)
        for _ in range(200):
            pawns = rng.getrandbits(64) & ~(chess.BB_RANK_1 | chess.BB_RANK_8)
            white = 0
            black = 0
            for sq in chess.scan_forward(pawns):
                white |= chess.BB_PAWN_ATTACKS[chess.WHITE][sq]
                black |= chess.BB_PAWN_ATTACKS[chess.BLACK][sq]
            self.assertEqual(int(hce_eval.white_pawn_attacks(pawns)), white)
            self.assertEqual(int(hce_eval.black_pawn_attacks(pawns)), black)


class EvaluationProperties(unittest.TestCase):
    def test_start_position_is_balanced(self) -> None:
        self.assertLessEqual(abs(evaluate_board(chess.Board())), hce_eval.TEMPO_MG)

    def test_material_gain_changes_sign_and_magnitude(self) -> None:
        base = chess.Board("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1")
        up_knight = chess.Board("4k3/pppppppp/8/8/8/8/PPPPPPPP/2N1K3 w - - 0 1")
        up_queen = chess.Board("4k3/pppppppp/8/8/8/8/PPPPPPPP/3QK3 w - - 0 1")
        down_rook = chess.Board("r3k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1")
        self.assertGreater(evaluate_board(up_knight), evaluate_board(base) + 200)
        self.assertGreater(evaluate_board(up_queen), evaluate_board(up_knight) + 400)
        self.assertLess(evaluate_board(down_rook), evaluate_board(base) - 350)
        # side-to-move view flips the sign
        self.assertEqual(evaluate_stm(up_queen), evaluate_board(up_queen))
        flipped = up_queen.copy()
        flipped.turn = chess.BLACK
        self.assertEqual(evaluate_stm(flipped), -evaluate_board(flipped))

    def test_mirrored_positions_are_antisymmetric(self) -> None:
        fens = [
            chess.STARTING_FEN,
            "r3k2r/pp1n1ppp/2pbpq2/3p4/3P1B2/2NBPN2/PPQ2PPP/R3K2R w KQkq - 0 10",
            "8/5k2/3p4/8/8/8/3K1P2/4R3 w - - 0 1",
            "r1bq1rk1/ppp2ppp/2n2n2/3pp3/1bPP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 7",
            "6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1",
            "8/8/4k3/8/2K5/8/8/8 w - - 0 1",
            "2r3k1/1p3ppp/p7/8/8/1P6/P4PPP/3R2K1 w - - 0 1",
        ]
        for fen in fens:
            board = chess.Board(fen)
            self.assertEqual(evaluate_board(board), -evaluate_board(mirror(board)), fen)
            self.assertEqual(evaluate_stm(board), evaluate_stm(mirror(board)), fen)

    def test_random_positions_are_antisymmetric(self) -> None:
        rng = random.Random(5)
        for _ in range(60):
            board = chess.Board()
            for _ in range(rng.randrange(0, 60)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            self.assertEqual(evaluate_board(board), -evaluate_board(mirror(board)), board.fen())

    def test_king_safety_fades_in_endgame(self) -> None:
        # identical material: a king walked out to e3 in front of its pawns costs a lot with
        # queens and rooks on the board and little (or gains) once they are gone
        def exposed(fen: str) -> chess.Board:
            board = chess.Board(fen)
            board.remove_piece_at(chess.G1)
            board.set_piece_at(chess.E3, chess.Piece(chess.KING, chess.WHITE))
            return board

        mg_fen = "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 1"
        eg_fen = "6k1/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/6K1 w - - 0 1"
        mg_gap = evaluate_board(chess.Board(mg_fen)) - evaluate_board(exposed(mg_fen))
        eg_gap = evaluate_board(chess.Board(eg_fen)) - evaluate_board(exposed(eg_fen))
        self.assertGreater(mg_gap, 40)
        self.assertLess(eg_gap, mg_gap // 2)

    def test_passed_pawn_bonus_grows_with_advancement(self) -> None:
        previous = None
        for rank in range(2, 7):
            board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
            board.set_piece_at(chess.square(3, rank - 1), chess.Piece(chess.PAWN, chess.WHITE))
            score = evaluate_board(board)
            if previous is not None:
                self.assertGreater(score, previous, f"rank {rank}")
            previous = score

    def test_passed_pawn_beats_blocked_pawn(self) -> None:
        passed = chess.Board("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1")
        blocked = chess.Board("4k3/3p4/8/3P4/8/8/8/4K3 w - - 0 1")
        self.assertGreater(evaluate_board(passed), evaluate_board(blocked) + 100)

    def test_no_mate_constants(self) -> None:
        # a mated and a stalemated side get an ordinary score, terminal logic lives in search
        mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        self.assertTrue(mated.is_checkmate())
        self.assertLess(abs(evaluate_board(mated)), 2000)
        stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(stalemate.is_stalemate())
        self.assertLess(abs(evaluate_board(stalemate)), 2000)

    def test_bishop_pair_and_open_file(self) -> None:
        pair = chess.Board("4k3/pppppppp/8/8/8/8/PPPPPPPP/2B1KB2 w - - 0 1")
        knight_bishop = chess.Board("4k3/pppppppp/8/8/8/8/PPPPPPPP/2N1KB2 w - - 0 1")
        self.assertGreater(evaluate_board(pair), evaluate_board(knight_bishop))
        open_file = chess.Board("4k3/pppp1ppp/8/8/8/8/PPPP1PPP/4RK2 w - - 0 1")
        closed_file = chess.Board("4k3/pppp1ppp/8/8/8/8/PPPPPPPP/4RK2 w - - 0 1")
        self.assertGreater(
            evaluate_board(open_file) - 0, evaluate_board(closed_file) - hce_eval.MATERIAL_MG[1]
        )

    def test_lone_minor_piece_is_a_draw(self) -> None:
        self.assertEqual(evaluate_board(chess.Board("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1")), 0)
        self.assertEqual(evaluate_board(chess.Board("4k3/8/8/8/8/8/8/1N2K3 b - - 0 1")), 0)

    def test_mop_up_prefers_cornered_king(self) -> None:
        centre = chess.Board("8/8/8/3k4/8/8/8/R3K3 w - - 0 1")
        corner = chess.Board("k7/8/8/8/8/8/8/R3K3 w - - 0 1")
        self.assertGreater(evaluate_board(corner), evaluate_board(centre))

    def test_evaluation_speed(self) -> None:
        # measured relative to python-chess move generation so the test is robust to machine
        # load: the compiled evaluation must be cheaper than listing the legal moves
        import time

        board = chess.Board("r3k2r/pp1n1ppp/2pbpq2/3p4/3P1B2/2NBPN2/PPQ2PPP/R3K2R w KQkq - 0 10")
        evaluate_stm(board)
        n = 3000
        started = time.perf_counter()
        for _ in range(n):
            evaluate_stm(board)
        eval_us = (time.perf_counter() - started) / n * 1e6
        started = time.perf_counter()
        for _ in range(n):
            list(board.legal_moves)
        movegen_us = (time.perf_counter() - started) / n * 1e6
        ratio_limit = 1.0 if hce_eval.USING_NUMBA else 60.0
        self.assertLess(
            eval_us, movegen_us * ratio_limit,
            f"{eval_us:.1f} us per evaluation vs {movegen_us:.1f} us per move generation",
        )


if __name__ == "__main__":
    unittest.main()
