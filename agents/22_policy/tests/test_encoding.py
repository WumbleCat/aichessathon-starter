"""Encoding tests: move index round trip, every legal move has an index, planes agree."""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import chess  # noqa: E402
import numpy as np  # noqa: E402
from pn_encoding import (  # noqa: E402
    NUM_ACTIONS,
    board_to_codes,
    codes_to_planes,
    encode_board,
    index_to_move,
    move_to_index,
)


def random_positions(n: int, seed: int = 1) -> list[chess.Board]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(0, 80)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if b.legal_moves:
            out.append(b)
    return out


def test_move_roundtrip() -> None:
    boards = random_positions(300)
    boards.append(chess.Board("r3k2r/pPpp1ppp/8/8/8/8/1pPPPPPP/R3K2R w KQkq - 0 1"))
    boards.append(chess.Board("r3k2r/pPpp1ppp/8/8/8/8/1pPPPPPP/R3K2R b KQkq - 0 1"))
    seen = set()
    for b in boards:
        flip = b.turn == chess.BLACK
        for m in b.legal_moves:
            idx = move_to_index(m, flip)
            assert 0 <= idx < NUM_ACTIONS
            back = index_to_move(idx, flip, b.piece_type_at(m.from_square) == chess.PAWN)
            assert back == m, (b.fen(), m, back)
            seen.add(idx)
    # promotions of every kind
    b = chess.Board("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    for m in b.legal_moves:
        assert (
            index_to_move(
                move_to_index(m, False), False, b.piece_type_at(m.from_square) == chess.PAWN
            )
            == m
        )
    b = chess.Board("4k3/8/8/8/8/8/p7/1N2K3 b - - 0 1")
    for m in b.legal_moves:
        assert (
            index_to_move(
                move_to_index(m, True), True, b.piece_type_at(m.from_square) == chess.PAWN
            )
            == m
        )
    print("move roundtrip ok, distinct indices seen:", len(seen))


def test_planes_agree() -> None:
    boards = random_positions(200, seed=2)
    codes = []
    metas = []
    for b in boards:
        c, m = board_to_codes(b)
        codes.append(c)
        metas.append(m)
    batch = codes_to_planes(np.stack(codes), np.array(metas, dtype=np.int64))
    for i, b in enumerate(boards):
        single = encode_board(b)
        assert single.shape == (18, 8, 8)
        assert np.array_equal(single, batch[i]), b.fen()
    # sanity: own king plane has exactly one bit on rank 0 for the starting position
    x = encode_board(chess.Board())
    assert x[5].sum() == 1 and x[5][0][4] == 1
    y = encode_board(chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"))
    assert y[5][0][4] == 1  # black king appears on "my" back rank after the flip
    assert y[16].sum() == 1
    print("planes ok")


if __name__ == "__main__":
    test_move_roundtrip()
    test_planes_agree()
