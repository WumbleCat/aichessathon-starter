"""Compare dc_engine's legal move list with python-chess's for a few positions.

    .venv/Scripts/python.exe agents/29_deepchess/tools/movegen_diff.py [fen ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dc_engine as e  # noqa: E402

FENS = sys.argv[1:] or [
    "rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
]

w1 = np.zeros((774, 8), dtype=np.float32)
b1 = np.zeros(8, dtype=np.float32)


def engine_moves(board: chess.Board) -> set[str]:
    pos = e.Position(hidden=8)
    pos.set_board(board, w1, b1)
    end = e.gen_moves(pos.board, pos.state, pos.moves, 0, False, e.KNIGHT_T, e.KING_T,
                      e.BISHOP_RAYS, e.ROOK_RAYS, e.PAWN_ATTACKERS)
    legal = set()
    for i in range(end):
        code = int(pos.moves[i])
        e.make_move(pos.board, pos.state, pos.hash, code, pos.undo, pos.acc, w1, b1, e.ZOBRIST,
                    e.Z_CASTLE, e.Z_EP, e.Z_SIDE, e.CASTLE_MASK, pos.hist)
        ok = not e.left_king_in_check(pos.board, pos.state, e.KNIGHT_T, e.KING_T, e.BISHOP_RAYS,
                                      e.ROOK_RAYS, e.PAWN_ATTACKERS)
        e.unmake_move(pos.board, pos.state, pos.hash, pos.undo)
        if ok:
            legal.add(e.move_to_uci(code))
    return legal


for fen in FENS:
    board = chess.Board(fen)
    ref = {m.uci() for m in board.legal_moves}
    got = engine_moves(board)
    print(f"{fen}: engine {len(got)} python-chess {len(ref)}")
    if got != ref:
        print("  missing:", sorted(ref - got))
        print("  extra:  ", sorted(got - ref))
print("done")
