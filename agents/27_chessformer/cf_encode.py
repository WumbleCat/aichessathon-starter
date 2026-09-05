"""Board -> 64 square tokens, and move <-> flat policy index, in side-to-move perspective.

Orientation: the side to move is always "us" moving up the board. When Black is to move the
board is mirrored vertically (square ^ 56) and colours are swapped, so the network sees one
consistent geometry. Moves are mapped through the same mirror in both directions.

Policy index space (4192 entries, unique per move):
  0 .. 4095   from * 64 + to           (all non-promotion moves, in mirrored coordinates)
  4096 .. 4191 promotions: (from_file * 3 + (to_file - from_file + 1)) * 4 + piece
               piece: 0 = knight, 1 = bishop, 2 = rook, 3 = queen; from rank 7 to rank 8.
"""

import chess
import numpy as np

NUM_FEATURES = 19  # 13 piece classes + ep-target + 4 castling rights + in-check
NUM_MOVES = 4096 + 96
PROMO_BASE = 4096
PROMO_PIECE_INDEX = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3}
PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

_scan = chess.scan_forward


def _flip(square: int, mirror: bool) -> int:
    return square ^ 56 if mirror else square


def encode(board: chess.Board, out: np.ndarray | None = None) -> np.ndarray:
    """Return float32 [64, NUM_FEATURES] token features from the side to move's perspective."""
    if out is None:
        out = np.zeros((64, NUM_FEATURES), dtype=np.float32)
    else:
        out.fill(0.0)
    us = board.turn
    mirror = not us
    occ_us = board.occupied_co[us]
    occ_them = board.occupied_co[not us]
    for pt, bb in (
        (chess.PAWN, board.pawns),
        (chess.KNIGHT, board.knights),
        (chess.BISHOP, board.bishops),
        (chess.ROOK, board.rooks),
        (chess.QUEEN, board.queens),
        (chess.KING, board.kings),
    ):
        for sq in _scan(bb & occ_us):
            out[_flip(sq, mirror), pt] = 1.0
        for sq in _scan(bb & occ_them):
            out[_flip(sq, mirror), pt + 6] = 1.0
    # empty squares
    empty = ~board.occupied & chess.BB_ALL
    for sq in _scan(empty):
        out[_flip(sq, mirror), 0] = 1.0
    if board.ep_square is not None:
        out[_flip(board.ep_square, mirror), 13] = 1.0
    if board.has_kingside_castling_rights(us):
        out[:, 14] = 1.0
    if board.has_queenside_castling_rights(us):
        out[:, 15] = 1.0
    if board.has_kingside_castling_rights(not us):
        out[:, 16] = 1.0
    if board.has_queenside_castling_rights(not us):
        out[:, 17] = 1.0
    if board.is_check():
        out[:, 18] = 1.0
    return out


COMPACT_SIZE = 67  # 64 piece classes + castling bits + ep square (255 = none) + in-check


def compact(board: chess.Board, out: np.ndarray | None = None) -> np.ndarray:
    """Compact uint8 [COMPACT_SIZE] form of the same information encode() produces."""
    if out is None:
        out = np.zeros(COMPACT_SIZE, dtype=np.uint8)
    else:
        out.fill(0)
    us = board.turn
    mirror = not us
    occ_us = board.occupied_co[us]
    occ_them = board.occupied_co[not us]
    for pt, bb in (
        (chess.PAWN, board.pawns),
        (chess.KNIGHT, board.knights),
        (chess.BISHOP, board.bishops),
        (chess.ROOK, board.rooks),
        (chess.QUEEN, board.queens),
        (chess.KING, board.kings),
    ):
        for sq in _scan(bb & occ_us):
            out[_flip(sq, mirror)] = pt
        for sq in _scan(bb & occ_them):
            out[_flip(sq, mirror)] = pt + 6
    bits = 0
    if board.has_kingside_castling_rights(us):
        bits |= 1
    if board.has_queenside_castling_rights(us):
        bits |= 2
    if board.has_kingside_castling_rights(not us):
        bits |= 4
    if board.has_queenside_castling_rights(not us):
        bits |= 8
    out[64] = bits
    out[65] = 255 if board.ep_square is None else _flip(board.ep_square, mirror)
    out[66] = 1 if board.is_check() else 0
    return out


def expand(batch: np.ndarray) -> np.ndarray:
    """uint8 [B, COMPACT_SIZE] -> float32 [B, 64, NUM_FEATURES] (vectorised, for training)."""
    b = batch.shape[0]
    out = np.zeros((b, 64, NUM_FEATURES), dtype=np.float32)
    pieces = batch[:, :64].astype(np.int64)
    np.put_along_axis(out, pieces[:, :, None], 1.0, axis=2)
    ep = batch[:, 65].astype(np.int64)
    has_ep = ep != 255
    rows = np.nonzero(has_ep)[0]
    out[rows, ep[rows], 13] = 1.0
    bits = batch[:, 64].astype(np.int64)
    for i in range(4):
        out[:, :, 14 + i] = ((bits >> i) & 1)[:, None].astype(np.float32)
    out[:, :, 18] = batch[:, 66][:, None].astype(np.float32)
    return out


def move_index(move: chess.Move, mirror: bool) -> int:
    """Flat policy index of a move for a position where mirror == (black to move)."""
    f = _flip(move.from_square, mirror)
    t = _flip(move.to_square, mirror)
    if move.promotion is None:
        return f * 64 + t
    ff = f & 7
    tf = t & 7
    return PROMO_BASE + (ff * 3 + (tf - ff + 1)) * 4 + PROMO_PIECE_INDEX[move.promotion]


def index_move(index: int, mirror: bool) -> chess.Move:
    """Inverse of move_index."""
    if index < PROMO_BASE:
        f, t = divmod(index, 64)
        return chess.Move(_flip(f, mirror), _flip(t, mirror))
    rest, piece = divmod(index - PROMO_BASE, 4)
    ff, d = divmod(rest, 3)
    tf = ff + d - 1
    f = 6 * 8 + ff
    t = 7 * 8 + tf
    return chess.Move(_flip(f, mirror), _flip(t, mirror), promotion=PROMO_PIECES[piece])


def legal_move_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    moves = list(board.legal_moves)
    mirror = not board.turn
    idx = np.fromiter((move_index(m, mirror) for m in moves), dtype=np.int64, count=len(moves))
    return moves, idx


def promo_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For the 96 promotion indices: (base index from*64+to, destination square, piece)."""
    base = np.zeros(96, dtype=np.int64)
    dst = np.zeros(96, dtype=np.int64)
    piece = np.zeros(96, dtype=np.int64)
    for j in range(96):
        rest, p = divmod(j, 4)
        ff, d = divmod(rest, 3)
        tf = ff + d - 1
        f = 48 + ff
        t = 56 + tf
        if 0 <= tf <= 7:
            base[j] = f * 64 + t
            dst[j] = t
        else:  # impossible (off-board) promotions: point at something harmless
            base[j] = f * 64 + f
            dst[j] = f
        piece[j] = p
    return base, dst, piece


def mirror_tables() -> tuple[np.ndarray, np.ndarray]:
    """Left-right (file a <-> h) mirror: square permutation [64], move permutation [NUM_MOVES].

    Chess is symmetric under this mirror except for castling, so training may use it as data
    augmentation on positions without castling rights.
    """
    squares = np.arange(64, dtype=np.int64) ^ 7
    moves = np.zeros(NUM_MOVES, dtype=np.int64)
    for f in range(64):
        for t in range(64):
            moves[f * 64 + t] = (f ^ 7) * 64 + (t ^ 7)
    for j in range(96):
        rest, p = divmod(j, 4)
        ff, d = divmod(rest, 3)
        moves[PROMO_BASE + j] = PROMO_BASE + ((7 - ff) * 3 + (2 - d)) * 4 + p
    return squares, moves


def geometry_tables() -> dict[str, np.ndarray]:
    """Index tables [64, 64] describing the geometric relation between source and target squares."""
    dx = np.zeros((64, 64), dtype=np.int64)
    dy = np.zeros((64, 64), dtype=np.int64)
    rel = np.zeros((64, 64), dtype=np.int64)
    dist = np.zeros((64, 64), dtype=np.int64)
    for s in range(64):
        sf, sr = s & 7, s >> 3
        for t in range(64):
            tf, tr = t & 7, t >> 3
            ddx, ddy = tf - sf, tr - sr
            dx[s, t] = ddx + 7
            dy[s, t] = ddy + 7
            dist[s, t] = max(abs(ddx), abs(ddy))
            if s == t:
                r = 1
            elif ddy == 0:
                r = 2
            elif ddx == 0:
                r = 3
            elif ddx == ddy:
                r = 4
            elif ddx == -ddy:
                r = 5
            elif {abs(ddx), abs(ddy)} == {1, 2}:
                r = 6
            else:
                r = 0
            rel[s, t] = r
    return {"dx": dx, "dy": dy, "rel": rel, "dist": dist}


NUM_REL = 7
NUM_DELTA = 15
NUM_DIST = 8
