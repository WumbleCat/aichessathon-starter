"""Board and move encodings shared by training and inference.

Everything is expressed from the side to move's point of view: when Black is to move the board
is mirrored vertically and colours are swapped, so the network only ever sees "my pieces move
up the board".

Input: 18 planes of 8x8
    0-5   own P N B R Q K
    6-11  opponent P N B R Q K
    12-15 castling rights: own K-side, own Q-side, opponent K-side, opponent Q-side (constant)
    16    en-passant target square
    17    all ones

Actions: 64 from-squares x 73 move planes = 4672 (AlphaZero-style)
    0-55  queen-like: 8 directions x distance 1..7 (also queen promotions)
    56-63 knight jumps
    64-72 underpromotions: 3 directions (capture left, push, capture right) x (N, B, R)
"""

from __future__ import annotations

import numpy as np

import chess

NUM_PLANES = 18
NUM_MOVE_PLANES = 73
NUM_ACTIONS = 64 * NUM_MOVE_PLANES

# direction order: N, NE, E, SE, S, SW, W, NW as (d_file, d_rank)
_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
_KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_UNDER_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
_UNDER_DIRS = [-1, 0, 1]

# (from, to, promotion) -> action index, in the side-to-move frame (mover plays "up")
MOVE_INDEX: dict[tuple[int, int, int | None], int] = {}
# action index -> (from, to, promotion)
INDEX_MOVE: list[tuple[int, int, int | None]] = [(0, 0, None)] * NUM_ACTIONS


def _build_tables() -> None:
    for frm in range(64):
        ff, fr = frm & 7, frm >> 3
        for d, (df, dr) in enumerate(_DIRS):
            for dist in range(1, 8):
                tf, tr = ff + df * dist, fr + dr * dist
                if not (0 <= tf < 8 and 0 <= tr < 8):
                    break
                to = tr * 8 + tf
                plane = d * 7 + dist - 1
                idx = frm * NUM_MOVE_PLANES + plane
                MOVE_INDEX[(frm, to, None)] = idx
                INDEX_MOVE[idx] = (frm, to, None)
                if fr == 6 and tr == 7 and dist == 1:
                    # queen promotion shares the queen-move plane
                    MOVE_INDEX[(frm, to, chess.QUEEN)] = idx
        for k, (df, dr) in enumerate(_KNIGHT):
            tf, tr = ff + df, fr + dr
            if 0 <= tf < 8 and 0 <= tr < 8:
                to = tr * 8 + tf
                idx = frm * NUM_MOVE_PLANES + 56 + k
                MOVE_INDEX[(frm, to, None)] = idx
                INDEX_MOVE[idx] = (frm, to, None)
        if fr == 6:
            for di, df in enumerate(_UNDER_DIRS):
                tf = ff + df
                if not 0 <= tf < 8:
                    continue
                to = 7 * 8 + tf
                for pi, piece in enumerate(_UNDER_PIECES):
                    idx = frm * NUM_MOVE_PLANES + 64 + di * 3 + pi
                    MOVE_INDEX[(frm, to, piece)] = idx
                    INDEX_MOVE[idx] = (frm, to, piece)


_build_tables()


def move_to_index(move: chess.Move, flip: bool) -> int:
    """Action index of ``move``; ``flip`` is True when Black is to move."""
    frm, to = move.from_square, move.to_square
    if flip:
        frm ^= 56
        to ^= 56
    promo = move.promotion
    if promo is not None and promo == chess.QUEEN:
        return MOVE_INDEX[(frm, to, chess.QUEEN)]
    return MOVE_INDEX[(frm, to, promo)]


def index_to_move(idx: int, flip: bool, is_pawn: bool = False) -> chess.Move:
    """Inverse of move_to_index. Queen promotions share a plane with ordinary one-step moves,
    so ``is_pawn`` says whether the moving piece is a pawn."""
    frm, to, promo = INDEX_MOVE[idx]
    if promo is None and is_pawn and (frm >> 3) == 6 and (to >> 3) == 7:
        promo = chess.QUEEN
    if flip:
        frm ^= 56
        to ^= 56
    return chess.Move(frm, to, promotion=promo)


_BIT_LOOKUP = np.array(
    [[(i >> b) & 1 for b in range(8)] for i in range(256)], dtype=np.float32
)  # byte -> 8 bits, LSB first


def _bits(bb: int) -> np.ndarray:
    """64 float32 flags indexed by square."""
    return _BIT_LOOKUP[np.frombuffer(bb.to_bytes(8, "little"), dtype=np.uint8)].reshape(64)


def encode_board(board: chess.Board) -> np.ndarray:
    """(18, 8, 8) float32 planes from the side to move's point of view."""
    planes = np.zeros((NUM_PLANES, 64), dtype=np.float32)
    us = board.turn
    them = not us
    occ_us = board.occupied_co[us]
    occ_them = board.occupied_co[them]
    for i, bb in enumerate(
        (board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings)
    ):
        if bb & occ_us:
            planes[i] = _bits(bb & occ_us)
        if bb & occ_them:
            planes[6 + i] = _bits(bb & occ_them)
    cr = board.castling_rights
    if us:
        if cr & chess.BB_H1:
            planes[12] = 1.0
        if cr & chess.BB_A1:
            planes[13] = 1.0
        if cr & chess.BB_H8:
            planes[14] = 1.0
        if cr & chess.BB_A8:
            planes[15] = 1.0
    else:
        if cr & chess.BB_H8:
            planes[12] = 1.0
        if cr & chess.BB_A8:
            planes[13] = 1.0
        if cr & chess.BB_H1:
            planes[14] = 1.0
        if cr & chess.BB_A1:
            planes[15] = 1.0
    if board.ep_square is not None:
        planes[16, board.ep_square] = 1.0
    planes[17] = 1.0
    planes = planes.reshape(NUM_PLANES, 8, 8)
    if not us:
        planes = planes[:, ::-1, :]
    return np.ascontiguousarray(planes)


# ------------------------------------------------------------------ compact storage for datasets
# piece codes: 0 empty, 1-6 white PNBRQK, 7-12 black PNBRQK


def board_to_codes(board: chess.Board) -> tuple[np.ndarray, int]:
    """(64,) uint8 piece codes in the true frame + packed meta (turn, castling, ep file)."""
    codes = np.zeros(64, dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        codes[sq] = piece.piece_type + (0 if piece.color else 6)
    cr = board.castling_rights
    meta = (1 if board.turn else 0)
    meta |= (1 if cr & chess.BB_H1 else 0) << 1
    meta |= (1 if cr & chess.BB_A1 else 0) << 2
    meta |= (1 if cr & chess.BB_H8 else 0) << 3
    meta |= (1 if cr & chess.BB_A8 else 0) << 4
    ep = 15 if board.ep_square is None else chess.square_file(board.ep_square)
    meta |= ep << 5
    return codes, meta


def codes_to_planes(codes: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """Vectorised: (N, 64) uint8 codes + (N,) int meta -> (N, 18, 8, 8) float32 STM planes."""
    n = codes.shape[0]
    turn = (meta & 1).astype(bool)
    planes = np.zeros((n, NUM_PLANES, 64), dtype=np.float32)
    # own/opponent relabelling: for black to move, white pieces (1-6) become opponent (7-12)
    c = codes.astype(np.int16)
    swap = (~turn)[:, None]
    w = (c >= 1) & (c <= 6)
    b = c >= 7
    cs = np.where(swap & w, c + 6, np.where(swap & b, c - 6, c))
    # one-hot into planes 0..11
    rows = np.repeat(np.arange(n), 64)
    sqs = np.tile(np.arange(64), n)
    flat = cs.reshape(-1)
    mask = flat > 0
    planes[rows[mask], flat[mask] - 1, sqs[mask]] = 1.0
    own_k = np.where(turn, (meta >> 1) & 1, (meta >> 3) & 1)
    own_q = np.where(turn, (meta >> 2) & 1, (meta >> 4) & 1)
    opp_k = np.where(turn, (meta >> 3) & 1, (meta >> 1) & 1)
    opp_q = np.where(turn, (meta >> 4) & 1, (meta >> 2) & 1)
    planes[:, 12, :] = own_k[:, None]
    planes[:, 13, :] = own_q[:, None]
    planes[:, 14, :] = opp_k[:, None]
    planes[:, 15, :] = opp_q[:, None]
    ep_file = (meta >> 5) & 15
    has_ep = ep_file < 8
    ep_rank = np.where(turn, 5, 2)
    ep_sq = ep_rank * 8 + np.minimum(ep_file, 7)
    idx = np.nonzero(has_ep)[0]
    planes[idx, 16, ep_sq[idx]] = 1.0
    planes[:, 17, :] = 1.0
    planes = planes.reshape(n, NUM_PLANES, 8, 8)
    flip_idx = np.nonzero(~turn)[0]
    planes[flip_idx] = planes[flip_idx][:, :, ::-1, :]
    return planes
