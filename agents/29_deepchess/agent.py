"""DeepChess-style pairwise evaluation engine for AI Chessathon.

The evaluation is a small position encoder (773 binary piece-square features -> 256 -> 32
-> 32 -> 1) trained with the DeepChess pairwise preference objective
``sigmoid(V(A) - V(B))`` plus a value-regression auxiliary loss on Stockfish labels.
The scalar head is the leaf evaluation of a negamax alpha-beta search. Everything the net
needs at play time is a numba kernel that reads python-chess bitboards, so a leaf costs a
few microseconds.

Falls back to a handcrafted material + piece-square evaluation when no model file is
present. Both evaluations share the same kernel signature, so the search never changes.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import chess
import numpy as np
from numba import njit, uint64, int64, float32, int32

# ---------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------

MATE = 100_000
MATE_BOUND = MATE - 1000  # scores beyond this are mate scores
INF = MATE + 1
DRAW = 0
MAX_PLY = 64

PIECE_VALUE = np.array([0, 100, 320, 330, 500, 900, 20000], dtype=np.int32)

# Handcrafted piece-square tables, White's point of view, a1 = index 0. Written for this
# engine; they only exist so the search can play before a model is trained and to give
# the trained model a reference to beat.
_PST_PAWN = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_PST_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
_PST_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
_PST_ROOK = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_PST_QUEEN = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
_PST_KING_MG = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]
_PST_KING_EG = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
]

# PST[piece_type][square]; index 6 is the endgame king table.
PST = np.zeros((7, 64), dtype=np.int32)
for _i, _t in enumerate(
    (_PST_PAWN, _PST_KNIGHT, _PST_BISHOP, _PST_ROOK, _PST_QUEEN, _PST_KING_MG, _PST_KING_EG)
):
    PST[_i] = np.array(_t, dtype=np.int32)

# ---------------------------------------------------------------------------------------
# Numba evaluation kernels. Both take the raw python-chess bitboards.
# ---------------------------------------------------------------------------------------


@njit(int64(uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64, int64, int32[:, :]),
      cache=False, nogil=True)
def eval_handcrafted(pawns, knights, bishops, rooks, queens, kings, wocc, bocc, turn, pst):
    """Material + piece-square evaluation, side to move perspective, centipawns."""
    occ = wocc | bocc
    # game phase: non-pawn material of both sides, 0 (endgame) .. 24 (opening)
    phase = 0
    score_mg = 0
    score_eg = 0
    for sq in range(64):
        bit = uint64(1) << uint64(sq)
        if occ & bit == 0:
            continue
        white = (wocc & bit) != 0
        if pawns & bit:
            pt = 0
            val = 100
        elif knights & bit:
            pt = 1
            val = 320
            phase += 1
        elif bishops & bit:
            pt = 2
            val = 330
            phase += 1
        elif rooks & bit:
            pt = 3
            val = 500
            phase += 2
        elif queens & bit:
            pt = 4
            val = 900
            phase += 4
        else:
            pt = 5
            val = 0
        psq = sq if white else (sq ^ 56)
        if pt == 5:
            mg = val + pst[5, psq]
            eg = val + pst[6, psq]
        else:
            mg = val + pst[pt, psq]
            eg = mg
        if white:
            score_mg += mg
            score_eg += eg
        else:
            score_mg -= mg
            score_eg -= eg
    if phase > 24:
        phase = 24
    score = (score_mg * phase + score_eg * (24 - phase)) // 24
    # tempo
    score += 10 if turn == 1 else -10
    return score if turn == 1 else -score


@njit(int64(uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64, int64, uint64, int64,
            int32[:, :]), cache=False, nogil=True)
def features_to_indices(pawns, knights, bishops, rooks, queens, kings, wocc, bocc, turn,
                        castling, ep_flag, out):
    """Fill out[0, :] with active feature indices (side-to-move perspective); return count."""
    occ = wocc | bocc
    n = 0
    flip = 0 if turn == 1 else 56
    for sq in range(64):
        bit = uint64(1) << uint64(sq)
        if occ & bit == 0:
            continue
        white = (wocc & bit) != 0
        if pawns & bit:
            pt = 0
        elif knights & bit:
            pt = 1
        elif bishops & bit:
            pt = 2
        elif rooks & bit:
            pt = 3
        elif queens & bit:
            pt = 4
        else:
            pt = 5
        ours = white == (turn == 1)
        idx = (pt * 2 + (0 if ours else 1)) * 64 + (sq ^ flip)
        out[0, n] = idx
        n += 1
    # castling rights: our K, our Q, their K, their Q
    if turn == 1:
        ok = (castling >> 7) & 1
        oq = castling & 1
        tk = (castling >> 63) & 1
        tq = (castling >> 56) & 1
    else:
        ok = (castling >> 63) & 1
        oq = (castling >> 56) & 1
        tk = (castling >> 7) & 1
        tq = castling & 1
    if ok:
        out[0, n] = 768
        n += 1
    if oq:
        out[0, n] = 769
        n += 1
    if tk:
        out[0, n] = 770
        n += 1
    if tq:
        out[0, n] = 771
        n += 1
    if ep_flag:
        out[0, n] = 772
        n += 1
    return n


@njit(int64(uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64, int64, uint64, int64,
            float32[:, :], float32[:], float32[:, :], float32[:], float32[:, :], float32[:],
            float32[:], float32[:], float32[:], float32[:], float32[:]),
      cache=False, nogil=True)
def eval_network(pawns, knights, bishops, rooks, queens, kings, wocc, bocc, turn, castling,
                 ep_flag, w1, b1, w2, b2, w3, b3, w4, b4, acc, h2, h3):
    """DeepChess scalar head: 773 sparse binary features -> 256 -> 32 -> 32 -> 1.

    Returns centipawns from the side to move's perspective. The first layer is a sum of the
    active rows of w1 because every input feature is binary.
    """
    n1 = acc.shape[0]
    for j in range(n1):
        acc[j] = b1[j]
    occ = wocc | bocc
    flip = 0 if turn == 1 else 56
    for sq in range(64):
        bit = uint64(1) << uint64(sq)
        if occ & bit == 0:
            continue
        white = (wocc & bit) != 0
        if pawns & bit:
            pt = 0
        elif knights & bit:
            pt = 1
        elif bishops & bit:
            pt = 2
        elif rooks & bit:
            pt = 3
        elif queens & bit:
            pt = 4
        else:
            pt = 5
        ours = white == (turn == 1)
        idx = (pt * 2 + (0 if ours else 1)) * 64 + (sq ^ flip)
        for j in range(n1):
            acc[j] += w1[idx, j]
    if turn == 1:
        ok = (castling >> 7) & 1
        oq = castling & 1
        tk = (castling >> 63) & 1
        tq = (castling >> 56) & 1
    else:
        ok = (castling >> 63) & 1
        oq = (castling >> 56) & 1
        tk = (castling >> 7) & 1
        tq = castling & 1
    if ok:
        for j in range(n1):
            acc[j] += w1[768, j]
    if oq:
        for j in range(n1):
            acc[j] += w1[769, j]
    if tk:
        for j in range(n1):
            acc[j] += w1[770, j]
    if tq:
        for j in range(n1):
            acc[j] += w1[771, j]
    if ep_flag:
        for j in range(n1):
            acc[j] += w1[772, j]
    # clipped relu
    for j in range(n1):
        v = acc[j]
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        acc[j] = v
    n2 = h2.shape[0]
    for k in range(n2):
        s = b2[k]
        for j in range(n1):
            s += acc[j] * w2[j, k]
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        h2[k] = s
    n3 = h3.shape[0]
    for k in range(n3):
        s = b3[k]
        for j in range(n2):
            s += h2[j] * w3[j, k]
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        h3[k] = s
    out = b4[0]
    for j in range(n3):
        out += h3[j] * w4[j]
    # network output is in pawns; clamp to keep clear of mate scores
    cp = out * 100.0
    if cp > 30000.0:
        cp = 30000.0
    elif cp < -30000.0:
        cp = -30000.0
    return int(cp)


# ---------------------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "models" / "deepchess.npz"
_MODEL: dict[str, np.ndarray] | None = None
_ACC = np.zeros(256, dtype=np.float32)
_H2 = np.zeros(32, dtype=np.float32)
_H3 = np.zeros(32, dtype=np.float32)
EVAL_MODE = os.environ.get("DEEPCHESS_EVAL", "auto")  # auto | net | hand | blend
BLEND_NET_WEIGHT = float(os.environ.get("DEEPCHESS_BLEND", "0.5"))


def load_model(path: Path = MODEL_PATH) -> dict[str, np.ndarray] | None:
    global _MODEL, _ACC, _H2, _H3
    if not path.exists():
        _MODEL = None
        return None
    with np.load(path) as data:
        model = {k: np.ascontiguousarray(data[k].astype(np.float32)) for k in data.files}
    _MODEL = model
    _ACC = np.zeros(model["w1"].shape[1], dtype=np.float32)
    _H2 = np.zeros(model["w2"].shape[1], dtype=np.float32)
    _H3 = np.zeros(model["w3"].shape[1], dtype=np.float32)
    return model


def _use_net() -> bool:
    if _MODEL is None:
        return False
    return EVAL_MODE in ("auto", "net", "blend")


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns from the side to move's perspective."""
    turn = 1 if board.turn else 0
    wocc = board.occupied_co[chess.WHITE]
    bocc = board.occupied_co[chess.BLACK]
    if _MODEL is not None and EVAL_MODE != "hand":
        m = _MODEL
        net = eval_network(
            board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
            wocc, bocc, turn, board.castling_rights, 1 if board.ep_square is not None else 0,
            m["w1"], m["b1"], m["w2"], m["b2"], m["w3"], m["b3"], m["w4"], m["b4"],
            _ACC, _H2, _H3,
        )
        if EVAL_MODE == "blend":
            hand = eval_handcrafted(
                board.pawns, board.knights, board.bishops, board.rooks, board.queens,
                board.kings, wocc, bocc, turn, PST,
            )
            return int(BLEND_NET_WEIGHT * net + (1.0 - BLEND_NET_WEIGHT) * hand)
        return int(net)
    return int(eval_handcrafted(
        board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
        wocc, bocc, turn, PST,
    ))


# ---------------------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------------------


class OutOfTime(Exception):
    pass


TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2


class Searcher:
    def __init__(self) -> None:
        self.tt: dict[int, tuple[int, int, int, chess.Move | None]] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 2)]
        self.history: dict[tuple[bool, int, int], int] = {}
        self.nodes = 0
        self.qnodes = 0
        self.tt_hits = 0
        self.seldepth = 0
        self.deadline = 0.0
        self.armed = False
        self.path: dict[int, int] = {}  # repetition counts along the current search path
        self.game_history: dict[int, int] = {}

    # ---- helpers -------------------------------------------------------------------

    def _check_time(self) -> None:
        # perf_counter costs well under a microsecond; a node costs tens, so check every node
        if self.armed and time.perf_counter() >= self.deadline:
            raise OutOfTime()

    def _is_repetition(self, key: int) -> bool:
        return self.path.get(key, 0) + self.game_history.get(key, 0) >= 2

    @staticmethod
    def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
        victim = board.piece_type_at(move.to_square)
        if victim is None:
            if board.is_en_passant(move):
                victim = chess.PAWN
            else:
                victim = 0
        attacker = board.piece_type_at(move.from_square) or 1
        score = int(PIECE_VALUE[victim]) * 10 - attacker
        if move.promotion:
            score += int(PIECE_VALUE[move.promotion])
        return score

    def _order_moves(self, board: chess.Board, moves: list[chess.Move], ply: int,
                     tt_move: chess.Move | None) -> list[chess.Move]:
        killers = self.killers[ply]
        turn = board.turn
        hist = self.history
        scored = []
        for m in moves:
            if m == tt_move:
                s = 10_000_000
            elif board.is_capture(m) or m.promotion:
                s = 1_000_000 + self._mvv_lva(board, m)
            elif m == killers[0]:
                s = 900_000
            elif m == killers[1]:
                s = 800_000
            else:
                s = hist.get((turn, m.from_square, m.to_square), 0)
            scored.append((s, m))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored]

    # ---- quiescence ------------------------------------------------------------------

    def quiesce(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self.qnodes += 1
        self._check_time()
        if ply > self.seldepth:
            self.seldepth = ply
        in_check = board.is_check()
        if in_check:
            # evasions: full move list, no stand pat
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
            best = -INF
            moves = self._order_moves(board, moves, ply, None)
            for m in moves:
                board.push(m)
                score = -self.quiesce(board, -beta, -alpha, ply + 1)
                board.pop()
                if score > best:
                    best = score
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            break
            return best
        stand = evaluate(board)
        if stand >= beta:
            return stand
        if ply >= MAX_PLY - 1:
            return stand
        if stand > alpha:
            alpha = stand
        best = stand
        captures = []
        for m in board.generate_legal_moves(chess.BB_ALL, board.occupied_co[not board.turn]):
            captures.append((self._mvv_lva(board, m), m))
        # en passant and promotions
        for m in board.generate_legal_moves(board.pawns & board.occupied_co[board.turn],
                                            ~board.occupied_co[not board.turn]):
            if m.promotion == chess.QUEEN or board.is_en_passant(m):
                captures.append((self._mvv_lva(board, m), m))
        captures.sort(key=lambda t: t[0], reverse=True)
        for score_key, m in captures:
            # delta pruning
            victim = board.piece_type_at(m.to_square)
            gain = int(PIECE_VALUE[victim]) if victim else 100
            if m.promotion:
                gain += 800
            if stand + gain + 200 <= alpha:
                continue
            board.push(m)
            score = -self.quiesce(board, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    # ---- main search ------------------------------------------------------------------

    def search(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int,
               null_ok: bool = True) -> int:
        self.nodes += 1
        self._check_time()
        if ply >= MAX_PLY - 1:
            return evaluate(board)

        key = board._transposition_key()
        if ply > 0:
            if self._is_repetition(key) or board.halfmove_clock >= 100:
                return DRAW
            # mate distance pruning
            alpha = max(alpha, -MATE + ply)
            beta = min(beta, MATE - ply - 1)
            if alpha >= beta:
                return alpha

        in_check = board.is_check()
        if in_check:
            depth += 1

        if depth <= 0:
            return self.quiesce(board, alpha, beta, ply)

        tt_move = None
        entry = self.tt.get(key)
        if entry is not None:
            tt_depth, tt_flag, tt_score, tt_move = entry
            if tt_depth >= depth and ply > 0:
                score = tt_score
                if score > MATE_BOUND:
                    score -= ply
                elif score < -MATE_BOUND:
                    score += ply
                if tt_flag == TT_EXACT:
                    self.tt_hits += 1
                    return score
                if tt_flag == TT_LOWER and score >= beta:
                    self.tt_hits += 1
                    return score
                if tt_flag == TT_UPPER and score <= alpha:
                    self.tt_hits += 1
                    return score

        pv_node = beta - alpha > 1
        static_eval = None

        if not in_check and not pv_node and ply > 0:
            static_eval = evaluate(board)
            # reverse futility pruning
            if depth <= 3 and static_eval - 120 * depth >= beta and abs(beta) < MATE_BOUND:
                return static_eval
            # null move pruning
            if (null_ok and depth >= 2 and static_eval >= beta
                    and self._has_non_pawn_material(board)):
                r = 3 if depth >= 6 else 2
                board.push(chess.Move.null())
                nkey = board._transposition_key()
                self.path[nkey] = self.path.get(nkey, 0) + 1
                try:
                    score = -self.search(board, depth - 1 - r, -beta, -beta + 1, ply + 1, False)
                finally:
                    self.path[nkey] -= 1
                    board.pop()
                if score >= beta and abs(score) < MATE_BOUND:
                    return score

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else DRAW
        moves = self._order_moves(board, moves, ply, tt_move)

        best = -INF
        best_move = None
        alpha_orig = alpha
        self.path[key] = self.path.get(key, 0) + 1
        try:
            for i, m in enumerate(moves):
                quiet = not m.promotion and not board.is_capture(m)
                board.push(m)
                if quiet and board.is_check():
                    quiet = False  # checking moves are never reduced or pruned
                # futility pruning at frontier nodes
                if (quiet and not in_check and depth <= 2 and i > 0
                        and abs(alpha) < MATE_BOUND):
                    if static_eval is None:
                        board.pop()
                        static_eval = evaluate(board)
                        board.push(m)
                    if static_eval + 150 * depth <= alpha:
                        board.pop()
                        continue
                new_depth = depth - 1
                # late move reductions
                if (i >= 3 and depth >= 3 and quiet and not in_check
                        and m != self.killers[ply][0] and m != self.killers[ply][1]):
                    reduction = 1 if i < 8 else 2
                    score = -self.search(board, new_depth - reduction, -alpha - 1, -alpha, ply + 1)
                    if score > alpha:
                        score = -self.search(board, new_depth, -beta, -alpha, ply + 1)
                elif i == 0:
                    score = -self.search(board, new_depth, -beta, -alpha, ply + 1)
                else:
                    score = -self.search(board, new_depth, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -self.search(board, new_depth, -beta, -alpha, ply + 1)
                board.pop()
                if score > best:
                    best = score
                    best_move = m
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            if quiet:
                                k = self.killers[ply]
                                if k[0] != m:
                                    k[1] = k[0]
                                    k[0] = m
                                hk = (board.turn, m.from_square, m.to_square)
                                self.history[hk] = self.history.get(hk, 0) + depth * depth
                            break
        finally:
            self.path[key] -= 1

        # store
        store = best
        if store > MATE_BOUND:
            store += ply
        elif store < -MATE_BOUND:
            store -= ply
        if best <= alpha_orig:
            flag = TT_UPPER
        elif best >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        old = self.tt.get(key)
        if old is None or old[0] <= depth or flag == TT_EXACT:
            self.tt[key] = (depth, flag, store, best_move)
        return best

    @staticmethod
    def _has_non_pawn_material(board: chess.Board) -> bool:
        us = board.occupied_co[board.turn]
        return bool(us & (board.knights | board.bishops | board.rooks | board.queens))

    # ---- root ----------------------------------------------------------------------

    def search_root(self, board: chess.Board, depth: int, moves: list[chess.Move],
                    prev_best: chess.Move | None) -> tuple[int, chess.Move, list[chess.Move]]:
        """Return (score, best move, moves ordered with best first) for one iteration."""
        ordered = self._order_moves(board, moves, 0, prev_best)
        alpha = -INF
        beta = INF
        best = -INF
        best_move = ordered[0]
        key = board._transposition_key()
        self.path[key] = self.path.get(key, 0) + 1
        scored: list[tuple[int, chess.Move]] = []
        try:
            for i, m in enumerate(ordered):
                board.push(m)
                if i == 0:
                    score = -self.search(board, depth - 1, -beta, -alpha, 1)
                else:
                    score = -self.search(board, depth - 1, -alpha - 1, -alpha, 1)
                    if score > alpha:
                        score = -self.search(board, depth - 1, -beta, -alpha, 1)
                board.pop()
                scored.append((score, m))
                if score > best:
                    best = score
                    best_move = m
                    if score > alpha:
                        alpha = score
        finally:
            self.path[key] -= 1
        scored.sort(key=lambda t: t[0], reverse=True)
        self.tt[key] = (depth, TT_EXACT, best, best_move)
        return best, best_move, [m for _, m in scored]


# ---------------------------------------------------------------------------------------
# Game state and time management
# ---------------------------------------------------------------------------------------

_searcher = Searcher()
_game_history: dict[int, int] = {}
_last_fen: str | None = None
_expected_moves_left = 40
STATS: dict[str, float] = {}


def _reset_game_if_needed(board: chess.Board) -> None:
    """A new game starts when the position is not a plausible continuation of the last."""
    global _game_history, _searcher
    if board.fullmove_number <= 1 and len(_game_history) > 2:
        _game_history = {}
        _searcher = Searcher()
    elif board.fullmove_number == 1 and board.turn == chess.WHITE:
        _game_history = {}
        _searcher = Searcher()


def _time_budget_ms(board: chess.Board, time_left_ms: int) -> tuple[float, float]:
    """(soft, hard) budget in ms for this move."""
    increment = 500.0 if time_left_ms > 20_000 else 100.0
    moves_left = max(18, 40 - board.fullmove_number // 2)
    soft = time_left_ms / moves_left + 0.6 * increment
    soft = min(soft, time_left_ms * 0.15)
    hard = min(soft * 2.5, time_left_ms * 0.4)
    # keep a safety margin for process overhead
    margin = 60.0 if time_left_ms > 2000 else 25.0
    soft = max(1.0, soft - margin)
    hard = max(1.0, hard - margin)
    return soft, hard


def _quick_move(board: chess.Board, moves: list[chess.Move]) -> chess.Move:
    """One-ply static evaluation pick: the fallback for tiny clocks."""
    best_score = -INF
    best = moves[0]
    for m in moves:
        board.push(m)
        if board.is_checkmate():
            score = MATE
        else:
            score = -evaluate(board)
        board.pop()
        if score > best_score:
            best_score = score
            best = m
    return best


def get_move(fen: str, time_left_ms: int) -> str:
    start = time.perf_counter()
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"
    fallback = moves[0]
    if len(moves) == 1:
        _remember(board)
        return fallback.uci()

    _reset_game_if_needed(board)
    _remember(board)
    searcher = _searcher
    searcher.game_history = _game_history
    searcher.nodes = 0
    searcher.qnodes = 0
    searcher.tt_hits = 0
    searcher.seldepth = 0
    searcher.path = {}
    for k in searcher.killers:
        k[0] = k[1] = None
    # decay history
    if searcher.history:
        for hk in list(searcher.history):
            searcher.history[hk] //= 2

    if time_left_ms < 120:
        best = _quick_move(board, moves)
        _record_stats(board, start, 0, 0, searcher)
        return best.uci()

    soft, hard = _time_budget_ms(board, time_left_ms)
    searcher.deadline = start + hard / 1000.0
    searcher.armed = True
    best = _quick_move(board, moves)
    best_score = 0
    depth_reached = 0
    ordered = moves
    stack_len = len(board.move_stack)
    for depth in range(1, MAX_PLY):
        try:
            score, move, ordered = searcher.search_root(board, depth, ordered, best)
        except OutOfTime:
            while len(board.move_stack) > stack_len:
                board.pop()
            break
        best, best_score = move, score
        depth_reached = depth
        elapsed = (time.perf_counter() - start) * 1000.0
        if abs(score) > MATE_BOUND and depth >= 4:
            break
        if elapsed > soft:
            break
        # do not start a new iteration that is unlikely to finish
        if elapsed * 2.5 > hard:
            break
    _record_stats(board, start, depth_reached, best_score, searcher)
    if best not in moves:
        best = fallback
    return best.uci()


def _remember(board: chess.Board) -> None:
    key = board._transposition_key()
    _game_history[key] = _game_history.get(key, 0) + 1


def _record_stats(board: chess.Board, start: float, depth: int, score: int,
                  searcher: Searcher) -> None:
    elapsed = time.perf_counter() - start
    STATS.update({
        "depth": depth,
        "seldepth": searcher.seldepth,
        "score": score,
        "nodes": searcher.nodes,
        "qnodes": searcher.qnodes,
        "tt_hits": searcher.tt_hits,
        "time_ms": elapsed * 1000.0,
        "nps": searcher.nodes / elapsed if elapsed > 0 else 0.0,
    })
    if os.environ.get("DEEPCHESS_VERBOSE"):
        print(f"depth {depth}/{searcher.seldepth} score {score} nodes {searcher.nodes} "
              f"qnodes {searcher.qnodes} tt {searcher.tt_hits} "
              f"time {elapsed * 1000:.0f}ms nps {STATS['nps']:.0f}")


# ---------------------------------------------------------------------------------------
# Import-time warm-up: compile the kernels and load the model inside the init budget.
# ---------------------------------------------------------------------------------------

load_model()
_warm = chess.Board()
evaluate(_warm)
eval_handcrafted(_warm.pawns, _warm.knights, _warm.bishops, _warm.rooks, _warm.queens,
                 _warm.kings, _warm.occupied_co[True], _warm.occupied_co[False], 1, PST)
_idx = np.zeros((1, 40), dtype=np.int32)
features_to_indices(_warm.pawns, _warm.knights, _warm.bishops, _warm.rooks, _warm.queens,
                    _warm.kings, _warm.occupied_co[True], _warm.occupied_co[False], 1,
                    _warm.castling_rights, 0, _idx)
if _MODEL is not None:
    _m = _MODEL
    eval_network(_warm.pawns, _warm.knights, _warm.bishops, _warm.rooks, _warm.queens,
                 _warm.kings, _warm.occupied_co[True], _warm.occupied_co[False], 1,
                 _warm.castling_rights, 0, _m["w1"], _m["b1"], _m["w2"], _m["b2"], _m["w3"],
                 _m["b3"], _m["w4"], _m["b4"], _ACC, _H2, _H3)
del _warm
