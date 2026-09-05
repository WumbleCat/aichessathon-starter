"""numba alpha-beta search on top of ``dc_engine`` for the DeepChess agent.

The whole tree search runs compiled: negamax with alpha-beta and principal variation
search, a transposition table, killer and history move ordering, null-move pruning, late
move reductions, reverse futility and futility pruning, check extension and a quiescence
search with delta pruning. The leaf evaluation is the DeepChess network whose first layer
is kept incrementally in the accumulator that ``dc_engine.make_move`` maintains; the
handcrafted material + piece-square evaluation is available as well, alone or blended.

All arrays travel in one tuple (``ctx``, layout in ``CTX_FIELDS``). numba's type inference
cost grows much faster than linearly with the number of arguments and call sites of a big
recursive function; with the tuple the search compiles in seconds instead of minutes.
Recursive calls must not pass literal arguments (``True``/``False``): while a function is
being compiled its dispatcher is still open, and numba builds one extra copy of the whole
function per literal type it sees at a recursive call site.

Time control: the search cannot read the clock from compiled code, so the Python driver
arms a ``threading.Timer`` that writes into the ``stop`` array at the deadline (the search
runs with the GIL released) and additionally caps the node count. Every node checks both.
"""

from __future__ import annotations

import os as _os
import time as _time

from dc_engine import (
    DRAW,
    F_CAPTURE,
    F_EP,
    INF,
    MATE,
    MATE_BOUND,
    MAX_MOVES,
    MAX_PLY,
    NUMBA_CACHE,
    PIECE_VALUE,
    S_CASTLING,
    S_EP,
    S_HALFMOVE,
    S_HIST_LEN,
    S_PLY,
    S_ROOT_HIST,
    S_TURN,
    U_CASTLING,
    U_EP,
    U_HALF,
    U_HASH,
    U_MOVE,
    Z_SIDE,
    gen_moves,
    is_attacked,
    make_move,
    piece_type,
    unmake_move,
)
from numba import boolean, float32, int8, int32, int64, njit, types, uint64

_TIMING = bool(_os.environ.get("DEEPCHESS_COMPILE_TIMING"))
_t_last = _time.process_time()


def _mark(name: str) -> None:
    """Print the CPU seconds spent compiling since the previous mark (opt-in)."""
    global _t_last
    if _TIMING:
        now = _time.process_time()
        print(f"compile {name}: {now - _t_last:.1f}s cpu", flush=True)
        _t_last = now


# ------------------------------------------------------------------------------ constants

TT_BITS = 20
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1
SCORE_OFFSET = 1 << 17
TT_EXACT, TT_LOWER, TT_UPPER = 1, 2, 3

# stats vector
ST_NODES, ST_QNODES, ST_MAX_NODES, ST_ABORT, ST_TT_HITS, ST_SELDEPTH = 0, 1, 2, 3, 4, 5
STATS_SIZE = 8

# params vector
P_MODE, P_BLEND = 0, 1  # mode 0 net, 1 hand, 2 blend; blend = network weight in percent
PARAMS_SIZE = 4

WORK_SIZE = 256 + 32 + 32

# ctx tuple layout: (name, numba type). agent.NumbaSearcher builds the tuple in this order.
CTX_FIELDS = (
    ("board", types.Array(int8, 1, "C")),
    ("state", types.Array(int64, 1, "C")),
    ("hash", types.Array(uint64, 1, "C")),
    ("undo", types.Array(int64, 2, "C")),
    ("moves", types.Array(int32, 1, "C")),
    ("mscore", types.Array(int64, 1, "C")),
    ("acc_stack", types.Array(float32, 3, "C")),
    ("w1", types.Array(float32, 2, "C")),
    ("b1", types.Array(float32, 1, "C")),
    ("w2t", types.Array(float32, 2, "C")),  # (32, 256): layer 2 weights transposed
    ("b2", types.Array(float32, 1, "C")),
    ("w3t", types.Array(float32, 2, "C")),  # (32, 32): layer 3 weights transposed
    ("b3", types.Array(float32, 1, "C")),
    ("w4", types.Array(float32, 1, "C")),
    ("b4", types.Array(float32, 1, "C")),
    ("hist", types.Array(uint64, 1, "C")),
    ("work", types.Array(float32, 1, "C")),
    ("pst", types.Array(int32, 2, "C")),
    ("params", types.Array(int64, 1, "C")),
    ("stats", types.Array(int64, 1, "C")),
    ("stop", types.Array(int64, 1, "C")),
    ("knight_t", types.Array(int64, 2, "C")),
    ("king_t", types.Array(int64, 2, "C")),
    ("bishop_rays", types.Array(int64, 3, "C")),
    ("rook_rays", types.Array(int64, 3, "C")),
    ("pawn_attackers", types.Array(int64, 3, "C")),
    ("zobrist", types.Array(uint64, 2, "C")),
    ("z_castle", types.Array(uint64, 1, "C")),
    ("z_ep", types.Array(uint64, 1, "C")),
    ("castle_mask", types.Array(int64, 1, "C")),
    ("tt_key", types.Array(uint64, 1, "C")),
    ("tt_data", types.Array(int64, 1, "C")),
    ("killers", types.Array(int32, 2, "C")),
    ("history", types.Array(int64, 3, "C")),
)
CTX = types.Tuple([t for _, t in CTX_FIELDS])
(C_BOARD, C_STATE, C_HASH, C_UNDO, C_MOVES, C_MSCORE, C_ACC, C_W1, C_B1, C_W2, C_B2, C_W3,
 C_B3, C_W4, C_B4, C_HIST, C_WORK, C_PST, C_PARAMS, C_STATS, C_STOP, C_KNIGHT, C_KING,
 C_BISHOP, C_ROOK, C_PAWN, C_ZOBRIST, C_ZCASTLE, C_ZEP, C_CASTLE_MASK, C_TT_KEY, C_TT_DATA,
 C_KILLERS, C_HISTORY) = range(len(CTX_FIELDS))


# ---------------------------------------------------------------------- thin wrappers


@njit((CTX, int32), cache=NUMBA_CACHE, nogil=True)
def make(ctx, move):
    make_move(ctx[C_BOARD], ctx[C_STATE], ctx[C_HASH], move, ctx[C_UNDO], ctx[C_ACC], ctx[C_W1],
              ctx[C_B1], ctx[C_ZOBRIST], ctx[C_ZCASTLE], ctx[C_ZEP], Z_SIDE, ctx[C_CASTLE_MASK],
              ctx[C_HIST])


@njit((CTX,), cache=NUMBA_CACHE, nogil=True)
def unmake(ctx):
    unmake_move(ctx[C_BOARD], ctx[C_STATE], ctx[C_HASH], ctx[C_UNDO])


@njit(boolean(CTX, int64), cache=NUMBA_CACHE, nogil=True)
def king_attacked(ctx, side):
    """Is `side`'s king attacked (by the other side)?"""
    state = ctx[C_STATE]
    ksq = state[6] if side == 0 else state[7]  # S_KING_W, S_KING_B
    return is_attacked(ctx[C_BOARD], ksq, 1 - side, ctx[C_KNIGHT], ctx[C_KING], ctx[C_BISHOP],
                       ctx[C_ROOK], ctx[C_PAWN])


@njit(int64(CTX, int64, boolean), cache=NUMBA_CACHE, nogil=True)
def generate(ctx, start, captures_only):
    return gen_moves(ctx[C_BOARD], ctx[C_STATE], ctx[C_MOVES], start, captures_only,
                     ctx[C_KNIGHT], ctx[C_KING], ctx[C_BISHOP], ctx[C_ROOK], ctx[C_PAWN])


_mark("wrappers")


# ------------------------------------------------------------------------- repetition


@njit(int64(int64[:], uint64[:]), cache=NUMBA_CACHE, nogil=True)
def is_repetition(state, hist):
    """1 if the current position occurred before in the search path or game history.

    ``hist`` holds the positions of this game we were asked to move in (two plies apart),
    then the root, then the search path. The hash includes the side to move, so scanning
    every entry is safe; the halfmove clock bounds the reach.
    """
    n = state[S_HIST_LEN]
    h = hist[n - 1]
    root = state[S_ROOT_HIST] - 1  # index of the root entry
    half = state[S_HALFMOVE]
    # search path (root included): one entry per ply
    lo = n - 1 - half
    i = n - 3
    while i >= root and i >= lo:
        if hist[i] == h:
            return 1
        i -= 2
    if lo >= root:
        return 0
    # game part: entries are two plies apart
    reach = (half - (n - 1 - root)) // 2
    i = root - 1
    lo = root - reach
    while i >= 0 and i >= lo:
        if hist[i] == h:
            return 1
        i -= 1
    return 0


_mark("is_repetition")


# ------------------------------------------------------------------------- null move


@njit((CTX,), cache=NUMBA_CACHE, nogil=True)
def make_null(ctx):
    state = ctx[C_STATE]
    hash_arr = ctx[C_HASH]
    undo = ctx[C_UNDO]
    acc_stack = ctx[C_ACC]
    hist = ctx[C_HIST]
    ply = state[S_PLY]
    undo[ply, U_MOVE] = 0
    undo[ply, U_CASTLING] = state[S_CASTLING]
    undo[ply, U_EP] = state[S_EP]
    undo[ply, U_HALF] = state[S_HALFMOVE]
    undo[ply, U_HASH] = int64(hash_arr[0])
    acc = acc_stack[ply + 1]
    prev = acc_stack[ply]
    n = acc.shape[1]
    for j in range(n):
        acc[0, j] = prev[0, j]
        acc[1, j] = prev[1, j]
    h = hash_arr[0]
    if state[S_EP] >= 0:
        h ^= ctx[C_ZEP][state[S_EP] & 7]
    h ^= Z_SIDE
    state[S_EP] = -1
    state[S_TURN] = 1 - state[S_TURN]
    state[S_HALFMOVE] += 1
    state[S_PLY] = ply + 1
    hash_arr[0] = h
    hist[state[S_HIST_LEN]] = h
    state[S_HIST_LEN] += 1


@njit((CTX,), cache=NUMBA_CACHE, nogil=True)
def unmake_null(ctx):
    state = ctx[C_STATE]
    undo = ctx[C_UNDO]
    ply = state[S_PLY] - 1
    state[S_CASTLING] = undo[ply, U_CASTLING]
    state[S_EP] = undo[ply, U_EP]
    state[S_HALFMOVE] = undo[ply, U_HALF]
    ctx[C_HASH][0] = uint64(undo[ply, U_HASH])
    state[S_TURN] = 1 - state[S_TURN]
    state[S_PLY] = ply
    state[S_HIST_LEN] -= 1


_mark("null move")


# ------------------------------------------------------------------------ evaluation


@njit(int64(int8[:], int64, int32[:, :]), cache=NUMBA_CACHE, nogil=True)
def hand_eval(board, turn, pst):
    """Material + piece-square evaluation, side to move perspective (mirrors agent.py)."""
    phase = 0
    score_mg = 0
    score_eg = 0
    for sq in range(64):
        p = board[sq]
        if p == 0:
            continue
        white = p <= 6
        pt = piece_type(p)
        if pt == 0:
            val = 100
        elif pt == 1:
            val = 320
            phase += 1
        elif pt == 2:
            val = 330
            phase += 1
        elif pt == 3:
            val = 500
            phase += 2
        elif pt == 4:
            val = 900
            phase += 4
        else:
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
    score += 10 if turn == 0 else -10
    return score if turn == 0 else -score


@njit(int64(CTX), cache=NUMBA_CACHE, nogil=True, fastmath=True)  # fastmath: vectorised sums
def evaluate_pos(ctx):
    """Static evaluation in centipawns from the side to move's perspective."""
    state = ctx[C_STATE]
    params = ctx[C_PARAMS]
    mode = params[P_MODE]
    turn = state[S_TURN]
    net = 0
    if mode != 1:
        acc = ctx[C_ACC][state[S_PLY]]
        w1 = ctx[C_W1]
        work = ctx[C_WORK]
        n1 = w1.shape[1]
        for j in range(n1):
            work[j] = acc[turn, j]
        c = state[S_CASTLING]
        if turn == 0:
            ok = c & 1
            oq = (c >> 1) & 1
            tk = (c >> 2) & 1
            tq = (c >> 3) & 1
        else:
            ok = (c >> 2) & 1
            oq = (c >> 3) & 1
            tk = c & 1
            tq = (c >> 1) & 1
        if ok:
            for j in range(n1):
                work[j] += w1[768, j]
        if oq:
            for j in range(n1):
                work[j] += w1[769, j]
        if tk:
            for j in range(n1):
                work[j] += w1[770, j]
        if tq:
            for j in range(n1):
                work[j] += w1[771, j]
        if state[S_EP] >= 0:
            for j in range(n1):
                work[j] += w1[772, j]
        for j in range(n1):
            v = work[j]
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            work[j] = v
        w2 = ctx[C_W2]  # (out, in): contiguous inner loop
        b2 = ctx[C_B2]
        n2 = w2.shape[0]
        o2 = 256
        for k in range(n2):
            s = b2[k]
            for j in range(n1):
                s += work[j] * w2[k, j]
            if s < 0.0:
                s = 0.0
            elif s > 1.0:
                s = 1.0
            work[o2 + k] = s
        w3 = ctx[C_W3]  # (out, in)
        b3 = ctx[C_B3]
        n3 = w3.shape[0]
        o3 = 256 + 32
        for k in range(n3):
            s = b3[k]
            for j in range(n2):
                s += work[o2 + j] * w3[k, j]
            if s < 0.0:
                s = 0.0
            elif s > 1.0:
                s = 1.0
            work[o3 + k] = s
        w4 = ctx[C_W4]
        out = ctx[C_B4][0]
        for j in range(n3):
            out += work[o3 + j] * w4[j]
        cp = out * 100.0
        if cp > 30000.0:
            cp = 30000.0
        elif cp < -30000.0:
            cp = -30000.0
        net = int(cp)
    if mode == 0:
        return net
    hand = hand_eval(ctx[C_BOARD], turn, ctx[C_PST])
    if mode == 1:
        return hand
    w = params[P_BLEND]
    return (net * w + hand * (100 - w)) // 100


_mark("evaluate_pos")


@njit(boolean(int8[:], int64), cache=NUMBA_CACHE, nogil=True)
def has_non_pawn_material(board, turn):
    lo = 2 if turn == 0 else 8
    hi = 5 if turn == 0 else 11
    for sq in range(64):
        p = board[sq]
        if p >= lo and p <= hi:
            return True
    return False


@njit(int64(int8[:], int32), cache=NUMBA_CACHE, nogil=True)
def mvv_lva(board, move):
    """Capture ordering score: victim value first, cheapest attacker first."""
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 15
    flags = (move >> 16) & 15
    victim = 0
    if flags & F_EP:
        victim = 100
    elif flags & F_CAPTURE:
        victim = PIECE_VALUE[board[to]]
    attacker = piece_type(board[frm]) + 1
    score = victim * 10 - attacker
    if promo != 0:
        score += PIECE_VALUE[promo]
    return score


# ---------------------------------------------------------------------- move ordering


@njit((CTX, int64, int64, int32, int64, int64), cache=NUMBA_CACHE, nogil=True)
def score_moves(ctx, start, end, tt_move, ply, turn):
    """Ordering scores: TT move, captures/promotions by MVV-LVA, killers, history."""
    board = ctx[C_BOARD]
    moves = ctx[C_MOVES]
    mscore = ctx[C_MSCORE]
    killers = ctx[C_KILLERS]
    history = ctx[C_HISTORY]
    k0 = killers[ply, 0]
    k1 = killers[ply, 1]
    for i in range(start, end):
        m = moves[i]
        if m == tt_move:
            mscore[i] = 10_000_000
        elif ((m >> 16) & F_CAPTURE) != 0 or ((m >> 12) & 15) != 0:
            mscore[i] = 1_000_000 + mvv_lva(board, m)
        elif m == k0:
            mscore[i] = 900_000
        elif m == k1:
            mscore[i] = 800_000
        else:
            mscore[i] = history[turn, m & 63, (m >> 6) & 63]


@njit(int32(CTX, int64, int64), cache=NUMBA_CACHE, nogil=True)
def pick_move(ctx, i, end):
    """Selection sort step: swap the best remaining move into slot i and return it."""
    moves = ctx[C_MOVES]
    mscore = ctx[C_MSCORE]
    bi = i
    bs = mscore[i]
    for j in range(i + 1, end):
        if mscore[j] > bs:
            bs = mscore[j]
            bi = j
    if bi != i:
        tm = moves[i]
        moves[i] = moves[bi]
        moves[bi] = tm
        mscore[bi] = mscore[i]
        mscore[i] = bs
    return moves[i]


_mark("ordering")


# ------------------------------------------------------------------------- quiescence


@njit(int64(CTX, int64, int64, int64), cache=NUMBA_CACHE, nogil=True)
def quiesce(ctx, alpha, beta, ply):
    stats = ctx[C_STATS]
    stats[ST_NODES] += 1
    stats[ST_QNODES] += 1
    if ctx[C_STOP][0] != 0 or stats[ST_NODES] >= stats[ST_MAX_NODES]:
        stats[ST_ABORT] = 1
        return 0
    if ply > stats[ST_SELDEPTH]:
        stats[ST_SELDEPTH] = ply
    if ply >= MAX_PLY - 2:
        return evaluate_pos(ctx)
    state = ctx[C_STATE]
    board = ctx[C_BOARD]
    moves = ctx[C_MOVES]
    mscore = ctx[C_MSCORE]
    turn = state[S_TURN]
    checked = king_attacked(ctx, turn)
    start = state[S_PLY] * MAX_MOVES
    if checked:
        # evasions: every move, no stand pat
        end = generate(ctx, start, False)
        stand = -INF
        best = -INF
    else:
        stand = evaluate_pos(ctx)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        best = stand
        end = generate(ctx, start, True)
    legal = 0
    for i in range(start, end):
        mscore[i] = mvv_lva(board, moves[i])
    for i in range(start, end):
        m = pick_move(ctx, i, end)
        if not checked:
            # delta pruning
            flags = (m >> 16) & 15
            to = (m >> 6) & 63
            gain = 100 if (flags & F_EP) else PIECE_VALUE[board[to]]
            if ((m >> 12) & 15) != 0:
                gain += 800
            if stand + gain + 200 <= alpha:
                continue
        make(ctx, m)
        if king_attacked(ctx, turn):
            unmake(ctx)
            continue
        legal += 1
        score = -quiesce(ctx, -beta, -alpha, ply + 1)
        unmake(ctx)
        if stats[ST_ABORT] != 0:
            return 0
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    if checked and legal == 0:
        return -MATE + ply
    return best


_mark("quiesce")


# ------------------------------------------------------------------------ main search


@njit(int64(CTX, int64, int64, int64, int64), cache=NUMBA_CACHE, nogil=True)
def search(ctx, depth, alpha, beta, ply):
    stats = ctx[C_STATS]
    stats[ST_NODES] += 1
    if ctx[C_STOP][0] != 0 or stats[ST_NODES] >= stats[ST_MAX_NODES]:
        stats[ST_ABORT] = 1
        return 0
    state = ctx[C_STATE]
    if ply > 0:
        if state[S_HALFMOVE] >= 100 or is_repetition(state, ctx[C_HIST]) != 0:
            return DRAW
        # mate distance pruning
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    turn = state[S_TURN]
    checked = king_attacked(ctx, turn)
    if checked:
        depth += 1
    # at the ply limit quiesce returns the static evaluation straight away
    if depth <= 0 or ply >= MAX_PLY - 2:
        return quiesce(ctx, alpha, beta, ply)

    tt_key = ctx[C_TT_KEY]
    tt_data = ctx[C_TT_DATA]
    key = ctx[C_HASH][0]
    idx = int64(key & uint64(TT_MASK))
    tt_move = int32(0)
    if tt_key[idx] == key:
        data = tt_data[idx]
        tt_move = int32(data & 0xFFFFF)
        tt_depth = (data >> 20) & 0xFF
        tt_flag = (data >> 28) & 3
        tt_score = ((data >> 30) & 0x3FFFF) - SCORE_OFFSET
        if tt_depth >= depth and ply > 0:
            if tt_score > MATE_BOUND:
                tt_score -= ply
            elif tt_score < -MATE_BOUND:
                tt_score += ply
            if (tt_flag == TT_EXACT or (tt_flag == TT_LOWER and tt_score >= beta)
                    or (tt_flag == TT_UPPER and tt_score <= alpha)):
                stats[ST_TT_HITS] += 1
                return tt_score

    pv_node = beta - alpha > 1
    static_eval = -INF
    if not checked and (depth <= 2 or (not pv_node and ply > 0)):
        static_eval = evaluate_pos(ctx)
    if not checked and not pv_node and ply > 0:
        # reverse futility pruning
        if depth <= 3 and static_eval - 120 * depth >= beta and abs(beta) < MATE_BOUND:
            return static_eval
        # null move pruning (not twice in a row: a null move leaves U_MOVE 0 in undo)
        null_ok = ply == 0 or ctx[C_UNDO][ply - 1, U_MOVE] != 0
        if (null_ok and depth >= 2 and static_eval >= beta
                and has_non_pawn_material(ctx[C_BOARD], turn)):
            r = 3 if depth >= 6 else 2
            make_null(ctx)
            score = -search(ctx, depth - 1 - r, -beta, -beta + 1, ply + 1)
            unmake_null(ctx)
            if stats[ST_ABORT] != 0:
                return 0
            if score >= beta and abs(score) < MATE_BOUND:
                return score

    start = state[S_PLY] * MAX_MOVES
    end = generate(ctx, start, False)
    score_moves(ctx, start, end, tt_move, ply, turn)
    killers = ctx[C_KILLERS]
    k0 = killers[ply, 0]
    k1 = killers[ply, 1]

    best = -INF
    best_move = int32(0)
    alpha_orig = alpha
    legal = 0
    for i in range(start, end):
        m = pick_move(ctx, i, end)
        quiet = ((m >> 16) & F_CAPTURE) == 0 and ((m >> 12) & 15) == 0
        make(ctx, m)
        if king_attacked(ctx, turn):
            unmake(ctx)
            continue
        if quiet and king_attacked(ctx, 1 - turn):
            quiet = False  # checking moves are never reduced or pruned
        # futility pruning at frontier nodes
        if (quiet and not checked and depth <= 2 and legal > 0 and abs(alpha) < MATE_BOUND
                and static_eval + 150 * depth <= alpha):
            unmake(ctx)
            continue
        new_depth = depth - 1
        # first attempt: full window for the first move, a null window otherwise, reduced
        # for late quiet moves; a second, full-depth full-window search when it improves
        # alpha (LMR) or lands inside the window (PVS)
        if legal == 0:
            score = -search(ctx, new_depth, -beta, -alpha, ply + 1)
        else:
            d = new_depth
            if legal >= 3 and depth >= 3 and quiet and not checked and m != k0 and m != k1:
                d -= 1 if legal < 8 else 2
            score = -search(ctx, d, -alpha - 1, -alpha, ply + 1)
            if stats[ST_ABORT] == 0 and score > alpha and (d < new_depth or score < beta):
                score = -search(ctx, new_depth, -beta, -alpha, ply + 1)
        unmake(ctx)
        if stats[ST_ABORT] != 0:
            return 0
        legal += 1
        if score > best:
            best = score
            best_move = m
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if quiet:
                        if killers[ply, 0] != m:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = m
                        ctx[C_HISTORY][turn, m & 63, (m >> 6) & 63] += depth * depth
                    break
    if legal == 0:
        return -MATE + ply if checked else DRAW

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
    if tt_key[idx] != key or ((tt_data[idx] >> 20) & 0xFF) <= depth or flag == TT_EXACT:
        tt_key[idx] = key
        tt_data[idx] = (int64(best_move) | (int64(depth) << 20) | (int64(flag) << 28)
                        | (int64(store + SCORE_OFFSET) << 30))
    return best


_mark("search")
