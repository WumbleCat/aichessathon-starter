"""Chess agent: alpha-beta negamax with a transposition table keyed by Zobrist hashes.

Implements my-agents-readmes/07_transposition_table_zobrist.md on top of the earlier
steps (negamax, alpha-beta, piece-square evaluation, quiescence, iterative deepening,
MVV-LVA move ordering):

1. Zobrist hashing        a 64-bit key over piece/colour/square, side to move, castling
                          rights and the en-passant file, kept up to date incrementally
                          as moves are pushed and popped
2. transposition table    a fixed-size table of (key, depth, score, flag, best move, age)
                          entries; probed before every node, stored after every node
3. bound reuse            EXACT scores are returned outright, LOWERBOUND raises alpha,
                          UPPERBOUND lowers beta, and a cached best move goes first in
                          move ordering even when the score cannot be reused
4. mate normalisation     mate scores are stored relative to the node, not the root, so
                          the same position found at a different ply means the same thing
5. replacement            an empty slot, an entry from an older search (generation), or
                          an entry of equal or lesser depth is overwritten

Lives at my-agents/07_transposition_table_zobrist/agent.py so the harness can import it.
To submit, copy it to agent.py at the root of the repo (make zip puts that file at the
root of the zip).
"""

from __future__ import annotations

import math
import random
import time

import chess

# ---------------------------------------------------------------------------
# Evaluation: material plus piece-square tables, from the mover's point of view
# ---------------------------------------------------------------------------

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables written as a diagram with White at the bottom (rank 8 first).
# They are flipped below so index 0 is a1, matching python-chess square numbering.
_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]
_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]
_KING = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]


def _from_diagram(table: list[int]) -> list[int]:
    """Reorder a diagram-style table (rank 8 first) into a1..h8 square order."""
    return [table[chess.square_mirror(square)] for square in chess.SQUARES]


_WHITE_PST: dict[chess.PieceType, list[int]] = {
    chess.PAWN: _from_diagram(_PAWN),
    chess.KNIGHT: _from_diagram(_KNIGHT),
    chess.BISHOP: _from_diagram(_BISHOP),
    chess.ROOK: _from_diagram(_ROOK),
    chess.QUEEN: _from_diagram(_QUEEN),
    chess.KING: _from_diagram(_KING),
}
_BLACK_PST: dict[chess.PieceType, list[int]] = {
    piece: [table[chess.square_mirror(square)] for square in chess.SQUARES]
    for piece, table in _WHITE_PST.items()
}
PST: dict[chess.Color, dict[chess.PieceType, list[int]]] = {
    chess.WHITE: _WHITE_PST,
    chess.BLACK: _BLACK_PST,
}


# PST_VALUE[color][piece_type][square] = material + square bonus, so evaluation is one
# table lookup per piece. chess.BLACK is False and chess.WHITE is True, so the list is
# built in that order to be indexable by colour.
PST_VALUE: list[list[list[int]]] = [
    [[0] * 64]
    + [
        [PIECE_VALUE[piece] + PST[color][piece][square] for square in chess.SQUARES]
        for piece in chess.PIECE_TYPES
    ]
    for color in (chess.BLACK, chess.WHITE)
]
_PIECE_BITBOARDS = ("pawns", "knights", "bishops", "rooks", "queens", "kings")


def evaluate(board: chess.Board) -> int:
    """Static score from the perspective of the side to move. Positive: mover is better."""
    score = 0
    turn = board.turn
    for piece, name in enumerate(_PIECE_BITBOARDS, start=1):
        pieces: chess.Bitboard = getattr(board, name)
        table = PST_VALUE[turn][piece]
        for square in chess.scan_forward(pieces & board.occupied_co[turn]):
            score += table[square]
        table = PST_VALUE[not turn][piece]
        for square in chess.scan_forward(pieces & board.occupied_co[not turn]):
            score -= table[square]
    return score


# ---------------------------------------------------------------------------
# 1. Zobrist hashing
# ---------------------------------------------------------------------------
#
# Every feature of a position that matters to the set of legal continuations gets its
# own 64-bit random number. A position's key is the XOR of the numbers for the features
# it has. XOR is its own inverse, so a move updates the key by XORing out the features
# it removes and XORing in the features it adds; undoing the move repeats the same XORs.
#
# The random numbers are drawn from a fixed seed so keys are identical across runs.

_ZOBRIST_SEED = 0x5EED_C0DE_1234_5678
_MASK64 = (1 << 64) - 1

_rng = random.Random(_ZOBRIST_SEED)

# ZOBRIST_PIECE[color][piece_type][square]. Index 0 of the piece axis is unused so the
# table can be indexed directly by chess.PAWN..chess.KING (1..6).
ZOBRIST_PIECE: list[list[list[int]]] = [
    [[_rng.getrandbits(64) for _ in chess.SQUARES] for _ in range(7)] for _ in range(2)
]
ZOBRIST_SIDE: int = _rng.getrandbits(64)  # XORed in when Black is to move
# One number per castling right, keyed by the rook's home square (a1, h1, a8, h8).
ZOBRIST_CASTLING: dict[chess.Square, int] = {
    square: _rng.getrandbits(64) for square in (chess.A1, chess.H1, chess.A8, chess.H8)
}
# One number per en-passant file. Only used when an en-passant capture is actually legal,
# so two positions that differ only in a meaningless ep marker share a key.
ZOBRIST_EP: list[int] = [_rng.getrandbits(64) for _ in range(8)]
del _rng


def _castling_hash(castling_rights: chess.Bitboard) -> int:
    key = 0
    for square, value in ZOBRIST_CASTLING.items():
        if castling_rights & chess.BB_SQUARES[square]:
            key ^= value
    return key


def _ep_hash(board: chess.Board) -> int:
    if board.ep_square is not None and board.has_legal_en_passant():
        return ZOBRIST_EP[chess.square_file(board.ep_square)]
    return 0


def zobrist_key(board: chess.Board) -> int:
    """Compute the full key of a position from scratch."""
    key = 0
    for square, piece in board.piece_map().items():
        key ^= ZOBRIST_PIECE[piece.color][piece.piece_type][square]
    if board.turn == chess.BLACK:
        key ^= ZOBRIST_SIDE
    key ^= _castling_hash(board.castling_rights)
    key ^= _ep_hash(board)
    return key


def zobrist_push(board: chess.Board, key: int, move: chess.Move) -> int:
    """Push `move` onto `board` and return the key of the resulting position.

    Updates the key incrementally from the parts of the position the move touches:
    the moving piece, a captured piece (including the en-passant victim), a promotion,
    the rook of a castling move, castling rights, en-passant state and side to move.
    """
    mover = board.turn
    piece_type = board.piece_type_at(move.from_square)
    assert piece_type is not None, "no piece on the from-square"
    key ^= ZOBRIST_PIECE[mover][piece_type][move.from_square]

    captured = board.piece_type_at(move.to_square)
    if captured is not None:
        key ^= ZOBRIST_PIECE[not mover][captured][move.to_square]
    elif piece_type == chess.PAWN and move.to_square == board.ep_square:
        # En passant: the captured pawn sits behind the destination square.
        victim_square = move.to_square + (-8 if mover == chess.WHITE else 8)
        key ^= ZOBRIST_PIECE[not mover][chess.PAWN][victim_square]

    landing_piece = move.promotion if move.promotion else piece_type
    key ^= ZOBRIST_PIECE[mover][landing_piece][move.to_square]

    if piece_type == chess.KING and board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            rook_from, rook_to = chess.square(7, rank), chess.square(5, rank)
        else:
            rook_from, rook_to = chess.square(0, rank), chess.square(3, rank)
        key ^= ZOBRIST_PIECE[mover][chess.ROOK][rook_from]
        key ^= ZOBRIST_PIECE[mover][chess.ROOK][rook_to]

    # Castling rights and en-passant state depend on the whole position, so read them
    # before and after the push and XOR out the old and in the new.
    key ^= _castling_hash(board.castling_rights) ^ _ep_hash(board)
    board.push(move)
    key ^= _castling_hash(board.castling_rights) ^ _ep_hash(board)
    key ^= ZOBRIST_SIDE
    return key


# ---------------------------------------------------------------------------
# 2. Transposition table
# ---------------------------------------------------------------------------

EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

MATE_SCORE = 100_000  # bigger than any material swing
MATE_BOUND = MATE_SCORE - 1_000  # scores beyond this are "mate in n"
INFINITY = math.inf

TT_BITS = 20  # 2**20 slots, about 1 million entries


class TTEntry:
    """One cached search result. `score` is stored root-independent (see mate notes)."""

    __slots__ = ("age", "best_move", "depth", "flag", "key", "score")

    def __init__(
        self,
        key: int,
        depth: int,
        score: float,
        flag: int,
        best_move: chess.Move | None,
        age: int,
    ) -> None:
        self.key = key
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move
        self.age = age


class TranspositionTable:
    """A fixed-size table indexed by the low bits of the Zobrist key.

    The full key is stored in each entry and checked on probe, so a slot collision
    between two different positions is detected rather than silently trusted.
    """

    def __init__(self, bits: int = TT_BITS) -> None:
        self.mask = (1 << bits) - 1
        self.slots: list[TTEntry | None] = [None] * (1 << bits)
        self.generation = 0
        self.hits = 0
        self.stores = 0

    def new_search(self) -> None:
        """Called once per root search so old entries become preferred for replacement."""
        self.generation += 1

    def clear(self) -> None:
        self.slots = [None] * (self.mask + 1)
        self.generation = 0
        self.hits = self.stores = 0

    def probe(self, key: int) -> TTEntry | None:
        entry = self.slots[key & self.mask]
        if entry is not None and entry.key == key:
            self.hits += 1
            return entry
        return None

    def store(
        self,
        key: int,
        depth: int,
        score: float,
        flag: int,
        best_move: chess.Move | None,
    ) -> None:
        index = key & self.mask
        old = self.slots[index]
        # Replace an empty slot, an entry from an earlier search, or one that was
        # searched no deeper than this one. A deeper entry from this search survives.
        if old is None or old.age != self.generation or depth >= old.depth:
            self.slots[index] = TTEntry(key, depth, score, flag, best_move, self.generation)
            self.stores += 1
        elif old.key == key and best_move is not None and old.best_move is None:
            old.best_move = best_move


# Mate scores are "MATE_SCORE - ply" from the root: a mate found at ply 5 scores less
# than a mate found at ply 3, so the search prefers the quicker one. That number depends
# on where the root is, and the same position can be reached at different plies, so the
# table stores mates as a distance from the *node* instead. Converting on the way in
# and back out keeps the stored value meaningful wherever it is later probed.


def score_to_tt(score: float, ply: int) -> float:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def score_from_tt(score: float, ply: int) -> float:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


# ---------------------------------------------------------------------------
# 3. Move ordering: TT move first, then MVV-LVA captures and promotions
# ---------------------------------------------------------------------------


def _move_priority(board: chess.Board, move: chess.Move) -> int:
    """Higher is searched earlier. Good captures and promotions first."""
    priority = 0
    if move.promotion:
        priority += 10_000 + PIECE_VALUE[move.promotion]
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square)
        # An en-passant capture lands on an empty square; the victim is a pawn.
        victim_value = PIECE_VALUE[victim] if victim else PIECE_VALUE[chess.PAWN]
        attacker = board.piece_type_at(move.from_square)
        attacker_value = PIECE_VALUE[attacker] if attacker else 0
        priority += 1_000 + 10 * victim_value - attacker_value
    return priority


TT_MOVE_PRIORITY = 1_000_000  # outranks every capture and promotion

_PROMOTION_SQUARES = chess.BB_RANK_1 | chess.BB_RANK_8


def tactical_moves(board: chess.Board) -> list[chess.Move]:
    """Captures and promotions only, best guesses first, for quiescence search.

    Generating just these is several times cheaper than generating every legal move
    and discarding the quiet ones, and quiescence is where most nodes are.
    """
    moves = list(board.generate_legal_captures())
    pawns = board.pawns & board.occupied_co[board.turn]
    if pawns & (chess.BB_RANK_7 if board.turn == chess.WHITE else chess.BB_RANK_2):
        moves.extend(
            move
            for move in board.generate_legal_moves(pawns, _PROMOTION_SQUARES)
            if not board.is_capture(move)
        )
    moves.sort(key=lambda move: _move_priority(board, move), reverse=True)
    return moves


def ordered_moves(board: chess.Board, tt_move: chess.Move | None = None) -> list[chess.Move]:
    """All legal moves, best guesses first. The cached best move, if any, goes first."""
    moves = list(board.legal_moves)
    moves.sort(
        key=lambda move: TT_MOVE_PRIORITY if move == tt_move else _move_priority(board, move),
        reverse=True,
    )
    return moves


# ---------------------------------------------------------------------------
# 4. Search: negamax with alpha-beta, quiescence and the transposition table
# ---------------------------------------------------------------------------


class OutOfTime(Exception):
    """Raised inside the search when the move budget is spent."""


class Searcher:
    """One search for one root position. Holds the deadline, the table and counters."""

    def __init__(self, table: TranspositionTable, deadline: float, use_tt: bool = True) -> None:
        self.table = table
        self.deadline = deadline
        self.use_tt = use_tt
        self.nodes = 0
        self.tt_cutoffs = 0
        self.fallback: chess.Move | None = None  # best root move seen so far, any depth

    def _tick(self) -> None:
        # A node costs tens of microseconds here, so 64 nodes is about the finest
        # granularity worth paying for; 2048 would overshoot the budget by hundreds of
        # milliseconds and lose games on the clock.
        self.nodes += 1
        if self.nodes & 63 == 0 and time.monotonic() > self.deadline:
            raise OutOfTime

    def quiescence(self, board: chess.Board, alpha: float, beta: float) -> float:
        """Search only captures and promotions until the position is quiet."""
        self._tick()
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)

        for move in tactical_moves(board):
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def negamax(
        self,
        board: chess.Board,
        key: int,
        depth: int,
        ply: int,
        alpha: float,
        beta: float,
    ) -> float:
        """Best score the side to move can force within `depth` plies.

        `key` is the Zobrist key of `board`, maintained by the caller. `ply` is the
        distance from the root, used for mate scoring.
        """
        self._tick()

        # Draw by repetition or the fifty-move rule. These depend on the path, not the
        # position, so they are checked before the table is consulted.
        if ply > 0 and (board.is_repetition(2) or board.halfmove_clock >= 100):
            return 0.0

        # --- probe -------------------------------------------------------------
        original_alpha = alpha
        tt_move: chess.Move | None = None
        if self.use_tt:
            entry = self.table.probe(key)
            if entry is not None:
                tt_move = entry.best_move
                if entry.depth >= depth and ply > 0:
                    score = score_from_tt(entry.score, ply)
                    if entry.flag == EXACT:
                        self.tt_cutoffs += 1
                        return score
                    if entry.flag == LOWERBOUND:
                        alpha = max(alpha, score)
                    elif entry.flag == UPPERBOUND:
                        beta = min(beta, score)
                    if alpha >= beta:
                        self.tt_cutoffs += 1
                        return score

        moves = ordered_moves(board, tt_move)
        if not moves:
            # Checkmated: losing, and sooner is worse than later. Stalemate: draw.
            return -(MATE_SCORE - ply) if board.is_check() else 0.0

        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        # --- search ------------------------------------------------------------
        best_score = -INFINITY
        best_move: chess.Move | None = None
        for move in moves:
            child_key = zobrist_push(board, key, move)
            score = -self.negamax(board, child_key, depth - 1, ply + 1, -beta, -alpha)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # the opponent will never allow this line; stop looking

        # --- store -------------------------------------------------------------
        if self.use_tt:
            if best_score <= original_alpha:
                flag = UPPERBOUND  # every move failed low: the true score is at most this
            elif best_score >= beta:
                flag = LOWERBOUND  # a cutoff: the true score is at least this
            else:
                flag = EXACT
            self.table.store(key, depth, score_to_tt(best_score, ply), flag, best_move)
        return best_score

    def search_root(self, board: chess.Board, depth: int) -> tuple[chess.Move, float]:
        """Negamax at the root, remembering which move produced the best score.

        The root never takes a table cutoff (it must produce a move), but it does use
        the table's best move from the previous iteration to order its moves, which is
        where most of iterative deepening's ordering benefit comes from.
        """
        key = zobrist_key(board)
        tt_move: chess.Move | None = None
        if self.use_tt:
            entry = self.table.probe(key)
            if entry is not None:
                tt_move = entry.best_move

        best_move: chess.Move | None = None
        best_score = -INFINITY
        alpha, beta = -INFINITY, INFINITY

        moves = ordered_moves(board, tt_move)
        assert moves, "search_root called with no legal moves"
        if self.fallback is None:
            self.fallback = moves[0]  # something legal to play if even depth 1 is cut short
        for move in moves:
            child_key = zobrist_push(board, key, move)
            score = -self.negamax(board, child_key, depth - 1, 1, -beta, -alpha)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
                self.fallback = move
            alpha = max(alpha, score)

        assert best_move is not None
        if self.use_tt:
            self.table.store(key, depth, score_to_tt(best_score, 0), EXACT, best_move)
        return best_move, best_score


# ---------------------------------------------------------------------------
# 5. Iterative deepening and time management
# ---------------------------------------------------------------------------

MAX_DEPTH = 64
MOVES_TO_GO = 30  # assume the game lasts about this many more of our moves
MIN_BUDGET_MS = 20
MAX_BUDGET_MS = 8_000
# Each iteration costs several times the one before, so an iteration started with less
# than this fraction of the budget left would almost certainly be aborted anyway.
NEW_ITERATION_FRACTION = 0.5

# The table lives on the module so it survives from one move to the next within a game.
# The process is fresh for every game, so nothing leaks between games.
TABLE = TranspositionTable()


def move_budget_ms(time_left_ms: int) -> int:
    """How long to think on this move. Spend a slice of what is left, never all of it."""
    budget = time_left_ms // MOVES_TO_GO
    return max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, budget))


def choose_move(
    board: chess.Board,
    time_left_ms: int,
    table: TranspositionTable = TABLE,
    max_depth: int = MAX_DEPTH,
    verbose: bool = True,
) -> chess.Move:
    """Deepen one ply at a time until the budget runs out; keep the last finished depth."""
    started = time.monotonic()
    budget_s = move_budget_ms(time_left_ms) / 1000.0
    deadline = started + budget_s
    table.new_search()
    searcher = Searcher(table, deadline)

    # Depth 1 is normally instant. If the clock is so low that even it is cut short,
    # play the best root move seen so far rather than lose on time.
    try:
        best_move, best_score = searcher.search_root(board, 1)
    except OutOfTime:
        assert searcher.fallback is not None
        return searcher.fallback

    for depth in range(2, max_depth + 1):
        if time.monotonic() - started > NEW_ITERATION_FRACTION * budget_s:
            break
        try:
            best_move, best_score = searcher.search_root(board, depth)
        except OutOfTime:
            break
        if verbose:
            print(
                f"depth {depth:2d}  score {best_score:8.0f}  best {best_move.uci()}  "
                f"nodes {searcher.nodes}  tt hits {table.hits}  tt cutoffs {searcher.tt_cutoffs}"
            )
        if abs(best_score) >= MATE_BOUND:
            break  # a forced mate was found; deeper search cannot improve it
    return best_move


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    return choose_move(board, time_left_ms).uci()
