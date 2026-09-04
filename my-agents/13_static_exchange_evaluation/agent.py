"""Static Exchange Evaluation (SEE) chess agent.

Implements my-agents-readmes/13_static_exchange_evaluation.md on top of a plain
iterative-deepening alpha-beta searcher with quiescence.

SEE answers one question without touching the real board: if this capture is played
and both sides keep recapturing on that square with their least valuable piece, each
side free to stop whenever stopping is better, how much material does the mover end up
with? The answer is a centipawn estimate:

    pawn takes an undefended queen        -> +900
    queen takes a pawn defended by a pawn -> -800
    pawn takes pawn, pawn takes back      ->    0

It is used in two places:

    move ordering     captures with SEE >= 0 go first, captures with SEE < 0 go last
    quiescence        captures with SEE < 0 are not searched at all

The swap-list algorithm below handles x-ray attackers (a rook behind a rook joins the
exchange once the front rook has moved), absolutely pinned pieces (they cannot
recapture unless the target lies on the pin line), promotions (a capture onto the last
rank turns the pawn into a queen), en passant, and king legality (a king may only
recapture when the square is no longer attacked).

Lives at my-agents/13_static_exchange_evaluation/agent.py so the harness can import it.
"""

import math
import time

import chess

# ---------------------------------------------------------------------------
# 1. Piece values and evaluation
# ---------------------------------------------------------------------------

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# The king can never actually be captured inside SEE (legality is enforced), but it
# needs a value for the swap list arithmetic. Anything larger than a queen works.
SEE_VALUE: dict[chess.PieceType, int] = {**PIECE_VALUE, chess.KING: 20_000}

# Attacker order for the exchange: cheapest piece first.
_ATTACKER_ORDER: tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

# Piece-square tables, written as they appear on a diagram with White at the bottom
# (rank 8 first). They are flipped below so index 0 is a1, matching python-chess.
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

MATE_SCORE = 100_000
INFINITY = math.inf


def evaluate(board: chess.Board) -> int:
    """Material plus piece-square tables, from the side to move's point of view."""
    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type] + PST[piece.color][piece.piece_type][square]
        score += value if piece.color == board.turn else -value
    return score


# ---------------------------------------------------------------------------
# 2. Static Exchange Evaluation
# ---------------------------------------------------------------------------


def _is_pinned(
    board: chess.Board, color: chess.Color, square: chess.Square, occupied: int, target_bb: int
) -> bool:
    """Would moving the piece on `square` to the target expose its own king?

    Evaluated against the simulated occupancy, so a pin disappears once the pinning
    piece has itself been spent in the exchange. A pin by the piece standing on the
    target square never counts: capturing the pinner along the pin line is legal.
    """
    king = board.king(color)
    if king is None:
        return False
    enemy_sliders = (
        (board.rooks | board.bishops | board.queens)
        & board.occupied_co[not color]
        & occupied
        & ~target_bb
    )
    if not enemy_sliders:
        return False
    before = board.attackers_mask(not color, king, occupied) & enemy_sliders
    without = occupied ^ chess.BB_SQUARES[square]
    after = board.attackers_mask(not color, king, without) & enemy_sliders
    return bool(after & ~before)


def _least_valuable_attacker(
    board: chess.Board, side: chess.Color, target: chess.Square, occupied: int
) -> tuple[chess.Square, chess.PieceType] | None:
    """Cheapest piece of `side` that can legally capture on `target` right now.

    `occupied` is the simulated occupancy: pieces already used in the exchange have
    been removed, which is what lets x-ray attackers behind them show up.
    """
    target_bb = chess.BB_SQUARES[target]
    attackers = board.attackers_mask(side, target, occupied) & occupied
    if not attackers:
        return None
    for piece_type in _ATTACKER_ORDER:
        candidates = attackers & board.pieces_mask(piece_type, side)
        while candidates:
            square = chess.lsb(candidates)
            candidates &= candidates - 1
            if piece_type == chess.KING:
                # A king may only capture if nobody can capture it back.
                after = occupied ^ chess.BB_SQUARES[square]
                if board.attackers_mask(not side, target, after) & after:
                    return None
                return square, piece_type
            if not _is_pinned(board, side, square, occupied, target_bb):
                return square, piece_type
    return None


def see(board: chess.Board, move: chess.Move) -> int:
    """Centipawn estimate of the exchange started by `move`, for the side to move.

    Positive is good for the mover. The real board is never modified: the exchange is
    simulated on a bitboard of occupied squares. Works for quiet moves too, in which
    case it tells whether the moved piece is simply hanging on its new square.
    """
    target = move.to_square
    us = board.turn
    attacker_type = board.piece_type_at(move.from_square)
    if attacker_type is None:
        return 0

    occupied = board.occupied & ~chess.BB_SQUARES[move.from_square]
    if board.is_en_passant(move):
        captured_square = target - 8 if us == chess.WHITE else target + 8
        occupied &= ~chess.BB_SQUARES[captured_square]
        victim_value = SEE_VALUE[chess.PAWN]
    else:
        victim = board.piece_type_at(target)
        victim_value = SEE_VALUE[victim] if victim is not None else 0

    # gain[d] is the material the side making capture d ends up with if the other side
    # then stops. It is filled in speculatively and corrected on the way back.
    gain: list[int] = [victim_value]
    on_square = SEE_VALUE[attacker_type]  # value of the piece now sitting on the target
    if move.promotion:
        gain[0] += SEE_VALUE[move.promotion] - SEE_VALUE[chess.PAWN]
        on_square = SEE_VALUE[move.promotion]

    side = not us
    last_rank = chess.BB_RANK_1 | chess.BB_RANK_8
    while True:
        next_attacker = _least_valuable_attacker(board, side, target, occupied)
        if next_attacker is None:
            break
        square, piece_type = next_attacker
        bonus = 0
        value = SEE_VALUE[piece_type]
        if piece_type == chess.PAWN and chess.BB_SQUARES[target] & last_rank:
            bonus = SEE_VALUE[chess.QUEEN] - SEE_VALUE[chess.PAWN]
            value = SEE_VALUE[chess.QUEEN]
        gain.append(on_square + bonus - gain[-1])
        on_square = value
        occupied ^= chess.BB_SQUARES[square]
        side = not side

    # Each side may stop when continuing would be worse than stopping.
    for depth in range(len(gain) - 1, 0, -1):
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
    return gain[0]


# ---------------------------------------------------------------------------
# 3. Move ordering
# ---------------------------------------------------------------------------

USE_SEE_ORDERING = True  # captures sorted by SEE, losing captures searched last
USE_SEE_QS_PRUNING = True  # quiescence skips captures with SEE < 0


def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_type_at(move.to_square)
    victim_value = PIECE_VALUE[victim] if victim else PIECE_VALUE[chess.PAWN]
    attacker = board.piece_type_at(move.from_square)
    attacker_value = PIECE_VALUE[attacker] if attacker else 0
    return 10 * victim_value - attacker_value


def _move_priority(board: chess.Board, move: chess.Move) -> int:
    """Higher is searched earlier.

    good captures (SEE >= 0)   30_000 + SEE, ties broken by MVV-LVA
    promotions                 20_000 + promoted piece value
    quiet moves                10_000 + piece-square gain
    bad captures (SEE < 0)     SEE (negative, so always last)
    """
    if board.is_capture(move):
        if USE_SEE_ORDERING:
            exchange = see(board, move)
            if exchange >= 0:
                return 30_000 + exchange * 16 + _mvv_lva(board, move)
            return exchange
        return 30_000 + _mvv_lva(board, move)
    if move.promotion:
        return 20_000 + PIECE_VALUE[move.promotion]
    piece = board.piece_type_at(move.from_square)
    if piece is None:
        return 10_000
    table = PST[board.turn][piece]
    return 10_000 + table[move.to_square] - table[move.from_square]


def ordered_moves(board: chess.Board) -> list[chess.Move]:
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: _move_priority(board, move), reverse=True)
    return moves


def tactical_moves(board: chess.Board) -> list[chess.Move]:
    """Captures and promotions for quiescence, best exchange first.

    With USE_SEE_QS_PRUNING, captures that lose material outright (SEE < 0) are
    dropped: they cannot raise the score above stand pat, so searching them only
    burns nodes. SEE is computed once per capture here rather than again per move.
    """
    scored: list[tuple[int, chess.Move]] = []
    for move in board.legal_moves:
        if board.is_capture(move):
            exchange = see(board, move) if USE_SEE_ORDERING or USE_SEE_QS_PRUNING else 0
            if USE_SEE_QS_PRUNING and exchange < 0:
                continue
            scored.append((exchange * 16 + _mvv_lva(board, move), move))
        elif move.promotion:
            scored.append((PIECE_VALUE[move.promotion], move))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [move for _, move in scored]


# ---------------------------------------------------------------------------
# 4. Search: negamax, alpha-beta, quiescence
# ---------------------------------------------------------------------------


class OutOfTime(Exception):
    """Raised inside the search when the move budget is spent."""


class Searcher:
    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.nodes = 0

    def _tick(self) -> None:
        # Nodes are slow in Python, so poll the clock often enough that the overrun
        # past the deadline stays well under the per-move increment.
        self.nodes += 1
        if self.nodes & 127 == 0 and time.monotonic() > self.deadline:
            raise OutOfTime

    def quiescence(self, board: chess.Board, alpha: float, beta: float) -> float:
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

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        self._tick()
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0.0

        moves = ordered_moves(board)
        if not moves:
            return -(MATE_SCORE + depth) if board.is_check() else 0.0
        if depth == 0:
            return self.quiescence(board, alpha, beta)

        best = -INFINITY
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def search_root(self, board: chess.Board, depth: int) -> tuple[chess.Move, float]:
        best_move: chess.Move | None = None
        best_score = -INFINITY
        alpha, beta = -INFINITY, INFINITY
        for move in ordered_moves(board):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        assert best_move is not None, "search_root called with no legal moves"
        return best_move, best_score


# ---------------------------------------------------------------------------
# 5. Iterative deepening and time management
# ---------------------------------------------------------------------------

MAX_DEPTH = 64
MOVES_TO_GO = 30
MIN_BUDGET_MS = 50
MAX_BUDGET_MS = 8_000


def move_budget_ms(time_left_ms: int) -> int:
    budget = time_left_ms // MOVES_TO_GO
    return max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, budget))


def choose_move(board: chess.Board, time_left_ms: int) -> chess.Move:
    started = time.monotonic()
    budget_s = move_budget_ms(time_left_ms) / 1000.0
    searcher = Searcher(started + budget_s)

    best_move, best_score = searcher.search_root(board, 1)
    for depth in range(2, MAX_DEPTH + 1):
        # A new iteration costs several times the previous one. If more than half
        # the budget is gone it will almost surely be aborted, so stop here instead.
        if time.monotonic() - started > budget_s / 2:
            break
        try:
            best_move, best_score = searcher.search_root(board, depth)
        except OutOfTime:
            break
        print(
            f"depth {depth:2d}  score {best_score:8.0f}  best {best_move.uci()}  "
            f"nodes {searcher.nodes}"
        )
        if abs(best_score) >= MATE_SCORE:
            break
    return best_move


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    return choose_move(board, time_left_ms).uci()
