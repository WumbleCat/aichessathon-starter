"""Quiescence-search chess agent.

Implements my-agents-readmes/04_quiescence_search.md on top of a plain negamax /
alpha-beta search with a handcrafted material + piece-square evaluation.

The point of this bot is the leaf handling. A fixed-depth search that calls the
static evaluation at depth 0 suffers from the horizon effect: it stops in the middle
of an exchange and happily reports "queen takes rook, +500" when the rook was
defended and the queen is lost on the very next ply. Quiescence search keeps
resolving tactical moves (captures and promotions) past the nominal depth until the
position is quiet, and only then trusts the static evaluation.

Rules the quiescence search follows here:

- stand pat: the mover may decline all captures and take the static score
- if the side to move is in check there is no stand pat; every legal evasion is
  searched so the engine can never "evaluate its way out" of a check
- captures are ordered most-valuable-victim / least-valuable-attacker
- only queen promotions are generated as tactical moves, to keep the tree small
- a hard ply cap guarantees termination even in pathological positions
- qnodes are counted separately from normal search nodes

Lives at my-agents/04_quiescence_search/agent.py so the harness can import it.
"""

from __future__ import annotations

import time

import chess

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USE_QUIESCENCE = True  # flip to False to see the horizon effect come back
QS_MAX_PLY = 32  # hard cap on quiescence depth; captures are finite but be safe
QS_INCLUDE_CHECKS = False  # checking moves in quiescence are left for a later stage

MAX_DEPTH = 64
MOVES_TO_GO = 30  # spend roughly 1/30 of the remaining clock per move
MIN_BUDGET_MS = 40
MAX_BUDGET_MS = 6_000

# ---------------------------------------------------------------------------
# Evaluation: material + piece-square tables, side-to-move perspective
# ---------------------------------------------------------------------------

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

MATE_SCORE = 100_000  # mate found now; mates further away score slightly less
MATE_BOUND = MATE_SCORE - 1_000  # anything beyond this magnitude is a mate score
INFINITY = 10 * MATE_SCORE

# Tables are written as seen on a diagram, White at the bottom (rank 8 first), and are
# flipped below so that index 0 is a1, which is python-chess' square numbering.
# fmt: off
_PAWN_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
_BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
_ROOK_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
]
_QUEEN_TABLE = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
_KING_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]
# fmt: on


def _diagram_to_a1(table: list[int]) -> list[int]:
    """Reorder a rank-8-first diagram into a1..h8 order."""
    return [table[chess.square_mirror(square)] for square in chess.SQUARES]


_WHITE_PST: dict[chess.PieceType, list[int]] = {
    chess.PAWN: _diagram_to_a1(_PAWN_TABLE),
    chess.KNIGHT: _diagram_to_a1(_KNIGHT_TABLE),
    chess.BISHOP: _diagram_to_a1(_BISHOP_TABLE),
    chess.ROOK: _diagram_to_a1(_ROOK_TABLE),
    chess.QUEEN: _diagram_to_a1(_QUEEN_TABLE),
    chess.KING: _diagram_to_a1(_KING_TABLE),
}
_BLACK_PST: dict[chess.PieceType, list[int]] = {
    piece: [table[chess.square_mirror(square)] for square in chess.SQUARES]
    for piece, table in _WHITE_PST.items()
}
# PST[colour][piece_type][square]: material plus square bonus, precombined.
PST: dict[chess.Color, dict[chess.PieceType, list[int]]] = {
    chess.WHITE: {p: [PIECE_VALUE[p] + v for v in t] for p, t in _WHITE_PST.items()},
    chess.BLACK: {p: [PIECE_VALUE[p] + v for v in t] for p, t in _BLACK_PST.items()},
}


_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def evaluate(board: chess.Board) -> int:
    """Static score in centipawns from the perspective of the side to move.

    Walks the piece bitboards directly rather than building a piece map: this runs at
    every quiescence node, so it is the hottest function in the engine.
    """
    white_pst = PST[chess.WHITE]
    black_pst = PST[chess.BLACK]
    score = 0
    for piece_type in _PIECE_TYPES:
        table = white_pst[piece_type]
        for square in chess.scan_forward(board.pieces_mask(piece_type, chess.WHITE)):
            score += table[square]
        table = black_pst[piece_type]
        for square in chess.scan_forward(board.pieces_mask(piece_type, chess.BLACK)):
            score -= table[square]
    return score if board.turn == chess.WHITE else -score


# ---------------------------------------------------------------------------
# Move generation and ordering
# ---------------------------------------------------------------------------


def _victim_value(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_type_at(move.to_square)
    if victim is None:
        # En passant: the captured pawn is not on the destination square.
        return PIECE_VALUE[chess.PAWN] if board.is_en_passant(move) else 0
    return PIECE_VALUE[victim]


def mvv_lva(board: chess.Board, move: chess.Move) -> int:
    """Most valuable victim, least valuable attacker. Higher means search earlier."""
    attacker = board.piece_type_at(move.from_square)
    attacker_value = PIECE_VALUE[attacker] if attacker is not None else 0
    score = 10 * _victim_value(board, move) - attacker_value
    if move.promotion is not None:
        score += 10_000 + PIECE_VALUE[move.promotion]
    return score


_SEVENTH_RANKS = chess.BB_RANK_7 | chess.BB_RANK_2
_BACK_RANKS = chess.BB_RANK_8 | chess.BB_RANK_1


def tactical_moves(board: chess.Board) -> list[chess.Move]:
    """Legal captures and queen promotions, best-looking first.

    Captures come straight from python-chess' capture generator. Promotions are
    generated only for pawns on their seventh rank; underpromotions are dropped so
    each promoting pawn adds one move, not four.
    """
    moves = [
        move
        for move in board.generate_legal_captures()
        if move.promotion is None or move.promotion == chess.QUEEN
    ]
    pawns_about_to_promote = board.pawns & board.occupied_co[board.turn] & _SEVENTH_RANKS
    if pawns_about_to_promote:
        for move in board.generate_legal_moves(pawns_about_to_promote, _BACK_RANKS):
            if move.promotion == chess.QUEEN and not board.is_capture(move):
                moves.append(move)
    moves.sort(key=lambda move: mvv_lva(board, move), reverse=True)
    return moves


def _full_move_priority(board: chess.Board, move: chess.Move) -> int:
    if move.promotion is not None or board.is_capture(move):
        return 1_000_000 + mvv_lva(board, move)
    return 0


def ordered_moves(board: chess.Board) -> list[chess.Move]:
    """All legal moves; tactical ones first so alpha-beta cuts early."""
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: _full_move_priority(board, move), reverse=True)
    return moves


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class OutOfTime(Exception):
    """Raised from inside the search once the move budget is spent."""


class Searcher:
    """One search from one root position. Owns the deadline and the node counters."""

    def __init__(self, deadline: float | None = None, use_quiescence: bool = USE_QUIESCENCE):
        self.deadline = deadline
        self.use_quiescence = use_quiescence
        self.nodes = 0  # negamax nodes
        self.qnodes = 0  # quiescence nodes, tracked separately as the readme asks
        self.max_qply = 0  # deepest quiescence ply reached, for diagnostics

    def _check_time(self) -> None:
        total = self.nodes + self.qnodes
        if self.deadline is not None and total & 511 == 0 and time.monotonic() > self.deadline:
            raise OutOfTime

    # -- quiescence --------------------------------------------------------

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int = 0) -> int:
        """Resolve captures and promotions until the position is quiet.

        alpha/beta are from the mover's point of view. Returns a fail-hard score in
        [alpha, beta] except for mate scores, which are exact.
        """
        self.qnodes += 1
        self._check_time()
        if ply > self.max_qply:
            self.max_qply = ply

        in_check = board.is_check()

        if in_check:
            # Stand pat is meaningless: the mover has to get out of check. Search every
            # legal evasion, and if there are none it is checkmate.
            moves = ordered_moves(board)
            if not moves:
                return -(MATE_SCORE - ply)
            if ply >= QS_MAX_PLY:
                # Depth cap reached while still in check. Evasions exist, so the position
                # is at least not mate; fall back to the static score.
                return evaluate(board)
        else:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
            if ply >= QS_MAX_PLY:
                return alpha
            moves = tactical_moves(board)

        for move in moves:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # -- negamax with alpha-beta ------------------------------------------

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self._check_time()

        # Draws that python-chess can tell us about cheaply. Checked before the leaf so a
        # line that "wins" material but allows a repetition is scored as the draw it is.
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0

        if depth <= 0:
            if self.use_quiescence:
                return self.quiescence(board, alpha, beta, 0)
            return evaluate(board)

        moves = ordered_moves(board)
        if not moves:
            return -(MATE_SCORE - ply) if board.is_check() else 0

        best = -INFINITY
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def search_root(
        self, board: chess.Board, depth: int, visited: frozenset[str] = frozenset()
    ) -> tuple[chess.Move, int]:
        """Full-width search at the root that also remembers the best move.

        `visited` holds keys of positions already seen this game. The platform sends
        a bare FEN, so the board carries no history and the in-tree repetition check
        cannot see game-level repetitions. A root move that re-enters a visited
        position is scored at best as a draw, so a winning side never shuffles into
        a threefold repetition while a losing side is still free to take one.
        """
        moves = ordered_moves(board)
        if not moves:
            raise ValueError("search_root called on a position with no legal moves")

        best_move = moves[0]
        best_score = -INFINITY
        alpha, beta = -INFINITY, INFINITY
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            if visited and position_key(board) in visited:
                score = min(score, 0)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score
        return best_move, best_score


# ---------------------------------------------------------------------------
# Iterative deepening and time management
# ---------------------------------------------------------------------------


def move_budget_ms(time_left_ms: int) -> int:
    budget = time_left_ms // MOVES_TO_GO
    return max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, budget))


def position_key(board: chess.Board) -> str:
    """Placement, side to move, castling rights and en passant square: what repetition
    rules compare. The move counters are dropped."""
    return board.fen().rsplit(" ", 2)[0]


# Positions this game has passed through. The process lives for exactly one game, so
# module state is the game's memory; it is never carried into another game.
_visited: set[str] = set()


def reset_history() -> None:
    _visited.clear()


def choose_move(
    board: chess.Board, time_left_ms: int, visited: frozenset[str] = frozenset()
) -> chess.Move:
    """Deepen one ply at a time; keep the move from the last completed depth.

    An aborted iteration raises OutOfTime from deep inside the tree, which skips the
    board.pop() calls on the way up. The board is unwound back to the root here so
    the caller always gets its own position back.
    """
    deadline = time.monotonic() + move_budget_ms(time_left_ms) / 1000.0
    root_ply = len(board.move_stack)

    # A one-ply static search is fast in any position and guarantees an answer even
    # when the clock is nearly gone. Everything after this is on the deadline.
    best_move, best_score = Searcher(None, use_quiescence=False).search_root(board, 1, visited)

    searcher = Searcher(deadline, use_quiescence=USE_QUIESCENCE)
    for depth in range(1, MAX_DEPTH + 1):
        try:
            best_move, best_score = searcher.search_root(board, depth, visited)
        except OutOfTime:
            while len(board.move_stack) > root_ply:
                board.pop()
            break
        print(
            f"depth {depth:2d}  score {best_score:7d}  best {best_move.uci()}  "
            f"nodes {searcher.nodes}  qnodes {searcher.qnodes}  max_qply {searcher.max_qply}"
        )
        if abs(best_score) >= MATE_BOUND:
            break
    return best_move


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform: return a legal move in UCI."""
    board = chess.Board(fen)
    _visited.add(position_key(board))
    move = choose_move(board, time_left_ms, frozenset(_visited))
    if move not in board.legal_moves:  # belt and braces; the search only uses legal moves
        move = next(iter(board.legal_moves))
    board.push(move)
    _visited.add(position_key(board))
    return move.uci()
