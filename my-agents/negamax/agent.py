"""Negamax chess agent.

Implements my-agents-readmes/negamax.md, in the order that document recommends:

1. board evaluation      material plus piece-square tables, from the mover's point of view
2. negamax               one maximising function; a child's score is negated on the way up
3. alpha-beta pruning    skip branches the opponent would never allow
4. move ordering         captures first (most valuable victim, least valuable attacker),
                         then the best move of the previous iteration at the root
5. iterative deepening   depth 1, 2, 3 ... until the move budget runs out
6. quiescence search     at depth 0 keep resolving captures so the evaluation is never
                         taken halfway through an exchange

Interface: get_move(fen, time_left_ms) -> uci string, as the harness runner expects.
The process lives for one game, so module state (the repetition history) survives
between moves of the same game.
"""

import time

import chess

# ---------------------------------------------------------------------------
# 1. Evaluation
# ---------------------------------------------------------------------------

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables written as they look on a diagram with White at the bottom
# (rank 8 first). They are reordered below so index 0 is a1, matching python-chess.
# fmt: off
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
# Middlegame king: stay tucked behind the pawns.
_KING_MIDDLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
# Endgame king: walk to the centre and help the pawns.
_KING_END = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]
# fmt: on


def _from_diagram(table: list[int]) -> list[int]:
    """Reorder a diagram-style table (rank 8 first) into a1..h8 square order."""
    return [table[chess.square_mirror(square)] for square in chess.SQUARES]


def _for_black(white_table: list[int]) -> list[int]:
    """Black uses the same table mirrored top to bottom."""
    return [white_table[chess.square_mirror(square)] for square in chess.SQUARES]


_WHITE_MIDDLE: dict[chess.PieceType, list[int]] = {
    chess.PAWN: _from_diagram(_PAWN),
    chess.KNIGHT: _from_diagram(_KNIGHT),
    chess.BISHOP: _from_diagram(_BISHOP),
    chess.ROOK: _from_diagram(_ROOK),
    chess.QUEEN: _from_diagram(_QUEEN),
    chess.KING: _from_diagram(_KING_MIDDLE),
}
_WHITE_END: dict[chess.PieceType, list[int]] = {
    **_WHITE_MIDDLE,
    chess.KING: _from_diagram(_KING_END),
}

# PST[phase][colour][piece_type][square] -> bonus for that piece standing on that square.
PST: dict[str, dict[chess.Color, dict[chess.PieceType, list[int]]]] = {
    "middle": {
        chess.WHITE: _WHITE_MIDDLE,
        chess.BLACK: {piece: _for_black(table) for piece, table in _WHITE_MIDDLE.items()},
    },
    "end": {
        chess.WHITE: _WHITE_END,
        chess.BLACK: {piece: _for_black(table) for piece, table in _WHITE_END.items()},
    },
}

# Once the board holds this little non-pawn material in total, the kings should come out.
ENDGAME_MATERIAL = 2 * (2 * PIECE_VALUE[chess.ROOK] + PIECE_VALUE[chess.KNIGHT])

MATE_SCORE = 100_000  # bigger than any material swing
INFINITY = MATE_SCORE * 10  # a finite "infinity" keeps every score an int


def _is_endgame(board: chess.Board) -> bool:
    heavy = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        count = chess.popcount(board.pieces_mask(piece_type, chess.WHITE))
        count += chess.popcount(board.pieces_mask(piece_type, chess.BLACK))
        heavy += PIECE_VALUE[piece_type] * count
    return heavy <= ENDGAME_MATERIAL


def evaluate(board: chess.Board) -> int:
    """Static score from the perspective of the side to move.

    Positive means the mover is better. Negamax needs exactly this convention so a
    child's score can simply be negated.
    """
    tables = PST["end" if _is_endgame(board) else "middle"]
    mover = board.turn
    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type] + tables[piece.color][piece.piece_type][square]
        score += value if piece.color == mover else -value
    return score


# ---------------------------------------------------------------------------
# 4. Move ordering
# ---------------------------------------------------------------------------


def move_priority(board: chess.Board, move: chess.Move) -> int:
    """Higher is searched earlier: promotions, then captures by MVV-LVA, then the rest.

    Alpha-beta prunes the most when the best move is tried first. We do not know the
    best move, but "capture the queen with a pawn" is a fine guess.
    """
    priority = 0
    if move.promotion:
        priority += 10_000 + PIECE_VALUE[move.promotion]
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square)
        # An en passant capture lands on an empty square; the victim is a pawn.
        victim_value = PIECE_VALUE[victim] if victim else PIECE_VALUE[chess.PAWN]
        attacker = board.piece_type_at(move.from_square)
        attacker_value = PIECE_VALUE[attacker] if attacker else 0
        priority += 1_000 + 10 * victim_value - attacker_value
    return priority


def ordered_moves(board: chess.Board, first: chess.Move | None = None) -> list[chess.Move]:
    """All legal moves, best guesses first. `first` (if legal) goes to the front."""
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: move_priority(board, move), reverse=True)
    if first is not None and first in moves:
        moves.remove(first)
        moves.insert(0, first)
    return moves


def capture_moves(board: chess.Board) -> list[chess.Move]:
    """Captures and promotions only, best guesses first. Used by quiescence."""
    moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    moves.sort(key=lambda move: move_priority(board, move), reverse=True)
    return moves


# ---------------------------------------------------------------------------
# 2 + 3 + 6. Negamax with alpha-beta pruning and quiescence
# ---------------------------------------------------------------------------


class OutOfTime(Exception):
    """Raised inside the search when the move budget is spent."""


class Searcher:
    """One search for one root position. Holds the deadline and a node counter."""

    # time.monotonic is cheap but not free; look at the clock every this many nodes.
    # python-chess searches on the order of 10k-40k nodes/s, so this is a few
    # tens of milliseconds between checks.
    TICK_MASK = 511

    def __init__(self, deadline: float | None = None) -> None:
        self.deadline = deadline
        self.nodes = 0
        # Best fully-searched root move of the iteration in progress, and how many
        # root moves that iteration has finished. Read after an OutOfTime abort.
        self.partial: tuple[chess.Move, int] | None = None
        self.partial_count = 0

    def _tick(self) -> None:
        self.nodes += 1
        if (
            self.deadline is not None
            and self.nodes & self.TICK_MASK == 0
            and time.monotonic() > self.deadline
        ):
            raise OutOfTime

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """Search only captures (and promotions) until the position is quiet.

        Without this, a depth-limited search stops in the middle of an exchange and
        the evaluation sees a queen "won" that is about to be taken back.
        """
        self._tick()

        # Mate cannot be recognised by the stand-pat score, so check for it.
        if board.is_check() and not any(board.legal_moves):
            return -(MATE_SCORE - ply)

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return stand_pat  # already good enough; the opponent avoids this line
        alpha = max(alpha, stand_pat)

        best = stand_pat
        for move in capture_moves(board):
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        """Best score the side to move can force within `depth` more plies.

        This is the function from the readme, plus alpha-beta and quiescence:

            for every legal move:
                make the move
                score = -negamax(child)      # the opponent's best, negated
                undo the move
                best = max(best, score)
        """
        self._tick()

        # Draw by repetition inside the line we are searching, or by the fifty-move
        # rule. Checked before generating moves so we never "win" material in a line
        # the opponent can simply repeat out of.
        if ply > 0 and (board.halfmove_clock >= 100 or board.is_repetition(2)):
            return 0

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        moves = ordered_moves(board)
        if not moves:
            # Checkmated: losing, and sooner is worse than later so subtract the ply.
            # Stalemate: draw.
            return -(MATE_SCORE - ply) if board.is_check() else 0

        best = -INFINITY
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if score > best:
                best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break  # the opponent will never allow this line; stop looking
        return best

    def search_root(
        self,
        board: chess.Board,
        depth: int,
        first: chess.Move | None = None,
        draw_moves: frozenset[chess.Move] = frozenset(),
    ) -> tuple[chess.Move, int]:
        """Negamax at the root, but remember which move produced the best score.

        `first` is the best move of the previous iteration and is tried first.
        `draw_moves` repeat a position already seen in the game; they score 0.
        """
        best_move: chess.Move | None = None
        best_score = -INFINITY
        alpha, beta = -INFINITY, INFINITY
        self.partial = None
        self.partial_count = 0

        for move in ordered_moves(board, first):
            if move in draw_moves:
                score = 0
            else:
                board.push(move)
                score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
                board.pop()
            if score > best_score:
                best_score = score
                best_move = move
                self.partial = (move, score)
            self.partial_count += 1
            alpha = max(alpha, score)

        if best_move is None:
            raise ValueError("search_root called with no legal moves")
        return best_move, best_score


# ---------------------------------------------------------------------------
# 5. Iterative deepening and time management
# ---------------------------------------------------------------------------

MAX_DEPTH = 64
MOVES_TO_GO = 30  # assume the game lasts about this many more of our moves
MIN_BUDGET_MS = 10
FLOOR_BUDGET_MS = 300  # on a low clock, still spend up to this (or a quarter of the clock)
MAX_BUDGET_MS = 8_000
SAFETY_MS = 60  # the search overruns its deadline by at most one tick; keep this spare


def move_budget_ms(time_left_ms: int) -> int:
    """How long to think on this move. Spend a slice of what is left, never all of it.

    The increment (0.5 s on the platform) refills a low clock, so when little is left
    it is better to keep spending a modest amount than to play depth-0 moves.
    """
    budget = max(time_left_ms // MOVES_TO_GO, min(FLOOR_BUDGET_MS, time_left_ms // 4))
    budget = min(budget, MAX_BUDGET_MS, time_left_ms // 2 - SAFETY_MS)
    return max(MIN_BUDGET_MS, budget)


def static_best(board: chess.Board) -> chess.Move:
    """One-ply lookahead with the static evaluation. Cheap, so always available."""
    best_move: chess.Move | None = None
    best_score = -INFINITY
    for move in ordered_moves(board):
        board.push(move)
        score = MATE_SCORE if board.is_checkmate() else -evaluate(board)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    if best_move is None:
        raise ValueError("no legal moves")
    return best_move


def choose_move(
    board: chess.Board,
    time_left_ms: int,
    draw_moves: frozenset[chess.Move] = frozenset(),
    max_depth: int = MAX_DEPTH,
) -> chess.Move:
    """Deepen one ply at a time until the budget runs out; keep the last finished depth.

    The search at depth d is aborted by OutOfTime when the clock expires. Because
    each completed depth was a full search, the move from the deepest finished
    iteration is always a sound answer. A static one-ply pick is the fallback so a
    legal move exists even if depth 1 could not finish.
    """
    deadline = time.monotonic() + move_budget_ms(time_left_ms) / 1000.0
    searcher = Searcher(deadline)
    root_depth = len(board.move_stack)

    best_move = static_best(board)
    best_score = 0
    for depth in range(1, max_depth + 1):
        try:
            best_move, best_score = searcher.search_root(board, depth, best_move, draw_moves)
        except OutOfTime:
            # The abort unwound through pushed moves without popping them; undo them.
            while len(board.move_stack) > root_depth:
                board.pop()
            # The previous best move was searched first at this depth. If it finished
            # and a later move beat it, that later move is the better answer.
            if searcher.partial is not None and searcher.partial_count >= 1:
                best_move, best_score = searcher.partial
            break
        print(
            f"depth {depth:2d}  score {best_score:7d}  best {best_move.uci()}  "
            f"nodes {searcher.nodes}"
        )
        if abs(best_score) >= MATE_SCORE - MAX_DEPTH:
            break  # a forced mate was found; deeper search cannot improve it
    return best_move


# ---------------------------------------------------------------------------
# Game state: positions seen so far, so we do not repeat when ahead
# ---------------------------------------------------------------------------

_seen_positions: dict[str, int] = {}
_last_fullmove = 0


def _position_key(board: chess.Board) -> str:
    # epd() drops the move counters and only shows an en passant square when a
    # capture is actually legal, which is what the repetition rule looks at.
    return board.epd()


def _remember(board: chess.Board) -> None:
    global _last_fullmove
    if board.fullmove_number < _last_fullmove:
        _seen_positions.clear()  # a new game in the same process (local harness only)
    _last_fullmove = board.fullmove_number
    key = _position_key(board)
    _seen_positions[key] = _seen_positions.get(key, 0) + 1


def _repeating_moves(board: chess.Board) -> frozenset[chess.Move]:
    """Root moves that lead to a position this game has already seen."""
    repeating = []
    for move in board.legal_moves:
        board.push(move)
        if _position_key(board) in _seen_positions:
            repeating.append(move)
        board.pop()
    return frozenset(repeating)


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    board = chess.Board(fen)
    _remember(board)
    move = choose_move(board, time_left_ms, _repeating_moves(board))
    board.push(move)
    _remember(board)  # the position we hand the opponent counts towards repetition too
    return move.uci()
