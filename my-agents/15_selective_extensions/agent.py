"""Selective-extensions chess agent (roadmap step 15).

A complete alpha-beta engine (iterative deepening, PVS, transposition table, killers,
history, null move, LMR, futility, quiescence) whose distinguishing feature is a set of
*selective extensions*: forcing lines are searched one ply deeper than the nominal depth.

    check extension        a move that gives check gets one more ply
    recapture extension    a capture that immediately takes back on the same square
    passed-pawn extension  a passed pawn stepping onto the seventh rank
    singular extension     the TT move is much better than every alternative

Every extension is toggled independently through `SearchConfig`, and a per-line
*extension budget* (`max_extensions`) bounds how many plies a single line can gain so
that a run of checks cannot explode the tree.

Interface: `get_move(fen, time_left_ms) -> str` (UCI), as required by the harness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess

# ---------------------------------------------------------------------------
# Evaluation: material + tapered piece-square tables + a few pawn terms
# ---------------------------------------------------------------------------

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
# Value used for SEE and move ordering, indexed by piece type. The king is "priceless".
SEE_VALUE: list[int] = [0, 100, 320, 330, 500, 900, 20_000]

# Tables are written as they appear on a diagram (rank 8 first) and flipped below.
_PAWN_MG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_PAWN_EG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    90, 90, 90, 90, 90, 90, 90, 90,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    15, 15, 15, 15, 15, 15, 15, 15,
     5,  5,  5,  5,  5,  5,  5,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
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
_KING_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
_KING_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]


def _from_diagram(table: list[int]) -> list[int]:
    """Reorder a diagram-style table (rank 8 first) into a1..h8 square order."""
    return [table[chess.square_mirror(square)] for square in chess.SQUARES]


def _mirror(table: list[int]) -> list[int]:
    return [table[chess.square_mirror(square)] for square in chess.SQUARES]


def _build(color: chess.Color, endgame: bool) -> list[list[int]]:
    """PST[piece_type] -> table indexed by square, with material folded in."""
    white = {
        chess.PAWN: _from_diagram(_PAWN_EG if endgame else _PAWN_MG),
        chess.KNIGHT: _from_diagram(_KNIGHT),
        chess.BISHOP: _from_diagram(_BISHOP),
        chess.ROOK: _from_diagram(_ROOK),
        chess.QUEEN: _from_diagram(_QUEEN),
        chess.KING: _from_diagram(_KING_EG if endgame else _KING_MG),
    }
    tables: list[list[int]] = [[0] * 64]
    for piece_type in chess.PIECE_TYPES:
        table = white[piece_type] if color == chess.WHITE else _mirror(white[piece_type])
        tables.append([PIECE_VALUE[piece_type] + bonus for bonus in table])
    return tables


# Indexed by colour (BLACK = 0, WHITE = 1), then piece type, then square.
PST_MG: list[list[list[int]]] = [_build(chess.BLACK, False), _build(chess.WHITE, False)]
PST_EG: list[list[list[int]]] = [_build(chess.BLACK, True), _build(chess.WHITE, True)]

# Game phase weights: 24 = full middlegame, 0 = bare kings and pawns.
PHASE_WEIGHT: list[int] = [0, 0, 1, 1, 2, 4, 0]
MAX_PHASE = 24


def _passed_masks(color: chess.Color) -> list[int]:
    """Squares an enemy pawn must not occupy for a pawn on `square` to be passed."""
    masks = []
    for square in chess.SQUARES:
        file, rank = chess.square_file(square), chess.square_rank(square)
        files = chess.BB_FILES[file]
        if file > 0:
            files |= chess.BB_FILES[file - 1]
        if file < 7:
            files |= chess.BB_FILES[file + 1]
        ahead = 0
        ranks = range(rank + 1, 8) if color == chess.WHITE else range(rank - 1, -1, -1)
        for r in ranks:
            ahead |= chess.BB_RANKS[r]
        masks.append(files & ahead)
    return masks


PASSED_MASK: list[list[int]] = [_passed_masks(chess.BLACK), _passed_masks(chess.WHITE)]
ADJACENT_FILES: list[int] = [
    (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
    for f in range(8)
]
PASSED_BONUS_EG = [0, 10, 20, 35, 60, 100, 150, 0]  # by relative rank
PASSED_BONUS_MG = [0, 5, 10, 15, 25, 40, 60, 0]

MATE_SCORE = 100_000
MATE_BOUND = MATE_SCORE - 1_000  # anything beyond this is a mate score
INF = 1_000_000


def evaluate(board: chess.Board) -> int:
    """Static score from the perspective of the side to move (centipawns)."""
    mg = 0
    eg = 0
    phase = 0
    occupied_co = board.occupied_co
    pawns = board.pawns
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        own = occupied_co[color]
        pst_mg = PST_MG[color]
        pst_eg = PST_EG[color]
        side_mg = 0
        side_eg = 0
        for piece_type, bb in (
            (chess.KNIGHT, board.knights),
            (chess.BISHOP, board.bishops),
            (chess.ROOK, board.rooks),
            (chess.QUEEN, board.queens),
        ):
            pieces = bb & own
            if not pieces:
                continue
            table_mg = pst_mg[piece_type]
            table_eg = pst_eg[piece_type]
            count = 0
            for square in chess.scan_forward(pieces):
                side_mg += table_mg[square]
                side_eg += table_eg[square]
                count += 1
            phase += PHASE_WEIGHT[piece_type] * count
            if piece_type == chess.BISHOP and count >= 2:
                side_mg += 30
                side_eg += 40
            elif piece_type == chess.ROOK:
                for square in chess.scan_forward(pieces):
                    if not chess.BB_FILES[chess.square_file(square)] & pawns:
                        side_mg += 15
                        side_eg += 10
        king = board.king(color)
        if king is not None:
            side_mg += pst_mg[chess.KING][king]
            side_eg += pst_eg[chess.KING][king]
        own_pawns = pawns & own
        enemy_pawns = pawns & occupied_co[not color]
        table_mg = pst_mg[chess.PAWN]
        table_eg = pst_eg[chess.PAWN]
        passed = PASSED_MASK[color]
        for square in chess.scan_forward(own_pawns):
            side_mg += table_mg[square]
            side_eg += table_eg[square]
            file = chess.square_file(square)
            if not passed[square] & enemy_pawns:
                rank = chess.square_rank(square)
                relative = rank if color == chess.WHITE else 7 - rank
                side_mg += PASSED_BONUS_MG[relative]
                side_eg += PASSED_BONUS_EG[relative]
            if not ADJACENT_FILES[file] & own_pawns:
                side_mg -= 12
                side_eg -= 18
            if chess.BB_FILES[file] & own_pawns & ~chess.BB_SQUARES[square]:
                side_mg -= 8
                side_eg -= 12
        mg += sign * side_mg
        eg += sign * side_eg
    phase = min(phase, MAX_PHASE)
    score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    return score if board.turn == chess.WHITE else -score


def non_pawn_material(board: chess.Board, color: chess.Color) -> int:
    own = board.occupied_co[color]
    return (
        320 * chess.popcount(board.knights & own)
        + 330 * chess.popcount(board.bishops & own)
        + 500 * chess.popcount(board.rooks & own)
        + 900 * chess.popcount(board.queens & own)
    )


# ---------------------------------------------------------------------------
# Static exchange evaluation
# ---------------------------------------------------------------------------


def _least_valuable_attacker(board: chess.Board, attackers: int) -> tuple[int, int]:
    """Return (square, piece_type) of the cheapest piece in `attackers`."""
    for piece_type, bb in (
        (chess.PAWN, board.pawns),
        (chess.KNIGHT, board.knights),
        (chess.BISHOP, board.bishops),
        (chess.ROOK, board.rooks),
        (chess.QUEEN, board.queens),
        (chess.KING, board.kings),
    ):
        subset = attackers & bb
        if subset:
            return chess.lsb(subset), piece_type
    return -1, 0


def see(board: chess.Board, move: chess.Move) -> int:
    """Static exchange evaluation: net material the mover gains on move.to_square.

    A quiet move scores 0 when the piece is safe on its new square, or negative when it
    can be taken for free. Sliding x-rays are exposed by removing pieces from `occupied`.
    """
    target = move.to_square
    attacker_type = board.piece_type_at(move.from_square)
    if attacker_type is None:
        return 0
    if board.is_en_passant(move):
        captured = SEE_VALUE[chess.PAWN]
    else:
        victim = board.piece_type_at(target)
        captured = SEE_VALUE[victim] if victim else 0
    if move.promotion:
        captured += SEE_VALUE[move.promotion] - SEE_VALUE[chess.PAWN]
        attacker_type = move.promotion

    gain = [captured]
    occupied = board.occupied & ~chess.BB_SQUARES[move.from_square]
    side = not board.turn
    on_square = SEE_VALUE[attacker_type]
    while True:
        attackers = board.attackers_mask(side, target, occupied) & occupied
        if not attackers:
            break
        square, piece_type = _least_valuable_attacker(board, attackers)
        if piece_type == chess.KING and board.attackers_mask(not side, target, occupied) & occupied:
            break  # the king cannot capture into a defended square
        gain.append(on_square - gain[-1])
        occupied &= ~chess.BB_SQUARES[square]
        on_square = SEE_VALUE[piece_type]
        side = not side
    # Resolve the swap list backwards: each side may decline to continue the exchange.
    while len(gain) > 1:
        last = gain.pop()
        gain[-1] = min(gain[-1], -last)
    return gain[0]


# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------


@dataclass
class SearchConfig:
    """Every extension and pruning rule is a switch so each can be measured alone."""

    # --- selective extensions -------------------------------------------------
    check_extension: bool = True
    recapture_extension: bool = True
    passed_pawn_extension: bool = True
    singular_extension: bool = True
    max_extensions: int = 2          # extension budget: plies one line may gain in total
    check_ext_min_depth: int = 1     # do not extend checks below this nominal depth
    singular_min_depth: int = 5      # singular test needs a trustworthy TT entry
    singular_margin: int = 40        # scaled by depth/4: how much better the TT move must be
    # --- pruning / reductions -------------------------------------------------
    null_move: bool = True
    lmr: bool = True
    futility: bool = True
    reverse_futility: bool = True
    tt: bool = True
    pvs: bool = True
    quiescence_checks: int = 2       # plies of check evasions resolved inside quiescence


EXACT, LOWER, UPPER = 0, 1, 2
MAX_PLY = 96

TTEntry = tuple[int, int, int, chess.Move | None]  # depth, score, flag, best move


class OutOfTime(Exception):
    """Raised inside the search when the move budget is spent."""


class Searcher:
    """One search for one root position. Holds the clock, tables and statistics."""

    def __init__(
        self,
        config: SearchConfig | None = None,
        tt: dict[object, TTEntry] | None = None,
        deadline: float | None = None,
        game_history: set[object] | None = None,
    ) -> None:
        self.config = config or SearchConfig()
        self.tt: dict[object, TTEntry] = tt if tt is not None else {}
        self.deadline = deadline if deadline is not None else float("inf")
        self.game_history = game_history or set()
        self.nodes = 0
        self.qnodes = 0
        self.max_main_ply = 0  # deepest ply reached by the main search (not quiescence)
        self.extensions: dict[str, int] = {
            "check": 0, "recapture": 0, "passed_pawn": 0, "singular": 0, "budget_denied": 0,
        }
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 2)]
        self.history: list[int] = [0] * (2 * 64 * 64)
        self.path: list[object] = []

    # -- utilities ---------------------------------------------------------

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes & 1023 == 0 and time.monotonic() > self.deadline:
            raise OutOfTime

    @staticmethod
    def _key(board: chess.Board) -> object:
        # python-chess's own transposition key: pieces, turn, castling rights, en passant.
        return board._transposition_key()

    def _is_draw(self, board: chess.Board, key: object) -> bool:
        if board.halfmove_clock >= 100:
            return True
        if board.halfmove_clock >= 4:
            # Same position earlier in this line, or earlier in the game, counts as a draw.
            window = self.path[-(board.halfmove_clock):]
            if key in window or key in self.game_history:
                return True
        # Kings plus at most one minor piece in total cannot mate.
        return (
            not (board.pawns | board.rooks | board.queens)
            and chess.popcount(board.knights | board.bishops) <= 1
        )

    # -- move ordering -----------------------------------------------------

    def _order(
        self, board: chess.Board, moves: list[chess.Move], tt_move: chess.Move | None, ply: int
    ) -> list[chess.Move]:
        killers = self.killers[ply]
        history = self.history
        colour_offset = 4096 if board.turn else 0
        scored = []
        for move in moves:
            if move == tt_move:
                score = 10_000_000
            elif board.is_capture(move):
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                attacker = board.piece_type_at(move.from_square) or chess.PAWN
                score = 1_000_000 + SEE_VALUE[victim] * 10 - SEE_VALUE[attacker]
                if move.promotion:
                    score += SEE_VALUE[move.promotion]
            elif move.promotion:
                score = 900_000 + SEE_VALUE[move.promotion]
            elif move == killers[0]:
                score = 800_000
            elif move == killers[1]:
                score = 790_000
            else:
                score = history[colour_offset + move.from_square * 64 + move.to_square]
            scored.append((score, move))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored]

    # -- extensions ----------------------------------------------------------

    def extension(
        self,
        board: chess.Board,
        move: chess.Move,
        depth: int,
        gives_check: bool,
        is_capture: bool,
        last_capture: tuple[int, int] | None,
        ext_used: int,
    ) -> int:
        """Decide how many plies (0 or 1) the line through `move` gains.

        Called *after* `move` has been pushed on `board`. `last_capture` is the
        (square, victim value) of the previous ply's capture, if any. Each rule is
        independently toggled; the shared budget `max_extensions` bounds the total
        number of extensions along one line.
        """
        config = self.config
        kind: str | None = None
        see_score: int | None = None

        if config.check_extension and gives_check and depth >= config.check_ext_min_depth:
            # Only sound checks: a checking piece that simply hangs is not worth a ply.
            see_score = _see_after_push(board, move)
            if see_score >= 0:
                kind = "check"

        if kind is None and config.recapture_extension and is_capture and last_capture is not None:
            square, value = last_capture
            if move.to_square == square:
                if see_score is None:
                    see_score = _see_after_push(board, move)
                captured = _captured_value_after_push(board, move)
                if see_score >= 0 and abs(captured - value) <= 150:
                    kind = "recapture"

        if kind is None and config.passed_pawn_extension:
            mover = not board.turn
            if board.piece_type_at(move.to_square) == chess.PAWN:
                rank = chess.square_rank(move.to_square)
                relative = rank if mover == chess.WHITE else 7 - rank
                enemy_pawns = board.pawns & board.occupied_co[board.turn]
                if relative == 6 and not PASSED_MASK[mover][move.to_square] & enemy_pawns:
                    if see_score is None:
                        see_score = _see_after_push(board, move)
                    if see_score >= 0:
                        kind = "passed_pawn"

        if kind is None:
            return 0
        if ext_used >= config.max_extensions:
            self.extensions["budget_denied"] += 1
            return 0
        self.extensions[kind] += 1
        return 1

    # -- quiescence --------------------------------------------------------

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int, checks: int) -> int:
        self._tick()
        self.qnodes += 1
        if ply >= MAX_PLY:
            return evaluate(board)

        in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE_SCORE + ply
        if in_check and checks < self.config.quiescence_checks:
            best = -INF
            for move in self._order(board, moves, None, ply):
                board.push(move)
                score = -self.quiescence(board, -beta, -alpha, ply + 1, checks + 1)
                board.pop()
                if score > best:
                    best = score
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            break
            return best

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        best = stand_pat

        promotions = board.pawns & board.occupied_co[board.turn] & (
            chess.BB_RANK_7 if board.turn == chess.WHITE else chess.BB_RANK_2
        )
        candidates = list(board.generate_legal_captures())
        if promotions:
            for move in board.generate_legal_moves(promotions):
                if move.promotion == chess.QUEEN and not board.is_capture(move):
                    candidates.append(move)
        if not candidates:
            return best
        for move in self._order(board, candidates, None, ply):
            if move.promotion and move.promotion != chess.QUEEN:
                continue
            victim = board.piece_type_at(move.to_square) or chess.PAWN
            if stand_pat + SEE_VALUE[victim] + 200 < alpha and not move.promotion:
                continue  # delta pruning: even winning this piece cannot raise alpha
            if see(board, move) < 0:
                continue  # losing capture
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1, checks)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    # -- main search -------------------------------------------------------

    def search(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        ext_used: int,
        last_capture: tuple[int, int] | None = None,
        null_allowed: bool = True,
        excluded: chess.Move | None = None,
    ) -> int:
        """Negamax with PVS. Returns the score for the side to move."""
        self._tick()
        config = self.config
        if ply > self.max_main_ply:
            self.max_main_ply = ply
        if ply >= MAX_PLY:
            return evaluate(board)

        key = self._key(board)
        if ply > 0:
            if self._is_draw(board, key):
                return 0
            # Mate distance pruning: no mate found from here can beat one already known.
            alpha = max(alpha, -MATE_SCORE + ply)
            beta = min(beta, MATE_SCORE - ply - 1)
            if alpha >= beta:
                return alpha

        in_check = board.is_check()
        if depth <= 0:
            # Quiescence resolves captures and a bounded number of check evasions. Going
            # there even when in check keeps the extension budget the only way to get
            # deeper: an unbudgeted "in check at the frontier" rule chains forever.
            return self.quiescence(board, alpha, beta, ply, 0)

        pv_node = beta - alpha > 1
        original_alpha = alpha

        # --- transposition table ---
        tt_move: chess.Move | None = None
        tt_entry: TTEntry | None = None
        if config.tt and excluded is None:
            tt_entry = self.tt.get(key)
            if tt_entry is not None:
                tt_depth, tt_score, tt_flag, tt_move = tt_entry
                if tt_score > MATE_BOUND:
                    tt_score -= ply
                elif tt_score < -MATE_BOUND:
                    tt_score += ply
                if tt_depth >= depth and ply > 0 and not pv_node:
                    if tt_flag == EXACT:
                        return tt_score
                    if tt_flag == LOWER and tt_score >= beta:
                        return tt_score
                    if tt_flag == UPPER and tt_score <= alpha:
                        return tt_score

        static_eval = evaluate(board) if not in_check else -INF

        # --- static pruning (never at PV nodes, never in check, never near mate) ---
        if not pv_node and not in_check and excluded is None and abs(beta) < MATE_BOUND:
            if config.reverse_futility and depth <= 3 and static_eval - 120 * depth >= beta:
                return static_eval
            if (
                config.null_move
                and null_allowed
                and depth >= 2
                and static_eval >= beta
                and non_pawn_material(board, board.turn) > 0
            ):
                reduction = 2 + depth // 4
                board.push(chess.Move.null())
                self.path.append(key)
                try:
                    score = -self.search(
                        board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, ext_used,
                        None, null_allowed=False,
                    )
                finally:
                    self.path.pop()
                    board.pop()
                if score >= beta and score < MATE_BOUND:
                    return beta

        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply if in_check else 0
        moves = self._order(board, moves, tt_move, ply)

        futility_prune = (
            config.futility
            and not pv_node
            and not in_check
            and depth <= 2
            and static_eval + 150 * depth <= alpha
        )

        best = -INF
        best_move: chess.Move | None = None
        searched = 0
        killers = self.killers[ply]
        for index, move in enumerate(moves):
            if move == excluded:
                continue
            is_capture = board.is_capture(move)
            is_promotion = move.promotion is not None

            # --- singular extension: is the TT move much better than every alternative? ---
            extension = 0
            if (
                config.singular_extension
                and move == tt_move
                and tt_entry is not None
                and excluded is None
                and ply > 0
                and depth >= config.singular_min_depth
                and tt_entry[2] != UPPER
                and tt_entry[0] >= depth - 3
                and abs(tt_entry[1]) < MATE_BOUND
                and ext_used < config.max_extensions
            ):
                singular_beta = tt_entry[1] - config.singular_margin * depth // 4
                # Reduced-depth search of this node with the TT move excluded. If nothing
                # else comes close to the TT score, the TT move is singular: extend it.
                singular_score = self.search(
                    board, (depth - 1) // 2, singular_beta - 1, singular_beta, ply, ext_used,
                    last_capture, null_allowed=False, excluded=move,
                )
                if singular_score < singular_beta:
                    extension = 1
                    self.extensions["singular"] += 1

            if is_capture:
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                capture_info: tuple[int, int] | None = (move.to_square, SEE_VALUE[victim])
            else:
                capture_info = None

            board.push(move)
            self.path.append(key)
            try:
                gives_check = board.is_check()
                if extension == 0:
                    extension = self.extension(
                        board, move, depth, gives_check, is_capture, last_capture, ext_used
                    )

                # --- futility: skip quiet moves that cannot raise alpha ---
                if (
                    futility_prune
                    and searched > 0
                    and not is_capture
                    and not is_promotion
                    and not gives_check
                    and extension == 0
                ):
                    continue

                new_depth = depth - 1 + extension
                new_ext = ext_used + extension

                # --- late move reductions ---
                reduction = 0
                if (
                    config.lmr
                    and depth >= 3
                    and index >= 3
                    and not is_capture
                    and not is_promotion
                    and not in_check
                    and not gives_check
                    and extension == 0
                    and move != killers[0]
                    and move != killers[1]
                ):
                    reduction = 2 if index >= 8 and depth >= 5 else 1

                if searched == 0 or not config.pvs:
                    score = -self.search(
                        board, new_depth, -beta, -alpha, ply + 1, new_ext, capture_info
                    )
                else:
                    score = -self.search(
                        board, new_depth - reduction, -alpha - 1, -alpha, ply + 1, new_ext,
                        capture_info,
                    )
                    if reduction and score > alpha:
                        score = -self.search(
                            board, new_depth, -alpha - 1, -alpha, ply + 1, new_ext, capture_info
                        )
                    if alpha < score < beta:
                        score = -self.search(
                            board, new_depth, -beta, -alpha, ply + 1, new_ext, capture_info
                        )
            finally:
                self.path.pop()
                board.pop()
            searched += 1

            if score > best:
                best = score
                best_move = move
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if not is_capture and not is_promotion:
                            if killers[0] != move:
                                killers[1] = killers[0]
                                killers[0] = move
                            offset = (4096 if board.turn else 0) + move.from_square * 64
                            self.history[offset + move.to_square] += depth * depth
                        break

        if searched == 0:
            # Every move was excluded or pruned; report the bound we could not beat.
            return alpha

        if config.tt and excluded is None:
            if best <= original_alpha:
                flag = UPPER
            elif best >= beta:
                flag = LOWER
            else:
                flag = EXACT
            stored = best
            if stored > MATE_BOUND:
                stored += ply
            elif stored < -MATE_BOUND:
                stored -= ply
            previous = self.tt.get(key)
            if previous is None or previous[0] <= depth or flag == EXACT:
                self.tt[key] = (depth, stored, flag, best_move)
        return best

    # -- root ------------------------------------------------------------------

    def search_root(
        self, board: chess.Board, depth: int, alpha: int = -INF, beta: int = INF
    ) -> tuple[chess.Move, int]:
        """Search the root; returns (best move, score). Raises OutOfTime if aborted."""
        key = self._key(board)
        tt_entry = self.tt.get(key) if self.config.tt else None
        tt_move = tt_entry[3] if tt_entry is not None else None
        moves = self._order(board, list(board.legal_moves), tt_move, 0)
        best_move = moves[0]
        best = -INF
        for index, move in enumerate(moves):
            is_capture = board.is_capture(move)
            if is_capture:
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                capture_info: tuple[int, int] | None = (move.to_square, SEE_VALUE[victim])
            else:
                capture_info = None
            board.push(move)
            self.path.append(key)
            try:
                gives_check = board.is_check()
                extension = self.extension(board, move, depth, gives_check, is_capture, None, 0)
                new_depth = depth - 1 + extension
                if index == 0 or not self.config.pvs:
                    score = -self.search(
                        board, new_depth, -beta, -alpha, 1, extension, capture_info
                    )
                else:
                    score = -self.search(
                        board, new_depth, -alpha - 1, -alpha, 1, extension, capture_info
                    )
                    if alpha < score < beta:
                        score = -self.search(
                            board, new_depth, -beta, -alpha, 1, extension, capture_info
                        )
            finally:
                self.path.pop()
                board.pop()
            if score > best:
                best = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        if self.config.tt:
            flag = EXACT if alpha < beta else LOWER
            self.tt[key] = (depth, best, flag, best_move)
        return best_move, best


def _see_after_push(board: chess.Board, move: chess.Move) -> int:
    """SEE for a move that is already on the board: pop, evaluate, push back."""
    board.pop()
    try:
        return see(board, move)
    finally:
        board.push(move)


def _captured_value_after_push(board: chess.Board, move: chess.Move) -> int:
    board.pop()
    try:
        if board.is_en_passant(move):
            return SEE_VALUE[chess.PAWN]
        victim = board.piece_type_at(move.to_square)
        return SEE_VALUE[victim] if victim else 0
    finally:
        board.push(move)


# ---------------------------------------------------------------------------
# Iterative deepening and time management
# ---------------------------------------------------------------------------

MAX_DEPTH = 64
MOVES_TO_GO = 28
MIN_BUDGET_MS = 30
MAX_BUDGET_MS = 12_000
SAFETY_MS = 150
ASPIRATION = 50


def move_budget_ms(time_left_ms: int) -> int:
    """Spend a slice of the clock, never all of it, and keep a safety margin."""
    budget = time_left_ms // MOVES_TO_GO
    budget = max(MIN_BUDGET_MS, min(MAX_BUDGET_MS, budget))
    return max(1, min(budget, time_left_ms - SAFETY_MS))


def choose_move(
    board: chess.Board,
    time_left_ms: int,
    config: SearchConfig | None = None,
    tt: dict[object, TTEntry] | None = None,
    game_history: set[object] | None = None,
    max_depth: int = MAX_DEPTH,
    verbose: bool = True,
) -> tuple[chess.Move, Searcher]:
    """Deepen one ply at a time until the budget runs out; keep the last finished depth."""
    budget_s = move_budget_ms(time_left_ms) / 1000.0
    started = time.monotonic()
    searcher = Searcher(config, tt, started + budget_s, game_history)

    legal = list(board.legal_moves)
    if len(legal) == 1:
        return legal[0], searcher

    root_stack = len(board.move_stack)
    best_move, best_score = searcher.search_root(board, 1)
    for depth in range(2, max_depth + 1):
        elapsed = time.monotonic() - started
        if elapsed > budget_s * 0.45:
            break  # the next iteration will not finish; do not start it
        try:
            alpha, beta = best_score - ASPIRATION, best_score + ASPIRATION
            move, score = searcher.search_root(board, depth, alpha, beta)
            if score <= alpha or score >= beta:
                move, score = searcher.search_root(board, depth, -INF, INF)
        except OutOfTime:
            # The abort can fire anywhere in the tree; unwind whatever is still pushed.
            while len(board.move_stack) > root_stack:
                board.pop()
            searcher.path.clear()
            break
        best_move, best_score = move, score
        if verbose:
            print(
                f"depth {depth:2d} score {best_score:7d} best {best_move.uci()} "
                f"nodes {searcher.nodes} seldepth {searcher.max_main_ply} "
                f"ext {searcher.extensions} time {time.monotonic() - started:.2f}s"
            )
        if abs(best_score) >= MATE_BOUND:
            break
    return best_move, searcher


# ---------------------------------------------------------------------------
# Module state: persists between moves of one game, reset when a new game starts
# ---------------------------------------------------------------------------

_TT: dict[object, TTEntry] = {}
_HISTORY: set[object] = set()
_LAST_FULLMOVE = 0
TT_MAX_ENTRIES = 1_500_000


def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point required by the platform. Return a legal move in UCI."""
    global _LAST_FULLMOVE
    board = chess.Board(fen)
    if board.fullmove_number < _LAST_FULLMOVE:
        _TT.clear()  # a new game started in this process
        _HISTORY.clear()
    if board.halfmove_clock == 0:
        _HISTORY.clear()  # an irreversible move: no earlier position can repeat
    _LAST_FULLMOVE = board.fullmove_number
    if len(_TT) > TT_MAX_ENTRIES:
        _TT.clear()

    root_key = board._transposition_key()
    move, _ = choose_move(board, time_left_ms, tt=_TT, game_history=_HISTORY)

    _HISTORY.add(root_key)
    board.push(move)
    _HISTORY.add(board._transposition_key())
    return move.uci()
