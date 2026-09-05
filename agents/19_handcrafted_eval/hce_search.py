"""Alpha-beta search around the handcrafted evaluation.

Iterative deepening, principal variation search, a transposition table that survives between
moves, MVV-LVA / killer / history move ordering, quiescence search with delta pruning, null-move
pruning, check extension, late move reductions and futility pruning. The clock is checked every
few hundred nodes and an `OutOfTime` exception unwinds the search; the caller always holds a
legal move before the clocked part begins.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import chess

MATE = 100_000
MATE_BOUND = MATE - 2_000
INF = 1_000_000
MAX_PLY = 96
QS_MAX_PLY = 10

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
TT_MAX_ENTRIES = 2_000_000

# victim values for MVV-LVA and delta pruning (index = python-chess piece type)
CAPTURE_VALUE = (0, 100, 320, 330, 500, 950, 20_000)
DELTA_MARGIN = 200
FUTILITY_MARGIN = (0, 140, 280)
REVERSE_FUTILITY_MARGIN = (0, 90, 180, 280)
NODES_PER_CLOCK_CHECK = 31  # power of two minus one, used as a mask; time.monotonic is ~50 ns

NULL_MOVE = chess.Move.null()


class OutOfTime(Exception):
    pass


class Searcher:
    def __init__(self, evaluate: Callable[[chess.Board], int]) -> None:
        self.evaluate = evaluate  # side-to-move centipawns; swapped for the compiled one when ready
        self.tt: dict[object, tuple[int, int, int, chess.Move | None]] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 2)]
        self.history: dict[tuple[bool, int, int], int] = {}
        self.game_keys: set[object] = set()
        self.path: list[object] = []
        self.deadline = float("inf")
        self.reset_stats()

    # -------------------------------------------------------------------------------------
    # bookkeeping
    # -------------------------------------------------------------------------------------

    def reset_stats(self) -> None:
        self.nodes = 0
        self.qnodes = 0
        self.tt_hits = 0
        self.tt_cuts = 0
        self.eval_calls = 0
        self.seldepth = 0
        self.depth_reached = 0

    def new_search(self) -> None:
        self.reset_stats()
        self.path = []
        if len(self.tt) > TT_MAX_ENTRIES:
            self.tt.clear()
        # age history so old positions do not dominate ordering forever
        if self.history:
            for key in self.history:
                self.history[key] //= 2

    def _check_clock(self) -> None:
        if (self.nodes & NODES_PER_CLOCK_CHECK) == 0 and time.monotonic() >= self.deadline:
            raise OutOfTime

    # -------------------------------------------------------------------------------------
    # root
    # -------------------------------------------------------------------------------------

    def search_root(
        self,
        board: chess.Board,
        soft_deadline: float,
        hard_deadline: float,
        max_depth: int = 64,
        info: bool = False,
    ) -> tuple[chess.Move, int]:
        """Iterative deepening under the hard deadline.

        Every iteration, depth 1 included, can be cut short by the clock: with a slow
        evaluation (the pure-Python fallback) or a loaded core even a depth-1 search with its
        quiescence tail can cost seconds, which is a lost game at a 50 ms clock. A move always
        exists because the fallback is the best-ordered root move (transposition-table move,
        then the most valuable capture) before any search starts, and an aborted iteration
        keeps its best fully-searched move when that beat the previous iteration.
        """
        self.new_search()
        root_len = len(board.move_stack)
        started = time.monotonic()
        if not any(board.generate_legal_moves()):
            raise ValueError("search_root called with no legal moves")
        entry = self.tt.get(board._transposition_key())
        best_move = self._ordered_moves(board, entry[3] if entry is not None else None, 0)[0]
        best_score = -INF
        score = 0
        for depth in range(1, max_depth + 1):
            self.deadline = hard_deadline
            self.iteration_best: chess.Move | None = None
            self.iteration_score = -INF
            self.iteration_moves_done = 0
            alpha, beta = -INF, INF
            delta = 30
            if depth >= 4 and abs(score) < MATE_BOUND:
                alpha, beta = score - delta, score + delta
            try:
                while True:
                    score = self._root(board, depth, alpha, beta, best_move)
                    if alpha < score < beta:
                        break
                    # aspiration failure: widen the window on the failing side and retry
                    delta *= 3
                    if score <= alpha:
                        alpha = score - delta
                    else:
                        beta = score + delta
                    if delta > 600 or abs(score) >= MATE_BOUND:
                        alpha, beta = -INF, INF
            except OutOfTime:
                while len(board.move_stack) > root_len:
                    board.pop()
                self.path = []
                # a move that beat the previous best in an unfinished iteration is safe to use
                if self.iteration_best is not None and self.iteration_moves_done >= 1 and (
                    self.iteration_best != best_move and self.iteration_score > best_score
                ):
                    best_move, best_score = self.iteration_best, self.iteration_score
                break
            assert self.iteration_best is not None
            best_move, best_score = self.iteration_best, score
            self.depth_reached = depth
            if info:
                elapsed = time.monotonic() - started
                print(
                    f"depth {depth} seldepth {self.seldepth} score {score} nodes {self.nodes} "
                    f"qnodes {self.qnodes} time {elapsed * 1000:.0f}ms move {best_move.uci()}",
                    flush=True,
                )
            now = time.monotonic()
            if abs(score) >= MATE_BOUND:
                break
            # the next iteration usually costs several times the whole search so far
            if now - started > (soft_deadline - started) * 0.45:
                break
            if now >= soft_deadline:
                break
        return best_move, best_score

    def _root(
        self, board: chess.Board, depth: int, alpha: int, beta: int, first: chess.Move
    ) -> int:
        key = board._transposition_key()
        moves = self._ordered_moves(board, first, 0)
        best = -INF
        in_check = board.is_check()
        if in_check:
            depth += 1
        for moves_done, move in enumerate(moves):
            board.push(move)
            self.path.append(key)
            try:
                if not board.is_check() and not any(board.generate_legal_moves()):
                    score = 0  # stalemate; the leaf evaluation cannot see it at depth 1
                elif moves_done == 0:
                    score = -self._search(board, depth - 1, -beta, -alpha, 1, True, True)
                else:
                    score = -self._search(board, depth - 1, -alpha - 1, -alpha, 1, False, True)
                    if alpha < score < beta:
                        score = -self._search(board, depth - 1, -beta, -alpha, 1, True, True)
            finally:
                board.pop()
                self.path.pop()
            if score > best:
                best = score
                if score > alpha or moves_done == 0:
                    self.iteration_best = move
                    self.iteration_score = score
                    self.iteration_moves_done = moves_done + 1
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        flag = TT_EXACT if alpha < beta else TT_LOWER
        self.tt[key] = (depth, best, flag, self.iteration_best)
        return best

    # -------------------------------------------------------------------------------------
    # main search
    # -------------------------------------------------------------------------------------

    def _search(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        pv_node: bool,
        allow_null: bool,
    ) -> int:
        self.nodes += 1
        self._check_clock()

        key = board._transposition_key()
        # draws by repetition (game history or the current line) and the fifty-move rule
        if board.halfmove_clock >= 100 or key in self.game_keys or key in self.path:
            return 0
        if board.occupied.bit_count() <= 4 and board.is_insufficient_material():
            return 0
        # mate distance pruning
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

        in_check = board.is_check()
        if in_check:
            depth += 1
        if depth <= 0 or ply >= MAX_PLY:
            return self._qsearch(board, alpha, beta, ply, 0)

        tt_move: chess.Move | None = None
        entry = self.tt.get(key)
        if entry is not None:
            self.tt_hits += 1
            tt_depth, tt_score, tt_flag, tt_move = entry
            if tt_depth >= depth and not pv_node:
                if tt_score > MATE_BOUND:
                    tt_score -= ply
                elif tt_score < -MATE_BOUND:
                    tt_score += ply
                if tt_flag == TT_EXACT:
                    self.tt_cuts += 1
                    return tt_score
                if tt_flag == TT_LOWER and tt_score >= beta:
                    self.tt_cuts += 1
                    return tt_score
                if tt_flag == TT_UPPER and tt_score <= alpha:
                    self.tt_cuts += 1
                    return tt_score

        static_eval = 0
        have_static = False
        if not in_check and not pv_node and abs(beta) < MATE_BOUND:
            static_eval = self.evaluate(board)
            self.eval_calls += 1
            have_static = True
            # reverse futility: the position is so good that a quiet move keeps it above beta
            if depth <= 3 and static_eval - REVERSE_FUTILITY_MARGIN[depth] >= beta:
                return static_eval
            # null move pruning
            if (
                allow_null
                and depth >= 2
                and static_eval >= beta
                and (board.occupied_co[board.turn] & ~(board.pawns | board.kings))
            ):
                reduction = 2 + depth // 4 + (1 if static_eval - beta > 200 else 0)
                board.push(NULL_MOVE)
                self.path.append(key)
                try:
                    score = -self._search(
                        board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, False, False
                    )
                finally:
                    board.pop()
                    self.path.pop()
                if score >= beta:
                    return beta if score >= MATE_BOUND else score

        moves = self._ordered_moves(board, tt_move, ply)
        if not moves:
            return -MATE + ply if in_check else 0

        futile = False
        if depth <= 2 and not in_check and not pv_node and abs(alpha) < MATE_BOUND:
            if not have_static:
                static_eval = self.evaluate(board)
                self.eval_calls += 1
                have_static = True
            futile = static_eval + FUTILITY_MARGIN[depth] <= alpha

        best = -INF
        best_move: chess.Move | None = None
        flag = TT_UPPER
        moves_done = 0
        killers = self.killers[ply]
        for move in moves:
            is_capture = board.is_capture(move)
            quiet = not is_capture and move.promotion is None
            board.push(move)
            self.path.append(key)
            try:
                gives_check = board.is_check()
                if futile and quiet and moves_done > 0 and not gives_check:
                    continue
                new_depth = depth - 1
                if moves_done == 0:
                    score = -self._search(board, new_depth, -beta, -alpha, ply + 1, pv_node, True)
                else:
                    reduction = 0
                    if (
                        quiet
                        and depth >= 3
                        and moves_done >= 3
                        and not in_check
                        and not gives_check
                        and move not in killers
                    ):
                        reduction = 1
                        if moves_done >= 8:
                            reduction += 1
                        if depth >= 7 and moves_done >= 14:
                            reduction += 1
                    score = -self._search(
                        board, new_depth - reduction, -alpha - 1, -alpha, ply + 1, False, True
                    )
                    if reduction and score > alpha:
                        score = -self._search(
                            board, new_depth, -alpha - 1, -alpha, ply + 1, False, True
                        )
                    if pv_node and alpha < score < beta:
                        score = -self._search(
                            board, new_depth, -beta, -alpha, ply + 1, True, True
                        )
            finally:
                board.pop()
                self.path.pop()
            moves_done += 1
            if score > best:
                best = score
                best_move = move
                if score > alpha:
                    alpha = score
                    flag = TT_EXACT
                    if alpha >= beta:
                        flag = TT_LOWER
                        if quiet:
                            if killers[0] != move:
                                killers[1] = killers[0]
                                killers[0] = move
                            hkey = (board.turn, move.from_square, move.to_square)
                            self.history[hkey] = self.history.get(hkey, 0) + depth * depth
                        break

        if best_move is None:  # every move was futility-pruned
            return static_eval if have_static else alpha
        stored = best
        if stored > MATE_BOUND:
            stored += ply
        elif stored < -MATE_BOUND:
            stored -= ply
        self.tt[key] = (depth, stored, flag, best_move)
        return best

    # -------------------------------------------------------------------------------------
    # quiescence
    # -------------------------------------------------------------------------------------

    def _qsearch(self, board: chess.Board, alpha: int, beta: int, ply: int, qply: int) -> int:
        self.nodes += 1
        self.qnodes += 1
        self._check_clock()
        if ply > self.seldepth:
            self.seldepth = ply

        in_check = board.is_check()
        if in_check:
            if qply >= QS_MAX_PLY or ply >= MAX_PLY:
                return self.evaluate(board)
            moves = list(board.generate_legal_moves())
            if not moves:
                return -MATE + ply
            moves.sort(key=lambda m: self._capture_score(board, m), reverse=True)
            best = -INF
        else:
            stand = self.evaluate(board)
            self.eval_calls += 1
            if stand >= beta:
                return stand
            if stand > alpha:
                alpha = stand
            if qply >= QS_MAX_PLY or ply >= MAX_PLY:
                return stand
            best = stand
            scored = []
            for move in board.generate_legal_captures():
                victim = self._victim_value(board, move)
                if move.promotion is None and stand + victim + DELTA_MARGIN <= alpha:
                    continue  # delta pruning: even winning this piece cannot raise alpha
                attacker = board.piece_type_at(move.from_square) or 1
                scored.append((victim * 16 - CAPTURE_VALUE[attacker] // 100, move))
            promos = board.pawns & board.occupied_co[board.turn] & (
                chess.BB_RANK_7 if board.turn else chess.BB_RANK_2
            )
            if promos:
                for move in board.generate_legal_moves(promos, ~board.occupied):
                    if move.promotion == chess.QUEEN:
                        scored.append((CAPTURE_VALUE[5], move))
            scored.sort(key=lambda t: t[0], reverse=True)
            moves = [m for _, m in scored]

        for move in moves:
            board.push(move)
            try:
                score = -self._qsearch(board, -beta, -alpha, ply + 1, qply + 1)
            finally:
                board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    # -------------------------------------------------------------------------------------
    # move ordering
    # -------------------------------------------------------------------------------------

    def _victim_value(self, board: chess.Board, move: chess.Move) -> int:
        victim = board.piece_type_at(move.to_square)
        if victim is None:  # en passant
            return CAPTURE_VALUE[1]
        value = CAPTURE_VALUE[victim]
        if move.promotion is not None:
            value += CAPTURE_VALUE[move.promotion] - CAPTURE_VALUE[1]
        return value

    def _capture_score(self, board: chess.Board, move: chess.Move) -> int:
        if board.is_capture(move):
            attacker = board.piece_type_at(move.from_square) or 1
            return 1_000_000 + self._victim_value(board, move) * 16 - attacker
        if move.promotion is not None:
            return 900_000 + CAPTURE_VALUE[move.promotion]
        return 0

    def _ordered_moves(
        self, board: chess.Board, tt_move: chess.Move | None, ply: int
    ) -> list[chess.Move]:
        killers = self.killers[ply]
        history = self.history
        turn = board.turn
        scored = []
        for move in board.generate_legal_moves():
            if move == tt_move:
                score = 10_000_000
            elif board.is_capture(move):
                attacker = board.piece_type_at(move.from_square) or 1
                score = 1_000_000 + self._victim_value(board, move) * 16 - attacker
            elif move.promotion is not None:
                score = 900_000 + CAPTURE_VALUE[move.promotion]
            elif move == killers[0]:
                score = 800_000
            elif move == killers[1]:
                score = 790_000
            else:
                score = history.get((turn, move.from_square, move.to_square), 0)
            scored.append((score, move))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored]
