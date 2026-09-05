"""Alpha-beta search that calls a plug-in leaf evaluator.

The search is deliberately independent of the evaluator so that the learned Giraffe
network and the handcrafted control evaluation can be compared with everything else held
fixed. Features: iterative deepening, principal variation search, transposition table,
MVV-LVA / killer / history move ordering, check extension, null-move pruning, late move
reductions, and a capture-only quiescence search with delta pruning. Mate, stalemate,
repetition and fifty-move scores are decided by the rules, never by the evaluator.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import chess

Evaluator = Callable[[chess.Board], int]

MATE = 100_000
MATE_BOUND = MATE - 1_000
INF = MATE + 1
DRAW = 0

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

MAX_PLY = 96
QS_MAX_PLY = 12
NODES_PER_CLOCK_CHECK = 128

MVV = (0, 100, 320, 330, 500, 900, 20_000)
LVA = (0, 1, 3, 3, 5, 9, 10)
DELTA_MARGIN = 200
NULL_MOVE_REDUCTION = 2
LMR_MIN_DEPTH = 3
LMR_MIN_MOVE = 4
FUTILITY_MARGIN = (0, 150, 350)


class OutOfTime(Exception):
    """Raised inside the search when the deadline passes."""


class SearchStats:
    """Counters reset at every root search; read by benchmarks and RESULTS.md."""

    def __init__(self) -> None:
        self.nodes = 0
        self.qnodes = 0
        self.depth = 0
        self.seldepth = 0
        self.tt_hits = 0
        self.elapsed = 0.0
        self.score = 0

    def summary(self) -> str:
        nps = self.nodes / self.elapsed if self.elapsed > 0 else 0.0
        return (
            f"depth {self.depth} seldepth {self.seldepth} score {self.score} "
            f"nodes {self.nodes} qnodes {self.qnodes} tt_hits {self.tt_hits} "
            f"nps {nps:.0f} time {self.elapsed:.3f}s"
        )


class Searcher:
    """One search engine instance; keeps the TT and heuristics across moves of a game."""

    def __init__(self, evaluate: Evaluator) -> None:
        self.evaluate = evaluate
        self.tt: dict[object, tuple[int, int, int, chess.Move | None]] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 1)]
        self.history: dict[tuple[bool, int, int], int] = {}
        self.game_keys: set[object] = set()  # positions seen in the game so far
        self.path: list[object] = []
        self.stats = SearchStats()
        self.deadline = float("inf")
        self.nodes_since_check = 0

    # ------------------------------------------------------------------ public API

    def search(
        self, board: chess.Board, budget_s: float, max_depth: int = 64
    ) -> tuple[chess.Move, int]:
        """Iterative deepening; returns the best move and its score."""
        self.stats = SearchStats()
        started = time.monotonic()
        self.deadline = float("inf")
        self.nodes_since_check = 0
        self.path = []
        if len(self.tt) > 400_000:
            self.tt.clear()
        for row in self.killers:
            row[0] = row[1] = None

        best_move: chess.Move | None = None
        best_score = 0
        base_len = len(board.move_stack)
        for depth in range(1, max_depth + 1):
            if depth > 1:
                self.deadline = started + budget_s
                # no point starting a depth that cannot finish
                if time.monotonic() - started > budget_s * 0.5:
                    break
            try:
                score = self._root(board, depth)
            except OutOfTime:
                self._restore(board, base_len)
                break
            best_move = self.tt_move(board)
            best_score = score
            self.stats.depth = depth
            self.stats.score = score
            if abs(score) >= MATE_BOUND:
                break
        self.stats.elapsed = time.monotonic() - started
        if best_move is None:  # depth 1 cannot time out, so this is only defensive
            best_move = next(iter(board.legal_moves))
        return best_move, best_score

    def tt_move(self, board: chess.Board) -> chess.Move | None:
        entry = self.tt.get(board._transposition_key())
        return entry[3] if entry is not None else None

    def remember(self, board: chess.Board) -> None:
        """Record a position that has occurred in the game (for repetition scoring)."""
        self.game_keys.add(board._transposition_key())

    def principal_variation(self, board: chess.Board, max_len: int) -> list[chess.Move]:
        """Walk the transposition table from ``board``; used by TD-Leaf to find the leaf."""
        pv: list[chess.Move] = []
        seen: set[object] = set()
        board = board.copy(stack=False)
        while len(pv) < max_len:
            key = board._transposition_key()
            if key in seen:
                break
            seen.add(key)
            move = self.tt_move(board)
            if move is None or move not in board.legal_moves:
                break
            pv.append(move)
            board.push(move)
        return pv

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _restore(board: chess.Board, base_len: int) -> None:
        # a timeout can unwind through pushed moves; pop back to the root position
        while len(board.move_stack) > base_len:
            board.pop()

    def _check_clock(self) -> None:
        self.nodes_since_check += 1
        if self.nodes_since_check >= NODES_PER_CLOCK_CHECK:
            self.nodes_since_check = 0
            if time.monotonic() >= self.deadline:
                raise OutOfTime

    def _root(self, board: chess.Board, depth: int) -> int:
        key = board._transposition_key()
        moves = self._ordered_moves(board, self.tt_move(board), 0)
        alpha, beta = -INF, INF
        best = -INF
        best_move = moves[0]
        self.path.append(key)
        for i, move in enumerate(moves):
            board.push(move)
            if i == 0:
                score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            else:
                score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                if score > alpha:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            if score > best:
                best = score
                best_move = move
                if score > alpha:
                    alpha = score
        self.path.pop()
        self.tt[key] = (depth, TT_EXACT, best, best_move)
        return best

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self._check_clock()
        self.stats.nodes += 1
        if ply > self.stats.seldepth:
            self.stats.seldepth = ply

        key = board._transposition_key()
        if key in self.game_keys or key in self.path or board.halfmove_clock >= 100:
            return DRAW
        if board.is_insufficient_material():
            return DRAW

        in_check = board.is_check()
        if in_check:
            depth += 1  # check extension
        if depth <= 0 or ply >= MAX_PLY:
            return self._quiescence(board, alpha, beta, ply)

        # mate distance pruning
        alpha = max(alpha, -MATE + ply)
        beta = min(beta, MATE - ply - 1)
        if alpha >= beta:
            return alpha

        entry = self.tt.get(key)
        tt_move: chess.Move | None = None
        if entry is not None:
            e_depth, e_flag, e_score, tt_move = entry
            if e_depth >= depth:
                self.stats.tt_hits += 1
                if e_flag == TT_EXACT:
                    return e_score
                if e_flag == TT_LOWER and e_score >= beta:
                    return e_score
                if e_flag == TT_UPPER and e_score <= alpha:
                    return e_score

        pv_node = beta - alpha > 1
        static_eval: int | None = None

        if not pv_node and not in_check:
            static_eval = self.evaluate(board)
            # reverse futility pruning
            if depth <= 2 and static_eval - FUTILITY_MARGIN[depth] >= beta:
                return static_eval
            # null-move pruning, only with non-pawn material for the side to move
            if depth >= 3 and static_eval >= beta and self._has_pieces(board):
                board.push(chess.Move.null())
                self.path.append(key)
                try:
                    score = -self._negamax(
                        board, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, ply + 1
                    )
                finally:
                    self.path.pop()
                    board.pop()
                if score >= beta and abs(score) < MATE_BOUND:
                    return score

        moves = self._ordered_moves(board, tt_move, ply)
        if not moves:
            return -MATE + ply if in_check else DRAW

        best = -INF
        best_move: chess.Move | None = None
        flag = TT_UPPER
        self.path.append(key)
        try:
            for i, move in enumerate(moves):
                is_capture = board.is_capture(move)
                gives_check = board.gives_check(move)
                quiet = not is_capture and move.promotion is None and not gives_check
                # futility pruning of late quiet moves at shallow depth
                if (
                    quiet
                    and not in_check
                    and depth <= 2
                    and i > 0
                    and static_eval is not None
                    and static_eval + FUTILITY_MARGIN[depth] <= alpha
                ):
                    continue
                board.push(move)
                reduce = 0
                if quiet and depth >= LMR_MIN_DEPTH and i >= LMR_MIN_MOVE and not in_check:
                    reduce = 1 if i < 12 else 2
                if i == 0:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self._negamax(board, depth - 1 - reduce, -alpha - 1, -alpha, ply + 1)
                    if score > alpha and reduce:
                        score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if score > alpha and score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
                board.pop()
                if score > best:
                    best = score
                    best_move = move
                if score > alpha:
                    alpha = score
                    flag = TT_EXACT
                    if alpha >= beta:
                        flag = TT_LOWER
                        if quiet or not is_capture:
                            self._store_killer(ply, move)
                            hk = (board.turn, move.from_square, move.to_square)
                            self.history[hk] = self.history.get(hk, 0) + depth * depth
                        break
        finally:
            self.path.pop()

        self.tt[key] = (depth, flag, best, best_move)
        return best

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self._check_clock()
        self.stats.nodes += 1
        self.stats.qnodes += 1
        if ply > self.stats.seldepth:
            self.stats.seldepth = ply

        in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
            if ply >= MAX_PLY:
                return DRAW
            best = -INF
            moves.sort(key=lambda m: -self._mvv_lva(board, m))
            for move in moves:
                board.push(move)
                score = -self._quiescence(board, -beta, -alpha, ply + 1)
                board.pop()
                if score > best:
                    best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
            return best

        stand_pat = self.evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        if ply >= MAX_PLY:
            return stand_pat

        captures = list(board.generate_legal_captures())
        # promotions that are not captures matter too
        for move in board.generate_legal_moves(board.pawns & board.occupied_co[board.turn]):
            if move.promotion == chess.QUEEN and not board.is_capture(move):
                captures.append(move)
        captures.sort(key=lambda m: -self._mvv_lva(board, m))
        best = stand_pat
        for move in captures:
            victim = board.piece_type_at(move.to_square)
            if victim is not None:
                gain = MVV[victim]
            elif move.promotion is None:
                gain = 100  # en passant
            else:
                gain = 800  # promotion
            if stand_pat + gain + DELTA_MARGIN <= alpha and move.promotion is None:
                continue  # delta pruning
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
        return best

    # ------------------------------------------------------------------ ordering

    def _mvv_lva(self, board: chess.Board, move: chess.Move) -> int:
        victim = board.piece_type_at(move.to_square)
        attacker = board.piece_type_at(move.from_square) or 1
        score = 0
        if victim is not None:
            score = MVV[victim] * 10 - LVA[attacker]
        elif board.is_en_passant(move):
            score = MVV[chess.PAWN] * 10 - LVA[chess.PAWN]
        if move.promotion is not None:
            score += MVV[move.promotion]
        return score

    def _ordered_moves(
        self, board: chess.Board, tt_move: chess.Move | None, ply: int
    ) -> list[chess.Move]:
        killers = self.killers[ply]
        turn = board.turn
        scored: list[tuple[int, chess.Move]] = []
        for move in board.legal_moves:
            if move == tt_move:
                score = 10_000_000
            else:
                victim = board.piece_type_at(move.to_square)
                if victim is not None or board.is_en_passant(move):
                    score = 1_000_000 + self._mvv_lva(board, move)
                elif move.promotion is not None:
                    score = 900_000 + MVV[move.promotion]
                elif move == killers[0]:
                    score = 800_000
                elif move == killers[1]:
                    score = 790_000
                else:
                    score = self.history.get((turn, move.from_square, move.to_square), 0)
            scored.append((score, move))
        scored.sort(key=lambda item: -item[0])
        return [move for _, move in scored]

    def _store_killer(self, ply: int, move: chess.Move) -> None:
        row = self.killers[ply]
        if row[0] != move:
            row[1] = row[0]
            row[0] = move

    @staticmethod
    def _has_pieces(board: chess.Board) -> bool:
        us = board.occupied_co[board.turn]
        return bool(us & (board.knights | board.bishops | board.rooks | board.queens))
