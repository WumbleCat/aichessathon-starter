"""Alpha-beta search with an optional policy-network prior for move ordering.

Iterative deepening, aspiration windows, principal variation search, transposition table,
quiescence search, killer and history heuristics, null-move pruning, late move reductions,
futility pruning and check extensions. Everything is written from scratch for this project.

The policy hook is a callable ``prior(board) -> dict[chess.Move, float]``. It is consulted at the
root and at interior nodes whose remaining depth is at least ``policy_min_depth``; those are the
nodes where ordering matters most and where one network call is cheap relative to the subtree.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import chess
from pn_eval import MATE, MATE_BOUND, PIECE_VALUE, evaluate

Prior = Callable[[chess.Board], dict[chess.Move, float]]

MAX_PLY = 96
INF = MATE + 1
TT_MAX_ENTRIES = 1_500_000
EXACT, LOWER, UPPER = 0, 1, 2
NODE_CHECK_MASK = 127  # check the clock every 128 nodes

FUTILITY_MARGIN = (0, 120, 260, 420)
RFP_MARGIN = 110
NULL_MIN_DEPTH = 3
LMR_MIN_DEPTH = 3
LMR_MIN_MOVES = 3
DELTA_MARGIN = 200
QS_MAX_PLY = 8
LABEL_WINDOW = 300  # cp below the best root move beyond which a label only needs a bound
POLICY_HIGH = 0.20
POLICY_LOW = 0.02


class OutOfTime(Exception):
    pass


@dataclass
class Stats:
    nodes: int = 0
    qnodes: int = 0
    tt_hits: int = 0
    seldepth: int = 0
    policy_calls: int = 0
    depth: int = 0


@dataclass
class SearchResult:
    move: chess.Move | None
    score: int
    depth: int
    pv: list[chess.Move]
    elapsed: float
    stats: Stats
    root_scores: dict[chess.Move, int] = field(default_factory=dict)


def _first(t: tuple) -> int:
    return t[0]


def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_type_at(move.to_square)
    if victim is None:
        if board.is_en_passant(move):
            victim = chess.PAWN
        else:
            return 0
    attacker = board.piece_type_at(move.from_square) or chess.PAWN
    return PIECE_VALUE[victim] * 8 - attacker


class Searcher:
    def __init__(
        self,
        prior: Prior | None = None,
        policy_min_depth: int = 4,
        policy_root: bool = True,
        policy_lmr: bool = True,
    ) -> None:
        self.prior = prior
        self.policy_min_depth = policy_min_depth
        self.policy_root = policy_root
        self.policy_lmr = policy_lmr
        self.tt: dict[object, tuple[int, int, int, chess.Move | None]] = {}
        self.policy_cache: dict[object, dict[chess.Move, float]] = {}
        self.history = [[0] * 4096, [0] * 4096]
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 2)]
        self.rep: dict[object, int] = {}  # repetition counts (game history + search path)
        self.stats = Stats()
        self.deadline = 0.0
        self.node_limit = 0
        self.clocked = False
        self.board = chess.Board()

    # ---------------------------------------------------------------- game-level state
    def new_game(self) -> None:
        self.tt.clear()
        self.policy_cache.clear()
        self.rep.clear()
        self.history = [[0] * 4096, [0] * 4096]
        self.killers = [[None, None] for _ in range(MAX_PLY + 2)]

    def remember_position(self, board: chess.Board) -> None:
        key = board._transposition_key()
        self.rep[key] = self.rep.get(key, 0) + 1

    def _age_history(self) -> None:
        for side in self.history:
            for i in range(4096):
                side[i] >>= 1

    # ---------------------------------------------------------------- public search
    def search(
        self,
        board: chess.Board,
        max_depth: int,
        time_budget: float,
        want_root_scores: bool = False,
        node_limit: int = 0,
    ) -> SearchResult:
        started = time.perf_counter()
        self.deadline = started + time_budget
        self.node_limit = node_limit
        self.board = board.copy(stack=False)
        b = self.board
        self.stats = Stats()
        self._age_history()
        if len(self.tt) > TT_MAX_ENTRIES:
            self.tt.clear()
        if len(self.policy_cache) > 200_000:
            self.policy_cache.clear()

        root_moves = list(b.legal_moves)
        if not root_moves:
            return SearchResult(None, 0, 0, [], 0.0, self.stats)
        root_key = b._transposition_key()
        root_prior: dict[chess.Move, float] | None = None
        if self.prior is not None and self.policy_root:
            root_prior = self._get_prior(b, root_key)
        # first-iteration ordering: prior if available, otherwise captures first
        if root_prior:
            root_moves.sort(key=lambda m: root_prior.get(m, 0.0), reverse=True)
        else:
            root_moves.sort(key=lambda m: _mvv_lva(b, m), reverse=True)

        best_move = root_moves[0]
        best_score = -INF
        pv: list[chess.Move] = [best_move]
        root_scores: dict[chess.Move, int] = {}
        completed_depth = 0
        alpha, beta = -INF, INF
        depth = 1
        while depth <= max_depth:
            self.clocked = depth > 1
            self.stats.depth = depth
            try:
                score, move, scores = self._root(
                    b, root_moves, depth, alpha, beta, root_prior, want_root_scores
                )
            except OutOfTime:
                break
            if score <= alpha or score >= beta:
                # aspiration failed: re-search with a full window at the same depth
                alpha, beta = -INF, INF
                continue
            best_score, best_move = score, move
            root_scores = scores
            completed_depth = depth
            pv = self._extract_pv(b, best_move)
            # put the best move first for the next iteration
            root_moves.remove(best_move)
            root_moves.insert(0, best_move)
            if abs(best_score) >= MATE_BOUND:
                break
            if depth >= 4:
                alpha, beta = best_score - 35, best_score + 35
            else:
                alpha, beta = -INF, INF
            depth += 1
            # do not start an iteration that is very unlikely to finish
            elapsed = time.perf_counter() - started
            if self.clocked and elapsed > time_budget * 0.55:
                break
        elapsed = time.perf_counter() - started
        return SearchResult(
            best_move, best_score, completed_depth, pv, elapsed, self.stats, root_scores
        )

    # ---------------------------------------------------------------- root
    def _root(
        self,
        b: chess.Board,
        moves: list[chess.Move],
        depth: int,
        alpha: int,
        beta: int,
        prior: dict[chess.Move, float] | None,
        want_scores: bool,
    ) -> tuple[int, chess.Move, dict[chess.Move, int]]:
        best_score = -INF
        best_move = moves[0]
        scores: dict[chess.Move, int] = {}
        orig_alpha = alpha
        key = b._transposition_key()
        for i, move in enumerate(moves):
            b.push(move)
            child_key = b._transposition_key()
            self.rep[child_key] = self.rep.get(child_key, 0) + 1
            try:
                if want_scores:
                    # Labels: moves more than LABEL_WINDOW below the best so far only need an
                    # upper bound (their softmax weight is negligible), so give them a bounded
                    # window; everything inside the window gets an exact score.
                    if i == 0 or best_score <= -MATE_BOUND:
                        score = -self._negamax(b, depth - 1, -INF, INF, 1, True, True)
                    else:
                        floor = best_score - LABEL_WINDOW
                        score = -self._negamax(b, depth - 1, -INF, -floor, 1, True, True)
                elif i == 0:
                    score = -self._negamax(b, depth - 1, -beta, -alpha, 1, True, True)
                else:
                    score = -self._negamax(b, depth - 1, -alpha - 1, -alpha, 1, False, True)
                    if alpha < score < beta:
                        score = -self._negamax(b, depth - 1, -beta, -alpha, 1, True, True)
            finally:
                self.rep[child_key] -= 1
                b.pop()
            scores[move] = score
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
                if not want_scores and alpha >= beta:
                    break
        flag = EXACT if orig_alpha < best_score < beta else (LOWER if best_score >= beta else UPPER)
        self.tt[key] = (depth, flag, best_score, best_move)
        return best_score, best_move, scores

    def _extract_pv(self, b: chess.Board, first: chess.Move) -> list[chess.Move]:
        pv = [first]
        b.push(first)
        pushed = 1
        seen = set()
        try:
            while pushed < 20:
                key = b._transposition_key()
                if key in seen:
                    break
                seen.add(key)
                entry = self.tt.get(key)
                if entry is None or entry[3] is None or entry[3] not in b.legal_moves:
                    break
                pv.append(entry[3])
                b.push(entry[3])
                pushed += 1
        finally:
            for _ in range(pushed):
                b.pop()
        return pv

    # ---------------------------------------------------------------- helpers
    def _check_time(self) -> None:
        if self.clocked and (
            time.perf_counter() >= self.deadline
            or (self.node_limit and self.stats.nodes >= self.node_limit)
        ):
            raise OutOfTime()

    def _get_prior(self, b: chess.Board, key: object) -> dict[chess.Move, float]:
        cached = self.policy_cache.get(key)
        if cached is not None:
            return cached
        assert self.prior is not None
        self.stats.policy_calls += 1
        result = self.prior(b)
        self.policy_cache[key] = result
        return result

    def _ordered_moves(
        self,
        b: chess.Board,
        moves: list[chess.Move],
        tt_move: chess.Move | None,
        ply: int,
        prior: dict[chess.Move, float] | None,
    ) -> list[tuple[int, chess.Move, bool]]:
        """Score every move for ordering. Returns (score, move, is_quiet) sorted best first."""
        killer0, killer1 = self.killers[ply]
        us = b.turn
        them = not us
        hist = self.history[us]
        them_occ = b.occupied_co[them]
        ep = b.ep_square
        pawns = b.pawns
        knights = b.knights
        bishops = b.bishops
        rooks = b.rooks
        queens = b.queens
        scored: list[tuple[int, chess.Move, bool]] = []
        for move in moves:
            if move == tt_move:
                scored.append((10_000_000, move, False))
                continue
            to = move.to_square
            frm = move.from_square
            to_bb = 1 << to
            from_bb = 1 << frm
            if from_bb & pawns:
                attacker = 1
            elif from_bb & knights:
                attacker = 2
            elif from_bb & bishops:
                attacker = 3
            elif from_bb & rooks:
                attacker = 4
            elif from_bb & queens:
                attacker = 5
            else:
                attacker = 6
            if to_bb & them_occ:
                if to_bb & pawns:
                    victim = 1
                elif to_bb & knights:
                    victim = 2
                elif to_bb & bishops:
                    victim = 3
                elif to_bb & rooks:
                    victim = 4
                else:
                    victim = 5
            elif attacker == 1 and to == ep:
                victim = 1
            else:
                victim = 0
            if victim:
                if PIECE_VALUE[victim] >= PIECE_VALUE[attacker] or not b.is_attacked_by(them, to):
                    s = 5_000_000 + PIECE_VALUE[victim] * 8 - attacker
                else:
                    s = -1_000_000 + PIECE_VALUE[victim] * 8 - attacker
                if move.promotion == 5:
                    s += 900
                scored.append((s, move, False))
                continue
            promo = move.promotion
            if promo is not None:
                scored.append((4_000_000 if promo == 5 else 100, move, False))
                continue
            if move == killer0:
                s = 3_000_000
            elif move == killer1:
                s = 2_900_000
            else:
                s = hist[frm * 64 + to]
            if prior is not None:
                # quiet move: the policy prior dominates, killers and history break ties
                s = int(prior.get(move, 0.0) * 1_000_000) + (
                    30_000 if s >= 2_900_000 else min(s, 9_999)
                )
            scored.append((s, move, True))
        scored.sort(key=_first, reverse=True)
        return scored

    # ---------------------------------------------------------------- main search
    def _negamax(
        self,
        b: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        pv_node: bool,
        can_null: bool,
    ) -> int:
        stats = self.stats
        stats.nodes += 1
        if (stats.nodes & NODE_CHECK_MASK) == 0:
            self._check_time()
        if ply > stats.seldepth:
            stats.seldepth = ply

        key = b._transposition_key()
        # draws by repetition or the fifty-move rule
        if self.rep.get(key, 0) > 1 or b.halfmove_clock >= 100:
            return 0
        if ply >= MAX_PLY:
            return evaluate(b)

        # mate distance pruning
        alpha = max(alpha, -MATE + ply)
        beta = min(beta, MATE - ply - 1)
        if alpha >= beta:
            return alpha

        in_check = b.is_check()
        if in_check:
            depth += 1  # check extension

        if depth <= 0:
            return self._qsearch(b, alpha, beta, ply, 0)

        tt_move: chess.Move | None = None
        entry = self.tt.get(key)
        if entry is not None:
            tt_depth, tt_flag, tt_score, tt_move = entry
            if tt_depth >= depth and not pv_node:
                stats.tt_hits += 1
                if tt_flag == EXACT:
                    return tt_score
                if tt_flag == LOWER and tt_score >= beta:
                    return tt_score
                if tt_flag == UPPER and tt_score <= alpha:
                    return tt_score

        static_eval = 0
        if not in_check:
            static_eval = evaluate(b)
            if not pv_node and abs(beta) < MATE_BOUND:
                # reverse futility pruning
                if depth <= 3 and static_eval - RFP_MARGIN * depth >= beta:
                    return static_eval
                # null move pruning
                if (
                    can_null
                    and depth >= NULL_MIN_DEPTH
                    and static_eval >= beta
                    and self._has_non_pawn_material(b)
                ):
                    reduction = 2 + depth // 4
                    b.push(chess.Move.null())
                    try:
                        score = -self._negamax(
                            b, depth - 1 - reduction, -beta, -beta + 1, ply + 1, False, False
                        )
                    finally:
                        b.pop()
                    if score >= beta:
                        return beta if score < MATE_BOUND else score

        moves = list(b.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0

        prior: dict[chess.Move, float] | None = None
        if self.prior is not None and depth >= self.policy_min_depth:
            prior = self._get_prior(b, key)

        ordered = self._ordered_moves(b, moves, tt_move, ply, prior)
        best_score = -INF
        best_move: chess.Move | None = None
        orig_alpha = alpha
        futile = (
            not pv_node
            and not in_check
            and depth <= 3
            and static_eval + FUTILITY_MARGIN[depth] <= alpha
        )
        quiets_tried: list[chess.Move] = []
        hist = self.history[b.turn]

        for i, (order_score, move, is_quiet) in enumerate(ordered):
            if futile and is_quiet and i > 0 and best_score > -MATE_BOUND:
                continue
            b.push(move)
            child_key = b._transposition_key()
            self.rep[child_key] = self.rep.get(child_key, 0) + 1
            try:
                gives_check = b.is_check()
                new_depth = depth - 1
                reduction = 0
                if (
                    depth >= LMR_MIN_DEPTH
                    and i >= LMR_MIN_MOVES
                    and is_quiet
                    and not in_check
                    and not gives_check
                ):
                    reduction = 1
                    if i >= 6:
                        reduction += 1
                    if depth >= 7 and i >= 12:
                        reduction += 1
                    if not pv_node:
                        reduction += 1
                    if prior is not None and self.policy_lmr:
                        p = prior.get(move, 0.0)
                        if p >= POLICY_HIGH:
                            reduction -= 1
                        elif p < POLICY_LOW and depth >= 4:
                            reduction += 1
                    elif order_score >= 2_900_000:
                        reduction -= 1
                    if reduction < 0:
                        reduction = 0
                    if reduction > new_depth - 1:
                        reduction = max(0, new_depth - 1)
                if i == 0:
                    score = -self._negamax(b, new_depth, -beta, -alpha, ply + 1, pv_node, True)
                else:
                    score = -self._negamax(
                        b, new_depth - reduction, -alpha - 1, -alpha, ply + 1, False, True
                    )
                    if score > alpha and reduction > 0:
                        score = -self._negamax(
                            b, new_depth, -alpha - 1, -alpha, ply + 1, False, True
                        )
                    if score > alpha and score < beta and pv_node:
                        score = -self._negamax(b, new_depth, -beta, -alpha, ply + 1, True, True)
            finally:
                self.rep[child_key] -= 1
                b.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if is_quiet:
                        killers = self.killers[ply]
                        if killers[0] != move:
                            killers[1] = killers[0]
                            killers[0] = move
                        bonus = depth * depth
                        hist[move.from_square * 64 + move.to_square] += bonus
                        for q in quiets_tried:
                            hist[q.from_square * 64 + q.to_square] -= bonus
                    break
            if is_quiet:
                quiets_tried.append(move)

        flag = EXACT if orig_alpha < best_score < beta else (LOWER if best_score >= beta else UPPER)
        old = self.tt.get(key)
        if old is None or old[0] <= depth or flag == EXACT:
            self.tt[key] = (depth, flag, best_score, best_move)
        return best_score

    @staticmethod
    def _has_non_pawn_material(b: chess.Board) -> bool:
        own = b.occupied_co[b.turn]
        return bool(own & (b.knights | b.bishops | b.rooks | b.queens))

    # ---------------------------------------------------------------- quiescence
    def _qsearch(self, b: chess.Board, alpha: int, beta: int, ply: int, qply: int) -> int:
        stats = self.stats
        stats.nodes += 1
        stats.qnodes += 1
        if (stats.nodes & NODE_CHECK_MASK) == 0:
            self._check_time()
        if ply > stats.seldepth:
            stats.seldepth = ply
        if ply >= MAX_PLY or qply >= QS_MAX_PLY:
            return evaluate(b)

        in_check = b.is_check()
        if in_check:
            # evade check: search every legal move
            moves = list(b.legal_moves)
            if not moves:
                return -MATE + ply
            best = -INF
            moves.sort(key=lambda m: _mvv_lva(b, m), reverse=True)
            for move in moves:
                b.push(move)
                try:
                    score = -self._qsearch(b, -beta, -alpha, ply + 1, qply + 1)
                finally:
                    b.pop()
                if score > best:
                    best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
            return best

        stand_pat = evaluate(b)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        best = stand_pat

        captures: list[tuple[int, chess.Move]] = []
        opp = not b.turn
        for move in b.generate_legal_captures():
            victim = b.piece_type_at(move.to_square)
            if victim is None:
                victim = chess.PAWN
            attacker = b.piece_type_at(move.from_square) or chess.PAWN
            gain = PIECE_VALUE[victim]
            if move.promotion == chess.QUEEN:
                gain += 800
            # delta pruning
            if stand_pat + gain + DELTA_MARGIN <= alpha:
                continue
            # skip obviously losing captures
            if PIECE_VALUE[victim] < PIECE_VALUE[attacker] and b.is_attacked_by(
                opp, move.to_square
            ):
                continue
            captures.append((PIECE_VALUE[victim] * 8 - attacker, move))
        # queen promotions without capture
        seventh = b.pawns & b.occupied_co[b.turn] & (chess.BB_RANK_7 if b.turn else chess.BB_RANK_2)
        if seventh and stand_pat + 800 + DELTA_MARGIN > alpha:
            for move in b.generate_legal_moves(seventh, ~b.occupied):
                if move.promotion == chess.QUEEN:
                    captures.append((7_000, move))
        captures.sort(key=lambda t: t[0], reverse=True)
        for _, move in captures:
            b.push(move)
            try:
                score = -self._qsearch(b, -beta, -alpha, ply + 1, qply + 1)
            finally:
                b.pop()
            if score > best:
                best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
        return best
