"""Alpha-beta search used both as the runtime engine and as the training-data teacher.

Iterative deepening, principal variation search, transposition table, quiescence with delta
pruning, MVV-LVA / killer / history ordering, null-move pruning, reverse futility, futility,
late move reductions and a check extension. A policy callback (the Chessformer network) can take
over move ordering and steer reductions at nodes with enough remaining depth.

Written from scratch for this project; only textbook algorithms are used.
"""

import time
from collections.abc import Callable

import chess
from cf_eval import MATE, MATE_BOUND, PIECE_VALUE_MG, evaluate, material_only

INF = MATE + 1
MAX_PLY = 96
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
NODE_CHECK_MASK = 127

PolicyFn = Callable[[chess.Board], dict[chess.Move, float]]


class OutOfTime(Exception):
    pass


class SearchResult:
    __slots__ = ("depth", "move", "nodes", "nps", "qnodes", "score", "seldepth", "time")

    def __init__(self) -> None:
        self.move: chess.Move | None = None
        self.score = 0
        self.depth = 0
        self.seldepth = 0
        self.nodes = 0
        self.qnodes = 0
        self.time = 0.0
        self.nps = 0.0


class Searcher:
    def __init__(
        self,
        policy_fn: PolicyFn | None = None,
        policy_min_depth: int = 3,
        tt_max_entries: int = 2_000_000,
    ) -> None:
        self.policy_fn = policy_fn
        self.policy_min_depth = policy_min_depth
        self.tt: dict[object, tuple[int, int, int, chess.Move | None]] = {}
        self.tt_max_entries = tt_max_entries
        self.history = [[[0] * 64 for _ in range(64)] for _ in range(2)]
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 2)]
        self.game_history: set[object] = set()
        self.path: list[object] = []
        self.deadline = 0.0
        self.nodes = 0
        self.qnodes = 0
        self.tt_hits = 0
        self.policy_calls = 0
        self.seldepth = 0
        self.policy_cache: dict[object, dict[chess.Move, float]] = {}
        # node-budget mode (for load-independent A/B matches): the search stops after this many
        # nodes, and every network call is charged policy_node_cost nodes so that the cost of
        # consulting the network is part of the budget
        self.node_limit = float("inf")
        self.policy_node_cost = 0
        # consult the network only at PV nodes (open window) instead of every node deep enough
        self.policy_pv_only = False
        # ... and only within policy_rel_depth plies of the root, where subtrees are large enough
        # for better ordering to pay for a network call (64 = everywhere)
        self.policy_rel_depth = 64
        self.root_depth = 0
        # harvesting of exact nodes for training data (list of (fen, best_uci, score, depth))
        self.harvest: list[tuple[str, str, int, int]] | None = None
        self.harvest_min_depth = 2

    # ------------------------------------------------------------------ public

    def note_position(self, board: chess.Board) -> None:
        """Remember a position that occurred in the game (for repetition detection)."""
        self.game_history.add(board._transposition_key())

    def search(
        self,
        board: chess.Board,
        budget_s: float,
        max_depth: int = 64,
        verbose: bool = False,
        max_nodes: int | None = None,
    ) -> SearchResult:
        """Iterative deepening. Depth 1 is never interrupted, so a legal move always results.

        The search stops at the earlier of budget_s seconds and max_nodes nodes (if given).
        """
        start = time.perf_counter()
        self.deadline = start + budget_s
        node_budget = float(max_nodes) if max_nodes else float("inf")
        self.nodes = 0
        self.qnodes = 0
        self.tt_hits = 0
        self.policy_calls = 0
        self.seldepth = 0
        self.path = []
        self.policy_cache.clear()
        if len(self.tt) > self.tt_max_entries:
            self.tt.clear()
        for side in self.history:
            for row in side:
                for i in range(64):
                    row[i] >>= 1
        result = SearchResult()
        legal = list(board.legal_moves)
        if not legal:
            return result
        result.move = legal[0]
        prev_score = 0
        for depth in range(1, max_depth + 1):
            self.root_depth = depth
            self.root_best: chess.Move | None = None
            self.root_score = -INF
            self.root_moves_done = 0
            if depth == 1:
                self.deadline = float("inf")
                self.node_limit = float("inf")
            else:
                self.deadline = start + budget_s
                self.node_limit = node_budget
            try:
                if depth >= 4:
                    score = self._aspiration(board, depth, prev_score)
                else:
                    score = self._root(board, depth, -INF, INF)
            except OutOfTime:
                del self.path[:]
                if self.root_best is not None and self.root_moves_done > 0:
                    result.move = self.root_best
                    result.score = self.root_score
                break
            result.move = self.root_best if self.root_best is not None else result.move
            result.score = score
            result.depth = depth
            prev_score = score
            elapsed = time.perf_counter() - start
            if verbose:
                print(
                    f"depth {depth} score {score} move {result.move} nodes {self.nodes} "
                    f"q {self.qnodes} seldepth {self.seldepth} time {elapsed:.2f}"
                )
            if abs(score) >= MATE_BOUND and depth >= 3:
                break
            # a further iteration usually costs 2-4x the last one; do not start one we cannot end
            if elapsed > budget_s * 0.45 or self.nodes > node_budget * 0.45:
                break
        result.nodes = self.nodes
        result.qnodes = self.qnodes
        result.seldepth = self.seldepth
        result.time = time.perf_counter() - start
        result.nps = self.nodes / result.time if result.time > 0 else 0.0
        return result

    # ------------------------------------------------------------------ root

    def _aspiration(self, board: chess.Board, depth: int, prev: int) -> int:
        window = 30
        alpha, beta = prev - window, prev + window
        while True:
            score = self._root(board, depth, alpha, beta)
            if score <= alpha:
                alpha = max(-INF, alpha - window * 2)
            elif score >= beta:
                beta = min(INF, beta + window * 2)
            else:
                return score
            window *= 2
            if window > 600:
                alpha, beta = -INF, INF

    def _root(self, board: chess.Board, depth: int, alpha: int, beta: int) -> int:
        key = board._transposition_key()
        entry = self.tt.get(key)
        tt_move = entry[3] if entry is not None else None
        moves = self._order_moves(board, list(board.legal_moves), tt_move, 0, depth)
        best = -INF
        best_move = None
        self.root_moves_done = 0
        self.path = [key]
        for i, move in enumerate(moves):
            board.push(move)
            gives_check = board.is_check()
            self.path.append(board._transposition_key())
            try:
                if i == 0:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, 1, True, gives_check)
                else:
                    score = -self._negamax(
                        board, depth - 1, -alpha - 1, -alpha, 1, True, gives_check
                    )
                    if alpha < score < beta:
                        score = -self._negamax(
                            board, depth - 1, -beta, -alpha, 1, True, gives_check
                        )
            finally:
                self.path.pop()
                board.pop()
            self.root_moves_done += 1
            if score > best:
                best = score
                best_move = move
                if score > self.root_score or i == 0:
                    self.root_best = move
                    self.root_score = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
        if best_move is not None:
            flag = TT_EXACT if alpha < beta else TT_LOWER
            self.tt[key] = (depth, best, flag, best_move)
        return best

    # ------------------------------------------------------------------ main search

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        do_null: bool,
        in_check: bool,
    ) -> int:
        self.nodes += 1
        if (self.nodes & NODE_CHECK_MASK) == 0 and (
            self.nodes > self.node_limit or time.perf_counter() > self.deadline
        ):
            raise OutOfTime
        if ply > self.seldepth:
            self.seldepth = ply
        key = self.path[-1]
        # draws by repetition (search path or game history) and the fifty-move rule
        if board.halfmove_clock >= 100:
            return 0
        if key in self.game_history:
            return 0
        if board.halfmove_clock >= 4:
            path = self.path
            n = len(path) - 1
            i = n - 2
            stop = max(0, n - board.halfmove_clock)
            while i >= stop:
                if path[i] == key:
                    return 0
                i -= 2
        # mate distance pruning
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha
        if in_check:
            depth += 1
        if depth <= 0 or ply >= MAX_PLY:
            return self._qsearch(board, alpha, beta, ply, 0)

        pv_node = beta - alpha > 1
        entry = self.tt.get(key)
        tt_move = None
        if entry is not None:
            e_depth, e_score, e_flag, tt_move = entry
            if e_depth >= depth and not pv_node:
                self.tt_hits += 1
                if e_score > MATE_BOUND:
                    e_score -= ply
                elif e_score < -MATE_BOUND:
                    e_score += ply
                if e_flag == TT_EXACT:
                    return e_score
                if e_flag == TT_LOWER and e_score >= beta:
                    return e_score
                if e_flag == TT_UPPER and e_score <= alpha:
                    return e_score

        static = 0
        if not in_check:
            static = evaluate(board)
            if not pv_node and abs(beta) < MATE_BOUND:
                # reverse futility pruning
                if depth <= 3 and static - 90 * depth >= beta:
                    return static
                # null move pruning
                if do_null and depth >= 2 and static >= beta and material_only(board) > 0:
                    r = 3 + depth // 4
                    board.push(chess.Move.null())
                    self.path.append(board._transposition_key())
                    try:
                        score = -self._negamax(
                            board, depth - 1 - r, -beta, -beta + 1, ply + 1, False, False
                        )
                    finally:
                        self.path.pop()
                        board.pop()
                    if score >= beta:
                        return beta

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0
        moves = self._order_moves(board, moves, tt_move, ply, depth, pv_node)
        priors = self._priors_cache if self._priors_valid_key == key else None

        best = -INF
        best_move = None
        searched = 0
        alpha_orig = alpha
        futile = (
            not pv_node
            and not in_check
            and depth <= 2
            and abs(alpha) < MATE_BOUND
            and static + 120 * depth + 40 <= alpha
        )
        killers = self.killers[ply]
        hist = self.history[board.turn]
        for move in moves:
            is_capture = board.is_capture(move)
            is_quiet = not is_capture and move.promotion is None
            if futile and is_quiet and searched > 0:
                continue
            board.push(move)
            gives_check = board.is_check()
            self.path.append(board._transposition_key())
            try:
                if searched == 0:
                    score = -self._negamax(
                        board, depth - 1, -beta, -alpha, ply + 1, True, gives_check
                    )
                else:
                    reduction = 0
                    if (
                        depth >= 3
                        and is_quiet
                        and not in_check
                        and not gives_check
                        and searched >= 2
                    ):
                        reduction = 1
                        if searched >= 6:
                            reduction += 1
                        if depth >= 6 and searched >= 12:
                            reduction += 1
                        if priors is not None:
                            p = priors.get(move, 0.0)
                            if p < 0.02:
                                reduction += 1
                            elif p > 0.25:
                                reduction -= 1
                        if move == killers[0] or move == killers[1]:
                            reduction -= 1
                        if reduction < 0:
                            reduction = 0
                        if reduction > depth - 1:
                            reduction = depth - 1
                    score = -self._negamax(
                        board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, True, gives_check
                    )
                    if score > alpha and reduction > 0:
                        score = -self._negamax(
                            board, depth - 1, -alpha - 1, -alpha, ply + 1, True, gives_check
                        )
                    if alpha < score < beta:
                        score = -self._negamax(
                            board, depth - 1, -beta, -alpha, ply + 1, True, gives_check
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
                        if is_quiet:
                            if killers[0] != move:
                                killers[1] = killers[0]
                                killers[0] = move
                            hist[move.from_square][move.to_square] += depth * depth
                        break
        if best_move is None:
            # every move was pruned by futility; fall back to the static bound
            return alpha_orig
        if best <= alpha_orig:
            flag = TT_UPPER
        elif best >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
            if self.harvest is not None and depth >= self.harvest_min_depth:
                self.harvest.append((board.fen(), best_move.uci(), best, depth))
        store = best
        if store > MATE_BOUND:
            store += ply
        elif store < -MATE_BOUND:
            store -= ply
        self.tt[key] = (depth, store, flag, best_move)
        return best

    # ------------------------------------------------------------------ quiescence

    def _qsearch(self, board: chess.Board, alpha: int, beta: int, ply: int, qply: int) -> int:
        self.nodes += 1
        self.qnodes += 1
        if (self.nodes & NODE_CHECK_MASK) == 0 and (
            self.nodes > self.node_limit or time.perf_counter() > self.deadline
        ):
            raise OutOfTime
        if ply > self.seldepth:
            self.seldepth = ply
        in_check = board.is_check() and qply < 4
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
            best = -INF
            moves.sort(key=lambda m: self._mvv_lva(board, m), reverse=True)
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
        stand = evaluate(board)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        best = stand
        captures = list(board.generate_legal_captures())
        if not captures and qply == 0:
            # queen promotions are worth a look even when they do not capture
            for move in board.generate_legal_moves(
                board.pawns & chess.BB_RANK_7 | board.pawns & chess.BB_RANK_2
            ):
                if move.promotion == chess.QUEEN:
                    captures.append(move)
        captures.sort(key=lambda m: self._mvv_lva(board, m), reverse=True)
        for move in captures:
            victim = board.piece_type_at(move.to_square)
            gain = PIECE_VALUE_MG[victim] if victim is not None else 82
            if move.promotion is not None:
                gain += PIECE_VALUE_MG[move.promotion] - 82
            if stand + gain + 200 <= alpha:
                continue  # delta pruning
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

    # ------------------------------------------------------------------ ordering

    _priors_cache: dict[chess.Move, float] = {}  # noqa: RUF012 (reassigned per node, never mutated)
    _priors_valid_key: object = None

    @staticmethod
    def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
        victim = board.piece_type_at(move.to_square)
        v = (
            PIECE_VALUE_MG[victim]
            if victim is not None
            else (82 if board.is_en_passant(move) else 0)
        )
        attacker = board.piece_type_at(move.from_square)
        a = PIECE_VALUE_MG[attacker] if attacker is not None else 0
        s = v * 16 - a // 8
        if move.promotion is not None:
            s += PIECE_VALUE_MG[move.promotion]
        return s

    def _order_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        tt_move: chess.Move | None,
        ply: int,
        depth: int,
        pv_node: bool = True,
    ) -> list[chess.Move]:
        self._priors_valid_key = None
        killers = self.killers[ply]
        hist = self.history[board.turn]
        priors: dict[chess.Move, float] | None = None
        if (
            self.policy_fn is not None
            and depth >= self.policy_min_depth
            and depth >= self.root_depth - self.policy_rel_depth
            and len(moves) > 1
            and (pv_node or not self.policy_pv_only)
        ):
            key = self.path[-1] if self.path else board._transposition_key()
            priors = self.policy_cache.get(key)
            if priors is None:
                self.policy_calls += 1
                self.nodes += self.policy_node_cost
                priors = self.policy_fn(board)
                self.policy_cache[key] = priors
            self._priors_cache = priors
            self._priors_valid_key = key

        # staged ordering: hash move, captures/promotions by MVV-LVA, killers, then quiet moves.
        # Quiet moves go by the network prior when one is available (history breaks ties),
        # otherwise by history alone.
        def score(m: chess.Move) -> float:
            if m == tt_move:
                return 1_000_000_000.0
            if board.is_capture(m) or m.promotion is not None:
                return 1_000_000.0 + self._mvv_lva(board, m)
            if m == killers[0]:
                return 900_000.0
            if m == killers[1]:
                return 890_000.0
            h = hist[m.from_square][m.to_square]
            if priors is not None:
                return priors.get(m, 0.0) * 500_000.0 + min(h, 99_999)
            return float(h)

        moves.sort(key=score, reverse=True)
        return moves
