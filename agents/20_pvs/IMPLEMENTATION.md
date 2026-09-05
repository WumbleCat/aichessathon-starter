# 20_pvs implementation plan

Architecture: iterative deepening -> aspiration -> PVS negamax with TT, null move,
reverse futility / futility, LMR, check extension, staged move ordering (TT, SEE captures,
promotions, killers, history, bad captures) and quiescence with delta/SEE pruning.

## Key design decision: numba bitboard core, python-chess safety net

Measured facts (see memory notes and probes on 2026-09-04):
- python-chess searchers on this machine reach only ~4-7k nodes/s, so depth 4-5 at
  contest time controls. The whole selective-search architecture pays off only with
  ordering-sensitive depth, which needs 50-100x more nodes.
- numba is preinstalled on the platform, and a jitted recursive search can read the wall
  clock through `objmode` at negligible cost (probe: 48M calls/s with a clock check every
  4096 calls).
- python-chess's `BB_*_ATTACKS` tables are edge-stripped: rank/file masks are 6 bits and
  the bishop mask is at most 9 bits, so compact PEXT-style tables (64x64, 64x64, 64x512)
  can be built at import in well under a second.

So the engine is:

1. `pvs_board.py`  numba board: bitboards per piece, mailbox, side, castling, ep, halfmove,
   Zobrist hash and incremental tapered material+PST score. Pseudo-legal generation plus
   make/`is own king attacked`/unmake for legality. Move = from | to<<6 | promo<<12 |
   flags<<16.
2. `pvs_eval.py`   evaluation: incremental material + tapered PST, plus pawn structure
   (passed/isolated/doubled), bishop pair, rook on open file, simple king shelter.
3. `pvs_search.py` the jitted search: negamax PVS with all selective features behind
   integer feature flags in a `params` array so A/B games can disable any one heuristic.
   TT as two int64 numpy arrays (2^20 entries, 16 MB). Killers, history, repetition stack.
4. `agent.py`      entrypoint. At import: build tables, warm the JIT on a real position
   (compile lands in the 90 s init budget). `get_move`: parse FEN, pick a legal fallback
   move from python-chess FIRST, run the engine with a time budget, verify the result is
   in `board.legal_moves`, else return the fallback. Own repetition memory of game hashes.

## Phases

- Phase 1: minimal python-chess agent that returns a legal move (done first, kept as the
  fallback path so a JIT failure can never lose by crash).
- Phase 2: numba board + perft vs python-chess (correctness oracle), then search features in
  the README order A..K with a toggle for each.
- Phase 3: tests in `tests/` (moves: capture, check, evasion, mate, stalemate, castling both
  sides, en passant, four promotions; clocks 50..120000 ms; repeated calls; init time).
- Phase 4: profile, cut per-node cost, tune time management.
- Phase 5/6: harness arena vs baselines both colours, record in RESULTS.md, fix weaknesses.

## Time management

Budget per move = clamp(time_left/30 + increment*0.8, lower, upper), never more than
time_left/4, and the jitted search checks the clock every 2048 nodes against a hard
deadline. Depth 1 always completes (no clock). Under 100 ms left: return the fallback or a
1-ply search only.
