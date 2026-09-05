# Numba compile time of the engine (2026-09-05, session 4)

The platform gives `import agent` 90 s.  The engine compiled in 119-153 s CPU on the shared
box (see RESULTS.md), so games started on the python fallback.  This session cut that.

Measured with `tools/prof_compile.py` (fresh process, no on-disk cache, same warm-up as
`agent._build_engine`; the box was running ~50 other python processes, wall time is inflated).
Numba's per-function pipeline times are nested (a caller's inference includes the callees it
compiles), so read the overload counts and the totals, not the sum of the rows.

| | wall | CPU | negamax overloads | qsearch | gen_moves | is_attacked | slider_attacks_dir | _add |
|---|---|---|---|---|---|---|---|---|
| before | 258.0 s | 116.8 s | 3 | 2 | 2 | 11 | 88 | 22 |
| after | 93.6 s | 44.2 s | 1 | 1 | 2 | 1 | 1 | inlined |

The test process built the engine in 44.1 s CPU (82.8 s wall) the same way.

## What cost the time

Numba compiles one specialisation per distinct argument *type*, and a Python int/bool constant
at a call site (or a module-level int global) is a `Literal` type, not `int64`.  So:

- `negamax(..., ply=0, ...)` from `search_root` and the `True`/`False` `do_null` arguments at
  the recursive call sites produced three full compiles of `negamax` (and two of `qsearch`,
  `do_move`, `static_eval`, `make_move`, ... via the literal `ply`).  Each compile of the 19-
  argument recursive function was dominated by type inference (`nopython_type_inference`
  222 s wall for the first overload).
- `_castle_ok(P, CASTLE_WK, WK_EMPTY, H1, WR, E1, F1, G1, them)` with global int constants
  compiled 4 versions of `_castle_ok`, 11 of `is_attacked` (one per literal square), 11 each
  of `bishop_attacks`/`rook_attacks`, and 88 of `slider_attacks_dir` (11 x 8 literal
  directions).
- `_add(out, n, frm, to, 0, F_CAPTURE)` and friends: 22 specialisations of the move-append
  helper, 8 of `_add_promos`.
- `nnue.copy_acc` did `acc[ply + 1] = acc[ply]`, an array-slice assignment that lowers to a
  generic broadcasting copy: 17.5 s of native lowering for a two-line function.

## What changed

- `csearch.qsearch/negamax/search_root` take one tuple `S` of the fourteen state arrays
  (indices `X_*`) instead of fourteen parameters; `Searcher.search` builds it once.
- The null-move permission is a control slot (`C_NULL_PLY`: the ply that must not answer a
  null move) instead of a literal `do_null` argument; the root call passes `np.int64(0)`.
- The five recursive call sites of the PVS/LMR move loop became one loop with two states
  (re-search at full depth, then with the full window), plus the null-move call: two sites.
- `cboard`: castling is table-driven (`CASTLE_TAB`), the slider directions come from a loop
  variable, `_add`/`_add_promos`/`make_piece` are `inline="always"` (`jitconf.jit_inline`),
  and the `captures_only` flag is passed as the int64 scalars `cb.ALL_MOVES`/`cb.CAPTURES_ONLY`
  everywhere (search, perft, PV extraction, tests).
- `nnue.copy_acc` copies with explicit loops.

None of this changes a search decision: `tools/compare_search.py` (fixed depth 6, cleared TT,
ten positions) gives identical move / score / depth / node lines for the reference engine
(branch `feature/agent-21-nnue`) and the refactored one, see `fingerprint_old.txt` and
`fingerprint_new.txt`.  All 26 unit tests pass (`tests_after.log`); ruff and mypy --strict are
clean.

## Still open

`search_root` + `negamax` + `qsearch` inference is now ~40 s of the 44 s CPU.  Splitting the
move loop of `negamax` into helpers (pruning decisions, LMR reduction, history/killer update)
would cut inference further if the platform core turns out slower than this box.
`INIT_WAIT_S` in agent.py stays at 70 s; the compile thread now finishes well inside it.
