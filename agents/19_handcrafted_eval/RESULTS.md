# 19 — Handcrafted Evaluation: measurements

Everything below was measured on the shared development machine while 60-80 other Python
processes (other agents' training and arena runs) kept all 16 cores at 100 %. Wall-clock numbers
are therefore 3-10x worse than an idle core and are noted as such; node counts at fixed depth
are load independent and are the yardstick for search changes.

## Status (2026-09-05)

| Check | Result |
| --- | --- |
| `unittest` (46 tests, numba path) | pass |
| `unittest` (46 tests, `HCE_NO_NUMBA=1` pure-Python path) | pass, after depth 1 was put under the clock |
| ruff / mypy strict on the agent directory | clean |
| `harness.arena` vs `baselines/greedy`, 6 games, 10 s + 0.1 s | +6 =0 -0, all by checkmate |
| `harness.arena` vs `baselines/minimax`, 6 games, 10 s + 0.1 s | +6 =0 -0, the baseline flagged every game under the load |
| `harness.arena` vs `baselines/numba`, 4 games, 10 s + 0.1 s | +4 =0 -0, the baseline flagged every game under the load |
| our clock over those 16 harness games at 100 % machine load | never flagged, never illegal, never crashed |
| submission zip | 20.5 KB compressed, 69 KB unzipped, `agent.py` at the root |
| init, cold numba cache, loaded machine | 23.8 s (compile), agent ready with the compiled evaluation |
| init, cache directory not creatable (`HCE_NO_CACHE` path) | 13.0 s, compiled uncached |
| init, warm cache, loaded machine | 18-20 s, almost all of it `import numba` (25.8 s measured alone under load) |

The init numbers are far below the 90 s budget even under this load; on an idle core `import
numba` is 1-3 s and the compile a few seconds.

## Fixed-depth node counts (`bench.py depth --depth 4`)

| Position | Move | Score | Nodes | Q-nodes | Evals |
| --- | --- | --- | --- | --- | --- |
| start | g1f3 | 12 | 1487 | 604 | 880 |
| Italian, move 4 | b1c3 | 12 | 4109 | 2862 | 2792 |
| middlegame with Bxd6 available | f4d6 | 1028 | 3611 | 2292 | 2549 |
| QGD-ish, move 7 | f3e5 | -380 | 6426 | 5169 | 5213 |
| rook ending | h2h4 | -1 | 3313 | 1452 | 1817 |
| K+R+P vs K+P | f2f4 | 595 | 2295 | 869 | 1216 |
| closed middlegame, move 11 | c4d5 | 38 | 10402 | 9131 | 9385 |
| tactical middlegame, Black to move | f5d3 | 80 | 12265 | 10417 | 8906 |

Total 43 908 nodes, 32 758 evaluation calls, 75 % of nodes in quiescence. Under the load this
ran at about 5 000 nodes/s; the same code on an idle core is several times faster.

## Bugs found by re-running the checks in this session

1. `bench.py` had a literal newline inside an f-string: every `bench.py` command died with a
   `SyntaxError`, so none of the benchmarks had actually run.
2. numba does not create a user-provided `NUMBA_CACHE_DIR`. With a fresh cache directory (the
   platform's state on every first game) the compile crashed while saving its index, the import
   failed, and the agent played the whole game with the material + PST fallback. It only worked
   locally because the directory already existed. `hce_eval` now creates the directory, compiles
   uncached if it cannot, and `agent.py` retries once without the cache.
3. Depth 1 ran with no deadline. With the pure-Python evaluation a depth-1 search with its
   quiescence tail took 4 s at a 50 ms clock. Every iteration is now clocked; the best-ordered
   root move is held before the search starts so a legal move always exists.

## Still to do

- Paired games against the baselines and the other `agents/` engines on a quiet machine
  (`bench.py arena`, `python -m harness.arena`), hundreds of games, both colours; record W/D/L,
  Elo estimate, nodes/s, average depth, median and p99 move time.
- Weight tuning (Texel-style or teacher regression) once a quiet machine gives stable results.
- Profile the search: 75 % of nodes are quiescence, so SEE-based pruning of losing captures
  is the most promising saving.
