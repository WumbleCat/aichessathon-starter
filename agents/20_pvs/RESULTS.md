# Results: 20_pvs

Measurements from 2026-09-04/05 on the shared development machine while it ran the other
agent-building sessions (CPU 100 %, 14-84 Python processes, 1.7-2.2 GB free RAM). Node counts
are deterministic; every wall-clock figure is pessimistic by a large, varying factor. CPU
time (`time.process_time`) is reported where it matters. Re-run on a quiet machine before
trusting a speed.

## Correctness

`tests/test_engine.py`: 30 tests, all passing (`-m unittest discover -s agents/20_pvs/tests`).

- perft on six positions (start, kiwipete, two endgames, two tactical) matches the known counts
- legal move lists match python-chess on castling / en passant / promotion positions
- incremental Zobrist hash equals a fresh hash after random move sequences
- mandatory chess tests: normal move, free capture, mate in one, check evasion, mate in two,
  stalemate avoidance when winning, one-legal-move position, both castlings, en passant
  (including a position where it is the only winning move), all four promotions, knight
  under-promotion fork, game-over positions return `0000` without raising
- clocks 50 / 100 / 500 / 1000 / 5000 / 30000 / 120000 ms never use more than half the clock
  (under 1 s) or 35 % of it, twelve consecutive calls in one process keep state valid
- mate scores, repetition avoided when ahead, node-limit stop, feature toggles (null move +
  LMR off costs more nodes in aggregate over four positions at depth 6)
- ponder thread stops within 0.5 s, fills the shared TT, leaves the game position untouched,
  and get_move works with pondering between calls

## Fixed-depth benchmark (`bench.py --depth 7`, six positions)

| position | depth/seldepth | nodes | q % | cpu s | knps (cpu) | first-move cut-offs |
|---|---|---|---|---|---|---|
| start | 7/12 | 8,466 | 50 | 0.03 | 271 | 93 % |
| kiwipete | 7/27 | 54,947 | 77 | 0.17 | 320 | 93 % |
| middlegame (bishop down) | 7/17 | 14,992 | 61 | 0.05 | 320 | 89 % |
| endgame | 7/12 | 4,852 | 46 | 0.03 | 155 | 91 % |
| rook endgame | 7/12 | 19,912 | 57 | 0.06 | 319 | 78 % |
| tactics | 7/18 | 13,150 | 64 | 0.06 | 210 | 91 % |
| total | avg 7.0/16.3 | 116,319 | 67 | 0.41 | 286 | 87 % |

At the platform time control the search reaches depth 10-14 in the 0.3-3 s it spends per
move (seen in the test logs: depth 14/27 in 3.0 s at 215 knps wall under load).

## Initialisation

| what | time |
|---|---|
| import + compile + warm-up, cold, loaded machine (wall) | 128-132 s |
| `import agent` returns (compile thread joined for at most 60 s) | 60.1 s |
| import + warm-up when the modules were already compiled in-process | 0.1 s |

The compile runs in a daemon thread; on this loaded machine the first 60-70 s of a game are
played by the python-chess fallback, on the platform's dedicated core the compile is expected
to finish inside the join. numba's on-disk cache was tried and rejected: a second process
loading the cached recursive search crashed with an access violation (see README).

## Games through the harness

| match | control | result |
|---|---|---|
| play as White vs `baselines/greedy` | 120 s + 0.5 s | win by checkmate |
| arena vs `baselines/greedy`, 4 games | 10 s + 0.1 s | +0 =3 -1: the compile outlasted each game, so the fallback played |
| arena vs `baselines/numba`, 6 games (3 per colour) | 60 s + 0.5 s | +6 =0 -0: 2 checkmates, 4 wins on the baseline's clock (it has no time management); no crash, illegal move or flag on our side |

| arena: ponder on vs `variants/noponder` (same engine, pondering off), 6 games (3 per colour) | 60 s + 0.5 s | +4 =1 -1 for pondering: 5 checkmates, 1 threefold; no flag on either side, so stopping the ponder thread costs nothing measurable |

Six games say little about the size of the pondering gain, only that it does not break the
clock handling; it is left on because the rules explicitly allow it.

## Self-play A/B (`selfplay.py`, 4000 nodes per move, 100 games, colours swapped per opening)

A = default parameters, B = one feature disabled. At a fixed node budget a pruning feature
can only show its accuracy cost, not its speed gain, so a small positive number is the
expected outcome; a clearly negative one would mean a bug. 100 games give roughly +-60 Elo.

| feature off in B | A result | score | Elo (95 %) | note |
|---|---|---|---|---|
| null move (`P_NULL`) | +42 =21 -37 | 52.5 % | +17 (-44..+79) | 1513 s wall / 140 s cpu |
| late move reductions (`P_LMR`) | +42 =27 -31 | 55.5 % | +38 (-20..+99) | |

The sweep continues over the other toggles in `results/selfplay/off_<P_NAME>.log`.

## What is next

1. Arena robustness runs against all four baselines (illegal / crash / flag must stay at 0).
2. Self-play A/B for every toggle (null move, LMR, futility, razoring, SEE pruning, IID,
   aspiration) at 4000 nodes per move; drop or retune anything that loses.
3. Measure the pondering gain with a real-clock match (ponder on vs off).
4. Build the zip from `agents/20_pvs` and check the unzipped size.
