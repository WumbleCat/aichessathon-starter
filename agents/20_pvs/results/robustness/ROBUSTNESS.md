# 20_pvs robustness runs (branch `worktree-agent-20-robustness`)

Run on 2026-09-05 from a worktree with `feature/agent-20-pvs` at 23f7bbb merged in, on the
shared machine while the other agent-building sessions were running (CPU saturated). Wall
times are pessimistic; the point of these runs is the crash / illegal / flag count on our
side, which must stay at zero, not the score.

## Unit tests

`-m unittest discover -s agents/20_pvs/tests`: 30 tests, OK in 26 s (after the 60 s bounded
compile wait at import; the compile was still running when the tests started, so the clock
tests exercised the fallback path as well as the engine).

## Zip, platform style

`harness.package` from `agents/20_pvs`: 27,043 bytes zipped, 95,046 bytes unzipped, six
files (`agent.py`, `bench.py`, `pvs_board.py`, `pvs_eval.py`, `pvs_search.py`,
`selfplay.py`), `agent.py` at the root. None of the names shadows a module the agent
imports.

Smoke test the way the platform loads it: extract into a fresh directory, put that directory
first on `sys.path`, `chdir` somewhere unrelated, `import agent`, play eight plies from the
start position with a 120 s clock, then one call with 300 ms left.

| step | result |
|---|---|
| `import agent` | returned after 60.1 s (bounded join; compile still running on the loaded box) |
| ply 0, 120 s clock | `e2e4` after 37.6 s: the designed compile wait (30 % of the clock, at most 40 s), then depth 14 |
| plies 1-7 | 1.8-2.6 s each, depth 13-15, all legal |
| 300 ms clock | legal move in 0.076 s |

So the engine tolerates a compile that overruns the init budget by ~37 s: it waits a bounded
share of the clock, then plays. On the platform's dedicated core the compile is expected to
finish inside the 60 s join, so move 1 would not pay this.

## Arena, 60 s + 0.5 s, 6 games per baseline (3 per colour), one arena at a time

The arena exits 1 whenever *either* side flags, so the per-game lines are what count.

| baseline | result (as White / as Black) | terminations | wall for 6 games | our crash / illegal / flag |
|---|---|---|---|---|
| `baselines/random` | +6 =0 -0 (3-0 / 3-0) | checkmate 6 | 10.6 min | 0 / 0 / 0 |
| `baselines/greedy` | +6 =0 -0 (3-0 / 3-0) | checkmate 6 | 11.0 min | 0 / 0 / 0 |
| `baselines/minimax` | +6 =0 -0 (3-0 / 3-0) | checkmate 6 | 15.1 min | 0 / 0 / 0 |

Together with the earlier `baselines/numba` run on `feature/agent-20-pvs` (+6 =0 -0 at the
same control, four of them on the baseline's clock) every baseline has now been played six
times at 60 s + 0.5 s with no failure on our side. Every game here ended by checkmate, so no
game reached the 300-ply adjudication and no game was decided on the clock.

Logs: `arena_<baseline>_60s_6games.log` in this directory.
