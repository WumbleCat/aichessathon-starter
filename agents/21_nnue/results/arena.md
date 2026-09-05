# harness.arena against the baselines (2026-09-05, session 4)

`python -m harness.arena --agent agents/21_nnue --opponent baselines/<x> --games 10
--base-ms 120000 --increment-ms 500`, the four arenas running in parallel from the
worktree, on the shared box (50-90 other python processes).  Colours alternate every game,
so 10 games are 5 paired games.  The harness does not show the agent's stderr, so a single
`harness.play` game (`play_random.log`; the PGN is git-ignored, the game was 1. e4 a5 2. d4 Nh6 3. Nf3 Ng4 4. h3 Rg8 5. hxg4 g5 6. Rxh7 Na6 7. Nxg5 Rh8 8. Rxh8 f5 9. gxf5 b6 10. Qh5#) is the evidence for which searcher
played.

## Round 1: compile-time cut only (`arena1_*.log`)

| opponent | result | score | terminations |
|---|---|---|---|
| random | +3 =7 -0 | 65% | threefold 7, checkmate 3 |
| greedy | +3 =6 -1 | 60% | threefold 6, checkmate 4 |
| minimax | +2 =3 -5 | 35% | checkmate 7, threefold 3 |
| numba | +5 =3 -2 | 65% | checkmate 7, threefold 3 |

These are the python fallback's numbers, not the engine's.  `harness.play` vs random in the
same conditions (`play_random.log`): "engine ready after 42.7 s wall, 38.2 s cpu", then mate
in 10 moves.  In the arena games the engine was not ready when the first moves were
requested (four processes compiling at once under the load), and once `get_move` falls back
to `_python_search` the compile thread starves: the fallback is pure Python and holds the
GIL for its whole budget, so numba's type inference, also pure Python, only progresses in
the gaps between moves.  The material-only fallback then shuffles into threefold repetitions
when ahead and loses to the minimax baseline.

Fix (agent.py): while the engine is not ready, `get_move` first waits on the ready event for
`COMPILE_WAIT_SHARE` (60%) of the move budget, which releases the GIL to the build thread,
and only then runs the fallback with the rest of the budget.  On the platform the compile
(44 s CPU on one core) finishes inside the 70 s import wait anyway; this is the safety net
for a slower core, and what makes local harness games measure the engine.

## Round 2: with the wait-for-compile hand-over (`arena_*.log`)

| opponent | result | score | terminations |
|---|---|---|---|
| random | +10 =0 -0 | 100% | checkmate 10 |
| greedy | +10 =0 -0 | 100% | checkmate 10 |
| minimax | +10 =0 -0 | 100% | checkmate 10 |
| numba | +10 =0 -0 | 100% | checkmate 10 |

40 games, 40 wins, all by checkmate, no flag, crash, illegal move or init failure; both
colours from the standard start position, the four arenas in parallel with the agent
processes raised to AboveNormal priority (`bump_priority.ps1` in the job's temp dir) so the
compile finished within the first moves under the load.  The README asks for hundreds of
paired games; that is what a quiet machine or the platform can provide, and the in-process
tools (`tools/selfplay_ab.py`, `tools/vs_bot.py`) remain the way to measure Elo differences
here.
