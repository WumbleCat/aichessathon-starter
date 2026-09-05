# Bot win rates against the baselines

Every bot in this folder was played against all four baselines in `baselines/` using the
repository's own arena (`harness/arena.py`), which enforces the same protocol and clock as
the platform. This file records the outcome and what it tells us about each bot.

## How the numbers were produced

- **Opponents:** `baselines/random`, `baselines/greedy` (1-ply material), `baselines/minimax`
  (2-ply material + mobility, no time management), `baselines/numba` (minimax with a jitted
  evaluation).
- **Games:** 10 per bot-baseline pairing, colours alternating each game (5 as White, 5 as Black),
  from the standard start position. 40 games per bot, 640 games in total.
- **Time control:** the arena's fast setting, 10 s + 0.1 s per side. Rated games on the platform
  use 120 s + 0.5 s, so these numbers measure behaviour under a much tighter clock.
- **Score:** wins + draws / 2, expressed as a percentage.
- **Command used for each pairing:**

```powershell
& ".venv\Scripts\python.exe" -m harness.arena --agent my-agents/<bot> --opponent baselines/<baseline> --games 10
```

Up to four pairings were run at once on a 16-core machine. The raw arena output for every pairing
is in `results/logs/<bot>__<baseline>.log`, and the parsed numbers are in `results/results.json`.

The last two columns separate two kinds of flag result. "Won on baseline's clock" counts games the
bot won only because the minimax or numba baseline ran out of time, which they do at this time
control since they have no time management. "Lost on own clock" counts games the bot itself
forfeited on time.

## Score table

| Rank | Bot | vs random | vs greedy | vs minimax | vs numba | Overall score | W-D-L (40 games) | Won on baseline's clock | Lost on own clock |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `09_history_heuristic` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | **100.0%** | 40-0-0 | 0 | 0 |
| 2 | `10_principal_variation_search` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | **100.0%** | 40-0-0 | 0 | 0 |
| 3 | `11_null_move_pruning` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | **100.0%** | 40-0-0 | 0 | 0 |
| 4 | `12_late_move_reductions` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | **100.0%** | 40-0-0 | 0 | 0 |
| 5 | `13_static_exchange_evaluation` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | **100.0%** | 40-0-0 | 1 | 0 |
| 6 | `14_futility_pruning_and_razoring` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | **100.0%** | 40-0-0 | 1 | 0 |
| 7 | `08_killer_heuristic` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 95% (+9 =1 -0) | **98.8%** | 39-1-0 | 10 | 0 |
| 8 | `15_selective_extensions` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 95% (+9 =1 -0) | 100% (+10 =0 -0) | **98.8%** | 39-1-0 | 0 | 0 |
| 9 | `03_handcrafted_evaluation` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 90% (+9 =0 -1) | **97.5%** | 39-0-1 | 0 | 0 |
| 10 | `06_move_ordering` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 95% (+9 =1 -0) | 95% (+9 =1 -0) | **97.5%** | 38-2-0 | 0 | 0 |
| 11 | `05_iterative_deepening` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 90% (+9 =0 -1) | 95% (+9 =1 -0) | **96.2%** | 38-1-1 | 0 | 0 |
| 12 | `04_quiescence_search` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 95% (+9 =1 -0) | 85% (+7 =3 -0) | **95.0%** | 36-4-0 | 2 | 0 |
| 13 | `07_transposition_table_zobrist` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 90% (+9 =0 -1) | 90% (+9 =0 -1) | **95.0%** | 38-0-2 | 0 | 0 |
| 14 | `01_negamax` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 85% (+8 =1 -1) | 75% (+6 =3 -1) | **90.0%** | 34-4-2 | 4 | 0 |
| 15 | `02_alpha_beta_pruning` | 100% (+10 =0 -0) | 100% (+10 =0 -0) | 70% (+4 =6 -0) | 75% (+5 =5 -0) | **86.2%** | 29-11-0 | 1 | 0 |
| 16 | `negamax` | 75% (+7 =1 -2) | 80% (+8 =0 -2) | 25% (+2 =1 -7) | 40% (+4 =0 -6) | **55.0%** | 21-2-17 | 1 | 17 |

## What each bot is

| Bot | What it adds |
|---|---|
| `01_negamax` | Plain fixed-depth negamax on material and piece-square tables. No pruning. |
| `02_alpha_beta_pruning` | Alpha-beta cut-offs and a shallow iterative-deepening loop (depth cap 6). |
| `03_handcrafted_evaluation` | A richer hand-written evaluation: pawn structure, king safety, mobility, tapered piece-square tables. |
| `04_quiescence_search` | A capture-only quiescence search at the horizon. |
| `05_iterative_deepening` | Iterative deepening with real per-move time management. |
| `06_move_ordering` | MVV-LVA capture ordering plus promotion and check ordering. |
| `07_transposition_table_zobrist` | A Zobrist-hashed transposition table with a best-move hint. |
| `08_killer_heuristic` | Killer-move slots for quiet-move ordering. |
| `09_history_heuristic` | A history table for quiet-move ordering. |
| `10_principal_variation_search` | Null-window (PVS) search on top of iterative-deepening alpha-beta. |
| `11_null_move_pruning` | Null-move pruning with zugzwang guards. |
| `12_late_move_reductions` | Late-move reductions on the PVS search. |
| `13_static_exchange_evaluation` | Static exchange evaluation for capture ordering and pruning losing captures. |
| `14_futility_pruning_and_razoring` | Futility pruning, reverse futility pruning and razoring near the leaves. |
| `15_selective_extensions` | Check, recapture, passed-pawn and singular extensions. |
| `negamax` | Earlier standalone single-file bot (alpha-beta, ordering, deepening, quiescence) written before the numbered roadmap. |

## Reading the table

**Every numbered bot beats random and greedy 10-0 by checkmate.** The ladder in the top-level
README says beating greedy needs a search, and even the plain negamax at stage 01 clears that bar.
The two harder baselines, minimax and numba, are what separate the bots.

**Six bots are perfect at 40-0:** stages 09 through 14. Nothing below stage 09 manages it. That
matches the roadmap's ordering: the later stages stack better move ordering (history heuristic,
SEE) and safe pruning (PVS, null move, LMR, futility) on top of the same evaluation, so they
search deeper in the same 10 seconds and convert winning positions instead of drifting into
repetition. All but two of their 120 games against minimax and numba ended in checkmate; the
two exceptions (one each for stages 13 and 14) were numba running out of clock.

**Draws by threefold repetition are the main leak for the early stages.** Stage 02
(`02_alpha_beta_pruning`) has no losses but drew 11 of its 20 games against minimax and numba,
all by repetition. Its depth is capped at 6 and it has no repetition awareness, so once it is
ahead it happily shuffles pieces. Stages 01 and 04 show the same pattern at a smaller scale
(4 draws each). From stage 05 onward, where iterative deepening with real time management arrives,
draws drop to 0 or 1 per bot.

**Losses among the numbered bots are rare and all genuine checkmates.** Stage 01 lost 2, stage 07
lost 2, stages 03 and 05 lost 1 each, always against minimax or numba and always by being mated,
not by a fault. Stage 07's two losses are worth a look: a transposition table should not make a
bot weaker than stage 06, so it may be storing or probing entries incorrectly in some positions,
or the table is costing more per node than it saves at this depth.

**Stage 08 (`08_killer_heuristic`) leaned on the clock more than the others.** It scored 98.8% but
10 of its 20 wins against minimax and numba came from the baseline flagging rather than checkmate.
It is still winning those positions, but it converts more slowly than stages 09 to 14, which mated
in every game.

**The standalone `negamax` bot is the outlier and its problem is time, not chess.** It scored 55%
and lost 17 of 40 games on its own clock, including 2 against random and 2 against greedy. Its
search checks the clock only every 2048 nodes, which at its speed is a gap of well over 100 ms,
and it always finishes a depth-1 search plus an unbounded quiescence search before it looks at the
deadline. With a 10 s clock its per-move slice is a few hundred milliseconds, so it overshoots
again and again. At the platform's 120 s control this would matter far less, but it is a
correctness bug worth fixing before it is used for anything.

## Caveats

- 10 games per pairing is a small sample. A 95% versus 100% difference is one draw and should not
  be read as a real strength gap; the 86% to 100% spread between stage 02 and stages 09 to 14 is.
- All games started from the standard position. Rated games use curated openings, and the
  platform's 120 s clock will reward deeper search more than this 10 s test did.
- Baselines were run at the same 10 s control they were not designed for. Some of the wins in the
  "Won on baseline's clock" column would have been fought out at 120 s.
- These matches only measure each bot against the four baselines. They do not tell you how the
  bots rank against each other; a round-robin between the top bots would be the next test.

## Reproducing

```powershell
# one pairing
& ".venv\Scripts\python.exe" -m harness.arena --agent my-agents/09_history_heuristic --opponent baselines/numba --games 10

# a full bot against every baseline
foreach ($b in "random","greedy","minimax","numba") {
  & ".venv\Scripts\python.exe" -m harness.arena --agent my-agents/09_history_heuristic --opponent baselines/$b --games 10
}
```
