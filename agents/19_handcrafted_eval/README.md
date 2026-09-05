# 19 — Handcrafted Evaluation Engine

A tapered, handcrafted centipawn evaluation designed to be called tens of thousands of times a
move from an alpha-beta search, plus the search that turns it into Elo. No neural network, no
external engine, nothing learned; every weight is a readable integer in `hce_tables.py`.

Spec: `my-agents-readmes/19_HANDCRAFTED_EVALUATION.md`.

## Files

| File | Purpose |
| --- | --- |
| `agent.py` | `get_move(fen, time_left_ms) -> str`. Initialisation with a deadline, time management, game-history memory for repetition detection, legal fallback move. |
| `hce_eval.py` | The evaluation, written on bitboards. One source runs compiled by numba (explicit `uint64` signatures, on-disk cache under the temp dir) or as plain Python when numba is missing (`HCE_NO_NUMBA=1`). |
| `hce_tables.py` | Material values, piece-square tables (MG/EG), all feature weights. Pure numpy, instant import. |
| `hce_eval_simple.py` | Material + PST fallback used only while the compiled evaluation is still building. |
| `hce_search.py` | Iterative deepening, PVS alpha-beta, transposition table kept across moves, MVV-LVA / killer / history ordering, quiescence with delta pruning, null-move pruning, reverse futility, futility, late move reductions, check extension, aspiration windows. |
| `bench.py` | `eval` / `search` / `arena` benchmarks; the arena plays paired games (both colours from the same neutral openings) through the official harness sandbox and clock. |
| `tests/` | `unittest` suites: bitboard helpers against python-chess, the evaluation properties from the spec, the mandatory chess tests, clock tests, init fallback. |

## The evaluation

Score = White minus Black; the search negates for the side to move. Both colours are scored by
one routine written for White: Black's bitboards are flipped vertically and passed as "us", so
colour symmetry holds by construction (tests check exact antisymmetry on random positions).

Two accumulators, middlegame (MG) and endgame (EG), blended by a phase from the non-pawn
material (knight 1, bishop 1, rook 2, queen 4; total 24):

```
score = int((mg * phase + eg * (24 - phase)) / 24)
```

Feature groups, in the order the spec asks for:

1. **Material** — separate MG/EG values (pawn 95/118 ... queen 980/1010).
2. **Piece-square tables** — MG and EG tables per piece, mirrored for Black.
3. **Mobility** — squares reachable per piece that are neither occupied by own pieces nor
   attacked by enemy pawns, minus a per-piece baseline, weighted per piece type.
4. **Pawn structure** — doubled, isolated, backward, supported, phalanx.
5. **Passed pawns** — rank-scaled MG/EG bonus, extra when supported, EG king-distance terms
   (enemy king far from the stop square is good, own king close is good), free path bonus.
6. **King safety** — pawn shield on the three files around the king (rank 1 / rank 2 in front),
   penalty for a missing shield pawn and a second one if the file is fully open while the enemy
   has heavy pieces, pawn storm penalty; king-zone attack units from enemy knights, bishops,
   rooks and queens (quadratic in the units, halved with a single attacker or no queen, capped).
   The whole group lives in MG so it fades with the phase.
7. **Bishop pair, rook on open / semi-open file, rook on the seventh, knight and bishop
   outposts.**
8. **Endgame king activity** — EG king table (centralisation) and a mop-up term when one side
   has no pawns and the other is at least a rook up: drive the lone king to the edge, bring the
   attacking king close. Plus drawishness: a bare minor piece cannot win (score 0) and
   opposite-coloured bishop endings are halved.
9. **Tapered interpolation** and a tempo bonus for the side to move.

The evaluator never returns mate scores; checkmate, stalemate and draws are the search's job.

Cost with numba: about 8-13 µs per call including the Python-side attribute reads and the
dispatch (measured on an idle core; see RESULTS.md). The pure-Python path is roughly 30x
slower and exists only as a safety net.

## Initialisation

`import agent` starts the numba compile in a background thread and waits at most 55 s for it.
If the machine is too slow (or numba is missing) the agent reports ready with the simple
evaluation and swaps the compiled one in on the first `get_move` after it finishes. The
compiled code is cached under the system temp directory (`/tmp` on the platform), so a second
process on the same machine starts in well under a second.

numba does not create a user-provided cache directory, and a missing one makes every compile
crash while the cache index is saved. `hce_eval` therefore creates the directory itself, compiles
without a cache when that is impossible (or when `HCE_NO_CACHE=1`), and `agent.py` retries the
import once uncached after any failure. With `HCE_INFO=1` the init line names the evaluation in
use, so a fallback is visible in the validation log.

## Time management

`soft = (time_left - 25 ms) / 28`, `hard = min(time_left / 4, 3 * soft)`. Before any search
the best-ordered root move (transposition-table move, then the most valuable capture) is held
as the answer, so every iteration including depth 1 runs under the hard deadline; a depth-1
search with its quiescence tail can cost seconds on a slow evaluation or a loaded core. Deeper
iterations stop when the soft budget is spent or when the next iteration would clearly not fit;
the hard deadline aborts inside the search (clock checked every 32 nodes), the board is unwound,
and an aborted iteration keeps its best fully-searched move when that beat the previous one.
Repetitions of any position already seen in the game (the harness sends bare FENs, so the agent
keeps its own list) score as draws in the search.

## Running

From the repository root, with the project interpreter:

```powershell
$PY = "E:\sourcecode\ai-chess-original\aichessathon-starter\.venv\Scripts\python.exe"

# tests (add HCE_NO_NUMBA=1 in the environment to test the pure-Python evaluation)
& $PY -m unittest discover -s agents/19_handcrafted_eval/tests -v

# benchmarks
& $PY agents/19_handcrafted_eval/bench.py eval
& $PY agents/19_handcrafted_eval/bench.py search --ms 1000
& $PY agents/19_handcrafted_eval/bench.py arena --opponent baselines/minimax --games 20

# the official harness
& $PY -m harness.arena --agent agents/19_handcrafted_eval --opponent baselines/greedy --games 10
& $PY -m harness.play --white agents/19_handcrafted_eval --black baselines/minimax

# submission zip (run inside the agent directory; ~70 KB unzipped)
cd agents/19_handcrafted_eval; $env:PYTHONPATH = "..\.."; & $PY -m harness.package
```

Set `HCE_INFO=1` to get one `info depth ... nodes ... time ...` line per move on stderr.

## Legality

Everything here is written from scratch for this project: the bitboard attack generation, the
tables, the weights and the search. Concepts (tapered evaluation, MVV-LVA, PVS, null move, LMR,
king-zone attack units) are standard published ideas. There are no engine weights, no lookup
tables of engine output, no native binaries, and the zip is a few tens of kilobytes of Python.
