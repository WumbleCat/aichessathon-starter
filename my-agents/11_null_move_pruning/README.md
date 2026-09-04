# 11 — Null-Move Pruning

Implements `my-agents-readmes/11_null_move_pruning.md`.

## Files

- `agent.py` — the bot. Exposes `get_move(fen, time_left_ms) -> str` as the harness requires.
- `test_null_move.py` — the spec's test checklist plus a node-count benchmark.

## What is in the search

Null-move pruning needs a real alpha-beta tree to prune, so the agent carries a compact
conventional engine: material + piece-square evaluation, fail-soft negamax with alpha-beta,
quiescence on captures and promotions, a transposition table keyed by python-chess's polyglot
Zobrist hash, TT/MVV-LVA/killer/history move ordering, and iterative deepening on a clock
budget of one thirtieth of the remaining time (50 ms to 8 s).

The null move itself lives in `Searcher.negamax`, and every safeguard is in
`can_try_null_move`:

| guard                                       | why                                            |
|---------------------------------------------|------------------------------------------------|
| `SearchConfig.null_move` off                | feature toggle                                 |
| parent node already passed                  | two passes in a row prove nothing              |
| side to move in check                       | passing would leave the king en prise          |
| depth below `null_move_min_depth` (2)       | nothing to reduce                              |
| beta is a mate score                        | a pass can neither prove nor refute a mate     |
| mover's non-pawn material below 500 cp      | zugzwang guard; king-and-pawn endings are 0    |
| static eval below beta (toggle)             | the pass would almost surely fail low          |

Reduction is `R = 2`, rising to 3 from depth 6. A null-move cutoff returns `beta` (fail hard),
never the reduced search's raw score. No verification search, as the spec advises.

## Running the tests

From this directory, with the repo interpreter:

```
python test_null_move.py
```

Measured at depth 5 with a fresh table per run (pruning off vs on):

| position                    | nodes off | nodes on | ratio |
|-----------------------------|-----------|----------|-------|
| Kiwipete                    | 203,313   | 236,704  | 116%  |
| Italian, move 4             |  79,375   |  37,981  |  48%  |
| Ruy/Italian middlegame      | 165,115   | 105,550  |  64%  |
| Dragon-type middlegame      | 120,639   |  36,053  |  30%  |

Best move and score were identical with and without pruning in all four. Kiwipete is the
outlier: the pruned tree lands on different transposition-table contents and move order and
searches more nodes there, which is a known cost of any selective pruning at low depth.

## Arena

Fast time control (10 s + 0.1 s), two games each, alternating colours:

- vs `baselines/greedy`: +2 =0 -0, both by checkmate
- vs `baselines/minimax`: +2 =0 -0, both by checkmate

## Limitations

- Pure Python at roughly 7k nodes/s, so depth 4 to 5 in the fast arena. The Zobrist hash is
  recomputed at every node; an incremental hash would be the first speed-up.
- No verification search, so a rare zugzwang with pieces on the board can still be
  mispruned. The material guard only covers the common pawn-ending case.
- Depth-6 fixed searches do not find the deep king-and-pawn idea in the third zugzwang test
  position, with or without pruning; the test checks that pruning does not change the answer.
