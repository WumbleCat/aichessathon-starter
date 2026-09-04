# 15 — Selective Extensions

Bot for roadmap step `my-agents-readmes/15_selective_extensions.md`.

`agent.py` is a self-contained alpha-beta engine (iterative deepening, aspiration windows,
PVS, transposition table keyed by python-chess's transposition key, killer and history
ordering, null move, late move reductions, futility and reverse futility, quiescence with
SEE pruning and bounded check evasions). On top of that it implements four *selective
extensions*: forcing lines are searched one ply deeper than the nominal depth.

| Extension    | Fires when                                                           | Guard                              |
|--------------|----------------------------------------------------------------------|------------------------------------|
| check        | the move gives check                                                 | SEE >= 0, `check_ext_min_depth`    |
| recapture    | a capture on the square of the previous ply's capture                | SEE >= 0, victims within 150 cp    |
| passed pawn  | a passed pawn steps onto the seventh rank                            | SEE >= 0                           |
| singular     | the TT move beats every alternative by a margin in a reduced search  | `singular_min_depth`, TT depth     |

Every rule is a boolean in `SearchConfig`, and `max_extensions` is the per-line budget:
one line can gain at most that many plies, whatever mix of rules fires. Nodes at depth 0
always go to quiescence, so the budget is the *only* way to get deeper than nominal
depth. `Searcher.extensions` counts how often each rule fired and how often the budget
refused one, and `Searcher.max_main_ply` reports the deepest main-search ply, so the
effect of each rule can be measured.

## Run

```
uv run python -m harness.arena --agent my-agents/15_selective_extensions --opponent baselines/greedy --games 2
uv run python -m unittest my-agents/15_selective_extensions/test_extensions.py -v
```

To submit, copy `agent.py` to the repository root and run `make zip`.
