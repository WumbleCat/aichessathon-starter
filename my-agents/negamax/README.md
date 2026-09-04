# negamax

A chess bot built from `my-agents-readmes/negamax.md`, following the development order
that document recommends:

| step | where in `agent.py`                | what it does                                              |
|------|------------------------------------|-----------------------------------------------------------|
| 1    | `evaluate`                         | material + piece-square tables, scored for the side to move |
| 2    | `Searcher.negamax`                 | one maximising function; a child's score is negated       |
| 3    | `Searcher.negamax` (alpha, beta)   | alpha-beta pruning with the window negated and swapped    |
| 4    | `ordered_moves`, `move_priority`   | promotions, then captures by MVV-LVA, then quiet moves    |
| 5    | `choose_move`                      | iterative deepening under a per-move time budget          |
| 6    | `Searcher.quiescence`              | captures-only search at the leaves                        |

`get_move(fen, time_left_ms) -> str` is the only entry point; it matches the contract in
`AGENTS.md` and the harness runner. The module also remembers the positions it has been
asked about during the game and scores a move that repeats one of them as a draw, so it
does not shuffle into a threefold repetition when it is ahead.

## Run it

```
uv run python -m harness.play  --white my-agents/negamax --black baselines/greedy
uv run python -m harness.arena --agent my-agents/negamax --opponent baselines/greedy --games 2
uv run python my-agents/negamax/test_agent.py
```

To submit, copy `agent.py` to the repository root and run `make zip`.

## Time management

`move_budget_ms` spends about a thirtieth of the remaining clock per move, never more
than eight seconds and never more than half of what is left. The search checks the clock
every 512 nodes and aborts the current depth when the budget is spent; the deepest
completed depth (or the best fully-searched move of the aborted one) is played.
