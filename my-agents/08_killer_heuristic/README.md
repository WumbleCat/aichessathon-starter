# 08 — Killer heuristic bot

Implements `my-agents-readmes/08_killer_heuristic.md`.

The engine is a plain iterative-deepening alpha-beta searcher (material + piece-square
evaluation, MVV-LVA capture ordering, quiescence on captures and promotions, a hash-move
table so the previous iteration's best move is tried first). On top of that sits the
feature under test: a **killer table** with two slots per ply.

- A move that causes a beta cutoff and is neither a capture nor a promotion is stored
  as killer 0 for that ply; the previous killer 0 shifts to slot 1. Re-storing the current
  killer 0 is a no-op, so a move never occupies both slots.
- In move ordering killers are scored just below every capture and above every other
  quiet move: hash move > promotions > captures > killer 0 > killer 1 > rest.
- Killers only reorder the legal move list. Nothing is dropped, so the search value at a
  fixed depth is identical with the feature on or off.
- The table lives for one `get_move` call (all iterative-deepening iterations) and is
  rebuilt on the next call.
- `USE_KILLERS` at the top of `agent.py`, or `Searcher(use_killers=False)`, switches it off.

## Files

| file | purpose |
| --- | --- |
| `agent.py` | the bot; exposes `get_move(fen, time_left_ms) -> str` |
| `test_killers.py` | the checks listed in the spec, run as a script |
| `bench.py` | node counts with killers off vs on at a fixed depth |

## Run

From the repository root:

```
uv run python my-agents/08_killer_heuristic/test_killers.py
uv run python my-agents/08_killer_heuristic/bench.py 4            # with hash move
uv run python my-agents/08_killer_heuristic/bench.py 4 --no-hash  # killers alone
uv run python -m harness.arena --agent my-agents/08_killer_heuristic --opponent baselines/greedy --games 2
```

## Measured (depth 4, five positions)

| ordering | nodes, killers off | nodes, killers on | saving |
| --- | --- | --- | --- |
| MVV-LVA only | 243,660 | 131,989 | 45.8% |
| MVV-LVA + hash move | 113,282 | 108,593 | 4.1% |

Scores and chosen moves were identical in every pair. In the tactical Kiwipete position
every cutoff at this depth is a capture, so killers neither help nor hurt there.
