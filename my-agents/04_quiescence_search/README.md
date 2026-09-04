# 04 — Quiescence Search

Implements `my-agents-readmes/04_quiescence_search.md`.

A negamax / alpha-beta search with a material + piece-square evaluation, where the
leaves are handed to a quiescence search instead of the static evaluation. The
quiescence search resolves captures and queen promotions until the position is
quiet, handles in-check nodes by searching every legal evasion (no stand pat), orders
captures MVV-LVA, and counts its nodes separately as `qnodes`.

Iterative deepening with a per-move budget of one thirtieth of the remaining clock
keeps the bot inside the harness time control. A one-ply static search is always
completed first so there is a legal answer even when the clock is nearly gone.

The platform sends a bare FEN, so the board has no history. The module remembers
every position the current game has visited and, at the root, scores a move that
re-enters one of them as a draw at best. A winning side therefore never shuffles
into a threefold repetition; a losing side may still take one.

## Files

- `agent.py` — the bot; exposes `get_move(fen, time_left_ms) -> str`
- `tests/test_quiescence.py` — the readme's test positions, plus legality and timing

## Run

From the repository root:

```
uv run python my-agents/04_quiescence_search/tests/test_quiescence.py
uv run python -m harness.arena --agent my-agents/04_quiescence_search --opponent baselines/greedy --games 2
uv run ruff check my-agents/04_quiescence_search
uv run mypy --strict my-agents/04_quiescence_search/agent.py
```

## Knobs

- `USE_QUIESCENCE` — switch the leaf handling back to static evaluation
- `QS_MAX_PLY` — hard cap on quiescence depth, guarantees termination
- `QS_INCLUDE_CHECKS` — reserved; checking moves are not yet generated in quiescence

## Not done yet (by design, per the readme)

- delta pruning
- static exchange evaluation
- checking moves and non-capture evasions beyond in-check nodes
- stalemate detection at quiet leaves (only in-check leaves generate all moves)
