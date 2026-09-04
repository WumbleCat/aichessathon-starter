# 03 — Handcrafted evaluation

Stage 03 of `my-agents-readmes/`: a deterministic, rule-based evaluation in centipawns,
driven by the Negamax + alpha-beta search from stages 01 and 02.

```
agent.py         get_move(fen, time_left_ms) -> uci. Negamax, alpha-beta, MVV-LVA capture
                 ordering, iterative deepening on a per-move time budget.
evaluation.py    evaluate(board) -> int, side-to-move perspective.
                 evaluate_white(board) -> int, White's perspective (white - black).
tests/           test_evaluation.py, runnable directly or with pytest.
```

## What the evaluation counts

Every term is computed for both colours and blended between a middlegame and an
endgame value by the amount of non-pawn material left on the board (`phase`, 0..24).

| Term            | Details                                                                                 |
| --------------- | --------------------------------------------------------------------------------------- |
| Material        | P 100, N 320, B 330, R 500, Q 900. The king has no material value.                      |
| Piece-square    | One table per piece; separate middlegame and endgame king tables; pawn tables differ    |
|                 | too. Black uses the White tables mirrored, not duplicated values.                        |
| Bishop pair     | +30 mg / +50 eg for owning two bishops.                                                 |
| Pawn structure  | Doubled, isolated and backward penalties; passed-pawn bonus scaled by rank; connected   |
|                 | passed pawn bonus. Cached by pawn configuration because it only changes on pawn moves. |
| Mobility        | Pseudo-legal attacks per knight, bishop, rook and queen, not counting squares held by   |
|                 | friendly pieces or guarded by enemy pawns. Straight from python-chess attack tables.    |
| King safety     | Pawn shield one and two ranks ahead of the king, open and semi-open files next to the  |
|                 | king, and enemy pieces attacking the king zone (a super-linear penalty table).          |
|                 | Middlegame only; the phase blend fades it out.                                          |
| Rook activity   | Open file, semi-open file, relative seventh rank, connected rooks.                      |

All weights are module-level constants at the top of `evaluation.py`.

## Verify

```
uv run python my-agents/03_handcrafted_evaluation/tests/test_evaluation.py
uv run python -m harness.arena --agent my-agents/03_handcrafted_evaluation --opponent baselines/greedy --games 2
```

The tests cover the cases the spec asks for (extra queen about +900, extra pawn about
+100, centre knight beats corner knight, passed / doubled / isolated / backward pawns,
bishop pair, mirrored positions score the exact negative) plus determinism, board
restoration, leaf-node speed, legal moves from the agent, mate in one, and alpha-beta
returning the same score as plain Negamax.

## Limits

- No quiescence search yet (stage 04), so the horizon effect is the main source of
  blunders: a capture sequence that ends one ply past the search depth is misjudged.
- Pure Python: roughly 8,000 nodes per second, so depth 3 at the fast arena clock and
  depth 4 or 5 at the real time control.
- King safety is scaled by total phase, not by the attacker's remaining material, so the
  king will centralise while the opponent still has a few pieces.
