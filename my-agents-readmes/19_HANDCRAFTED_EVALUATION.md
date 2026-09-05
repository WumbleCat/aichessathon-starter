# Handcrafted Evaluation Function

## Goal

Create a fast scalar evaluation suitable for alpha-beta leaves. This is separate from a one-ply rule-based bot: the evaluator must be **cheap enough to run at very high frequency** and stable enough for pruning decisions.

## AI Chessathon constraints you MUST design around

Target the current competition API:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    # return one legal move in UCI notation
```

Deployment constraints from the official docs:

- ZIP size: **50 MB maximum uncompressed**.
- `agent.py` must be at the ZIP root.
- One dedicated **CPU core**, **2 GB RAM**, **no GPU**, **no network**.
- Python 3.12.
- Preinstalled: `torch` (CPU), `numpy`, `python-chess`, `onnxruntime`, `numba`.
- 90 s initialization budget before the game clock starts; load weights at import time.
- Main clock: 120 s + 0.5 s/move.
- Native binaries in the ZIP are rejected.
- Published third-party chess engines and ports/translations are prohibited.
- A shipped neural network must be **trained by this team from scratch**.
- Training on positions labelled by an existing engine is explicitly allowed.
- Do **not** ship Stockfish/Lc0/Maia weights, fine-tuned versions of them, or a runtime lookup database of engine moves/evaluations.
- `.onnx`, `.safetensors`, and `.pt` weights are allowed if they are your own trained network.

Official rules/docs: https://aichessathon.com/docs

### Required engineering rule

Concepts and algorithms from papers are fair game. **Do not copy, port, translate, wrap, or mechanically reproduce third-party engine source.** Implement the described algorithm independently and be able to explain it.


## Interface

```python
def evaluate(board) -> int:
    # centipawn-like score from one documented perspective
```

Search must convert the sign consistently.

## Feature groups

Implement incrementally:

1. material;
2. piece-square tables;
3. mobility;
4. pawn structure;
5. passed pawns;
6. king safety;
7. bishop pair / rook files;
8. endgame king activity;
9. tapered middlegame/endgame interpolation.

Do not use expensive tactical logic that search/quiescence already resolves.

## Tapered evaluation

Maintain two scores:

```text
MG = material_mg + positional_mg + ...
EG = material_eg + positional_eg + ...
phase = function(remaining pieces)
score = interpolate(MG, EG, phase)
```

## Incremental optimization

If Python evaluation becomes a bottleneck, maintain incremental state on move push/pop:

```text
material totals
piece-square totals
pawn hash/key
game phase
```

Compute expensive pawn structure once per pawn-hash entry if useful.

## Tuning

Three levels:

### Manual
Reasonable feature weights.

### Texel-style supervised tuning
Fit weights so evaluation maps to observed game results.

### Teacher regression
Fit feature weights to teacher engine labels.

The latter is legal offline and is a useful bridge to NNUE.

## Tests

- material gain changes sign/magnitude correctly;
- mirrored/color-swapped boards produce symmetric values where appropriate;
- opening king safety fades in endgame;
- passed-pawn bonus increases as it advances;
- evaluator never returns mate constants; terminal logic belongs to search.

## Mandatory validation

Before comparing Elo, make these tests pass:

1. `get_move(fen, time_left_ms)` always returns a legal UCI move.
2. Test castling, promotion, en-passant, check evasion, mate and stalemate positions.
3. Test at very low clocks (`time_left_ms=50`, `200`, `1000`) and verify no flag/crash.
4. Test repeated calls in the same process; model/search state must remain valid.
5. Verify initialization is below 90 s.
6. Measure median and p99 move time.
7. Run at least hundreds of paired games against the supplied baselines.
8. Run both colours from the same starting positions.
9. Track W/D/L, Elo estimate, nodes/s, average depth and crashes.
10. Build the final ZIP and verify **uncompressed** size is < 50 MB.
11. Run in an offline environment with only contest-approved packages.
12. Save training scripts, dataset-generation scripts, seeds and provenance so the team can explain any shipped model.


## References

1. Shannon, **Programming a Computer for Playing Chess**: https://doi.org/10.1080/14786445008521796
2. AlphaZero paper Methods, classical evaluation/search discussion: https://arxiv.org/abs/1712.01815
3. Giraffe for learned evaluation contrast: https://arxiv.org/abs/1509.01549
4. NNUE technical docs for modern learned-evaluation contrast: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html

## Coding-agent instruction

Optimize evaluation in the context of search. Track evaluation calls, time spent in evaluation, and Elo. This file should remain the fallback when neural inference is too slow.
