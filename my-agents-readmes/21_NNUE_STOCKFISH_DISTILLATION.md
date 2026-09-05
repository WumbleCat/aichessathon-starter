# Self-Trained NNUE with Engine-Label Distillation

## Goal

Train a **small CPU-friendly evaluation network from scratch** on positions labelled offline by a strong engine, then use the network as the leaf evaluator of your own alpha-beta/PVS search.

This is the highest-priority ML route for this contest.

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


## Legality / provenance

The competition explicitly allows training your network on positions labelled by an existing engine. It explicitly forbids shipping a published chess network or fine-tuned/re-exported version.

Therefore:

```text
LEGAL:
random initialization
+ your training code
+ positions
+ Stockfish labels used OFFLINE
-> your weights

NOT LEGAL:
download Stockfish .nnue
-> fine-tune / quantize / convert
-> submit
```

Keep a provenance file recording random seed, dataset source, teacher version/settings and training command.

## Why NNUE

NNUE is designed around:

1. sparse inputs;
2. small board changes between parent/child positions;
3. cheap low-precision CPU inference.

That matches alpha-beta, which calls evaluation many times per move.

## Contest-sized architecture

Do not reproduce a giant current Stockfish network. Start small.

### Simple HalfKP-like feature idea

Feature identity can depend on:

```text
side perspective
king square
piece type/color
piece square
```

Only a small number of input features are active for a position.

Possible experimental hidden sizes:

```text
128
256
384
512
```

Then a tiny dense head:

```text
sparse feature transformer
-> clipped activation
-> small dense layer
-> scalar value
```

Keep model comfortably below 50 MB; speed matters more than filling the ZIP.

## Training target

Generate records offline:

```text
FEN
teacher centipawn score
optional game result
optional best move / WDL
```

Prefer a large diverse set:

- opening;
- middlegame;
- tactical;
- quiet positional;
- endgame;
- balanced and imbalanced material;
- positions from self-play and public game PGNs.

Avoid huge duplication from adjacent plies.

### Teacher labelling

For each sampled position run your offline teacher with a consistent limit:

```text
fixed nodes OR fixed depth
```

Fixed nodes often gives more reproducible label cost than wall-clock time.

Clip extreme mate/centipawn values or convert to WDL-like targets.

## Loss

Start with either:

```text
Huber/MSE on scaled centipawn score
```

or a sigmoid/WDL-style target.

A useful experiment is mixing teacher evaluation with actual game result:

```text
target = lambda * teacher_value + (1-lambda) * result
```

but do not complicate the first model.

## Incremental inference

The major NNUE optimization is to cache the first-layer accumulator.

When a move changes only a few active features:

```text
parent accumulator
- removed feature weight columns
+ added feature weight columns
= child accumulator
```

Refresh from scratch when king-dependent feature perspective changes in a way your representation requires.

### Important Python warning

Naively evaluating a large sparse PyTorch layer node-by-node may be too slow.

Benchmark three routes:

1. NumPy / Numba integer accumulator;
2. manually serialized int8/int16 weights + Numba inference;
3. small ONNX network.

The best route is the one with highest **complete search Elo**, not lowest neural loss.

## Quantization

Train FP32, then test integer inference:

```text
weights: int8/int16
accumulators: int16/int32 as required
activation: clipped integer
output: int32 -> scaled centipawns
```

Check maximum possible accumulator magnitude to prevent overflow.

## Integration

Search should call:

```python
score = nnue.evaluate(position_or_accumulator)
```

The board/search owns push/pop. Store accumulator state per ply or update/revert it with the move stack.

## Training plan

### Phase A
100k-1M labelled positions, tiny network. Verify it learns material and sign correctly.

### Phase B
Millions of diverse positions. Tune network width and target scaling.

### Phase C
Quantize and benchmark inference nodes/s.

### Phase D
Replace handcrafted evaluation in your tested PVS engine.

### Phase E
Run many A/B games:
- handcrafted vs NNUE;
- NNUE size 128 vs 256 vs 384;
- FP32 vs quantized;
- different label depths.

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


## NNUE-specific debugging

- Position and color flip should produce sensible sign behavior.
- Incremental accumulator must equal full recomputation within exact integer equality (or tiny floating tolerance).
- Test captures, promotions, castling and en-passant feature deltas.
- Compare Python reference inference to optimized Numba/ONNX inference on thousands of random positions.
- Check model file byte size.
- Measure evaluation calls/second separately, then full engine nodes/second.

## Papers / references

1. Yu Nasu, **Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi** (2018), English translation: https://oscarbalcells.com/assets/nnue_paper_english.pdf
2. Stockfish NNUE technical documentation (architecture, sparse features, accumulators, quantization): https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html
3. Stockfish NNUE PyTorch project, useful for understanding training concepts — **do not ship or port Stockfish code/weights**: https://github.com/official-stockfish/nnue-pytorch
4. AI Chessathon rules, especially model provenance: https://aichessathon.com/docs

## Coding-agent instruction

Build an independent small NNUE-like evaluator specifically for this contest. Do not load any published chess weights. Train from random initialization on your own generated dataset, and preserve scripts proving provenance.
