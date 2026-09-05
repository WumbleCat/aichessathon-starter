# Hybrid Policy + Value Neural Guidance for Alpha-Beta

## Goal

Combine the best contest-compatible ideas:

- a very cheap learned **value** evaluator;
- a learned **policy** used sparingly for move ordering;
- PVS/alpha-beta for tactical verification.

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


## Architecture

```text
                 position
                    |
          +---------+---------+
          |                   |
      cheap value          policy prior
       (NNUE)           (root/shallow only)
          |                   |
          +---------+---------+
                    |
              PVS alpha-beta
                    |
                 best move
```

Do not assume the two heads must share a network. For CPU efficiency, two specialized models can be better:

```text
value.nnue-like weights
policy.onnx (small)
```

or one compact shared trunk if benchmarked faster.

## Why this can beat pure neural play

Policy answers:

> Which moves look promising?

Alpha-beta answers:

> Do those moves survive adversarial tactical search?

Value answers:

> How good is the quiet leaf position?

The combination attacks different bottlenecks.

## Build plan

1. Finish `03_PVS_SELECTIVE_SEARCH.md`.
2. Finish a small value model from `04_NNUE_STOCKFISH_DISTILLATION.md`.
3. Train policy model from `05_SUPERVISED_POLICY_NETWORK.md`.
4. Add policy only at the root.
5. Measure Elo and time.
6. If positive, allow policy at first 2 plies or selected high-depth nodes.
7. Add cache keyed by Zobrist hash for policy outputs.
8. Never call policy in qsearch.

## Policy score integration

Do not blindly sort by neural probability only.

Combine:

```text
TT move: absolute priority
forcing good capture: large bonus
policy prior: medium bonus
killer/history: bonus
bad SEE capture: penalty
```

Tune scale by arena games.

## Value integration

Replace handcrafted static evaluation gradually:

```text
experiment A: 100% handcrafted
experiment B: 100% neural
experiment C: blend neural + cheap material sanity term
```

If neural model has out-of-distribution failures, a small material/check sanity layer may improve robustness.

## Optional multi-task training

Train one trunk with:

```text
value head -> teacher centipawn/WDL
policy head -> teacher move distribution
```

Loss:

```text
L = value_weight * L_value
  + policy_weight * L_policy
  + regularization
```

But the shared model is only justified if inference remains cheap.

## Caching

A root policy can be cached per Zobrist key. Since each game process persists between moves, reuse information when a position is revisited.

## Failure modes

- policy inference makes search shallower;
- policy overconfidently demotes a tactical move;
- value and policy trained on narrow opening distributions;
- action-index bugs around promotions;
- model is accurate offline but loses Elo.

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


## Ablation table to demand from the coding agent

```text
variant                   Elo   nodes/s   depth   ms/move
classical
+ value NNUE
+ root policy
+ policy first 2 plies
+ shared policy/value
```

## References

1. AlphaZero policy/value + MCTS concept: https://arxiv.org/abs/1712.01815
2. ChessBench supervised action/value distillation: https://arxiv.org/abs/2402.04494
3. NNUE CPU evaluation principles: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html
4. AI Chessathon docs: https://aichessathon.com/docs

## Coding-agent instruction

Optimize for **whole-engine Elo per CPU second**. A neural component is only accepted if its guidance compensates for the nodes lost to inference latency.
