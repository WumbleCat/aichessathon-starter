# AI Chessathon Architecture Playbook

These READMEs describe the major chess-engine approaches discussed for the AI Chessathon. They are written as implementation specifications for an AI coding agent.

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


## Recommended order

| Priority | Architecture | Contest fit | Why |
|---|---|---|---|
| 1 | `03_PVS_SELECTIVE_SEARCH.md` | Excellent | Strong classical search on one CPU |
| 2 | `04_NNUE_STOCKFISH_DISTILLATION.md` | Excellent | Best ML fit for CPU alpha-beta |
| 3 | `06_POLICY_VALUE_ALPHA_BETA_HYBRID.md` | Very good | Learned move ordering + cheap value |
| 4 | `02_MINIMAX_NEGAMAX_ALPHA_BETA.md` | Excellent baseline | Reliable foundation |
| 5 | `05_SUPERVISED_POLICY_NETWORK.md` | Good experiment | Cheap root guidance |
| 6 | `08_POLICY_VALUE_PUCT_MCTS.md` | Medium | Neural search, CPU-limited |
| 7 | `07_ALPHAZERO_SELF_PLAY.md` | Medium/low | Strong idea, expensive training |
| 8 | `11_SEARCHLESS_TRANSFORMER_CHESSBENCH.md` | Low/medium | Searchless inference, model-size pressure |
| 9 | `10_CHESSFORMER.md` | Low | Cutting-edge but CPU/size unfriendly |
| 10 | `12_DEEPCHESS_PAIRWISE.md` | Low/medium | Historically interesting learned evaluator |
| 11 | `13_GIRAFFE_DEEP_RL.md` | Low/medium | Historical learned evaluation route |
| 12 | `14_TDLEAF_KNIGHTCAP.md` | Medium research route | Learn eval from game-tree outcomes |
| 13 | `09_MUZERO.md` | Poor | Learns dynamics you already know |
| 14 | `01_RULE_BASED_ENGINE.md` | Baseline only | Useful for sanity/debugging |

## Suggested project strategy

Build a strong classical engine first, then improve evaluation rather than betting the entire contest on a large neural model:

```text
python-chess legal moves
        |
iterative deepening
        |
PVS / negamax alpha-beta
        |
TT + move ordering + pruning + reductions
        |
quiescence + SEE
        |
self-trained NNUE value
```

A second learned policy head can be tested for root/shallow move ordering, but never let slow inference destroy search depth.

## Contents

1. `01_RULE_BASED_ENGINE.md`
2. `02_MINIMAX_NEGAMAX_ALPHA_BETA.md`
3. `02A_HANDCRAFTED_EVALUATION.md`
4. `03_PVS_SELECTIVE_SEARCH.md`
5. `04_NNUE_STOCKFISH_DISTILLATION.md`
6. `05_SUPERVISED_POLICY_NETWORK.md`
7. `06_POLICY_VALUE_ALPHA_BETA_HYBRID.md`
8. `07_ALPHAZERO_SELF_PLAY.md`
9. `08_POLICY_VALUE_PUCT_MCTS.md`
10. `09_MUZERO.md`
11. `10_CHESSFORMER.md`
12. `11_SEARCHLESS_TRANSFORMER_CHESSBENCH.md`
13. `12_DEEPCHESS_PAIRWISE.md`
14. `13_GIRAFFE_DEEP_RL.md`
15. `14_TDLEAF_KNIGHTCAP.md`
16. `15_AUXILIARY_BOOKS_TABLEBASES.md`

## General experimental protocol

Change one component at a time. For each change record:

```text
git commit / experiment id
opponent
number of games
paired openings
W-D-L
estimated Elo difference
average time/move
nodes/s
depth
model size
peak RAM
crashes/flags/illegal moves
```

Do not accept a feature because it "looks smarter"; accept it because it wins statistically meaningful games under the **actual contest time control and hardware assumptions**.
