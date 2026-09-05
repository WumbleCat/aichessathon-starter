# Rule-Based / Handcrafted Chess Engine

## Goal

Create the simplest reliable legal chess agent with a human-designed evaluation function. This is not expected to be the final strongest entry; it is the reference implementation used to validate the harness, time manager and evaluation pipeline.

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


## Core architecture

```text
FEN
 -> python-chess Board
 -> generate legal moves
 -> score each move using handcrafted rules
 -> choose best legal move
```

Start with one-ply evaluation, then optionally add two-ply minimax. Keep this version intentionally simple.

## Position evaluation

Implement a score from the side-to-move or White perspective and be consistent everywhere.

Start with:

```text
material
+ piece-square activity
+ mobility
+ pawn structure
+ bishop pair
+ rook open/semi-open files
+ passed pawns
+ king safety
+ endgame king activity
```

Typical approximate material values are a starting point only:

```text
pawn   100
knight 320
bishop 330
rook   500
queen  900
```

Tune values by self-play/arena matches rather than assuming textbook numbers are optimal.

Use tapered evaluation:

```text
score = phase * middlegame_score + (1-phase) * endgame_score
```

where phase depends on remaining non-pawn material.

## Implementation steps

1. Use `python-chess` for legality and FEN parsing.
2. Implement `evaluate(board) -> int`.
3. Write small pure functions for each feature.
4. Ensure the sign convention is documented.
5. At `get_move`, make each legal move, evaluate the resulting position, undo, and return the best.
6. Add mate/stalemate handling before heuristic evaluation.
7. Add deterministic tie-breaking during experiments; random tie-breaking makes regressions harder to reproduce.
8. Add a time guard even though this version is fast.

## Useful handcrafted features

### Material
Count pieces.

### Piece-square tables
Give bonuses/penalties by piece and square. Mirror squares for Black.

### Mobility
Count safe or pseudo-legal moves for knights/bishops/rooks/queen. Do not make mobility so expensive that evaluation dominates runtime.

### Pawn structure
Penalize isolated/doubled/backward pawns; reward connected and passed pawns.

### King safety
Reward pawn cover and castled positions in the middlegame. Penalize exposed king files/diagonals.

### Endgame
Increase king centralization and passed-pawn value as material disappears.

## Debugging

Create positions where exactly one term should change. Example:

- same board except one doubled pawn;
- same material but knight center vs rim;
- same endgame but king center vs corner.

Print the term-by-term score offline. Do not print heavily during rated games.

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


## What the coding agent should NOT do

- Do not copy a third-party engine evaluation function.
- Do not port Stockfish's current evaluation/search code.
- Do not spend weeks hand-tuning this before search is working.

## Papers / references

1. Claude E. Shannon, **Programming a Computer for Playing Chess** (1950). Foundational discussion of Type A/Type B search and chess evaluation. DOI: https://doi.org/10.1080/14786445008521796
2. David Silver et al., **Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm**. Its methods section concisely describes the classical handcrafted-feature + alpha-beta paradigm that AlphaZero replaced: https://arxiv.org/abs/1712.01815
3. AI Chessathon docs: https://aichessathon.com/docs

## Coding-agent instruction

Implement a clean, independently written handcrafted baseline first. Make correctness and deterministic tests the priority. Keep the evaluator modular so it can later be replaced by NNUE without rewriting search.
