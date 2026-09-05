# Minimax -> Negamax -> Alpha-Beta Search

## Goal

Build the correct full-width search foundation. Minimax is the concept, negamax is the compact zero-sum implementation, and alpha-beta is the essential pruning optimization.

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


## Algorithm progression

### Minimax

```text
MAX chooses maximum child value.
MIN chooses minimum child value.
```

### Negamax

Chess is zero-sum, so:

```text
score(position) = -score(position_after_opponent_move)
```

This collapses MAX/MIN into one recursive routine.

### Alpha-beta

Maintain:

- `alpha`: best score already guaranteed to the side to move.
- `beta`: score at which the opponent has a better alternative.

When `alpha >= beta`, stop searching that branch.

## Core pseudocode

```python
def negamax(board, depth, alpha, beta, ply):
    if terminal(board):
        return terminal_score(board, ply)

    if depth == 0:
        return evaluate(board)

    best = -INF
    for move in ordered_legal_moves(board):
        board.push(move)
        score = -negamax(board, depth-1, -beta, -alpha, ply+1)
        board.pop()

        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return best
```

## Build sequence

1. Implement fixed-depth minimax depth 1-3.
2. Convert to negamax.
3. Add alpha-beta; verify returned moves/scores are identical to unpruned negamax.
4. Add iterative deepening:
   - search depth 1, 2, 3, ...
   - preserve last fully completed best move;
   - abort safely on time.
5. Add mate-distance scores (`MATE - ply`) so shorter mates are preferred.
6. Add move ordering.
7. Only after all tests pass, proceed to the stronger PVS/selective-search README.

## Move ordering

Alpha-beta efficiency depends enormously on ordering. Basic order:

```text
previous iteration / TT best move
promotions
winning captures
killer moves
history-scored quiets
bad captures
```

Even a weak ordering heuristic can greatly improve depth.

## Time management

Never begin a depth that can destroy the clock without a fallback.

Suggested architecture:

```python
best_completed_move = legal_moves[0]
for depth in range(1, MAX_DEPTH+1):
    try:
        move, score = search_depth(...)
        best_completed_move = move
    except SearchTimeout:
        break
return best_completed_move
```

Check time every N nodes rather than calling `time.perf_counter()` at every node.

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


## Specific correctness tests

- Compare minimax and alpha-beta on random shallow positions: same score/move set.
- Verify checkmate is not passed into static evaluation.
- Verify mate in 1 is preferred over winning material.
- Verify a shorter mate beats a longer mate.
- Verify search timeout always returns the last complete result.

## Papers / references

1. Claude E. Shannon, **Programming a Computer for Playing Chess** (1950): https://doi.org/10.1080/14786445008521796
2. Donald E. Knuth & Ronald W. Moore, **An Analysis of Alpha-Beta Pruning**, Artificial Intelligence 6(4), 1975: https://doi.org/10.1016/0004-3702(75)90019-3
3. Judea Pearl, **Asymptotic Properties of Minimax Trees and Game-Searching Procedures**, Artificial Intelligence 14(2), 1980: https://doi.org/10.1016/0004-3702(80)90037-5
4. AlphaZero paper methods section for a compact description of classical chess search: https://arxiv.org/abs/1712.01815

## Coding-agent instruction

Do not add aggressive pruning until plain alpha-beta is proven correct. Establish a regression suite where alpha-beta exactly matches full minimax at shallow depths.
