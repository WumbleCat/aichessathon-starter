# Negamax for a Chess Bot

Negamax is a simplified version of the Minimax algorithm for two-player, zero-sum games such as chess. It works because a position that is good for one player is equally bad for the other player.

The key idea is:

```text
my_score = -opponent_score
```

Instead of having separate logic for a maximizing player and a minimizing player, Negamax always tries to maximize the score for the player whose turn it is.

---

## How Negamax Works

### 1. Start from the current board position

The bot receives the current chess position and chooses a search depth.

For example:

```python
depth = 4
```

A depth of 4 means the bot searches four plies ahead.

A **ply** is one move by one player.

```text
Depth 4: Your move
Depth 3: Opponent move
Depth 2: Your move
Depth 1: Opponent move
Depth 0: Evaluate the board
```

---

### 2. Generate every legal move

From the current position, generate all legal moves.

For example:

```text
e4
d4
Nf3
Nc3
...
```

The bot will test every move and determine which one produces the best result.

---

### 3. Make one move temporarily

Take the first legal move and apply it to the board.

For example:

```python
board.push(move)
```

If White plays:

```text
e4
```

the board is updated and it becomes Black's turn.

---

### 4. Search from the opponent's perspective

Negamax recursively searches the new position:

```python
negamax(board, depth - 1)
```

The recursive call is effectively asking:

> What is the best score the opponent can achieve from this position?

---

### 5. Negate the opponent's score

This is the most important part of Negamax.

```python
score = -negamax(board, depth - 1)
```

Suppose the opponent evaluates the position as:

```text
+3
```

That means the position is good for them.

From your perspective, the same position is:

```text
-3
```

Therefore:

```text
Opponent: +3
You:      -3
```

This works because chess is a zero-sum game.

---

### 6. Repeat the same process for the opponent

The opponent now:

1. Generates all legal moves.
2. Makes each move.
3. Searches the next position.
4. Negates the returned score.
5. Keeps the highest score.

The same Negamax function works for both White and Black.

This is the main advantage over traditional Minimax.

---

### 7. Continue until the maximum search depth is reached

The algorithm continues recursively:

```text
Current Position
      |
      |-- Move A
      |      |
      |      |-- Opponent Move A
      |      |       |
      |      |       |-- Your Move A
      |      |       |-- Your Move B
      |      |
      |      |-- Opponent Move B
      |
      |-- Move B
      |
      |-- Move C
```

Eventually the algorithm reaches:

```python
depth == 0
```

At this point it stops searching deeper.

---

### 8. Evaluate the board

At depth 0, the bot uses an evaluation function.

A simple evaluation function might use piece values:

```text
Pawn   = 100
Knight = 320
Bishop = 330
Rook   = 500
Queen  = 900
```

For example:

```python
score = white_material - black_material
```

A stronger chess bot could also evaluate:

- Material advantage
- King safety
- Piece activity
- Centre control
- Pawn structure
- Passed pawns
- Mobility
- Bishop pair
- Checkmate or stalemate

For Negamax, the returned evaluation should be from the perspective of the player whose turn it is.

For example:

```text
+300 = good for current player
   0 = roughly equal
-300 = bad for current player
```

---

### 9. Return the score back up the search tree

Suppose a position at the bottom of the tree gives:

```text
White = +4
```

When the score returns to Black:

```text
Black = -4
```

When it returns to White again:

```text
White = +4
```

So each level flips the sign:

```text
+4
 |
 negate
 |
-4
 |
 negate
 |
+4
```

---

### 10. Choose the highest score at every position

Suppose the bot considers three moves:

```text
Move A -> -3
Move B -> +2
Move C -> -1
```

Negamax chooses:

```text
Move B
```

because:

```python
max(-3, 2, -1) == 2
```

The algorithm always chooses the move with the highest score from the current player's perspective.

---

### 11. Undo the move

After evaluating a move, restore the board before testing the next move.

For example with `python-chess`:

```python
board.push(move)

score = -negamax(board, depth - 1)

board.pop()
```

`push()` makes the move.

`pop()` restores the previous position.

This allows the same board object to be reused throughout the search.

---

### 12. Remember the best move at the root

Inside Negamax, the algorithm mainly needs to return scores.

At the top level, the bot also needs to remember which move produced the highest score.

```python
best_score = -float("inf")
best_move = None

for move in board.legal_moves:
    board.push(move)

    score = -negamax(board, depth - 1)

    board.pop()

    if score > best_score:
        best_score = score
        best_move = move
```

After every legal move has been tested:

```python
return best_move
```

---

## Basic Negamax Implementation

```python
def negamax(board, depth):
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    best_score = -float("inf")

    for move in board.legal_moves:
        board.push(move)

        score = -negamax(board, depth - 1)

        board.pop()

        best_score = max(best_score, score)

    return best_score
```

---

## Selecting the Best Move

A separate function can call Negamax for each legal move:

```python
def choose_move(board, depth):
    best_move = None
    best_score = -float("inf")

    for move in board.legal_moves:
        board.push(move)

        score = -negamax(board, depth - 1)

        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move
```

The chess bot can then do:

```python
move = choose_move(board, depth=4)
```

---

## Example

Imagine the bot has three possible moves.

After searching, the opponent's best scores are:

```text
Move A -> opponent gets +3
Move B -> opponent gets -2
Move C -> opponent gets +1
```

Negamax flips the scores:

```text
Move A -> -3 for us
Move B -> +2 for us
Move C -> -1 for us
```

Therefore the bot chooses:

```text
Move B
```

because `+2` is the highest score.

---

## Negamax vs Minimax

Traditional Minimax normally has two cases:

```text
MAX player -> choose highest score
MIN player -> choose lowest score
```

Negamax removes the separate MIN case.

Instead, both players do:

```text
Choose the highest score.
```

The perspective changes using:

```python
score = -negamax(child)
```

Therefore Negamax and Minimax produce the same result when implemented correctly.

---

## Negamax with Alpha-Beta Pruning

Basic Negamax becomes very slow at larger depths because the number of chess positions grows rapidly.

A common improvement is **alpha-beta pruning**.

Alpha-beta pruning skips branches that cannot affect the final decision.

A Negamax version looks like:

```python
def negamax(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    best_score = -float("inf")

    for move in board.legal_moves:
        board.push(move)

        score = -negamax(
            board,
            depth - 1,
            -beta,
            -alpha
        )

        board.pop()

        best_score = max(best_score, score)
        alpha = max(alpha, score)

        if alpha >= beta:
            break

    return best_score
```

Notice:

```python
-beta, -alpha
```

When the search changes to the opponent's perspective, the alpha-beta search window must also be negated and reversed.

Alpha-beta pruning returns the same best move as normal Negamax, but can search much faster.

---

## Recommended Development Order

A good progression for a chess bot is:

```text
1. Basic board evaluation
        |
        v
2. Negamax
        |
        v
3. Alpha-Beta Pruning
        |
        v
4. Move Ordering
        |
        v
5. Iterative Deepening
        |
        v
6. Transposition Table
        |
        v
7. Quiescence Search
        |
        v
8. Better Evaluation Function
```

---

## Mental Model

The easiest way to understand Negamax is:

```text
For every legal move:

    Make the move

    Ask:
        "What is the best score my opponent
         can achieve from here?"

    Negate their answer

    Undo the move

Choose the move that gives me the highest score.
```

In code, the heart of Negamax is simply:

```python
my_score = -opponent_best_score
```

or recursively:

```python
best_score = max(
    -negamax(child)
    for child in possible_positions
)
```

---

## Resources

Useful references for learning more:

- Chess Programming Wiki — Negamax  
  https://www.chessprogramming.org/Negamax

- Chess Programming Wiki — Alpha-Beta  
  https://www.chessprogramming.org/Alpha-Beta

- Chess Programming Wiki — Evaluation  
  https://www.chessprogramming.org/Evaluation

- Wikipedia — Negamax  
  https://en.wikipedia.org/wiki/Negamax

- Chess Programming Wiki — Search  
  https://www.chessprogramming.org/Search

---

## Summary

Negamax searches every possible move recursively, evaluates positions at a chosen depth, and assumes that the opponent will always choose their best possible response. Because a good position for one player is a bad position for the other, every recursive result is negated. This allows both players to use the same maximization logic:

```python
score = -negamax(child)
```

The bot ultimately selects the move with the highest Negamax score. In a practical chess engine, Negamax is normally combined with alpha-beta pruning and other search optimizations so that the bot can search deeper positions efficiently.
