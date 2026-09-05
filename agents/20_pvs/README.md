# 20_pvs: PVS selective-search engine (numba bitboards)

Entry point: `agent.py` exposing `get_move(fen: str, time_left_ms: int) -> str`.

Everything in this directory was written for this project. Concepts come from the
published literature listed in `my-agents-readmes/20_PVS_SELECTIVE_SEARCH.md` (Pearl's
SCOUT / PVS, Zobrist hashing, Donninger's null move, Heinz's adaptive null move, Beal's
quiescence, the CPW description of SEE). No third-party engine code was consulted or
ported. python-chess is used only to build attack tables at import, to parse FENs, and as
the legality safety net.

## Files

| file | role |
| --- | --- |
| `agent.py` | contract entrypoint, time management, python-chess fallback, warm-up |
| `pvs_board.py` | numba bitboard position: tables, move generation, make/unmake, Zobrist, perft |
| `pvs_eval.py` | tapered evaluation (material + PST incremental, pawns, mobility, king, mop-up) |
| `pvs_search.py` | jitted PVS search with all selective features, Python iterative-deepening driver |
| `tests/test_engine.py` | perft, mandatory chess tests, clock tests, search behaviour |
| `bench.py` | node/depth benchmark and feature A/B on fixed positions |
| `selfplay.py` | paired self-play between two parameter sets at a fixed node budget (load-independent A/B) |
| `IMPLEMENTATION.md` | plan and design decisions |
| `RESULTS.md` | measurements and arena results |

## Architecture

```text
get_move
  python-chess fallback move chosen first (mate > capture > castle)
  Position.set_fen  ->  Searcher.search(time budget)
      iterative deepening, depth 1 unclocked
        aspiration window (+-20, x3 on failure, full window after 3 failures)
        negamax PVS (jitted, recursive)
          repetition / 50-move draw detection (own game history + search path)
          mate-distance pruning
          transposition table (2^21 entries, 32 MB, depth-preferred replacement, aged)
          check extension
          static eval refined by TT bound
          reverse futility pruning        depth <= 7, margin 85 * depth
          razoring                        depth <= 2, margin 200 * depth -> qsearch
          null-move pruning               R = 3 + depth/4 (+1 when eval >> beta), never with only pawns
          internal iterative deepening    PV nodes without a hash move, depth >= 5
          move ordering: TT move > good captures/queen promotions (MVV-LVA, SEE filter)
                         > killers > history (gravity-bounded) > bad captures > under-promotions
          futility pruning                depth <= 4, margin 110 * depth + 50, quiets only
          late-move pruning               depth <= 3, after 3 + 2 * depth^2 quiets
          SEE pruning of losing captures  depth <= 3
          late-move reductions            log table, less for PV / killers / good history
          PVS: null-window scout, re-search on fail high
          killer + history updates with malus for earlier quiets
        quiescence: stand pat, TT probe/store, captures + queen promotions, evasions in check,
                    delta pruning (150), SEE < 0 skipped
      best move from the last completed depth (or a fully searched better root move)
  engine move verified in board.legal_moves, else fallback
```

Every heuristic has an integer toggle in `pvs_search.default_params()` (`P_*` indices), so
A/B games with one feature disabled are a one-line change.

## Position representation

int64 numpy arrays only (numba would silently promote a mixed uint64/int64 expression to
float). Piece codes 1..6 white, 7..12 black. Sliding attacks use compact PEXT-style tables
built from python-chess's edge-stripped masks (rank/file 64 entries, bishop 512 entries per
square). All lookup tables live in one flat array passed as an argument; numba embeds global
arrays as constants and that costs seconds of compile time per function.

## Time management

`think_time_ms = clamp(time_left / max(20, 40 - move/2) + 0.8 * increment, 15 ms,
time_left / 4)`, capped at `time_left - 100 ms`. Depth 1 runs without a clock so a move
always exists; deeper iterations check the wall clock every 2048 nodes through a numba
`objmode` call. A new iteration is not started when more than 55 % of the budget is spent.
Below 120 ms on the clock the python-chess fallback move is played instantly.

## Initialisation

All jitted functions carry explicit signatures and compile eagerly when `pvs_board` and
`pvs_search` are imported. `agent.py` does that import in a daemon thread started at its own
import, joins it for at most 60 s, and returns; the platform then sees the agent as ready
inside the 90 s budget no matter how slow the compile is. `get_move` waits a bounded share of
a long clock (30 %, at most 40 s, never on a clock under 30 s) for the thread and otherwise
plays the one-ply python-chess fallback until the engine is ready. Two short warm-up searches
run inside the thread. numba's on-disk cache is deliberately off: cached code for the
recursive search crashed with an access violation when a second process loaded it, so every
process compiles from source. See RESULTS.md for the measured times.

## Status

Merge into `main` is pending: the code lives on `feature/agent-20-pvs` (commits made through a
throwaway index because the shared checkout stays on `main`), and the working-tree copy in
`agents/20_pvs/` matches it. See RESULTS.md for what has been measured and what is next.

## Running

```powershell
& .venv\Scripts\python.exe -m unittest discover -s agents/20_pvs/tests -v
& .venv\Scripts\python.exe agents/20_pvs/bench.py
& .venv\Scripts\python.exe agents/20_pvs/selfplay.py --b P_NULL=0 --nodes 4000 --games 100
& .venv\Scripts\python.exe -m harness.arena --agent agents/20_pvs --opponent baselines/minimax --games 10
```
