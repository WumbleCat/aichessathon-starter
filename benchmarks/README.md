# Engine benchmark

Every agent in this repository, played against a real engine at a fixed strength, so the agents
can be compared with one number instead of only against each other.

```powershell
$py = "E:\sourcecode\ai-chess-original\aichessathon-starter\.venv\Scripts\python.exe"
& $py -m tools.engine_bench --games 100 --elo 1320 2200 --workers 6      # the full sweep
& $py -m tools.engine_bench --agents 31 --games 20 --elo 2200            # one agent
& $py -m tools.engine_bench --report                                     # table only, no games
& $py -m tools.engine_bench --games 100 --elo 1320 --redo-failures       # repair a loaded run
```

## The opponent

Stockfish, run locally as the opponent only. The competition forbids *shipping* a third-party
engine or a table of its moves; it does not forbid measuring yourself against one, and the
repository's `.gitignore` already keeps engine binaries out of git. The binary used is whichever
copy another agent's training pipeline already vendored (`agents/21_nnue/training/teacher/`,
`agents/23_hybrid/training/tools/`); nothing is downloaded and nothing is committed.

It plays with:

| Setting | Value | Why |
|---|---|---|
| `UCI_LimitStrength` + `UCI_Elo` | 1320 (the floor) and 2200 | one calibrated opponent, so a score converts to an implied Elo |
| `go movetime` | 100 ms | identical opponent in every game, independent of what the agent does with its clock |
| `Threads` | 1 | so eight games in parallel do not fight over cores |
| `Hash` | 16 MB | small enough that many games fit in memory at once |

Two levels because one saturates: at 1320 the strongest agents score 100 %, which puts no
ordering on them, and at 2200 the weakest score 0 %. Each level ranks the half of the field it
can actually separate.

## The games

Games are played by `harness.referee.play_match`, the same referee the local arena uses, so an
agent is started through `harness/runner.py`, gets the real 90 s init budget, is charged wall
time, and loses on a flag, an illegal move or a crash exactly as it would on the platform. The
engine side is a small object with the same `start`/`move`/`stop` interface, so `harness/` is
imported and never modified. Colours alternate: even-numbered games the agent is white.

Agent clock is the local arena's fast control, 10 s + 0.1 s, not the platform's 120 s + 0.5 s.
At the real control the sweep would take days. It is the same control for every agent, but it
does penalise agents that spend a long time initialising or that lean on deep search.

`implied Elo` is the rating that explains the score against an opponent of known rating,
`opponent_elo - 400 log10(1/score - 1)`, with a normal-approximation 95 % half-width. It
inherits whatever error Stockfish's own `UCI_Elo` calibration has, so treat the numbers as a
consistent ranking rather than as FIDE ratings.

## Reading the failures column

It counts games that ended in `crash`, `flag`, `illegal`, `init` or a harness error rather than
by a chess result. **A non-zero number there usually means the machine, not the agent.** The
first full sweep here reported crashes in 24 % of games, spread evenly over all 32 agents
including the simplest ones, because it ran against other sessions' training with about 2 GB
free; the same agents failed zero games run serially, at 8 workers alone, and with the
torch-heavy ones together. Those games are scored as losses and drag an agent's score down by
tens of points, so a run with a populated failures column is not a strength measurement.

If it happens: wait for the machine to be quiet, lower `--workers`, and re-run with
`--redo-failures`, which replays exactly those games and keeps the good ones. Failed games
store the agent's stderr tail in the JSONL, which is the first thing to read when deciding
whether a failure was really the agent.

## Files

| File | What it is |
|---|---|
| `engine_bench.jsonl` | one line per game; the last line for a key wins, so replays supersede |
| `REPORT.md` | the rendered table, rewritten at the end of a sweep |
| `run.log` | the sweep's own progress output |

The JSONL is the record: a re-run skips games already in it, so raising `--games` tops the
sweep up instead of starting it over, and an interrupted sweep loses only the game in flight.
Jobs are ordered round-robin across agents, so a partial run still covers all of them.
