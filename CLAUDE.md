@AGENTS.md

## Python Environment

This repository must use the following Python interpreter:

E:\sourcecode\ai-chess-original\aichessathon-starter\.venv\Scripts\python.exe

For ALL Python commands, tests, scripts, and harness executions, use this interpreter explicitly.

Do NOT use:

- `python`
- `python3`
- `py`
- `uv run python`

Examples:

```powershell
& "E:\sourcecode\ai-chess-original\aichessathon-starter\.venv\Scripts\python.exe" -m pytest

& "E:\sourcecode\ai-chess-original\aichessathon-starter\.venv\Scripts\python.exe" -m harness.arena --agent my-agents/01_negamax --opponent baselines/greedy --games 10
```
