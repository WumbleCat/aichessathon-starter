$root = "E:\sourcecode\ai-chess-original\aichessathon-starter"

$python = "E:\sourcecode\ai-chess-original\aichessathon-starter\.venv\Scripts\python.exe"

Set-Location $root

# Find README files 17 -> 32
$readmes = Get-ChildItem -Path $root -Filter "*.md" |
    Where-Object {
        if ($_.Name -match '^(\d+)_') {
            $num = [int]$matches[1]
            return ($num -ge 17 -and $num -le 32)
        }

        return $false
    } |
    Sort-Object Name

foreach ($readme in $readmes) {

    $readmeName = $readme.Name

    if ($readmeName -match '^(\d+)_') {
        $number = $matches[1]
    }

    $sessionName = "chess-$number"

    $prompt = @"
Read the following README in full:

$readmeName

You are an independent Claude coding agent responsible ONLY for implementing
the chess bot described in this README.

Your task:

1. Read $readmeName completely.
2. Inspect the existing repository and harness.
3. Understand the required agent.py interface.
4. Implement the algorithm described in the README.
5. Put your implementation in its own appropriately named agent directory.
6. Do NOT modify unrelated bots.
7. Test the bot using the supplied chess harness.
8. Debug any errors yourself.
9. Continue improving the implementation until it works correctly.
10. Record useful progress/checkpoint information in the repository if needed.

PYTHON ENVIRONMENT
==================

For ALL Python commands use exactly:

$python

Do NOT use:
- system Python
- another .venv
- Conda
- another uv environment

Example:

& "$python" -m harness.arena --agent <your-agent> --opponent baselines/minimax --games 20


INTERRUPTION / TOKEN LIMIT HANDLING
===================================

This task may have been interrupted previously because the Claude usage/token
limit was reached.

If previous work already exists for this bot:

DO NOT START AGAIN FROM SCRATCH.

Instead:

1. Inspect git status.
2. Inspect git diff.
3. Inspect the bot directory.
4. Inspect files already created.
5. Inspect TODO comments.
6. Inspect previous test results or progress notes.
7. Determine what has already been completed.
8. Continue from the latest working checkpoint.

If the session approaches its context limit:

- save all code to disk
- save useful progress notes
- make sure the repository reflects the latest working state
- do not intentionally abandon the task early

Continue autonomously implementing, testing and debugging the bot.
"@

    Write-Host ""
    Write-Host "========================================"
    Write-Host "Starting Claude agent: $sessionName"
    Write-Host "README: $readmeName"
    Write-Host "========================================"

    claude --bg --name $sessionName $prompt

    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "All Claude agents have been launched."
Write-Host ""
Write-Host "Run:"
Write-Host "    claude agents"
Write-Host ""
Write-Host "to monitor them."