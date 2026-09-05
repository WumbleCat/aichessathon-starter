$ErrorActionPreference = "Continue"

$repo = "E:\sourcecode\ai-chess-original\aichessathon-starter"

Set-Location $repo

Write-Host ""
Write-Host "=========================================="
Write-Host " CLAUDE RESUME + CONTINUE ALL"
Write-Host "=========================================="
Write-Host ""

# ------------------------------------------------------------
# 1. Get all Claude background sessions
# ------------------------------------------------------------

Write-Host "[1/4] Finding Claude sessions..."

$allAgents = @(
    claude agents --json --all | ConvertFrom-Json
)

# Only target background agents belonging to this project
# which are STOPPED or FAILED.
$targets = @(
    $allAgents | Where-Object {

        $inRepo = $false

        if ($_.cwd) {
            $inRepo = $_.cwd.StartsWith(
                $repo,
                [System.StringComparison]::OrdinalIgnoreCase
            )
        }

        $_.kind -eq "background" `
        -and $inRepo `
        -and $_.state -in @("stopped", "failed")
    }
)

if ($targets.Count -eq 0) {

    Write-Host ""
    Write-Host "No stopped/failed agents found."
    Write-Host ""
    Write-Host "Current sessions:"

    $allAgents |
        Where-Object { $_.kind -eq "background" } |
        Format-Table name, state, id, pid

    exit
}

Write-Host ""
Write-Host "Found $($targets.Count) stopped/failed agents:"
Write-Host ""

$targets |
    Format-Table name, state, id

# ------------------------------------------------------------
# 2. Respawn every stopped Claude session
# ------------------------------------------------------------

Write-Host ""
Write-Host "[2/4] Respawning agents..."
Write-Host ""

foreach ($agent in $targets) {

    Write-Host "RESPAWN -> $($agent.name) [$($agent.id)]"

    claude respawn $agent.id

    Start-Sleep -Milliseconds 500
}

# ------------------------------------------------------------
# 3. Give the agents time to start
# ------------------------------------------------------------

Write-Host ""
Write-Host "[3/4] Waiting for Claude processes to start..."

Start-Sleep -Seconds 10

# Check their status again

$afterRespawn = @(
    claude agents --json --all | ConvertFrom-Json
)

$targetIds = @($targets.id)

$restarted = @(
    $afterRespawn | Where-Object {
        $_.id -in $targetIds -and $_.pid
    }
)

Write-Host ""
Write-Host "$($restarted.Count)/$($targets.Count) agents now have running processes."
Write-Host ""

$afterRespawn |
    Where-Object { $_.id -in $targetIds } |
    Format-Table name, state, id, pid

# ------------------------------------------------------------
# 4. Use one Claude dispatcher to send "continue"
# ------------------------------------------------------------

Write-Host ""
Write-Host "[4/4] Sending CONTINUE to every restarted Claude..."
Write-Host ""

$names = @(
    $targets |
        Where-Object { $_.name } |
        Select-Object -ExpandProperty name -Unique
)

$nameList = ($names | ForEach-Object {
    "- $_"
}) -join "`n"

$dispatcherPrompt = @"
Use ListAgents to discover my other LOCAL Claude Code sessions.

I previously had multiple Claude Code background agents stop because my usage
limit was reached. They have now been restarted.

Send the exact message:

continue

using SendMessage to EACH of the following sessions:

$nameList

Rules:

1. Use ListAgents first.
2. Only message LOCAL Claude Code sessions.
3. Do not message yourself.
4. Send exactly "continue" to every target session you can find.
5. Use SendMessage separately for every target.
6. Do not skip a session just because it is idle.
7. Do not ask me any questions.
8. After sending all messages, report how many sessions received the message.
"@

# Use the same bypass-permission class as your unattended agents
claude -p --dangerously-skip-permissions $dispatcherPrompt

Write-Host ""
Write-Host "=========================================="
Write-Host " FINISHED"
Write-Host "=========================================="
Write-Host ""
Write-Host "Opening Agent View..."
Write-Host ""

claude agents