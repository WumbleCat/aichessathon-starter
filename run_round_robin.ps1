$repo = "E:\sourcecode\ai-chess-original\aichessathon-starter"
$python = "$repo\.venv\Scripts\python.exe"
$agentRoot = "$repo\agents"
$resultRoot = "$repo\round_robin_results"

Set-Location $repo

New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$log = "$resultRoot\round_robin_$timestamp.log"

$agents = Get-ChildItem $agentRoot -Directory | Sort-Object Name

Write-Host "Found $($agents.Count) agents."
"Round robin started: $(Get-Date)" | Tee-Object -FilePath $log
"Agents: $($agents.Name -join ', ')" | Tee-Object -FilePath $log -Append

for ($i = 0; $i -lt $agents.Count; $i++) {

    for ($j = $i + 1; $j -lt $agents.Count; $j++) {

        $agentA = "agents/$($agents[$i].Name)"
        $agentB = "agents/$($agents[$j].Name)"

        Write-Host ""
        Write-Host "========================================"
        Write-Host "$agentA VS $agentB"
        Write-Host "100 games"
        Write-Host "========================================"

        "" | Tee-Object -FilePath $log -Append
        "========================================" | Tee-Object -FilePath $log -Append
        "$agentA VS $agentB" | Tee-Object -FilePath $log -Append
        "========================================" | Tee-Object -FilePath $log -Append

        & $python -m harness.arena `
            --agent $agentA `
            --opponent $agentB `
            --games 100 2>&1 |
            Tee-Object -FilePath $log -Append
    }
}

"Round robin finished: $(Get-Date)" | Tee-Object -FilePath $log -Append
