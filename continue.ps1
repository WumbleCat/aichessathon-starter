$agents = claude agents --json --all | ConvertFrom-Json

$agents |
    Where-Object {
        $_.kind -eq "background" -and
        $_.state -eq "done" -and
        -not $_.pid -and
        $_.sessionId
    } |
    ForEach-Object {
        Write-Host "Continuing $($_.name)..."
        claude --bg --resume $_.sessionId "continue"
    }