Set-Location "E:\sourcecode\ai-chess-original\aichessathon-starter"

Write-Host "[$(Get-Date)] Restarting Claude agents..."

claude respawn --all

Write-Host "[$(Get-Date)] Finished."