
Set-Location "E:\sourcecode\ai-chess-original\aichessathon-starter"

$Host.UI.RawUI.WindowTitle = "Chess Agent 24 - AlphaZero-style self-play engine"

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host " CLAUDE CHESS AGENT 24" -ForegroundColor Cyan
Write-Host " AlphaZero-style self-play engine" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "README:"
Write-Host "E:\sourcecode\ai-chess-original\aichessathon-starter\my-agents-readmes\24_ALPHAZERO_SELF_PLAY.md" -ForegroundColor Yellow

Write-Host ""

Write-Host "Implementation:"
Write-Host "E:\sourcecode\ai-chess-original\aichessathon-starter\agents\24_alphazero" -ForegroundColor Green

Write-Host ""

$Prompt = Get-Content -Raw "E:\sourcecode\ai-chess-original\aichessathon-starter\.claude_prompts\24.txt"

claude "$Prompt"

