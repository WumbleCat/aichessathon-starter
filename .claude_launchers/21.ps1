
Set-Location "E:\sourcecode\ai-chess-original\aichessathon-starter"

$Host.UI.RawUI.WindowTitle = "Chess Agent 21 - self-trained NNUE engine"

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host " CLAUDE CHESS AGENT 21" -ForegroundColor Cyan
Write-Host " self-trained NNUE engine" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "README:"
Write-Host "E:\sourcecode\ai-chess-original\aichessathon-starter\my-agents-readmes\21_NNUE_STOCKFISH_DISTILLATION.md" -ForegroundColor Yellow

Write-Host ""

Write-Host "Implementation:"
Write-Host "E:\sourcecode\ai-chess-original\aichessathon-starter\agents\21_nnue" -ForegroundColor Green

Write-Host ""

$Prompt = Get-Content -Raw "E:\sourcecode\ai-chess-original\aichessathon-starter\.claude_prompts\21.txt"

claude "$Prompt"

