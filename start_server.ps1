#!/usr/bin/env pwsh

Write-Host "Starting MCP Semantic Segmentation Server..." -ForegroundColor Green

# プロジェクトディレクトリに移動
Set-Location "C:\projects\mcp_for_semantic_segmentation"

# uvが利用可能かチェック
try {
    $uvVersion = uv --version
    Write-Host "Using uv: $uvVersion" -ForegroundColor Cyan
} catch {
    Write-Host "Error: uv is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# 仮想環境の確認
if (Test-Path ".venv") {
    Write-Host "Virtual environment found" -ForegroundColor Green
} else {
    Write-Host "Warning: Virtual environment not found" -ForegroundColor Yellow
}

# サーバー起動
Write-Host "Starting server..." -ForegroundColor Blue
try {
    uv run python -m src
} catch {
    Write-Host "Error starting server: $_" -ForegroundColor Red
    exit 1
}