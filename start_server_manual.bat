@echo off
echo Starting MCP Semantic Segmentation Server...
echo Press Ctrl+C to stop the server.
echo.
cd /d "C:\projects\mcp_for_semantic_segmentation"
uv run python -m src
pause