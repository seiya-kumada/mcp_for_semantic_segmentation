#!/usr/bin/env python3
"""
MCP Semantic Segmentation Server - Main Entry Point

This module serves as the main entry point for the MCP semantic segmentation server.
It can be invoked using: python -m src or uv run python -m src
"""

import asyncio
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.server import main

if __name__ == "__main__":
    # サーバーの起動
    asyncio.run(main())