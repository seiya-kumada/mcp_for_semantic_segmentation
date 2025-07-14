#!/usr/bin/env python3

import sys
import asyncio
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_server_startup():
    """サーバーの起動テスト"""
    try:
        from src.server import MCPSemanticSegmentationServer
        
        print("Creating server instance...")
        server = MCPSemanticSegmentationServer()
        print("[SUCCESS] Server instance created successfully")
        
        # ツールリストの確認
        print("Testing list_tools...")
        tools = server.server.list_tools()
        print(f"[SUCCESS] Tools available: {len(tools)} tools")
        
        for tool in tools:
            print(f"   - {tool.name}: {tool.description}")
        
        print("[SUCCESS] Server startup test completed successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during server startup: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """メイン関数"""
    print("Testing MCP Server Startup...")
    print("=" * 50)
    
    success = await test_server_startup()
    
    if success:
        print("\n[SUCCESS] All tests passed - Server is ready to use")
    else:
        print("\n[ERROR] Tests failed - Please check the errors above")
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)