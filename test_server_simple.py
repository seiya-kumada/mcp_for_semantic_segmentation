#!/usr/bin/env python3

import sys
import asyncio
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_server_creation():
    """サーバーの作成テスト"""
    try:
        from src.server import MCPSemanticSegmentationServer
        
        print("Creating server instance...")
        server = MCPSemanticSegmentationServer()
        print("[SUCCESS] Server instance created successfully")
        
        # 基本的な属性の確認
        print(f"Server name: {server.server}")
        print(f"Static dir: {server.static_dir}")
        print(f"Input dir: {server.input_dir}")
        print(f"Output dir: {server.output_dir}")
        
        print("[SUCCESS] Server creation test completed successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during server creation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_function():
    """メイン関数のテスト"""
    try:
        from src.server import main
        print("Main function imported successfully")
        print("[SUCCESS] Main function test completed successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error importing main function: {e}")
        import traceback
        traceback.print_exc()
        return False

def main_test():
    """メイン関数"""
    print("Testing MCP Server Components...")
    print("=" * 50)
    
    success1 = test_server_creation()
    success2 = test_main_function()
    
    if success1 and success2:
        print("\n[SUCCESS] All tests passed - Server components are working")
        print("You can now try: uv run python -m src")
    else:
        print("\n[ERROR] Some tests failed - Please check the errors above")
    
    return success1 and success2

if __name__ == "__main__":
    result = main_test()
    sys.exit(0 if result else 1)