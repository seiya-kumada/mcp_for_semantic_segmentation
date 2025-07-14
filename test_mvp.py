#!/usr/bin/env python3

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.segmentation import SemanticSegmenter
from src.server import MCPSemanticSegmentationServer
from src.utils import (
    check_system_requirements,
    get_static_directories,
    validate_image_path,
    cleanup_old_files
)


class MVPTester:
    def __init__(self):
        self.test_results = []
        self.static_dirs = get_static_directories()
        
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """テスト結果をログに記録"""
        result = {
            "test": test_name,
            "success": success,
            "message": message
        }
        self.test_results.append(result)
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {test_name}: {message}")
    
    def create_test_image(self) -> str:
        """テスト用の画像を作成"""
        try:
            from PIL import Image
            import numpy as np
            
            # 簡単なテスト画像を作成
            width, height = 640, 480
            array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            image = Image.fromarray(array)
            
            # 一時ファイルに保存
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            image.save(temp_file.name)
            temp_file.close()
            
            return temp_file.name
        except Exception as e:
            raise RuntimeError(f"テスト画像作成エラー: {e}")
    
    def test_system_requirements(self):
        """システム要件のテスト"""
        try:
            requirements = check_system_requirements()
            
            # Python バージョン確認
            python_version = requirements["python_version"]
            major, minor = map(int, python_version.split('.'))
            if major >= 3 and minor >= 8:
                self.log_test("Python Version", True, f"Python {python_version}")
            else:
                self.log_test("Python Version", False, f"Python {python_version} (要求: >= 3.8)")
            
            # PyTorch確認
            if requirements["torch_available"]:
                cuda_status = "CUDA available" if requirements["cuda_available"] else "CPU only"
                self.log_test("PyTorch", True, cuda_status)
            else:
                self.log_test("PyTorch", False, "PyTorch not available")
            
            # メモリ確認
            memory_gb = requirements["available_memory_gb"]
            if memory_gb >= 2.0:
                self.log_test("Memory", True, f"{memory_gb:.1f}GB available")
            else:
                self.log_test("Memory", False, f"{memory_gb:.1f}GB available (要求: >= 2GB)")
            
            # 書き込み権限確認
            if requirements["static_dirs_writable"]:
                self.log_test("Directory Permissions", True, "Write access OK")
            else:
                self.log_test("Directory Permissions", False, "Write access failed")
                
        except Exception as e:
            self.log_test("System Requirements", False, str(e))
    
    def test_segmentation_basic(self):
        """基本的なセグメンテーション処理のテスト"""
        try:
            # テスト画像作成
            test_image_path = self.create_test_image()
            
            # セグメンターの初期化
            segmenter = SemanticSegmenter()
            self.log_test("Segmenter Initialization", True, "Model loaded successfully")
            
            # セグメンテーション実行
            result = segmenter.process_image(test_image_path, self.static_dirs["output"])
            
            if result["status"] == "success":
                self.log_test("Segmentation Processing", True, 
                            f"Processed in {result['processing_time']}s")
                
                # 出力ファイルの確認
                if os.path.exists(result["output_path"]):
                    self.log_test("Output File", True, "Output file created")
                else:
                    self.log_test("Output File", False, "Output file not found")
            else:
                self.log_test("Segmentation Processing", False, 
                            result.get("error_message", "Unknown error"))
            
            # クリーンアップ
            try:
                os.unlink(test_image_path)
            except:
                pass
                
        except Exception as e:
            self.log_test("Segmentation Basic", False, str(e))
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        try:
            segmenter = SemanticSegmenter()
            
            # 存在しないファイルのテスト
            result = segmenter.process_image("nonexistent.jpg", self.static_dirs["output"])
            if result["status"] == "error" and result["error_code"] == "FILE_NOT_FOUND":
                self.log_test("Error Handling - File Not Found", True, "Correctly handled")
            else:
                self.log_test("Error Handling - File Not Found", False, "Error not properly handled")
                
        except Exception as e:
            self.log_test("Error Handling", False, str(e))
    
    def test_utility_functions(self):
        """ユーティリティ関数のテスト"""
        try:
            # パス検証のテスト
            test_image_path = self.create_test_image()
            
            if validate_image_path(test_image_path):
                self.log_test("Utility - Path Validation", True, "Valid path correctly identified")
            else:
                self.log_test("Utility - Path Validation", False, "Valid path not recognized")
            
            # 無効なパスのテスト
            if not validate_image_path("invalid.txt"):
                self.log_test("Utility - Invalid Path", True, "Invalid path correctly rejected")
            else:
                self.log_test("Utility - Invalid Path", False, "Invalid path not rejected")
            
            # クリーンアップ
            try:
                os.unlink(test_image_path)
            except:
                pass
                
        except Exception as e:
            self.log_test("Utility Functions", False, str(e))
    
    def test_cleanup_functionality(self):
        """クリーンアップ機能のテスト"""
        try:
            # テスト用の古いファイルを作成
            test_file = os.path.join(self.static_dirs["output"], "old_test_file.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            
            # ファイルの作成時間を過去に設定
            import time
            old_time = time.time() - (8 * 24 * 60 * 60)  # 8日前
            os.utime(test_file, (old_time, old_time))
            
            # クリーンアップ実行
            deleted_files = cleanup_old_files(self.static_dirs["output"], days_old=7)
            
            if test_file in deleted_files:
                self.log_test("Cleanup Functionality", True, f"Cleaned up {len(deleted_files)} files")
            else:
                self.log_test("Cleanup Functionality", False, "Old file not cleaned up")
                
        except Exception as e:
            self.log_test("Cleanup Functionality", False, str(e))
    
    async def test_mcp_server_basic(self):
        """MCPサーバーの基本テスト"""
        try:
            # サーバーの初期化
            server = MCPSemanticSegmentationServer()
            self.log_test("MCP Server Initialization", True, "Server initialized successfully")
            
            # ツールリストの確認
            tools = await server.server.list_tools()
            if any(tool.name == "semantic_segmentation" for tool in tools):
                self.log_test("MCP Server - Tools", True, "semantic_segmentation tool available")
            else:
                self.log_test("MCP Server - Tools", False, "semantic_segmentation tool not found")
                
        except Exception as e:
            self.log_test("MCP Server Basic", False, str(e))
    
    def print_summary(self):
        """テスト結果の要約を出力"""
        print("\n" + "="*60)
        print("MVP TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print("\nFailed tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['message']}")
        
        print("\nMVP Status:", "READY" if failed_tests == 0 else "NEEDS FIXES")
        print("="*60)
    
    async def run_all_tests(self):
        """すべてのテストを実行"""
        print("Starting MVP Tests...")
        print("="*60)
        
        # 基本テスト
        self.test_system_requirements()
        self.test_segmentation_basic()
        self.test_error_handling()
        self.test_utility_functions()
        self.test_cleanup_functionality()
        
        # MCPサーバーテスト
        await self.test_mcp_server_basic()
        
        # 結果表示
        self.print_summary()


async def main():
    """メイン関数"""
    tester = MVPTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())