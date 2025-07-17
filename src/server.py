import asyncio
import json
import os

# import sys
from pathlib import Path
from typing import Any, Dict

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .segmentation import SemanticSegmenter


class MCPSemanticSegmentationServer:
    def __init__(self):
        self.server = Server("semantic-segmentation")
        self.segmenter = None
        self.static_dir = self._get_static_dir()
        self.input_dir = os.path.join(self.static_dir, "input")
        self.output_dir = os.path.join(self.static_dir, "output")

        # ディレクトリ作成
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self._setup_handlers()

    def _get_static_dir(self) -> str:
        """静的ファイルディレクトリのパス取得"""
        current_dir = Path(__file__).parent.parent
        return os.path.join(current_dir, "static")

    def _setup_handlers(self):
        """MCPサーバーのハンドラー設定"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """利用可能なツールのリスト"""
            return [
                Tool(
                    name="semantic_segmentation",
                    description="画像のセマンティックセグメンテーションを実行します",
                    inputSchema={
                        "type": "object",
                        "properties": {"image_path": {"type": "string", "description": "入力画像のパス"}},
                        "required": ["image_path"],
                    },
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> list[TextContent]:
            """ツールの実行"""
            if name == "semantic_segmentation":
                return await self._handle_segmentation(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _handle_segmentation(self, arguments: Dict[str, Any]) -> list[TextContent]:
        """セグメンテーション処理のハンドラー"""
        try:
            # 引数の取得
            image_path = arguments.get("image_path")
            if not image_path:
                raise ValueError("image_pathは必須です")

            # セグメンターの初期化（遅延初期化）
            if self.segmenter is None:
                self.segmenter = SemanticSegmenter()

            # 入力画像のコピー
            input_image_path = await self._copy_input_image(image_path)

            # セグメンテーション実行
            result = self.segmenter.process_image(input_image_path, self.output_dir)

            if result["status"] == "success":
                # 成功時のレスポンス
                file_url = f"file:///{result['output_path'].replace(os.sep, '/')}"
                json_url = f"file:///{result['json_path'].replace(os.sep, '/')}"

                # 検出されたクラスの情報を整理
                detected_classes_info = []
                for cls in result["detected_classes"]:
                    detected_classes_info.append(
                        {
                            "name": cls["name"],
                            "description": cls["description"],
                            "percentage": cls["percentage"],
                            "color": {"rgb": cls["color"]["rgb"], "hex": cls["color"]["hex"]},
                        }
                    )

                response = {
                    "result": {
                        "output_url": file_url,
                        "json_url": json_url,
                        "message": "セグメンテーションが完了しました",
                        "processing_time": result["processing_time"],
                        "input_size": result["input_size"],
                        "output_size": result["output_size"],
                        "detected_classes": detected_classes_info,
                        "total_pixels": result["total_pixels"],
                        "summary": self._generate_summary(detected_classes_info),
                    }
                }
            else:
                # エラー時のレスポンス
                response = {"error": {"message": result["error_message"], "code": result["error_code"]}}

            # Claude Desktopでの表示用にフォーマット
            if result["status"] == "success":
                display_text = f"""🎨 セマンティックセグメンテーション結果

📊 処理結果:
• 処理時間: {result['processing_time']}秒
• 画像サイズ: {result['input_size']}
• 総ピクセル数: {result['total_pixels']:,}

{response['result']['summary']}

📁 ファイル:
• 結果画像: {response['result']['output_url']}
• 詳細JSON: {response['result']['json_url']}

🔍 検出されたクラス詳細:"""

                for cls in detected_classes_info:
                    display_text += (
                        f"\n• {cls['name']} ({cls['description']}): {cls['percentage']}% - 色: {cls['color']['hex']}"
                    )

                return [TextContent(type="text", text=display_text)]
            else:
                return [TextContent(type="text", text=json.dumps(response, ensure_ascii=False, indent=2))]

        except Exception as e:
            error_response = {
                "error": {"message": f"処理中にエラーが発生しました: {str(e)}", "code": "PROCESSING_ERROR"}
            }
            return [TextContent(type="text", text=json.dumps(error_response, ensure_ascii=False, indent=2))]

    async def _copy_input_image(self, image_path: str) -> str:
        """入力画像をinputディレクトリにコピー"""
        import shutil
        import time

        # ファイル名の生成
        original_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(original_filename)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        new_filename = f"{name}_{timestamp}{ext}"

        # コピー先パス
        dest_path = os.path.join(self.input_dir, new_filename)

        # ファイルコピー
        shutil.copy2(image_path, dest_path)

        return dest_path

    def _generate_summary(self, detected_classes_info: list) -> str:
        """検出されたクラスの要約を生成"""
        if not detected_classes_info:
            return "オブジェクトが検出されませんでした。"

        # 背景以外のオブジェクトを抽出
        objects = [cls for cls in detected_classes_info if cls["name"] != "background"]
        background = next((cls for cls in detected_classes_info if cls["name"] == "background"), None)

        summary_parts = []

        # 背景情報
        if background:
            summary_parts.append(f"背景: {background['percentage']}%")

        # 検出されたオブジェクト
        if objects:
            summary_parts.append("検出されたオブジェクト:")
            for obj in objects:
                color_info = f"({obj['color']['hex']})"
                summary_parts.append(f"  • {obj['description']}: {obj['percentage']}% {color_info}")
        else:
            summary_parts.append("具体的なオブジェクトは検出されませんでした。")

        return "\n".join(summary_parts)

    async def run(self):
        """サーバーの実行"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="semantic-segmentation",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(), experimental_capabilities={}
                    ),
                ),
            )


async def main():
    """メイン関数"""
    server = MCPSemanticSegmentationServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
