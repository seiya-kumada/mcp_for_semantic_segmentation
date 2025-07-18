import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .segmentation import SemanticSegmenter


class ColorInfo(TypedDict):
    """色情報の型定義"""
    rgb: List[int]
    hex: str


class DetectedClass(TypedDict):
    """検出されたクラス情報の型定義"""
    name: str
    description: str
    percentage: float
    color: ColorInfo


class SegmentationResult(TypedDict):
    """セグメンテーション結果の型定義"""
    status: str
    output_path: str
    json_path: str
    processing_time: float
    input_size: str
    output_size: str
    detected_classes: List[DetectedClass]
    total_pixels: int
    error_message: Optional[str]
    error_code: Optional[str]


class ErrorCodes:
    """エラーコード定数"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    FILE_ERROR = "FILE_ERROR"


class ServerConfig:
    """サーバー設定定数"""
    SERVER_NAME = "semantic-segmentation"
    VERSION = "0.1.0"
    STATIC_DIR = "static"
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    TOOL_NAME = "semantic_segmentation"
    IMAGE_PATH_PARAM = "image_path"
    
    # ファイル名用タイムスタンプ形式
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    
    # ログ設定
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Messages:
    """メッセージ定数"""
    SEGMENTATION_COMPLETE = "セグメンテーションが完了しました"
    NO_OBJECTS_DETECTED = "オブジェクトが検出されませんでした。"
    IMAGE_PATH_REQUIRED = "image_path is required"
    UNKNOWN_TOOL = "Unknown tool: {}"
    BACKGROUND_LABEL = "background"
    NO_SPECIFIC_OBJECTS = "具体的なオブジェクトは検出されませんでした。"
    
    # 表示テンプレート
    RESULT_TEMPLATE = """セマンティックセグメンテーション結果

処理結果:
• 処理時間: {}秒
• 画像サイズ: {}
• 総ピクセル数: {:,}

{}

ファイル:
• 結果画像: {}
• 詳細JSON: {}

検出されたクラス詳細:"""


class MCPSemanticSegmentationServer:
    def __init__(self) -> None:
        self.server = Server(ServerConfig.SERVER_NAME)
        self.segmenter = None
        self.static_dir = self._get_static_dir()
        self.input_dir = os.path.join(self.static_dir, ServerConfig.INPUT_DIR)
        self.output_dir = os.path.join(self.static_dir, ServerConfig.OUTPUT_DIR)

        # ディレクトリ作成
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self._setup_handlers()
        self._setup_logging()

    def _setup_logging(self) -> None:
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format=ServerConfig.LOG_FORMAT
        )
        self.logger = logging.getLogger(__name__)

    def _handle_error(self, error: Exception, context: str, error_code: Optional[str] = None) -> Dict[str, Any]:
        """統一エラーハンドリング"""
        error_code = error_code or ErrorCodes.PROCESSING_ERROR
        error_message = str(error)
        
        # ログに記録
        self.logger.error(f"Error in {context}: {error_message}", exc_info=True)
        
        # エラーレスポンス構築
        return {
            "error": {
                "message": error_message,
                "code": error_code,
                "context": context
            }
        }

    def _get_static_dir(self) -> str:
        """静的ファイルディレクトリのパス取得"""
        current_dir = Path(__file__).parent.parent
        return os.path.join(current_dir, ServerConfig.STATIC_DIR)

    def _setup_handlers(self):
        """MCPサーバーのハンドラー設定"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """利用可能なツールのリスト"""
            return [
                Tool(
                    name=ServerConfig.TOOL_NAME,
                    description="画像のセマンティックセグメンテーションを実行します",
                    inputSchema={
                        "type": "object",
                        "properties": {ServerConfig.IMAGE_PATH_PARAM: {"type": "string", "description": "入力画像のパス"}},
                        "required": [ServerConfig.IMAGE_PATH_PARAM],
                    },
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> list[TextContent]:
            """ツールの実行"""
            try:
                if name == ServerConfig.TOOL_NAME:
                    return await self._handle_segmentation(arguments)
                else:
                    error_response = self._handle_error(
                        ValueError(Messages.UNKNOWN_TOOL.format(name)), 
                        "call_tool", 
                        ErrorCodes.TOOL_ERROR
                    )
                    return [TextContent(type="text", text=json.dumps(error_response, ensure_ascii=False, indent=2))]
            except Exception as e:
                error_response = self._handle_error(e, "call_tool")
                return [TextContent(type="text", text=json.dumps(error_response, ensure_ascii=False, indent=2))]

    def _validate_arguments(self, arguments: Dict[str, Any]) -> str:
        """引数の検証"""
        image_path = arguments.get(ServerConfig.IMAGE_PATH_PARAM)
        if not image_path:
            raise ValueError(Messages.IMAGE_PATH_REQUIRED)
        return image_path

    def _build_success_response(self, result: SegmentationResult) -> Dict[str, Any]:
        """成功時のレスポンス構築"""
        file_url = f"file:///{result['output_path'].replace(os.sep, '/')}"
        json_url = f"file:///{result['json_path'].replace(os.sep, '/')}"

        detected_classes_info: List[DetectedClass] = []
        for cls in result["detected_classes"]:
            detected_classes_info.append(
                {
                    "name": cls["name"],
                    "description": cls["description"],
                    "percentage": cls["percentage"],
                    "color": {"rgb": cls["color"]["rgb"], "hex": cls["color"]["hex"]},
                }
            )

        return {
            "result": {
                "output_url": file_url,
                "json_url": json_url,
                "message": Messages.SEGMENTATION_COMPLETE,
                "processing_time": result["processing_time"],
                "input_size": result["input_size"],
                "output_size": result["output_size"],
                "detected_classes": detected_classes_info,
                "total_pixels": result["total_pixels"],
                "summary": self._generate_summary(detected_classes_info),
            }
        }

    def _build_error_response(self, result: SegmentationResult) -> Dict[str, Any]:
        """エラー時のレスポンス構築"""
        return {"error": {"message": result["error_message"], "code": result["error_code"]}}

    def _format_display_text(self, result: SegmentationResult, response: Dict[str, Any]) -> str:
        """表示用テキストの生成"""
        detected_classes_info: List[DetectedClass] = response['result']['detected_classes']
        
        display_text = Messages.RESULT_TEMPLATE.format(
            result['processing_time'],
            result['input_size'],
            result['total_pixels'],
            response['result']['summary'],
            response['result']['output_url'],
            response['result']['json_url']
        )

        for cls in detected_classes_info:
            display_text += (
                f"\n• {cls['name']} ({cls['description']}): {cls['percentage']}% - 色: {cls['color']['hex']}"
            )

        return display_text

    async def _handle_segmentation(self, arguments: Dict[str, Any]) -> list[TextContent]:
        """セグメンテーション処理のハンドラー"""
        try:
            # 引数の検証
            image_path = self._validate_arguments(arguments)

            # セグメンターの初期化（遅延初期化）
            if self.segmenter is None:
                self.segmenter = SemanticSegmenter()

            # 入力画像のコピー
            input_image_path = await self._copy_input_image(image_path)

            # セグメンテーション実行
            result = self.segmenter.process_image(input_image_path, self.output_dir)

            # レスポンス構築
            if result["status"] == "success":
                response = self._build_success_response(result)
                display_text = self._format_display_text(result, response)
                return [TextContent(type="text", text=display_text)]
            else:
                response = self._build_error_response(result)
                return [TextContent(type="text", text=json.dumps(response, ensure_ascii=False, indent=2))]

        except Exception as e:
            error_response = self._handle_error(e, "_handle_segmentation")
            return [TextContent(type="text", text=json.dumps(error_response, ensure_ascii=False, indent=2))]

    async def _copy_input_image(self, image_path: str) -> str:
        """入力画像をinputディレクトリにコピー"""

        # ファイル名の生成
        original_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(original_filename)
        timestamp = time.strftime(ServerConfig.TIMESTAMP_FORMAT)
        new_filename = f"{name}_{timestamp}{ext}"

        # コピー先パス
        dest_path = os.path.join(self.input_dir, new_filename)

        # ファイルコピー
        shutil.copy2(image_path, dest_path)

        return dest_path

    def _generate_summary(self, detected_classes_info: List[DetectedClass]) -> str:
        """検出されたクラスの要約を生成"""
        if not detected_classes_info:
            return Messages.NO_OBJECTS_DETECTED

        # 背景以外のオブジェクトを抽出
        objects = [cls for cls in detected_classes_info if cls["name"] != Messages.BACKGROUND_LABEL]
        background = next((cls for cls in detected_classes_info if cls["name"] == Messages.BACKGROUND_LABEL), None)

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
            summary_parts.append(Messages.NO_SPECIFIC_OBJECTS)

        return "\n".join(summary_parts)

    async def run(self) -> None:
        """サーバーの実行"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=ServerConfig.SERVER_NAME,
                    server_version=ServerConfig.VERSION,
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(), experimental_capabilities={}
                    ),
                ),
            )


async def main() -> None:
    """メイン関数"""
    server = MCPSemanticSegmentationServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
