import asyncio
import json
import logging
import os
import shutil
import textwrap
import time
from pathlib import Path
from typing import Any, Optional

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from typing_extensions import TypedDict

from .segmentation import SemanticSegmenter


class ColorInfo(TypedDict):
    """
    色情報の型定義

    セマンティックセグメンテーションで検出された各クラスの表示色を定義する。
    視覚的な区別のため、RGB値と16進数表記の両方を含む。

    Attributes:
        rgb (list[int]): RGB値のリスト [R, G, B]。各値は0-255の範囲
        hex (str): 16進数表記の色コード（例: "#FF0000"）
    """

    rgb: list[int]
    hex: str


class DetectedClass(TypedDict):
    """
    検出されたクラス情報の型定義
    
    セマンティックセグメンテーションで識別された各オブジェクトクラスの情報を格納する。
    画像内の各領域がどのクラスに分類されたかと、その詳細情報を含む。
    
    Attributes:
        name (str): クラスの識別名（例: "person", "car", "background"）
        description (str): クラスの説明文（例: "人", "車", "背景"）
        percentage (float): 画像全体に占める割合（0.0-100.0の範囲）
        color (ColorInfo): 表示用の色情報（RGB値と16進数表記）
    """

    name: str
    description: str
    percentage: float
    color: ColorInfo


class SegmentationResult(TypedDict):
    """
    セグメンテーション結果の型定義
    
    セマンティックセグメンテーション処理の実行結果を格納する。
    成功時と失敗時の両方の情報を含む包括的な結果構造体。
    
    Attributes:
        status (str): 処理の状態（"success" または "error"）
        output_path (str): 出力画像ファイルのパス
        json_path (str): 詳細結果JSONファイルのパス
        processing_time (float): 処理時間（秒）
        input_size (str): 入力画像のサイズ（例: "640x480"）
        output_size (str): 出力画像のサイズ（例: "640x480"）
        detected_classes (list[DetectedClass]): 検出されたクラスのリスト
        total_pixels (int): 画像の総ピクセル数
        error_message (Optional[str]): エラー時のメッセージ（成功時はNone）
        error_code (Optional[str]): エラー時のコード（成功時はNone）
    """

    status: str
    output_path: str
    json_path: str
    processing_time: float
    input_size: str
    output_size: str
    detected_classes: list[DetectedClass]
    total_pixels: int
    error_message: Optional[str]
    error_code: Optional[str]


class ErrorCodes:
    """
    エラーコード定数
    
    アプリケーション全体で使用する標準化されたエラーコードを定義する。
    統一されたエラーハンドリングとログ出力のための定数集。
    
    Constants:
        VALIDATION_ERROR (str): 入力値の検証エラー（例：必須パラメータ不足）
        PROCESSING_ERROR (str): 処理実行時のエラー（例：セグメンテーション失敗）
        TOOL_ERROR (str): ツール呼び出しエラー（例：未知のツール名）
        FILE_ERROR (str): ファイル操作エラー（例：ファイル読み込み失敗）
    """

    VALIDATION_ERROR = "VALIDATION_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    FILE_ERROR = "FILE_ERROR"


class ServerConfig:
    """
    サーバー設定定数
    
    MCPセマンティックセグメンテーションサーバーの設定値を一元管理する。
    ハードコードされた値を集約し、設定変更時の保守性を向上させる。
    
    Constants:
        SERVER_NAME (str): MCPサーバーの識別名
        VERSION (str): アプリケーションのバージョン
        STATIC_DIR (str): 静的ファイルの保存ディレクトリ名
        INPUT_DIR (str): 入力ファイルの保存ディレクトリ名
        OUTPUT_DIR (str): 出力ファイルの保存ディレクトリ名
        TOOL_NAME (str): セグメンテーションツールの名前
        IMAGE_PATH_PARAM (str): 画像パスパラメータの名前
        TIMESTAMP_FORMAT (str): ファイル名用タイムスタンプの形式
        LOG_FORMAT (str): ログ出力の形式
    """

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
    """
    メッセージ定数
    
    アプリケーション内で使用される全てのメッセージを一元管理する。
    UI表示、エラーメッセージ、ログ出力などの文字列を集約し、国際化対応や
    メッセージ変更時の保守性を向上させる。
    
    Constants:
        SEGMENTATION_COMPLETE (str): セグメンテーション完了メッセージ
        NO_OBJECTS_DETECTED (str): オブジェクト未検出時のメッセージ
        IMAGE_PATH_REQUIRED (str): 画像パス必須エラーメッセージ
        UNKNOWN_TOOL (str): 未知のツール名エラーメッセージ（フォーマット用）
        BACKGROUND_LABEL (str): 背景クラスのラベル名
        NO_SPECIFIC_OBJECTS (str): 具体的オブジェクト未検出時のメッセージ
        RESULT_TEMPLATE (str): 結果表示用のテンプレート文字列
    """

    SEGMENTATION_COMPLETE = "セグメンテーションが完了しました"
    NO_OBJECTS_DETECTED = "オブジェクトが検出されませんでした。"
    IMAGE_PATH_REQUIRED = "image_path is required"
    UNKNOWN_TOOL = "Unknown tool: {}"
    BACKGROUND_LABEL = "background"
    NO_SPECIFIC_OBJECTS = "具体的なオブジェクトは検出されませんでした。"

    # 表示テンプレート
    RESULT_TEMPLATE = textwrap.dedent("""
        セマンティックセグメンテーション結果

        処理結果:
        • 処理時間: {}秒
        • 画像サイズ: {}
        • 総ピクセル数: {:,}

        {}

        ファイル:
        • 結果画像: {}
        • 詳細JSON: {}

        検出されたクラス詳細:
        """).strip()


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
        logging.basicConfig(level=logging.INFO, format=ServerConfig.LOG_FORMAT)
        self.logger = logging.getLogger(__name__)

    def _handle_error(self, error: Exception, context: str, error_code: Optional[str] = None) -> dict[str, Any]:
        """統一エラーハンドリング"""
        error_code = error_code or ErrorCodes.PROCESSING_ERROR
        error_message = str(error)

        # ログに記録
        self.logger.error(f"Error in {context}: {error_message}", exc_info=True)

        # エラーレスポンス構築
        return {"error": {"message": error_message, "code": error_code, "context": context}}

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
                        "properties": {
                            ServerConfig.IMAGE_PATH_PARAM: {"type": "string", "description": "入力画像のパス"}
                        },
                        "required": [ServerConfig.IMAGE_PATH_PARAM],
                    },
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """ツールの実行"""
            try:
                if name == ServerConfig.TOOL_NAME:
                    return await self._handle_segmentation(arguments)
                else:
                    error_response = self._handle_error(
                        ValueError(Messages.UNKNOWN_TOOL.format(name)), "call_tool", ErrorCodes.TOOL_ERROR
                    )
                    return [TextContent(type="text", text=json.dumps(error_response, ensure_ascii=False, indent=2))]
            except Exception as e:
                error_response = self._handle_error(e, "call_tool")
                return [TextContent(type="text", text=json.dumps(error_response, ensure_ascii=False, indent=2))]

    def _validate_arguments(self, arguments: dict[str, Any]) -> str:
        """引数の検証"""
        image_path = arguments.get(ServerConfig.IMAGE_PATH_PARAM)
        if not image_path:
            raise ValueError(Messages.IMAGE_PATH_REQUIRED)
        return image_path

    def _build_success_response(self, result: SegmentationResult) -> dict[str, Any]:
        """成功時のレスポンス構築"""
        file_url = f"file:///{result['output_path'].replace(os.sep, '/')}"
        json_url = f"file:///{result['json_path'].replace(os.sep, '/')}"

        detected_classes_info: list[DetectedClass] = []
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

    def _build_error_response(self, result: SegmentationResult) -> dict[str, Any]:
        """エラー時のレスポンス構築"""
        return {"error": {"message": result["error_message"], "code": result["error_code"]}}

    def _format_display_text(self, result: SegmentationResult, response: dict[str, Any]) -> str:
        """表示用テキストの生成"""
        detected_classes_info: list[DetectedClass] = response["result"]["detected_classes"]

        display_text = Messages.RESULT_TEMPLATE.format(
            result["processing_time"],
            result["input_size"],
            result["total_pixels"],
            response["result"]["summary"],
            response["result"]["output_url"],
            response["result"]["json_url"],
        )

        for cls in detected_classes_info:
            display_text += (
                f"\n• {cls['name']} ({cls['description']}): {cls['percentage']}% - 色: {cls['color']['hex']}"
            )

        return display_text

    async def _handle_segmentation(self, arguments: dict[str, Any]) -> list[TextContent]:
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

    def _generate_summary(self, detected_classes_info: list[DetectedClass]) -> str:
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
