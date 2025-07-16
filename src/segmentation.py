import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50


class SemanticSegmenter:
    # モデル設定の定数
    IMAGE_SIZE = (520, 520)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # ファイル制限の定数
    MAX_FILE_SIZE_MB = 10
    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
    
    # サポートされる画像形式
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    
    # 処理時間の小数点桁数
    PROCESSING_TIME_PRECISION = 2
    
    def __init__(self, device: str = "auto"):
        """SemanticSegmenterの初期化

        Args:
            device: 使用するデバイス ("auto", "cuda", "cpu")
                   "auto"の場合はCUDAが利用可能ならCUDA、そうでなければCPUを使用

        Raises:
            ValueError: 無効なdeviceが指定された場合
            RuntimeError: モデルの読み込みに失敗した場合
        """
        if device not in ["auto", "cuda", "cpu"]:
            raise ValueError(f"Invalid device: {device}. Must be 'auto', 'cuda', or 'cpu'")

        self.device = self._get_device(device)
        self.model: Optional[torch.nn.Module] = None
        self.transform: Optional[transforms.Compose] = None

        try:
            self._load_model()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SemanticSegmenter: {str(e)}")

    def _get_device(self, device: str) -> str:
        """デバイスの選択

        Args:
            device: デバイス指定 ("auto", "cuda", "cpu")

        Returns:
            実際に使用するデバイス名 ("cuda" または "cpu")

        Raises:
            RuntimeError: CUDAが指定されたが利用できない場合
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available")

        return device

    def _load_model(self) -> None:
        """事前学習済みDeepLabV3モデルをロード

        モデルを初期化し指定されたデバイスに配置し評価モードに設定する。
        さらに、データ変換パイプラインを初期化する。

        Raises:
            RuntimeError: モデルの読み込みまたは初期化に失敗した場合
        """
        try:
            self.model = deeplabv3_resnet50(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
            self._initialize_transform()
        except Exception as e:
            raise RuntimeError(f"モデル読み込みエラー: {str(e)}")
    
    def _initialize_transform(self) -> None:
        """データ変換パイプラインの初期化
        
        画像の前処理用の変換パイプラインを設定する。
        リサイズ、テンソル変換、正規化を含む。
        """
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.NORMALIZE_MEAN, std=self.NORMALIZE_STD),
            ]
        )

    def process_image(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """画像のセマンティックセグメンテーションを実行
        
        指定された画像に対してセマンティックセグメンテーションを実行し、
        結果画像とメタデータを保存する。
        
        Args:
            image_path: 入力画像のパス
            output_dir: 結果を保存するディレクトリ
            
        Returns:
            処理結果を含む辞書:
            - status: "success" または "error"
            - output_path: 生成された結果画像のパス（成功時）
            - json_path: 生成されたJSONファイルのパス（成功時）
            - processing_time: 処理時間（秒）（成功時）
            - input_size: 入力画像サイズ（成功時）
            - output_size: 出力画像サイズ（成功時）
            - detected_classes: 検出されたクラス情報（成功時）
            - total_pixels: 総ピクセル数（成功時）
            - error_message: エラーメッセージ（エラー時）
            - error_code: エラーコード（エラー時）
        """
        start_time = time.time()

        try:
            # 入力画像の読み込みと検証
            image, original_size = self._load_and_validate_image(image_path)

            # セグメンテーション実行
            segmentation_mask = self._segment_image(image)

            # 結果画像の生成と保存
            output_path = self._save_result(segmentation_mask, original_size, output_dir)

            # セグメンテーション統計情報の生成
            stats = self._generate_segmentation_stats(segmentation_mask)

            # JSONファイルの保存
            json_path = self._save_segmentation_json(output_path, stats, original_size, output_dir)

            processing_time = time.time() - start_time
            return self._create_success_response(output_path, json_path, processing_time, original_size, stats)

        except Exception as e:
            return self._create_error_response(e)
    
    def _create_success_response(self, output_path: str, json_path: str, processing_time: float, 
                               original_size: Tuple[int, int], stats: Dict[str, Any]) -> Dict[str, Union[str, float, int, list]]:
        """成功時のレスポンスを生成
        
        Args:
            output_path: 生成された結果画像のパス
            json_path: 生成されたJSONファイルのパス
            processing_time: 処理時間（秒）
            original_size: 元画像のサイズ
            stats: セグメンテーション統計情報
            
        Returns:
            成功レスポンス辞書
        """
        return {
            "status": "success",
            "output_path": output_path,
            "json_path": json_path,
            "processing_time": round(processing_time, self.PROCESSING_TIME_PRECISION),
            "input_size": f"{original_size[0]}x{original_size[1]}",
            "output_size": f"{original_size[0]}x{original_size[1]}",
            "detected_classes": stats["detected_classes"],
            "total_pixels": stats["total_pixels"],
        }
    
    def _create_error_response(self, error: Exception) -> Dict[str, str]:
        """エラー時のレスポンスを生成
        
        Args:
            error: 発生した例外
            
        Returns:
            エラーレスポンス辞書
        """
        return {
            "status": "error",
            "error_message": str(error),
            "error_code": self._get_error_code(error)
        }

    def _load_and_validate_image(self, image_path: str) -> Tuple[Image.Image, Tuple[int, int]]:
        """画像の読み込みと検証
        
        指定された画像ファイルを読み込み、ファイルサイズと形式を検証する。
        検証に成功した場合、RGB形式に変換した画像とそのサイズを返す。
        
        Args:
            image_path: 読み込む画像ファイルのパス
            
        Returns:
            読み込んだ画像とサイズのタプル:
            - Image.Image: RGB形式に変換された画像
            - Tuple[int, int]: 元画像のサイズ（幅、高さ）
            
        Raises:
            FileNotFoundError: 指定された画像ファイルが見つからない場合
            ValueError: ファイルサイズが制限を超える、サポートされていない形式、
                       または画像が破損している場合
        """
        image_path_obj = Path(image_path)
        
        # ファイル存在チェック
        if not image_path_obj.exists():
            raise FileNotFoundError("画像ファイルが見つかりません")
        
        # ファイル種別チェック
        if not image_path_obj.is_file():
            raise ValueError("指定されたパスはファイルではありません")

        # ファイルサイズチェック
        file_size = image_path_obj.stat().st_size
        if file_size > self.MAX_FILE_SIZE_BYTES:
            raise ValueError("ファイルサイズが10MBを超えています")

        # 画像形式チェック
        file_ext = image_path_obj.suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError("サポートされていない画像形式です")

        try:
            image = Image.open(image_path_obj).convert("RGB")
            original_size = image.size
            return image, original_size
        except Exception:
            raise ValueError("画像ファイルが破損しているか、読み込めません")

    def _segment_image(self, image: Image.Image) -> np.ndarray:
        """セグメンテーションの実行
        
        PIL画像に対してセマンティックセグメンテーションを実行し、
        各ピクセルのクラス予測を含む配列を返す。
        
        Args:
            image: セグメンテーションを実行するPIL画像
            
        Returns:
            各ピクセルのクラスIDを含むnumpy配列 (H, W)
            
        Raises:
            RuntimeError: モデルまたは変換パイプラインが初期化されていない場合
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        if self.transform is None:
            raise RuntimeError("Transform pipeline is not initialized")
            
        with torch.no_grad():
            # 前処理
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 推論
            output = self.model(input_tensor)["out"]

            # 後処理
            predictions = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

            return predictions

    def _save_result(self, mask: np.ndarray, original_size: Tuple[int, int], output_dir: str) -> str:
        """結果画像の保存"""
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)

        # ファイル名生成
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)

        # マスクを元のサイズにリサイズ
        mask_resized = cv2.resize(mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)

        # カラーマップ適用（可視化用）
        colormap = self._get_colormap()
        colored_mask = colormap[mask_resized]

        # 画像として保存
        result_image = Image.fromarray(colored_mask.astype(np.uint8))
        result_image.save(output_path)

        return output_path

    def _get_colormap(self) -> np.ndarray:
        """PASCAL VOCカラーマップの生成"""
        colormap = np.zeros((256, 3), dtype=np.uint8)

        # 主要クラスの色定義
        colors = self._get_class_colors()

        for i, color_info in enumerate(colors):
            colormap[i] = color_info["color"]

        return colormap

    def _generate_segmentation_stats(self, mask: np.ndarray) -> Dict[str, Any]:
        """セグメンテーション結果の統計情報を生成"""
        unique_classes, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.shape[0] * mask.shape[1]

        class_colors = self._get_class_colors()
        detected_classes = []

        for class_id, pixel_count in zip(unique_classes, counts):
            if class_id < len(class_colors):
                class_info = class_colors[class_id]
                percentage = (pixel_count / total_pixels) * 100

                detected_classes.append(
                    {
                        "id": int(class_id),
                        "name": class_info["name"],
                        "description": class_info["description"],
                        "color": {
                            "rgb": class_info["color"],
                            "hex": "#{:02x}{:02x}{:02x}".format(*class_info["color"]),
                        },
                        "pixel_count": int(pixel_count),
                        "percentage": round(percentage, 2),
                    }
                )

        # 面積の大きい順にソート
        detected_classes.sort(key=lambda x: x["pixel_count"], reverse=True)

        return {
            "detected_classes": detected_classes,
            "total_pixels": total_pixels,
            "num_classes": len(detected_classes),
        }

    def _save_segmentation_json(
        self, output_path: str, stats: Dict[str, Any], original_size: Tuple[int, int], output_dir: str
    ) -> str:
        """セグメンテーション結果をJSONファイルに保存"""
        # JSONファイル名の生成
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        json_filename = f"{base_name}_segmentation.json"
        json_path = os.path.join(output_dir, json_filename)

        # メタデータの追加
        result_data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "DeepLabV3_ResNet50",
                "dataset": "PASCAL VOC",
                "image_size": {"width": original_size[0], "height": original_size[1]},
                "output_image": os.path.basename(output_path),
                "total_pixels": stats["total_pixels"],
                "num_detected_classes": stats["num_classes"],
            },
            "color_mapping": {
                "description": "RGB colors used for each detected class in the segmentation result",
                "classes": [],
            },
            "segmentation_results": {
                "detected_classes": stats["detected_classes"],
                "class_statistics": self._generate_class_statistics(stats["detected_classes"]),
            },
        }

        # 検出されたクラスのみの色情報を追加
        for detected_class in stats["detected_classes"]:
            result_data["color_mapping"]["classes"].append(
                {
                    "id": detected_class["id"],
                    "name": detected_class["name"],
                    "description": detected_class["description"],
                    "color": detected_class["color"],
                }
            )

        # JSONファイルに保存
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        return json_path

    def _generate_class_statistics(self, detected_classes: list) -> Dict[str, Any]:
        """クラス統計情報の生成"""
        if not detected_classes:
            return {}

        percentages = [cls["percentage"] for cls in detected_classes]

        return {
            "dominant_class": detected_classes[0]["name"] if detected_classes else None,
            "dominant_percentage": detected_classes[0]["percentage"] if detected_classes else 0,
            "coverage_summary": {
                "background_percentage": next(
                    (cls["percentage"] for cls in detected_classes if cls["name"] == "background"), 0
                ),
                "foreground_percentage": sum(
                    cls["percentage"] for cls in detected_classes if cls["name"] != "background"
                ),
                "most_diverse": len(detected_classes) > 5,
                "primary_objects": [cls["name"] for cls in detected_classes[:3] if cls["name"] != "background"],
            },
        }

    def _get_class_colors(self) -> list:
        """PASCAL VOCクラスの色とラベル情報"""
        return [
            {"id": 0, "name": "background", "color": [0, 0, 0], "description": "背景"},
            {"id": 1, "name": "aeroplane", "color": [128, 0, 0], "description": "飛行機"},
            {"id": 2, "name": "bicycle", "color": [0, 128, 0], "description": "自転車"},
            {"id": 3, "name": "bird", "color": [128, 128, 0], "description": "鳥"},
            {"id": 4, "name": "boat", "color": [0, 0, 128], "description": "船"},
            {"id": 5, "name": "bottle", "color": [128, 0, 128], "description": "ボトル"},
            {"id": 6, "name": "bus", "color": [0, 128, 128], "description": "バス"},
            {"id": 7, "name": "car", "color": [128, 128, 128], "description": "車"},
            {"id": 8, "name": "cat", "color": [64, 0, 0], "description": "猫"},
            {"id": 9, "name": "chair", "color": [192, 0, 0], "description": "椅子"},
            {"id": 10, "name": "cow", "color": [64, 128, 0], "description": "牛"},
            {"id": 11, "name": "diningtable", "color": [192, 128, 0], "description": "ダイニングテーブル"},
            {"id": 12, "name": "dog", "color": [64, 0, 128], "description": "犬"},
            {"id": 13, "name": "horse", "color": [192, 0, 128], "description": "馬"},
            {"id": 14, "name": "motorbike", "color": [64, 128, 128], "description": "バイク"},
            {"id": 15, "name": "person", "color": [192, 128, 128], "description": "人"},
            {"id": 16, "name": "pottedplant", "color": [0, 64, 0], "description": "植物"},
            {"id": 17, "name": "sheep", "color": [128, 64, 0], "description": "羊"},
            {"id": 18, "name": "sofa", "color": [0, 192, 0], "description": "ソファ"},
            {"id": 19, "name": "train", "color": [128, 192, 0], "description": "電車"},
            {"id": 20, "name": "tvmonitor", "color": [0, 64, 128], "description": "テレビ・モニター"},
        ]

    def _get_error_code(self, error: Exception) -> str:
        """エラーコードの決定"""
        if isinstance(error, FileNotFoundError):
            return "FILE_NOT_FOUND"
        elif isinstance(error, ValueError):
            if "ファイルサイズ" in str(error):
                return "FILE_TOO_LARGE"
            elif "サポートされていない" in str(error):
                return "UNSUPPORTED_FORMAT"
            else:
                return "PROCESSING_ERROR"
        elif isinstance(error, RuntimeError):
            return "MODEL_ERROR"
        else:
            return "PROCESSING_ERROR"
