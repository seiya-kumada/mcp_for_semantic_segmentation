import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
import cv2


class SemanticSegmenter:
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.model = None
        self.transform = None
        self._load_model()
    
    def _get_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """事前学習済みDeepLabV3モデルをロード"""
        try:
            self.model = deeplabv3_resnet50(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((520, 520)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        except Exception as e:
            raise RuntimeError(f"モデル読み込みエラー: {str(e)}")
    
    def process_image(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """画像のセマンティックセグメンテーションを実行"""
        start_time = time.time()
        
        try:
            # 入力画像の読み込みと検証
            image, original_size = self._load_and_validate_image(image_path)
            
            # セグメンテーション実行
            segmentation_mask = self._segment_image(image)
            
            # 結果画像の生成と保存
            output_path = self._save_result(
                segmentation_mask, 
                original_size, 
                output_dir
            )
            
            # セグメンテーション統計情報の生成
            stats = self._generate_segmentation_stats(segmentation_mask)
            
            # JSONファイルの保存
            json_path = self._save_segmentation_json(
                output_path, 
                stats, 
                original_size,
                output_dir
            )
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "output_path": output_path,
                "json_path": json_path,
                "processing_time": round(processing_time, 2),
                "input_size": f"{original_size[0]}x{original_size[1]}",
                "output_size": f"{original_size[0]}x{original_size[1]}",
                "detected_classes": stats["detected_classes"],
                "total_pixels": stats["total_pixels"]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "error_code": self._get_error_code(e)
            }
    
    def _load_and_validate_image(self, image_path: str) -> Tuple[Image.Image, Tuple[int, int]]:
        """画像の読み込みと検証"""
        if not os.path.exists(image_path):
            raise FileNotFoundError("画像ファイルが見つかりません")
        
        # ファイルサイズチェック（10MB制限）
        file_size = os.path.getsize(image_path)
        if file_size > 10 * 1024 * 1024:
            raise ValueError("ファイルサイズが10MBを超えています")
        
        # 画像形式チェック
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        file_ext = Path(image_path).suffix.lower()
        if file_ext not in supported_formats:
            raise ValueError("サポートされていない画像形式です")
        
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            return image, original_size
        except Exception:
            raise ValueError("画像ファイルが破損しているか、読み込めません")
    
    def _segment_image(self, image: Image.Image) -> np.ndarray:
        """セグメンテーションの実行"""
        with torch.no_grad():
            # 前処理
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 推論
            output = self.model(input_tensor)['out']
            
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
        mask_resized = cv2.resize(
            mask.astype(np.uint8), 
            original_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
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
                
                detected_classes.append({
                    "id": int(class_id),
                    "name": class_info["name"],
                    "description": class_info["description"],
                    "color": {
                        "rgb": class_info["color"],
                        "hex": "#{:02x}{:02x}{:02x}".format(*class_info["color"])
                    },
                    "pixel_count": int(pixel_count),
                    "percentage": round(percentage, 2)
                })
        
        # 面積の大きい順にソート
        detected_classes.sort(key=lambda x: x["pixel_count"], reverse=True)
        
        return {
            "detected_classes": detected_classes,
            "total_pixels": total_pixels,
            "num_classes": len(detected_classes)
        }
    
    def _save_segmentation_json(self, output_path: str, stats: Dict[str, Any], 
                               original_size: Tuple[int, int], output_dir: str) -> str:
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
                "image_size": {
                    "width": original_size[0],
                    "height": original_size[1]
                },
                "output_image": os.path.basename(output_path),
                "total_pixels": stats["total_pixels"],
                "num_detected_classes": stats["num_classes"]
            },
            "color_mapping": {
                "description": "RGB colors used for each detected class in the segmentation result",
                "classes": []
            },
            "segmentation_results": {
                "detected_classes": stats["detected_classes"],
                "class_statistics": self._generate_class_statistics(stats["detected_classes"])
            }
        }
        
        # 検出されたクラスのみの色情報を追加
        for detected_class in stats["detected_classes"]:
            result_data["color_mapping"]["classes"].append({
                "id": detected_class["id"],
                "name": detected_class["name"],
                "description": detected_class["description"],
                "color": detected_class["color"]
            })
        
        # JSONファイルに保存
        with open(json_path, 'w', encoding='utf-8') as f:
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
                "background_percentage": next((cls["percentage"] for cls in detected_classes if cls["name"] == "background"), 0),
                "foreground_percentage": sum(cls["percentage"] for cls in detected_classes if cls["name"] != "background"),
                "most_diverse": len(detected_classes) > 5,
                "primary_objects": [cls["name"] for cls in detected_classes[:3] if cls["name"] != "background"]
            }
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
            {"id": 20, "name": "tvmonitor", "color": [0, 64, 128], "description": "テレビ・モニター"}
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