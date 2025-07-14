#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.segmentation import SemanticSegmenter
from src.utils import get_static_directories, validate_image_path


def test_natural_image():
    """自然画像でのセグメンテーションテスト"""
    
    # テスト画像のパス
    test_image_path = "./images/test.png"
    
    print("=" * 60)
    print("NATURAL IMAGE SEGMENTATION TEST")
    print("=" * 60)
    
    # 1. 画像の存在確認
    if not os.path.exists(test_image_path):
        print(f"[ERROR] Test image not found: {test_image_path}")
        print("Please ensure the image exists at ./images/test.png")
        return False
    
    # 2. 画像の妥当性確認
    if not validate_image_path(test_image_path):
        print(f"[ERROR] Invalid image format: {test_image_path}")
        return False
    
    # 3. 画像情報の表示
    try:
        from PIL import Image
        with Image.open(test_image_path) as img:
            width, height = img.size
            mode = img.mode
            format_name = img.format
            print(f"[SUCCESS] Image loaded successfully")
            print(f"   Path: {test_image_path}")
            print(f"   Size: {width}x{height}")
            print(f"   Mode: {mode}")
            print(f"   Format: {format_name}")
            
            # ファイルサイズ
            file_size = os.path.getsize(test_image_path)
            print(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.1f}MB)")
            
    except Exception as e:
        print(f"[ERROR] Error loading image: {e}")
        return False
    
    # 4. セグメンテーション実行
    print("\n" + "-" * 60)
    print("SEGMENTATION PROCESSING")
    print("-" * 60)
    
    try:
        # セグメンターの初期化
        print("[INFO] Initializing segmenter...")
        start_time = time.time()
        segmenter = SemanticSegmenter()
        init_time = time.time() - start_time
        print(f"[SUCCESS] Segmenter initialized in {init_time:.1f}s")
        
        # 出力ディレクトリの取得
        static_dirs = get_static_directories()
        output_dir = static_dirs["output"]
        
        # セグメンテーション実行
        print("[INFO] Processing image...")
        result = segmenter.process_image(test_image_path, output_dir)
        
        # 結果の表示
        if result["status"] == "success":
            print(f"[SUCCESS] Segmentation completed successfully!")
            print(f"   Processing time: {result['processing_time']}s")
            print(f"   Input size: {result['input_size']}")
            print(f"   Output size: {result['output_size']}")
            print(f"   Output file: {result['output_path']}")
            print(f"   JSON file: {result['json_path']}")
            print(f"   Detected classes: {result['detected_classes']}")
            print(f"   Total pixels: {result['total_pixels']:,}")
            
            # 出力ファイルの存在確認
            if os.path.exists(result["output_path"]):
                output_size = os.path.getsize(result["output_path"])
                print(f"   Output file size: {output_size:,} bytes ({output_size/1024/1024:.1f}MB)")
                
                # 結果画像の情報表示
                try:
                    with Image.open(result["output_path"]) as result_img:
                        print(f"   Result image mode: {result_img.mode}")
                        print(f"   Result image size: {result_img.size}")
                except Exception as e:
                    print(f"   [WARNING] Could not analyze result image: {e}")
                
                # JSONファイルの存在確認と内容表示
                if os.path.exists(result["json_path"]):
                    json_size = os.path.getsize(result["json_path"])
                    print(f"   JSON file size: {json_size:,} bytes")
                    
                    # JSON内容の一部を表示
                    try:
                        import json
                        with open(result["json_path"], 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        
                        print(f"   JSON content preview:")
                        print(f"     Detected {len(json_data['segmentation_results']['detected_classes'])} classes")
                        
                        # 上位3クラスの表示
                        top_classes = json_data['segmentation_results']['detected_classes'][:3]
                        for i, cls in enumerate(top_classes):
                            print(f"     {i+1}. {cls['name']} ({cls['description']}): {cls['percentage']}%")
                            print(f"        Color: RGB{cls['color']['rgb']} / {cls['color']['hex']}")
                        
                    except Exception as e:
                        print(f"   [WARNING] Could not read JSON file: {e}")
                
                return True
            else:
                print(f"[ERROR] Output file not found: {result['output_path']}")
                return False
                
        else:
            print(f"[ERROR] Segmentation failed:")
            print(f"   Error: {result['error_message']}")
            print(f"   Code: {result['error_code']}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Unexpected error during segmentation: {e}")
        return False


def display_result_comparison():
    """結果の比較表示"""
    print("\n" + "=" * 60)
    print("RESULT COMPARISON")
    print("=" * 60)
    
    test_image_path = "./images/test.png"
    static_dirs = get_static_directories()
    output_dir = static_dirs["output"]
    
    # 最新の結果ファイルを検索
    if os.path.exists(output_dir):
        result_files = [f for f in os.listdir(output_dir) if f.startswith("result_") and f.endswith(".png")]
        if result_files:
            # 最新のファイルを取得
            latest_result = max(result_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
            latest_result_path = os.path.join(output_dir, latest_result)
            
            print(f"Input image:  {test_image_path}")
            print(f"Output image: {latest_result_path}")
            print(f"\nVisual comparison:")
            print(f"   1. Open both images in an image viewer")
            print(f"   2. Compare the original image with the segmentation result")
            print(f"   3. Check if different objects/regions are properly segmented")
            
            # セグメンテーションクラスの説明
            print(f"\nSegmentation classes (PASCAL VOC):")
            classes = [
                "0: background (black)",
                "1: aeroplane (dark red)",
                "2: bicycle (green)", 
                "3: bird (yellow)",
                "4: boat (blue)",
                "5: bottle (purple)",
                "6: bus (cyan)",
                "7: car (light gray)",
                "8: cat (dark red)",
                "9: chair (red)",
                "10: cow (yellow-green)",
                "11: dining table (yellow-red)",
                "12: dog (blue-purple)",
                "13: horse (purple-red)",
                "14: motorbike (cyan-purple)",
                "15: person (light purple)",
                "16: potted plant (dark green)",
                "17: sheep (brown)",
                "18: sofa (light green)",
                "19: train (light yellow)",
                "20: tv/monitor (blue-cyan)"
            ]
            
            for cls in classes[:10]:  # 最初の10クラスだけ表示
                print(f"   {cls}")
            print("   ... (and 10 more classes)")
            
        else:
            print("No result files found in output directory")
    else:
        print("Output directory not found")


def main():
    """メイン関数"""
    success = test_natural_image()
    
    if success:
        display_result_comparison()
        print(f"\n[SUCCESS] Test completed successfully!")
        print(f"Next steps:")
        print(f"   1. View the output image to verify segmentation quality")
        print(f"   2. Test with Claude Desktop using the setup guide")
        print(f"   3. Try different images to test various scenarios")
    else:
        print(f"\n[ERROR] Test failed. Please check the error messages above.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()