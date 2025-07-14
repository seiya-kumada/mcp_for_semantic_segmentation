import os
import time
from pathlib import Path
from typing import List


def cleanup_old_files(directory: str, days_old: int = 7) -> List[str]:
    """指定した日数より古いファイルを削除"""
    deleted_files = []
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)
    
    if not os.path.exists(directory):
        return deleted_files
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_mtime = os.path.getmtime(file_path)
            
            if file_mtime < cutoff_time:
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                except OSError as e:
                    print(f"ファイル削除エラー {file_path}: {e}")
    
    return deleted_files


def validate_image_path(image_path: str) -> bool:
    """画像パスの妥当性を検証"""
    if not image_path:
        return False
    
    # パスの存在確認
    if not os.path.exists(image_path):
        return False
    
    # ファイルかどうかの確認
    if not os.path.isfile(image_path):
        return False
    
    # 拡張子の確認
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_extension = Path(image_path).suffix.lower()
    
    return file_extension in supported_extensions


def get_file_size_mb(file_path: str) -> float:
    """ファイルサイズをMB単位で取得"""
    if not os.path.exists(file_path):
        return 0.0
    
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def ensure_directory_exists(directory: str) -> bool:
    """ディレクトリの存在確認と作成"""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except OSError:
        return False


def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得"""
    current_file = Path(__file__)
    return current_file.parent.parent


def get_static_directories() -> dict:
    """静的ディレクトリのパス情報を取得"""
    project_root = get_project_root()
    static_dir = project_root / "static"
    
    return {
        "static": str(static_dir),
        "input": str(static_dir / "input"),
        "output": str(static_dir / "output")
    }


def format_processing_time(seconds: float) -> str:
    """処理時間を読みやすい形式でフォーマット"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def create_response_success(output_path: str, processing_time: float, 
                          input_size: str, output_size: str) -> dict:
    """成功レスポンスの生成"""
    file_url = f"file:///{output_path.replace(os.sep, '/')}"
    
    return {
        "status": "success",
        "output_path": output_path,
        "output_url": file_url,
        "processing_time": processing_time,
        "input_size": input_size,
        "output_size": output_size,
        "message": "セグメンテーションが完了しました"
    }


def create_response_error(error_message: str, error_code: str) -> dict:
    """エラーレスポンスの生成"""
    return {
        "status": "error",
        "error_message": error_message,
        "error_code": error_code
    }


def log_processing_info(image_path: str, processing_time: float, 
                       output_path: str, success: bool = True):
    """処理情報のログ出力"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    status = "SUCCESS" if success else "ERROR"
    
    log_message = (
        f"[{timestamp}] {status} - "
        f"Input: {image_path}, "
        f"Output: {output_path}, "
        f"Time: {format_processing_time(processing_time)}"
    )
    
    print(log_message)


def get_available_memory_gb() -> float:
    """利用可能メモリ容量をGB単位で取得"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.available / (1024 ** 3)
    except ImportError:
        return 0.0


def check_system_requirements() -> dict:
    """システム要件のチェック"""
    requirements = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "available_memory_gb": get_available_memory_gb(),
        "static_dirs_writable": True,
        "torch_available": False,
        "cuda_available": False
    }
    
    # PyTorchの確認
    try:
        import torch
        requirements["torch_available"] = True
        requirements["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    
    # 書き込み権限の確認
    static_dirs = get_static_directories()
    for dir_path in static_dirs.values():
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except OSError:
                requirements["static_dirs_writable"] = False
                break
    
    return requirements


import sys