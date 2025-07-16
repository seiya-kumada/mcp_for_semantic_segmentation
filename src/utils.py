# import os
import sys
import time
from pathlib import Path

import psutil
import torch


def cleanup_old_files(directory: str, days_old: int = 7) -> list[str]:
    """指定した日数より古いファイルを削除

    Args:
        directory: 対象ディレクトリのパス
        days_old: 何日前より古いファイルを削除するか（デフォルト: 7日）

    Returns:
        削除されたファイルパスのリスト
    """
    deleted_files: list[str] = []
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)

    dir_path = Path(directory)
    if not dir_path.is_dir():
        return deleted_files

    for file_path in dir_path.iterdir():
        if file_path.is_file():
            file_mtime = file_path.stat().st_mtime

            if file_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                except OSError as e:
                    print(f"ファイル削除エラー {file_path}: {e}")

    return deleted_files


def validate_image_path(image_path: str) -> bool:
    """画像パスの妥当性を検証

    Args:
        image_path: 検証する画像ファイルのパス

    Returns:
        画像パスが有効な場合True、無効な場合False
    """
    if not image_path:
        return False

    path = Path(image_path)

    # パスの存在とファイル確認を統合
    if not (path.exists() and path.is_file()):
        return False

    # 拡張子の確認
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return path.suffix.lower() in supported_extensions


def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得

    Returns:
        プロジェクトルートディレクトリのPath
    """
    current_file = Path(__file__)
    return current_file.parent.parent


def get_static_directories() -> dict[str, str]:
    """静的ディレクトリのパス情報を取得

    Returns:
        静的ディレクトリのパス情報を含む辞書
        - "static": 静的ファイル用ディレクトリ
        - "input": 入力ファイル用ディレクトリ
        - "output": 出力ファイル用ディレクトリ
    """
    project_root = get_project_root()
    static_directory = project_root / "static"

    return {
        "static": str(static_directory),
        "input": str(static_directory / "input"),
        "output": str(static_directory / "output"),
    }


def format_processing_time(seconds: float) -> str:
    """処理時間を読みやすい形式でフォーマット

    Args:
        seconds: 処理時間（秒単位）

    Returns:
        フォーマットされた処理時間文字列
        - 1秒未満: "XXXms"
        - 1分未満: "X.Xs"
        - 1分以上: "XmX.Xs"
    """
    MILLISECONDS_PER_SECOND = 1000
    SECONDS_PER_MINUTE = 60

    if seconds < 1:
        return f"{seconds * MILLISECONDS_PER_SECOND:.0f}ms"
    elif seconds < SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // SECONDS_PER_MINUTE)
        remaining_seconds = seconds % SECONDS_PER_MINUTE
        return f"{minutes}m {remaining_seconds:.1f}s"


def get_available_memory_gb() -> float:
    """利用可能メモリ容量をGB単位で取得

    Returns:
        利用可能メモリ容量（GB単位）
    """
    BYTES_PER_GB = 1024**3
    memory = psutil.virtual_memory()
    return memory.available / BYTES_PER_GB


def _check_torch_availability() -> tuple[bool, bool]:
    """PyTorchとCUDAの利用可能性をチェック

    Returns:
        (torch_available, cuda_available)のタプル
    """
    try:
        return True, torch.cuda.is_available()
    except Exception:
        return False, False


def _check_static_dirs_writable() -> bool:
    """静的ディレクトリの書き込み権限をチェック

    Returns:
        全ての静的ディレクトリが書き込み可能な場合True
    """
    static_dirs = get_static_directories()
    for dir_path in static_dirs.values():
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError:
                return False
    return True


def check_system_requirements() -> dict[str, str | float | bool]:
    """システム要件のチェック

    Returns:
        システム要件の情報を含む辞書
        - "python_version": Pythonバージョン
        - "available_memory_gb": 利用可能メモリ（GB）
        - "static_dirs_writable": 静的ディレクトリの書き込み権限
        - "torch_available": PyTorchの利用可能性
        - "cuda_available": CUDAの利用可能性
    """
    torch_available, cuda_available = _check_torch_availability()

    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "available_memory_gb": get_available_memory_gb(),
        "static_dirs_writable": _check_static_dirs_writable(),
        "torch_available": torch_available,
        "cuda_available": cuda_available,
    }
