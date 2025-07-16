import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils import (
    _check_static_dirs_writable,
    _check_torch_availability,
    check_system_requirements,
    cleanup_old_files,
    format_processing_time,
    get_available_memory_gb,
    get_project_root,
    get_static_directories,
    validate_image_path,
)

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCleanupOldFiles:
    """cleanup_old_files関数のテスト"""

    def test_cleanup_old_files_normal(self) -> None:
        """通常の古いファイル削除テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # テスト用ファイルを作成
            old_file1 = temp_path / "old_file1.txt"
            old_file2 = temp_path / "old_file2.txt"
            new_file = temp_path / "new_file.txt"

            old_file1.write_text("old content 1")
            old_file2.write_text("old content 2")
            new_file.write_text("new content")

            # 古いファイルの更新時刻を8日前に設定
            old_timestamp = time.time() - (8 * 24 * 60 * 60)
            os.utime(old_file1, (old_timestamp, old_timestamp))
            os.utime(old_file2, (old_timestamp, old_timestamp))

            # テスト実行
            deleted_files = cleanup_old_files(str(temp_path), days_old=7)

            # 結果検証
            assert len(deleted_files) == 2
            assert str(old_file1) in deleted_files
            assert str(old_file2) in deleted_files
            assert not old_file1.exists()
            assert not old_file2.exists()
            assert new_file.exists()

    def test_cleanup_nonexistent_directory(self) -> None:
        """存在しないディレクトリのテスト"""
        result = cleanup_old_files("/nonexistent/directory")
        assert result == []

    def test_cleanup_file_path_instead_of_directory(self) -> None:
        """ファイルパスを指定した場合のテスト"""
        with tempfile.NamedTemporaryFile() as temp_file:
            result = cleanup_old_files(temp_file.name)
            assert result == []

    def test_cleanup_empty_directory(self) -> None:
        """空のディレクトリのテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = cleanup_old_files(temp_dir)
            assert result == []


class TestValidateImagePath:
    """validate_image_path関数のテスト"""

    def test_validate_empty_path(self) -> None:
        """空のパスのテスト"""
        assert not validate_image_path("")
        assert not validate_image_path(None)  # type: ignore

    def test_validate_nonexistent_path(self) -> None:
        """存在しないパスのテスト"""
        assert not validate_image_path("/nonexistent/image.jpg")

    def test_validate_directory_path(self) -> None:
        """ディレクトリパスのテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            assert not validate_image_path(temp_dir)


class TestGetProjectRoot:
    """get_project_root関数のテスト"""

    def test_get_project_root(self) -> None:
        """プロジェクトルート取得のテスト"""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()


class TestGetStaticDirectories:
    """get_static_directories関数のテスト"""

    def test_get_static_directories(self) -> None:
        """静的ディレクトリ取得のテスト"""
        dirs = get_static_directories()

        assert isinstance(dirs, dict)
        assert "static" in dirs
        assert "input" in dirs
        assert "output" in dirs

        for path in dirs.values():
            assert isinstance(path, str)


class TestFormatProcessingTime:
    """format_processing_time関数のテスト"""

    def test_format_milliseconds(self) -> None:
        """ミリ秒のフォーマットテスト"""
        assert format_processing_time(0.5) == "500ms"
        assert format_processing_time(0.123) == "123ms"

    def test_format_seconds(self) -> None:
        """秒のフォーマットテスト"""
        assert format_processing_time(1.5) == "1.5s"
        assert format_processing_time(30.7) == "30.7s"

    def test_format_minutes(self) -> None:
        """分のフォーマットテスト"""
        assert format_processing_time(65) == "1m 5.0s"
        assert format_processing_time(125.3) == "2m 5.3s"


class TestGetAvailableMemoryGb:
    """get_available_memory_gb関数のテスト"""

    def test_get_available_memory_gb_with_psutil(self) -> None:
        """psutil利用可能時のテスト"""
        with patch("src.utils.psutil") as mock_psutil:
            mock_memory = MagicMock()
            mock_memory.available = 8 * (1024**3)  # 8GB
            mock_psutil.virtual_memory.return_value = mock_memory

            memory_gb = get_available_memory_gb()
            assert memory_gb == 8.0


class TestCheckTorchAvailability:
    """_check_torch_availability関数のテスト"""

    def test_torch_available_with_cuda(self) -> None:
        """PyTorchとCUDAが利用可能な場合のテスト"""
        with patch("src.utils.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True

            torch_available, cuda_available = _check_torch_availability()

            assert torch_available is True
            assert cuda_available is True

    def test_torch_available_without_cuda(self) -> None:
        """PyTorchは利用可能だがCUDAが利用不可な場合のテスト"""
        with patch("src.utils.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            torch_available, cuda_available = _check_torch_availability()

            assert torch_available is True
            assert cuda_available is False


class TestCheckStaticDirsWritable:
    """_check_static_dirs_writable関数のテスト"""

    @patch("src.utils.get_static_directories")
    @patch("pathlib.Path.exists")
    def test_all_dirs_exist(self, mock_exists, mock_get_static_dirs) -> None:
        """全ディレクトリが存在する場合のテスト"""
        mock_get_static_dirs.return_value = {"static": "/tmp/static", "input": "/tmp/input", "output": "/tmp/output"}
        mock_exists.return_value = True

        result = _check_static_dirs_writable()
        assert result is True

    @patch("src.utils.get_static_directories")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_create_missing_dirs_success(self, mock_mkdir, mock_exists, mock_get_static_dirs) -> None:
        """存在しないディレクトリの作成が成功する場合のテスト"""
        mock_get_static_dirs.return_value = {"static": "/tmp/static"}
        mock_exists.return_value = False
        mock_mkdir.return_value = None

        result = _check_static_dirs_writable()
        assert result is True
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("src.utils.get_static_directories")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_create_missing_dirs_failure(self, mock_mkdir, mock_exists, mock_get_static_dirs) -> None:
        """存在しないディレクトリの作成が失敗する場合のテスト"""
        mock_get_static_dirs.return_value = {"static": "/tmp/static"}
        mock_exists.return_value = False
        mock_mkdir.side_effect = OSError("Permission denied")

        result = _check_static_dirs_writable()
        assert result is False


class TestCheckSystemRequirements:
    """check_system_requirements関数のテスト"""

    @patch("src.utils._check_torch_availability")
    @patch("src.utils._check_static_dirs_writable")
    @patch("src.utils.get_available_memory_gb")
    def test_check_system_requirements(self, mock_get_memory, mock_check_dirs, mock_check_torch) -> None:
        """システム要件チェックのテスト"""
        mock_get_memory.return_value = 8.0
        mock_check_dirs.return_value = True
        mock_check_torch.return_value = (True, False)

        requirements = check_system_requirements()

        assert "python_version" in requirements
        assert "available_memory_gb" in requirements
        assert "static_dirs_writable" in requirements
        assert "torch_available" in requirements
        assert "cuda_available" in requirements

        assert requirements["available_memory_gb"] == 8.0
        assert requirements["static_dirs_writable"] is True
        assert requirements["torch_available"] is True
        assert requirements["cuda_available"] is False
