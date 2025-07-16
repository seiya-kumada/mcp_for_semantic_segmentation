import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segmentation import SemanticSegmenter


class TestSemanticSegmenterInit:
    """__init__メソッドのテスト"""

    def test_init_with_auto_device_cuda_available(self) -> None:
        """auto指定でCUDAが利用可能な場合のテスト"""
        with patch("torch.cuda.is_available", return_value=True):
            with patch.object(SemanticSegmenter, "_load_model"):
                segmenter = SemanticSegmenter(device="auto")
                assert segmenter.device == "cuda"

    def test_init_with_auto_device_cuda_not_available(self) -> None:
        """auto指定でCUDAが利用不可な場合のテスト"""
        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(SemanticSegmenter, "_load_model"):
                segmenter = SemanticSegmenter(device="auto")
                assert segmenter.device == "cpu"

    def test_init_with_cpu_device(self) -> None:
        """CPU明示指定のテスト"""
        with patch.object(SemanticSegmenter, "_load_model"):
            segmenter = SemanticSegmenter(device="cpu")
            assert segmenter.device == "cpu"

    def test_init_with_invalid_device(self) -> None:
        """無効なデバイス指定のテスト"""
        with pytest.raises(ValueError) as exc_info:
            SemanticSegmenter(device="invalid")
        assert "Invalid device: invalid" in str(exc_info.value)

    def test_init_model_load_failure(self) -> None:
        """モデル読み込み失敗時のテスト"""
        with patch.object(SemanticSegmenter, "_load_model", side_effect=Exception("Model load error")):
            with pytest.raises(RuntimeError) as exc_info:
                SemanticSegmenter()
            assert "Failed to initialize SemanticSegmenter" in str(exc_info.value)


class TestGetDevice:
    """_get_deviceメソッドのテスト"""

    def test_get_device_auto_cuda_available(self) -> None:
        """auto指定でCUDAが利用可能な場合のテスト"""
        with patch("torch.cuda.is_available", return_value=True):
            with patch.object(SemanticSegmenter, "_load_model"):
                segmenter = SemanticSegmenter()
                device = segmenter._get_device("auto")
                assert device == "cuda"

    def test_get_device_auto_cuda_not_available(self) -> None:
        """auto指定でCUDAが利用不可な場合のテスト"""
        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(SemanticSegmenter, "_load_model"):
                segmenter = SemanticSegmenter()
                device = segmenter._get_device("auto")
                assert device == "cpu"

    def test_get_device_cuda_not_available(self) -> None:
        """CUDA指定だが利用不可な場合のテスト"""
        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(SemanticSegmenter, "_load_model"):
                segmenter = SemanticSegmenter(device="cpu")
                with pytest.raises(RuntimeError) as exc_info:
                    segmenter._get_device("cuda")
                assert "CUDA device requested but CUDA is not available" in str(exc_info.value)

    def test_get_device_cpu(self) -> None:
        """CPU指定のテスト"""
        with patch.object(SemanticSegmenter, "_load_model"):
            segmenter = SemanticSegmenter(device="cpu")
            device = segmenter._get_device("cpu")
            assert device == "cpu"


class TestLoadModel:
    """_load_modelメソッドのテスト"""

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_load_model_success(self, mock_deeplabv3):
        """モデル読み込み成功のテスト"""
        # モックモデルの設定
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        with patch.object(SemanticSegmenter, "_initialize_transform"):
            segmenter = SemanticSegmenter(device="cpu")

            # モデルが正しく初期化されたか確認
            mock_deeplabv3.assert_called_once_with(pretrained=True)
            mock_model.to.assert_called_once_with("cpu")
            mock_model.eval.assert_called_once()
            assert segmenter.model == mock_model

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_load_model_failure(self, mock_deeplabv3):
        """モデル読み込み失敗のテスト"""
        mock_deeplabv3.side_effect = Exception("Model loading failed")

        with pytest.raises(RuntimeError) as exc_info:
            SemanticSegmenter(device="cpu")
        assert "モデル読み込みエラー" in str(exc_info.value)

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_load_model_calls_initialize_transform(self, mock_deeplabv3):
        """_initialize_transformが呼ばれることを確認"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        with patch.object(SemanticSegmenter, "_initialize_transform") as mock_init_transform:
            segmenter = SemanticSegmenter(device="cpu")
            mock_init_transform.assert_called_once()


class TestInitializeTransform:
    """_initialize_transformメソッドのテスト"""

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_initialize_transform(self, mock_deeplabv3):
        """transform初期化のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        # transformが正しく設定されたか確認
        assert segmenter.transform is not None
        assert hasattr(segmenter.transform, "transforms")
        assert len(segmenter.transform.transforms) == 3

        # 各transformの型を確認
        transforms = segmenter.transform.transforms
        assert transforms[0].__class__.__name__ == "Resize"
        assert transforms[1].__class__.__name__ == "ToTensor"
        assert transforms[2].__class__.__name__ == "Normalize"


class TestSegmentImage:
    """_segment_imageメソッドのテスト"""

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_segment_image_success(self, mock_deeplabv3):
        """正常なセグメンテーション実行のテスト"""
        # モックモデルの設定
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.squeeze.return_value = torch.zeros(21, 520, 520)  # ダミーの出力（21クラス）
        mock_model.return_value = {"out": mock_output}
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")
        
        # テスト用のPIL画像を作成
        test_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        
        # セグメンテーション実行
        result = segmenter._segment_image(test_image)
        
        # 結果の検証
        assert isinstance(result, np.ndarray)
        assert result.shape == (520, 520)
        mock_model.assert_called_once()

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_segment_image_model_not_initialized(self, mock_deeplabv3):
        """モデルが初期化されていない場合のテスト"""
        mock_deeplabv3.return_value = MagicMock()
        
        segmenter = SemanticSegmenter(device="cpu")
        segmenter.model = None  # モデルを無効化
        
        test_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        
        with pytest.raises(RuntimeError) as exc_info:
            segmenter._segment_image(test_image)
        assert "Model is not initialized" in str(exc_info.value)

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_segment_image_transform_not_initialized(self, mock_deeplabv3):
        """変換パイプラインが初期化されていない場合のテスト"""
        mock_deeplabv3.return_value = MagicMock()
        
        segmenter = SemanticSegmenter(device="cpu")
        segmenter.transform = None  # 変換パイプラインを無効化
        
        test_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        
        with pytest.raises(RuntimeError) as exc_info:
            segmenter._segment_image(test_image)
        assert "Transform pipeline is not initialized" in str(exc_info.value)

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_segment_image_tensor_operations(self, mock_deeplabv3):
        """テンソル操作が正しく行われることを確認"""
        mock_model = MagicMock()
        
        # モックの出力テンソルを設定
        mock_tensor = torch.ones(1, 21, 520, 520)  # 21クラス、520x520
        mock_output = MagicMock()
        mock_output.squeeze.return_value = mock_tensor.squeeze(0)
        mock_model.return_value = {"out": mock_output}
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")
        
        test_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        
        with patch("torch.argmax") as mock_argmax:
            mock_argmax.return_value.cpu.return_value.numpy.return_value = np.zeros((520, 520))
            
            result = segmenter._segment_image(test_image)
            
            # torch.argmaxが正しく呼ばれることを確認
            mock_argmax.assert_called_once()
            assert isinstance(result, np.ndarray)


class TestConstants:
    """定数のテスト"""

    def test_constants_defined(self) -> None:
        """必要な定数が定義されているか確認"""
        assert hasattr(SemanticSegmenter, "IMAGE_SIZE")
        assert SemanticSegmenter.IMAGE_SIZE == (520, 520)

        assert hasattr(SemanticSegmenter, "NORMALIZE_MEAN")
        assert SemanticSegmenter.NORMALIZE_MEAN == [0.485, 0.456, 0.406]

        assert hasattr(SemanticSegmenter, "NORMALIZE_STD")
        assert SemanticSegmenter.NORMALIZE_STD == [0.229, 0.224, 0.225]

        assert hasattr(SemanticSegmenter, "MAX_FILE_SIZE_MB")
        assert SemanticSegmenter.MAX_FILE_SIZE_MB == 10

        assert hasattr(SemanticSegmenter, "MAX_FILE_SIZE_BYTES")
        assert SemanticSegmenter.MAX_FILE_SIZE_BYTES == 10 * 1024 * 1024

        assert hasattr(SemanticSegmenter, "SUPPORTED_FORMATS")
        assert SemanticSegmenter.SUPPORTED_FORMATS == {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        assert hasattr(SemanticSegmenter, "PROCESSING_TIME_PRECISION")
        assert SemanticSegmenter.PROCESSING_TIME_PRECISION == 2


class TestProcessImage:
    """process_imageメソッドのテスト"""

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_process_image_success(self, mock_deeplabv3):
        """process_image成功時のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        # 各メソッドをモック
        mock_image = MagicMock()
        mock_original_size = (640, 480)
        mock_segmentation_mask = MagicMock()
        mock_stats = {
            "detected_classes": [{"id": 1, "name": "person"}],
            "total_pixels": 307200,
            "num_classes": 1
        }

        with patch.object(segmenter, "_load_and_validate_image", return_value=(mock_image, mock_original_size)):
            with patch.object(segmenter, "_segment_image", return_value=mock_segmentation_mask):
                with patch.object(segmenter, "_save_result", return_value="/output/result.png"):
                    with patch.object(segmenter, "_generate_segmentation_stats", return_value=mock_stats):
                        with patch.object(segmenter, "_save_segmentation_json", return_value="/output/result.json"):
                            result = segmenter.process_image("/input/test.jpg", "/output")

        assert result["status"] == "success"
        assert result["output_path"] == "/output/result.png"
        assert result["json_path"] == "/output/result.json"
        assert "processing_time" in result
        assert result["input_size"] == "640x480"
        assert result["output_size"] == "640x480"
        assert result["detected_classes"] == [{"id": 1, "name": "person"}]
        assert result["total_pixels"] == 307200

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_process_image_error(self, mock_deeplabv3):
        """process_image失敗時のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        # _load_and_validate_imageでエラーを発生
        with patch.object(segmenter, "_load_and_validate_image", side_effect=FileNotFoundError("画像ファイルが見つかりません")):
            result = segmenter.process_image("/input/nonexistent.jpg", "/output")

        assert result["status"] == "error"
        assert result["error_message"] == "画像ファイルが見つかりません"
        assert result["error_code"] == "FILE_NOT_FOUND"


class TestCreateSuccessResponse:
    """_create_success_responseメソッドのテスト"""

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_create_success_response(self, mock_deeplabv3):
        """成功レスポンス生成のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        output_path = "/output/result.png"
        json_path = "/output/result.json"
        processing_time = 1.23456
        original_size = (800, 600)
        stats = {
            "detected_classes": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
            "total_pixels": 480000,
            "num_classes": 2
        }

        response = segmenter._create_success_response(output_path, json_path, processing_time, original_size, stats)

        assert response["status"] == "success"
        assert response["output_path"] == "/output/result.png"
        assert response["json_path"] == "/output/result.json"
        assert response["processing_time"] == 1.23  # 小数点2桁で丸められる
        assert response["input_size"] == "800x600"
        assert response["output_size"] == "800x600"
        assert response["detected_classes"] == [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}]
        assert response["total_pixels"] == 480000


class TestCreateErrorResponse:
    """_create_error_responseメソッドのテスト"""

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_create_error_response_file_not_found(self, mock_deeplabv3):
        """FileNotFoundErrorのエラーレスポンス生成テスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        error = FileNotFoundError("画像ファイルが見つかりません")
        response = segmenter._create_error_response(error)

        assert response["status"] == "error"
        assert response["error_message"] == "画像ファイルが見つかりません"
        assert response["error_code"] == "FILE_NOT_FOUND"

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_create_error_response_value_error(self, mock_deeplabv3):
        """ValueErrorのエラーレスポンス生成テスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        error = ValueError("ファイルサイズが10MBを超えています")
        response = segmenter._create_error_response(error)

        assert response["status"] == "error"
        assert response["error_message"] == "ファイルサイズが10MBを超えています"
        assert response["error_code"] == "FILE_TOO_LARGE"

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_create_error_response_runtime_error(self, mock_deeplabv3):
        """RuntimeErrorのエラーレスポンス生成テスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        error = RuntimeError("モデルエラー")
        response = segmenter._create_error_response(error)

        assert response["status"] == "error"
        assert response["error_message"] == "モデルエラー"
        assert response["error_code"] == "MODEL_ERROR"


class TestLoadAndValidateImage:
    """_load_and_validate_imageメソッドのテスト"""

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_load_and_validate_image_success(self, mock_deeplabv3):
        """画像読み込み成功のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        # 正常な画像ファイルをモック
        mock_image = MagicMock()
        mock_image.size = (800, 600)
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 1024 * 1024  # 1MB
                    with patch("pathlib.Path.suffix", new_callable=lambda: ".jpg"):
                        with patch("PIL.Image.open", return_value=mock_image):
                            with patch.object(mock_image, "convert", return_value=mock_image):
                                result_image, original_size = segmenter._load_and_validate_image("test.jpg")
                                
                                assert result_image == mock_image
                                assert original_size == (800, 600)

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_load_and_validate_image_file_not_found(self, mock_deeplabv3):
        """ファイルが見つからない場合のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError) as exc_info:
                segmenter._load_and_validate_image("nonexistent.jpg")
            assert "画像ファイルが見つかりません" in str(exc_info.value)

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_load_and_validate_image_not_file(self, mock_deeplabv3):
        """ディレクトリが指定された場合のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_file", return_value=False):
                with pytest.raises(ValueError) as exc_info:
                    segmenter._load_and_validate_image("directory")
                assert "指定されたパスはファイルではありません" in str(exc_info.value)

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_load_and_validate_image_file_too_large(self, mock_deeplabv3):
        """ファイルサイズが制限を超える場合のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 15 * 1024 * 1024  # 15MB
                    with pytest.raises(ValueError) as exc_info:
                        segmenter._load_and_validate_image("large_file.jpg")
                    assert "ファイルサイズが10MBを超えています" in str(exc_info.value)

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_load_and_validate_image_unsupported_format(self, mock_deeplabv3):
        """サポートされていない画像形式の場合のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 1024 * 1024  # 1MB
                    with patch("pathlib.Path.suffix", new_callable=lambda: ".gif"):
                        with pytest.raises(ValueError) as exc_info:
                            segmenter._load_and_validate_image("image.gif")
                        assert "サポートされていない画像形式です" in str(exc_info.value)

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_load_and_validate_image_corrupted_file(self, mock_deeplabv3):
        """破損した画像ファイルの場合のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 1024 * 1024  # 1MB
                    with patch("pathlib.Path.suffix", new_callable=lambda: ".jpg"):
                        with patch("PIL.Image.open", side_effect=Exception("Corrupted image")):
                            with pytest.raises(ValueError) as exc_info:
                                segmenter._load_and_validate_image("corrupted.jpg")
                            assert "画像ファイルが破損しているか、読み込めません" in str(exc_info.value)
