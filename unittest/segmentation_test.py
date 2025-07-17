import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
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
        mock_stats = {"detected_classes": [{"id": 1, "name": "person"}], "total_pixels": 307200, "num_classes": 1}

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
        with patch.object(
            segmenter, "_load_and_validate_image", side_effect=FileNotFoundError("画像ファイルが見つかりません")
        ):
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
            "num_classes": 2,
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


class TestSaveResult:
    """_save_resultメソッドのテスト"""

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_save_result_success(self, mock_deeplabv3):
        """結果保存成功のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        # テスト用のマスクデータ
        mask = np.zeros((520, 520), dtype=np.uint8)
        original_size = (800, 600)
        output_dir = "/test/output"

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("time.strftime") as mock_strftime:
                with patch("cv2.resize") as mock_resize:
                    with patch.object(segmenter, "_get_colormap") as mock_get_colormap:
                        with patch("PIL.Image.fromarray") as mock_fromarray:
                            mock_strftime.return_value = "20231201_143022"
                            mock_resize.return_value = np.zeros((600, 800), dtype=np.uint8)
                            mock_get_colormap.return_value = np.zeros((256, 3), dtype=np.uint8)
                            mock_image = MagicMock()
                            mock_fromarray.return_value = mock_image

                            result_path = segmenter._save_result(mask, original_size, output_dir)

                            # 各関数が正しく呼ばれることを確認
                            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                            mock_strftime.assert_called_once_with(segmenter.TIMESTAMP_FORMAT)
                            mock_resize.assert_called_once()
                            mock_get_colormap.assert_called_once()
                            mock_fromarray.assert_called_once()
                            mock_image.save.assert_called_once()

                            # 結果パスが正しい形式であることを確認
                            assert result_path.endswith(".png")
                            assert "result_20231201_143022.png" in result_path

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_save_result_directory_creation(self, mock_deeplabv3):
        """出力ディレクトリ作成のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        mask = np.zeros((520, 520), dtype=np.uint8)
        original_size = (800, 600)
        output_dir = "/test/output"

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("time.strftime") as mock_strftime:
                with patch("cv2.resize") as mock_resize:
                    with patch.object(segmenter, "_get_colormap") as mock_get_colormap:
                        with patch("PIL.Image.fromarray") as mock_fromarray:
                            mock_strftime.return_value = "20231201_143022"
                            mock_resize.return_value = np.zeros((600, 800), dtype=np.uint8)
                            mock_get_colormap.return_value = np.zeros((256, 3), dtype=np.uint8)
                            mock_image = MagicMock()
                            mock_fromarray.return_value = mock_image

                            segmenter._save_result(mask, original_size, output_dir)

                            # ディレクトリ作成が正しく呼ばれることを確認
                            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_save_result_mask_resize(self, mock_deeplabv3):
        """マスクリサイズのテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        mask = np.zeros((520, 520), dtype=np.uint8)
        original_size = (800, 600)
        output_dir = "/test/output"

        with patch("pathlib.Path.mkdir"):
            with patch("time.strftime") as mock_strftime:
                with patch("cv2.resize") as mock_resize:
                    with patch.object(segmenter, "_get_colormap") as mock_get_colormap:
                        with patch("PIL.Image.fromarray") as mock_fromarray:
                            mock_strftime.return_value = "20231201_143022"
                            mock_resize.return_value = np.zeros((600, 800), dtype=np.uint8)
                            mock_get_colormap.return_value = np.zeros((256, 3), dtype=np.uint8)
                            mock_image = MagicMock()
                            mock_fromarray.return_value = mock_image

                            segmenter._save_result(mask, original_size, output_dir)

                            # cv2.resizeが正しい引数で呼ばれることを確認
                            mock_resize.assert_called_once()
                            call_args = mock_resize.call_args
                            assert call_args[0][0].dtype == np.uint8
                            assert call_args[0][1] == (800, 600)
                            assert call_args[1]["interpolation"] == cv2.INTER_NEAREST

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_save_result_colormap_application(self, mock_deeplabv3):
        """カラーマップ適用のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        mask = np.zeros((520, 520), dtype=np.uint8)
        original_size = (800, 600)
        output_dir = "/test/output"

        with patch("pathlib.Path.mkdir"):
            with patch("time.strftime") as mock_strftime:
                with patch("cv2.resize") as mock_resize:
                    with patch.object(segmenter, "_get_colormap") as mock_get_colormap:
                        with patch("PIL.Image.fromarray") as mock_fromarray:
                            mock_strftime.return_value = "20231201_143022"
                            mock_resize.return_value = np.zeros((600, 800), dtype=np.uint8)
                            mock_colormap = np.zeros((256, 3), dtype=np.uint8)
                            mock_get_colormap.return_value = mock_colormap
                            mock_image = MagicMock()
                            mock_fromarray.return_value = mock_image

                            segmenter._save_result(mask, original_size, output_dir)

                            # _get_colormapが呼ばれることを確認
                            mock_get_colormap.assert_called_once()

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_save_result_error_handling(self, mock_deeplabv3):
        """エラー処理のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        mask = np.zeros((520, 520), dtype=np.uint8)
        original_size = (800, 600)
        output_dir = "/test/output"

        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError) as exc_info:
                segmenter._save_result(mask, original_size, output_dir)
            assert "Permission denied" in str(exc_info.value)


class TestGetColormap:
    """_get_colormapメソッドのテスト"""

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_get_colormap_basic_structure(self, mock_deeplabv3):
        """カラーマップの基本構造のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        colormap = segmenter._get_colormap()

        # カラーマップが正しい形式であることを確認
        assert isinstance(colormap, np.ndarray)
        assert colormap.shape == (256, 3)
        assert colormap.dtype == np.uint8

        # カラーマップの値が適切な範囲内であることを確認
        assert np.all(colormap >= 0)
        assert np.all(colormap <= 255)

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_get_colormap_initialization(self, mock_deeplabv3):
        """カラーマップ初期化のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        colormap = segmenter._get_colormap()

        # カラーマップが正しく初期化されていることを確認
        assert colormap.shape == (256, 3)
        assert colormap.dtype == np.uint8

        # 未使用のインデックス（21-255）はゼロであることを確認
        assert np.all(colormap[21:] == 0)

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_get_colormap_class_colors_integration(self, mock_deeplabv3):
        """クラス色情報との統合テスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        # _get_class_colorsの戻り値をモック
        mock_colors = [
            {"id": 0, "name": "background", "color": [0, 0, 0], "description": "背景"},
            {"id": 1, "name": "person", "color": [192, 128, 128], "description": "人"},
            {"id": 2, "name": "car", "color": [128, 128, 128], "description": "車"},
        ]

        with patch.object(segmenter, "_get_class_colors", return_value=mock_colors):
            colormap = segmenter._get_colormap()

            # 各クラスの色が正しく設定されていることを確認
            assert np.array_equal(colormap[0], [0, 0, 0])  # background
            assert np.array_equal(colormap[1], [192, 128, 128])  # person
            assert np.array_equal(colormap[2], [128, 128, 128])  # car

            # 未使用のインデックスはゼロのままであることを確認
            assert np.array_equal(colormap[3], [0, 0, 0])
            assert np.array_equal(colormap[255], [0, 0, 0])

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_get_colormap_all_pascal_classes(self, mock_deeplabv3):
        """全PASCAL VOCクラスのテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        colormap = segmenter._get_colormap()

        # PASCAL VOCの主要クラスの色を確認
        # background
        assert np.array_equal(colormap[0], [0, 0, 0])
        # person
        assert np.array_equal(colormap[15], [192, 128, 128])
        # car
        assert np.array_equal(colormap[7], [128, 128, 128])
        # aeroplane
        assert np.array_equal(colormap[1], [128, 0, 0])
        # bicycle
        assert np.array_equal(colormap[2], [0, 128, 0])

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_get_colormap_data_type(self, mock_deeplabv3):
        """データ型のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        colormap = segmenter._get_colormap()

        # データ型がuint8であることを確認
        assert colormap.dtype == np.uint8

        # 各要素も整数型であることを確認
        for i in range(colormap.shape[0]):
            for j in range(colormap.shape[1]):
                assert isinstance(colormap[i, j], np.integer)

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_get_colormap_value_range(self, mock_deeplabv3):
        """値の範囲のテスト"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model

        segmenter = SemanticSegmenter(device="cpu")

        colormap = segmenter._get_colormap()

        # 全ての値が0-255の範囲内であることを確認
        assert np.all(colormap >= 0)
        assert np.all(colormap <= 255)

        # 実際のPASCAL VOCクラスの色が適切な範囲内であることを確認
        # person (ID: 15)
        person_color = colormap[15]
        assert np.all(person_color >= 0) and np.all(person_color <= 255)
        # car (ID: 7)
        car_color = colormap[7]
        assert np.all(car_color >= 0) and np.all(car_color <= 255)


class TestGenerateSegmentationStats:
    """_generate_segmentation_statsメソッドのテスト"""

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_single_class(self, mock_deeplabv3):
        """単一クラスのみのマスク"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        mask = np.zeros((10, 10), dtype=np.uint8)  # 全部背景（ID:0）
        stats = segmenter._generate_segmentation_stats(mask)
        assert stats["total_pixels"] == 100
        assert stats["num_classes"] == 1
        assert len(stats["detected_classes"]) == 1
        cls = stats["detected_classes"][0]
        assert cls["id"] == 0
        assert cls["pixel_count"] == 100
        assert cls["percentage"] == 100.0

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_multiple_classes(self, mock_deeplabv3):
        """複数クラスのマスク"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:5, 0:5] = 1  # クラス1: 25ピクセル
        mask[5:10, 5:10] = 2  # クラス2: 25ピクセル
        stats = segmenter._generate_segmentation_stats(mask)
        assert stats["total_pixels"] == 100
        assert stats["num_classes"] == 3
        ids = [c["id"] for c in stats["detected_classes"]]
        assert set(ids) == {0, 1, 2}
        # 面積順（同数ならID順）
        assert stats["detected_classes"][0]["pixel_count"] == 50
        assert stats["detected_classes"][1]["pixel_count"] == 25
        assert stats["detected_classes"][2]["pixel_count"] == 25

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_class_id_out_of_range(self, mock_deeplabv3):
        """クラスIDが範囲外の場合は無視される"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[0, 0] = 255  # 範囲外ID
        stats = segmenter._generate_segmentation_stats(mask)
        ids = [c["id"] for c in stats["detected_classes"]]
        assert 255 not in ids
        assert 0 in ids

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_empty_mask(self, mock_deeplabv3):
        """空マスク（全て背景）"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        mask = np.zeros((0, 0), dtype=np.uint8)
        stats = segmenter._generate_segmentation_stats(mask)
        assert stats["total_pixels"] == 0
        assert stats["num_classes"] == 0
        assert stats["detected_classes"] == []

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_percentage_and_sorting(self, mock_deeplabv3):
        """割合とソート順の検証"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, :] = 1  # クラス1: 20ピクセル
        mask[2:5, :] = 2  # クラス2: 30ピクセル
        # 背景: 50ピクセル
        stats = segmenter._generate_segmentation_stats(mask)
        # クラス2が最大
        assert stats["detected_classes"][0]["id"] == 0 or stats["detected_classes"][0]["pixel_count"] == 50
        assert stats["detected_classes"][1]["id"] == 2 or stats["detected_classes"][1]["pixel_count"] == 30
        assert stats["detected_classes"][2]["id"] == 1 or stats["detected_classes"][2]["pixel_count"] == 20
        # 割合の合計は100
        total = sum([c["percentage"] for c in stats["detected_classes"]])
        assert abs(total - 100.0) < 1e-2


class TestSaveSegmentationJson:
    """_save_segmentation_jsonメソッドのテスト"""

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_save_segmentation_json_success(self, mock_deeplabv3):
        """正常にJSONが保存される"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        output_path = "/test/output/result_20231201_143022.png"
        stats = {
            "detected_classes": [
                {
                    "id": 1,
                    "name": "person",
                    "description": "人",
                    "color": {"rgb": [192, 128, 128], "hex": "#c08080"},
                    "pixel_count": 100,
                    "percentage": 50.0,
                },
                {
                    "id": 2,
                    "name": "car",
                    "description": "車",
                    "color": {"rgb": [128, 128, 128], "hex": "#808080"},
                    "pixel_count": 100,
                    "percentage": 50.0,
                },
            ],
            "total_pixels": 200,
            "num_classes": 2,
        }
        original_size = (20, 10)
        output_dir = "/test/output"
        with patch("json.dump") as mock_json_dump:
            with patch("builtins.open", create=True) as mock_open:
                with patch("time.strftime") as mock_strftime:
                    mock_strftime.return_value = "2023-12-01 14:30:22"
                    mock_file = MagicMock()
                    mock_open.return_value.__enter__.return_value = mock_file
                    result_path = segmenter._save_segmentation_json(output_path, stats, original_size, output_dir)
                    # ファイル名・パス
                    assert result_path.endswith("_segmentation.json")
                    assert "result_20231201_143022_segmentation.json" in result_path
                    # open, json.dumpが呼ばれる
                    mock_open.assert_called_once()
                    mock_json_dump.assert_called_once()
                    # JSON内容の検証
                    call_args = mock_json_dump.call_args[0]
                    json_data = call_args[0]
                    assert json_data["metadata"]["timestamp"] == "2023-12-01 14:30:22"
                    assert json_data["metadata"]["image_size"] == {"width": 20, "height": 10}
                    assert json_data["metadata"]["output_image"] == "result_20231201_143022.png"
                    assert json_data["metadata"]["total_pixels"] == 200
                    assert json_data["metadata"]["num_detected_classes"] == 2
                    # 検出クラス色情報
                    color_classes = json_data["color_mapping"]["classes"]
                    assert len(color_classes) == 2
                    assert color_classes[0]["id"] == 1
                    assert color_classes[1]["id"] == 2
                    # segmentation_results
                    seg = json_data["segmentation_results"]
                    assert len(seg["detected_classes"]) == 2
                    assert seg["detected_classes"][0]["id"] == 1
                    assert seg["detected_classes"][1]["id"] == 2

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_save_segmentation_json_error(self, mock_deeplabv3):
        """書き込みエラー時の例外"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        output_path = "/test/output/result.png"
        stats = {"detected_classes": [], "total_pixels": 0, "num_classes": 0}
        original_size = (1, 1)
        output_dir = "/test/output"
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with pytest.raises(IOError) as exc_info:
                segmenter._save_segmentation_json(output_path, stats, original_size, output_dir)
            assert "Permission denied" in str(exc_info.value)


class TestGenerateClassStatistics:
    """_generate_class_statisticsメソッドのテスト"""

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_empty_detected_classes(self, mock_deeplabv3):
        """空の検出クラスリストの場合"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        detected_classes = []
        result = segmenter._generate_class_statistics(detected_classes)
        assert result == {}

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_single_class(self, mock_deeplabv3):
        """単一クラスの場合"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        detected_classes = [{"name": "person", "percentage": 100.0}]
        result = segmenter._generate_class_statistics(detected_classes)
        assert result["dominant_class"] == "person"
        assert result["dominant_percentage"] == 100.0
        assert result["coverage_summary"]["background_percentage"] == 0
        assert result["coverage_summary"]["foreground_percentage"] == 100.0
        assert result["coverage_summary"]["most_diverse"] == False
        assert result["coverage_summary"]["primary_objects"] == ["person"]

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_background_only(self, mock_deeplabv3):
        """背景のみの場合"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        detected_classes = [{"name": "background", "percentage": 100.0}]
        result = segmenter._generate_class_statistics(detected_classes)
        assert result["dominant_class"] == "background"
        assert result["dominant_percentage"] == 100.0
        assert result["coverage_summary"]["background_percentage"] == 100.0
        assert result["coverage_summary"]["foreground_percentage"] == 0
        assert result["coverage_summary"]["most_diverse"] == False
        assert result["coverage_summary"]["primary_objects"] == []

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_multiple_classes_with_background(self, mock_deeplabv3):
        """複数クラス（背景含む）の場合"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        detected_classes = [
            {"name": "person", "percentage": 60.0},
            {"name": "background", "percentage": 30.0},
            {"name": "car", "percentage": 10.0},
        ]
        result = segmenter._generate_class_statistics(detected_classes)
        assert result["dominant_class"] == "person"
        assert result["dominant_percentage"] == 60.0
        assert result["coverage_summary"]["background_percentage"] == 30.0
        assert result["coverage_summary"]["foreground_percentage"] == 70.0
        assert result["coverage_summary"]["most_diverse"] == False
        assert result["coverage_summary"]["primary_objects"] == ["person", "car"]

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_diverse_classes(self, mock_deeplabv3):
        """多様なクラス（5個以上）の場合"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        detected_classes = [
            {"name": "person", "percentage": 30.0},
            {"name": "car", "percentage": 25.0},
            {"name": "dog", "percentage": 20.0},
            {"name": "cat", "percentage": 15.0},
            {"name": "bird", "percentage": 10.0},
        ]
        result = segmenter._generate_class_statistics(detected_classes)
        assert result["dominant_class"] == "person"
        assert result["dominant_percentage"] == 30.0
        assert result["coverage_summary"]["background_percentage"] == 0
        assert result["coverage_summary"]["foreground_percentage"] == 100.0
        assert result["coverage_summary"]["most_diverse"] == False  # 5個なのでFalse
        assert result["coverage_summary"]["primary_objects"] == ["person", "car", "dog"]

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_background_in_middle(self, mock_deeplabv3):
        """背景が中間にある場合"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        detected_classes = [
            {"name": "person", "percentage": 50.0},
            {"name": "background", "percentage": 30.0},
            {"name": "car", "percentage": 20.0},
        ]
        result = segmenter._generate_class_statistics(detected_classes)
        assert result["coverage_summary"]["background_percentage"] == 30.0
        assert result["coverage_summary"]["foreground_percentage"] == 70.0
        assert result["coverage_summary"]["primary_objects"] == ["person", "car"]

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_no_background(self, mock_deeplabv3):
        """背景がない場合"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        detected_classes = [{"name": "person", "percentage": 60.0}, {"name": "car", "percentage": 40.0}]
        result = segmenter._generate_class_statistics(detected_classes)
        assert result["coverage_summary"]["background_percentage"] == 0
        assert result["coverage_summary"]["foreground_percentage"] == 100.0
        assert result["coverage_summary"]["primary_objects"] == ["person", "car"]

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_actually_diverse_classes(self, mock_deeplabv3):
        """実際に多様なクラス（6個以上）の場合"""
        mock_model = MagicMock()
        mock_deeplabv3.return_value = mock_model
        segmenter = SemanticSegmenter(device="cpu")
        detected_classes = [
            {"name": "person", "percentage": 25.0},
            {"name": "car", "percentage": 20.0},
            {"name": "dog", "percentage": 15.0},
            {"name": "cat", "percentage": 15.0},
            {"name": "bird", "percentage": 15.0},
            {"name": "aeroplane", "percentage": 10.0},
        ]
        result = segmenter._generate_class_statistics(detected_classes)
        assert result["dominant_class"] == "person"
        assert result["dominant_percentage"] == 25.0
        assert result["coverage_summary"]["background_percentage"] == 0
        assert result["coverage_summary"]["foreground_percentage"] == 100.0
        assert result["coverage_summary"]["most_diverse"] == True  # 6個なのでTrue
        assert result["coverage_summary"]["primary_objects"] == ["person", "car", "dog"]


class TestGetErrorCode:
    """_get_error_codeメソッドのテスト"""

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_file_not_found(self, mock_deeplabv3):
        segmenter = SemanticSegmenter(device="cpu")
        err = FileNotFoundError("not found")
        assert segmenter._get_error_code(err) == "FILE_NOT_FOUND"

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_value_error_file_size(self, mock_deeplabv3):
        segmenter = SemanticSegmenter(device="cpu")
        err = ValueError("ファイルサイズが大きい")
        assert segmenter._get_error_code(err) == "FILE_TOO_LARGE"

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_value_error_unsupported_format(self, mock_deeplabv3):
        segmenter = SemanticSegmenter(device="cpu")
        err = ValueError("サポートされていない形式")
        assert segmenter._get_error_code(err) == "UNSUPPORTED_FORMAT"

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_value_error_other(self, mock_deeplabv3):
        segmenter = SemanticSegmenter(device="cpu")
        err = ValueError("その他のエラー")
        assert segmenter._get_error_code(err) == "PROCESSING_ERROR"

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_runtime_error(self, mock_deeplabv3):
        segmenter = SemanticSegmenter(device="cpu")
        err = RuntimeError("モデルエラー")
        assert segmenter._get_error_code(err) == "MODEL_ERROR"

    @patch("src.segmentation.deeplabv3_resnet50")
    def test_other_exception(self, mock_deeplabv3):
        segmenter = SemanticSegmenter(device="cpu")
        err = Exception("unknown")
        assert segmenter._get_error_code(err) == "PROCESSING_ERROR"
