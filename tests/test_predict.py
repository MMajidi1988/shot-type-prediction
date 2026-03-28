"""Unit tests for prediction helpers (TensorFlow is mocked)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure `src` is used when pytest is invoked without editable install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from shot_type_prediction.predict import (  # noqa: E402
    CLASS_NAMES,
    PredictionResult,
    iter_image_paths,
    predict_image,
)


def test_iter_image_paths_sorts_and_filters(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "skip.txt").write_text("no")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "c.jpg").write_bytes(b"")

    paths = list(iter_image_paths(tmp_path))
    assert [p.name for p in paths] == ["a.jpg", "b.png"]


def test_iter_image_paths_not_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        list(iter_image_paths(f))


def test_prediction_result_to_dict() -> None:
    r = PredictionResult(
        path=Path("x/y.jpg"),
        predicted_index=1,
        label="medium",
        probabilities=np.array([0.1, 0.8, 0.1], dtype=np.float32),
    )
    d = r.to_dict()
    assert d["label"] == "medium"
    assert d["predicted_index"] == 1
    assert d["path"] == str(Path("x/y.jpg"))
    assert len(d["probabilities"]) == 3


def test_predict_image_with_mocks(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow", reason="TensorFlow required for Keras image pipeline")

    img = tmp_path / "shot.jpg"
    img.write_bytes(b"")

    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.05, 0.9, 0.05]], dtype=np.float32)

    mock_image = MagicMock()
    mock_arr = np.zeros((224, 224, 3), dtype=np.float32)

    with (
        patch("tensorflow.keras.preprocessing.image.load_img", return_value=mock_image),
        patch(
            "tensorflow.keras.preprocessing.image.img_to_array",
            return_value=mock_arr,
        ),
        patch(
            "tensorflow.keras.applications.resnet50.preprocess_input",
            side_effect=lambda x: x,
        ),
    ):
        r = predict_image(fake_model, img)

    assert r.predicted_index == 1
    assert r.label == CLASS_NAMES[1]
    fake_model.predict.assert_called_once()
