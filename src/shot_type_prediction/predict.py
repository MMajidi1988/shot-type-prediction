"""Load a trained Keras model and run shot-type classification on image files."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# TensorFlow is imported inside functions that need it so importing this module
# for tests/docs-only environments can still parse without TF installed.

DEFAULT_IMAGE_SIZE: tuple[int, int] = (224, 224)

# Indices must match the order used when training the model.
CLASS_NAMES: dict[int, str] = {
    0: "close-up",
    1: "medium",
    2: "full",
}

DEFAULT_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
)


@dataclass(frozen=True)
class PredictionResult:
    """One image file and its predicted shot type."""

    path: Path
    predicted_index: int
    label: str
    probabilities: np.ndarray

    def to_dict(self) -> dict[str, object]:
        probs = self.probabilities.tolist()
        return {
            "path": str(self.path),
            "predicted_index": self.predicted_index,
            "label": self.label,
            "probabilities": probs,
        }


def load_keras_model(model_path: str | os.PathLike[str]):
    """Load a saved Keras model from disk."""
    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for inference. Install: pip install 'shot-type-prediction[tf]' "
            "or pip install 'tensorflow>=2.12,<3'."
        ) from exc

    return load_model(os.fspath(model_path))


def iter_image_paths(
    folder: str | os.PathLike[str],
    extensions: frozenset[str] | None = None,
) -> Iterator[Path]:
    """Yield sorted image file paths under ``folder`` (non-recursive)."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    exts = extensions if extensions is not None else DEFAULT_EXTENSIONS
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _preprocess_array(
    image_array: np.ndarray,
) -> np.ndarray:
    from tensorflow.keras.applications.resnet50 import preprocess_input

    x = preprocess_input(image_array.astype(np.float32))
    return np.expand_dims(x, axis=0)


def predict_image(
    model,
    image_path: str | os.PathLike[str],
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    class_names: dict[int, str] | None = None,
) -> PredictionResult:
    """Load one image, run ``model.predict``, and return structured output."""
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    path = Path(image_path)
    image = load_img(path, target_size=image_size)
    arr = img_to_array(image)
    batch = _preprocess_array(arr)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    names = class_names if class_names is not None else CLASS_NAMES
    label = names.get(idx, f"class_{idx}")
    return PredictionResult(
        path=path,
        predicted_index=idx,
        label=label,
        probabilities=probs,
    )


def run_predictions(
    images_dir: str | os.PathLike[str],
    model_path: str | os.PathLike[str],
    *,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    extensions: frozenset[str] | None = None,
    class_names: dict[int, str] | None = None,
    show_plots: bool = False,
    verbose: bool = True,
) -> Sequence[PredictionResult]:
    """
    Classify every image in ``images_dir`` and optionally show matplotlib windows.

    When ``show_plots`` is True, uses a non-interactive backend if ``MPLBACKEND``
    is unset, so batch runs do not block on GUI on servers.
    """
    model = load_keras_model(model_path)
    if verbose:
        print(f"Loaded model from {os.fspath(model_path)}")

    names = class_names if class_names is not None else CLASS_NAMES
    results: list[PredictionResult] = []

    plt_module = None
    load_img = None
    img_to_array = None
    if show_plots:
        import matplotlib

        if "MPLBACKEND" not in os.environ:
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from tensorflow.keras.preprocessing.image import (
            img_to_array as _ita,
        )
        from tensorflow.keras.preprocessing.image import (
            load_img as _li,
        )

        plt_module = plt
        load_img = _li
        img_to_array = _ita

    for img_path in iter_image_paths(images_dir, extensions):
        r = predict_image(model, img_path, image_size=image_size, class_names=names)
        results.append(r)
        if verbose:
            print(f"{img_path.name}: {r.label} (index {r.predicted_index})")

        if show_plots and plt_module is not None and load_img is not None:
            image = load_img(img_path, target_size=image_size)
            arr = img_to_array(image)
            plt_module.figure(figsize=(8, 8))
            plt_module.imshow(arr.astype(np.uint8))
            plt_module.title(f"{img_path.name}\n{r.label}")
            plt_module.axis("off")
            plt_module.show()

    return results
