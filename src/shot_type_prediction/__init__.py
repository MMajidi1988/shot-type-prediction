"""Classify camera shot types (close-up, medium, full) from images using a Keras model."""

from shot_type_prediction.predict import (
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    iter_image_paths,
    load_keras_model,
    predict_image,
    run_predictions,
)

__all__ = [
    "CLASS_NAMES",
    "DEFAULT_IMAGE_SIZE",
    "iter_image_paths",
    "load_keras_model",
    "predict_image",
    "run_predictions",
]

__version__ = "1.0.0"
