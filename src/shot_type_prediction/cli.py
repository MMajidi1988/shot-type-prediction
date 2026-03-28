"""Command-line interface for batch shot-type prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shot_type_prediction import __version__
from shot_type_prediction.predict import (
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    run_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shot-type-predict",
        description=(
            "Classify shot types (close-up / medium / full) for images in a folder "
            "using a trained Keras model (e.g. ResNet50-based classifier)."
        ),
    )
    p.add_argument(
        "--images",
        "-i",
        type=Path,
        required=True,
        help="Directory containing image files (non-recursive).",
    )
    p.add_argument(
        "--model",
        "-m",
        type=Path,
        required=True,
        help="Path to the saved Keras model (.keras or SavedModel).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Open a matplotlib window for each image (local desktop use).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON object per line (NDJSON) to stdout instead of text.",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress messages (still prints JSON lines if --json).",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    images_dir = args.images
    model_path = args.model

    if not images_dir.is_dir():
        print(f"Error: not a directory: {images_dir}", file=sys.stderr)
        return 2
    if not model_path.is_file() and not model_path.is_dir():
        # SavedModel is a directory
        print(f"Error: model path not found: {model_path}", file=sys.stderr)
        return 2

    results = run_predictions(
        images_dir,
        model_path,
        image_size=DEFAULT_IMAGE_SIZE,
        class_names=CLASS_NAMES,
        show_plots=args.show,
        verbose=not args.quiet,
    )

    if args.json:
        for r in results:
            print(json.dumps(r.to_dict()))

    if not results and not args.quiet:
        print(
            "No images found. Supported extensions: .jpg, .jpeg, .png, .webp, .bmp, .gif",
            file=sys.stderr,
        )

    return 0
