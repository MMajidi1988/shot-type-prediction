"""CLI smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from shot_type_prediction.cli import main  # noqa: E402


def test_cli_errors_on_missing_images_dir(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    model = tmp_path / "m.keras"
    model.write_bytes(b"fake")
    code = main(["--images", str(missing), "--model", str(model)])
    assert code == 2


def test_cli_errors_on_missing_model(tmp_path: Path) -> None:
    d = tmp_path / "imgs"
    d.mkdir()
    code = main(["--images", str(d), "--model", str(tmp_path / "missing.keras")])
    assert code == 2
