# Shot type prediction

Classify **camera shot scale** in still frames: **close-up**, **medium**, or **full** body. The runtime uses a trained **Keras** classifier (for example **ResNet50** with ImageNet preprocessing, input size **224×224**).

This repository is a small, reusable **Python package** and **CLI** for batch inference. It replaces a single hard-coded script with typed APIs, tests, and clear installation instructions.

## Requirements

- Python **3.10+**
- A saved Keras model whose output layer has **three** units in this order:

  | Index | Label     |
  |------:|-----------|
  | 0     | close-up  |
  | 1     | medium    |
  | 2     | full      |

If your training used different class order, retrain with matching indices or adapt `CLASS_NAMES` in `src/shot_type_prediction/predict.py`.

## Install

From the repository root:

- **Inference** (TensorFlow + package):

  ```bash
  pip install -e ".[tf]"
  ```

- **Development** (lint, tests; TensorFlow needed for the full test suite):

  ```bash
  pip install -e ".[tf,dev]"
  ```

TensorFlow publishes wheels for common **64-bit** Python versions on Linux/macOS/Windows; if `pip install` fails, use **Python 3.10–3.12** from [python.org](https://www.python.org/downloads/) or your OS package manager.

`requirements.txt` mirrors the main runtime pins for non-editable setups.

## Usage

### Command line

```bash
shot-type-predict --images path/to/folder --model path/to/model.keras
```

Options:

- `--show` — open a **matplotlib** window per image (desktop only).
- `--json` — print **one JSON object per line** (path, label, probabilities).
- `-q` / `--quiet` — less console output.

Example:

```bash
shot-type-predict -i ./frames -m ./best_model.keras --json > results.ndjson
```

### Python API

```python
from pathlib import Path
from shot_type_prediction import load_keras_model, predict_image

model = load_keras_model("best_model.keras")
result = predict_image(model, Path("frame_001.jpg"))
print(result.label, result.probabilities)
```

### Module entry point

```bash
python -m shot_type_prediction --images ./frames --model ./best_model.keras
```

## Supported image formats

By default, files in the image folder with these extensions are processed (non-recursive): `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`.

## Development

```bash
pip install -e ".[tf,dev]"
ruff check src tests
pytest
```

## Project layout

```text
src/shot_type_prediction/   # Package: prediction logic + CLI
tests/                      # Pytest (TensorFlow mocked where possible)
pyproject.toml              # Metadata, dependencies, console script
```

## License

MIT — see [LICENSE](LICENSE).

## Customization

- Update **`[project.urls]`** in `pyproject.toml` with your GitHub repository URL.
- Set the **copyright year/name** in `LICENSE` if you publish the project publicly.
