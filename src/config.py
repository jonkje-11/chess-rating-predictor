"""Load the project configuration from config.yaml."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | str = ROOT / "config.yaml") -> dict[str, Any]:
    """Return config.yaml as a dict. Paths in it are relative to the repo root."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
