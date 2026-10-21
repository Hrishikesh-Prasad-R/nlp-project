"""Configuration loading and path resolution (no machine-specific absolute paths)."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(p: str | Path) -> Path:
    """Resolve a path relative to the project root (absolute paths are kept)."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def resolve_input_path(p: str | Path) -> Path:
    """Resolve a user-supplied data path: current directory first, then project root."""
    p = Path(p)
    if p.is_absolute() or p.exists():
        return p
    return PROJECT_ROOT / p
