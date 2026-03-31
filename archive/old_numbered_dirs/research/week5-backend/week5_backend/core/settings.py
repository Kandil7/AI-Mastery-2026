from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class Settings:
    app_version: str
    default_provider: str
    default_vector_store: str
    config_path: Path
    raw: Dict[str, Any]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_settings() -> Settings:
    default_path = Path(__file__).resolve().parents[1] / "config/settings.yaml"
    config_path = Path(os.getenv("WEEK5_BACKEND_CONFIG", str(default_path)))
    data = _load_yaml(config_path)
    return Settings(
        app_version=str(data.get("app_version", "0.1.0")),
        default_provider=str(data.get("default_provider", "openai")),
        default_vector_store=str(data.get("default_vector_store", "pgvector")),
        config_path=config_path,
        raw=data,
    )
