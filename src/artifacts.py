import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def update_json_section(path: Path, section: str, value: Any) -> None:
    data = _read_json(path)
    data[section] = value
    _write_json(path, data)


def save_feature_columns(models_dir: Path, model_name: str, columns: list[str]) -> Path:
    path = models_dir / "feature_columns.json"
    update_json_section(path, model_name, columns)
    return path


def save_preprocessing_config(models_dir: Path, model_name: str, config: dict[str, Any]) -> Path:
    path = models_dir / "preprocessing_config.json"
    update_json_section(path, model_name, config)
    return path
