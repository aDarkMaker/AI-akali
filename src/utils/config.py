from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(
    config_path: str = "config/bot.yaml", base_dir: Path = None
) -> Dict[str, Any]:
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent

    full_path = base_dir / config_path

    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
