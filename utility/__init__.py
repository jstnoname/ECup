from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utility.config import data_dir, load_config, set_secrets


@dataclass(frozen=True)
class Env:
    config: dict[str, Any]
    data_dir: Path


def load(path: str | Path | None = None) -> Env:
    """
    Initialize environment: secrets to os.environ, config, data dir.
    :param path: path to config
    """
    set_secrets()
    return Env(config=load_config(path), data_dir=data_dir())
