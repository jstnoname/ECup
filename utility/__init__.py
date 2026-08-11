from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utility.config import data_dir, load_config, set_secrets


@dataclass(frozen=True)
class Data:
    data_repo: str
    items: str
    items_human: str
    matches: str
    matches_llm: str


@dataclass(frozen=True)
class Model:
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    data: Data
    model: Model


@dataclass(frozen=True)
class Env:
    config: Config
    data_dir: Path


def load(path: str | Path | None = None) -> Env:
    """
    Initialize environment: secrets to os.environ, config, data dir.
    :param path: path to config
    """
    set_secrets()
    cfg = load_config(path)
    return Env(
        config=Config(
            data=Data(**cfg.get("data", {})),
            model=Model(params=cfg.get("model", {}))
        ),
        data_dir=data_dir(),
    )
