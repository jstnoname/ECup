from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tomllib

SECRET_KEYS = [
    "HF_TOKEN",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Read [tool.ecup] from pyproject.toml.
    :param path: path to config file
    :return dictionary of key and value config
    """
    start = Path(path) if path is not None else Path.cwd()
    for candidate in [start, *start.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            with pyproject.open("rb") as f:
                return tomllib.load(f).get("tool", {}).get("ecup", {})
    raise FileNotFoundError("pyproject.toml with [tool.ecup] not found")


def set_secrets(names: list[str] | None = None) -> None:
    """
    load secrets to os.environ,
    :param names: environment names
    """
    if names is None:
        names = SECRET_KEYS

    def from_kaggle(key: str) -> str | None:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore

            return UserSecretsClient().get_secret(key)
        except Exception:
            return None

    def from_userdata(key: str) -> str | None:
        try:
            from google.colab import userdata  # type: ignore

            return userdata.get(key)
        except Exception:
            return None

    for name in names:
        if name in os.environ:
            continue
        value = from_kaggle(name) or from_userdata(name)
        if value is not None:
            os.environ[name] = value

    from dotenv import load_dotenv

    load_dotenv()


def data_dir() -> Path:
    """
    get data path: /kaggle/working/data → /content/data → local ./data.
    """
    for root in ("/kaggle/working", "/content"):
        if Path(root).is_dir():
            return Path(root) / "data"
    return Path("data")
