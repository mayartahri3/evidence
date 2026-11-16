"""Loads configs/project.yaml and environment variables in one place."""
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Env(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    openalex_api_key: str = ""


@lru_cache
def load_project_config() -> dict:
    with open(ROOT / "configs" / "project.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def load_env() -> Env:
    return Env()
