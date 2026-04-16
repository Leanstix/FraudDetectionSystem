from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.constants import DEFAULT_MODEL, DEFAULT_RANDOM_SEED, DEFAULT_TEMPERATURE


@dataclass(slots=True)
class Settings:
    team_name: str = "reply-mirror-team"
    openrouter_api_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://challenges.reply.com/langfuse"
    langfuse_media_upload_enabled: bool = False
    default_model: str = DEFAULT_MODEL
    default_temperature: float = DEFAULT_TEMPERATURE
    llm_enabled: bool = True
    llm_cache_dir: str = "cache/llm"
    random_seed: int = DEFAULT_RANDOM_SEED
    agent_weights: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    output_dir: str = "outputs"
    cache_dir: str = "cache"

    @classmethod
    def from_env_and_file(cls, config_path: str | None = None) -> "Settings":
        load_dotenv()
        cfg_path = Path(config_path or "configs/default.yaml")
        file_cfg: dict[str, Any] = {}
        if cfg_path.exists():
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                file_cfg = loaded

        def env_or_file(env_name: str, file_key: str, default: Any = None) -> Any:
            env_val = os.getenv(env_name)
            if env_val is not None:
                return env_val
            return file_cfg.get(file_key, default)

        def as_bool(value: Any, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

        def as_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def as_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        settings = cls(
            team_name=str(env_or_file("TEAM_NAME", "team_name", "reply-mirror-team")),
            openrouter_api_key=(
                env_or_file("OPENROUTER_API_KEY", "openrouter_api_key")
                or os.getenv("OPENROUTER_KEY_FOR_DATASET_1-3")
            ),
            langfuse_public_key=env_or_file("LANGFUSE_PUBLIC_KEY", "langfuse_public_key"),
            langfuse_secret_key=(
                env_or_file("LANGFUSE_SECRET_KEY", "langfuse_secret_key")
                or os.getenv("LANGFUSE_PRIVATE_KEY")
            ),
            langfuse_host=str(env_or_file("LANGFUSE_HOST", "langfuse_host", "https://challenges.reply.com/langfuse")),
            langfuse_media_upload_enabled=as_bool(
                env_or_file("LANGFUSE_MEDIA_UPLOAD_ENABLED", "langfuse_media_upload_enabled", False),
                default=False,
            ),
            default_model=str(env_or_file("DEFAULT_MODEL", "default_model", DEFAULT_MODEL)),
            default_temperature=as_float(
                env_or_file("DEFAULT_TEMPERATURE", "default_temperature", DEFAULT_TEMPERATURE),
                DEFAULT_TEMPERATURE,
            ),
            llm_enabled=as_bool(env_or_file("LLM_ENABLED", "llm_enabled", True), default=True),
            llm_cache_dir=str(env_or_file("LLM_CACHE_DIR", "llm_cache_dir", "cache/llm")),
            random_seed=as_int(env_or_file("RANDOM_SEED", "random_seed", DEFAULT_RANDOM_SEED), DEFAULT_RANDOM_SEED),
            agent_weights=dict(file_cfg.get("agent_weights", {})),
            thresholds=dict(file_cfg.get("thresholds", {})),
            output_dir=str(env_or_file("OUTPUT_DIR", "output_dir", "outputs")),
            cache_dir=str(env_or_file("CACHE_DIR", "cache_dir", "cache")),
        )

        if not settings.agent_weights:
            settings.agent_weights = {
                "transaction_behavior_score": 0.28,
                "temporal_sequence_score": 0.20,
                "geospatial_score": 0.16,
                "communication_risk_score": 0.18,
                "novelty_drift_score": 0.18,
            }
        if not settings.thresholds:
            settings.thresholds = {
                "target_flag_rate": 0.12,
                "min_flag_rate": 0.02,
                "max_flag_rate": 0.45,
                "min_score": 0.15,
                "max_score": 0.98,
            }

        settings.ensure_directories()
        return settings

    def is_llm_enabled(self) -> bool:
        return bool(self.llm_enabled and self.openrouter_api_key)

    def ensure_directories(self) -> None:
        for path in [self.output_dir, self.cache_dir, self.llm_cache_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
