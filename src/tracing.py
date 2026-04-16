from __future__ import annotations

import re

import ulid

from src.config import Settings

try:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler
except Exception:
    Langfuse = None  # type: ignore[assignment]
    CallbackHandler = None  # type: ignore[assignment]


class TracingManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._session_id: str | None = None
        self._langfuse = None
        self._handler = None
        if self.is_enabled() and Langfuse is not None:
            self._langfuse = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )

    def generate_session_id(self) -> str:
        team = re.sub(r"[^A-Za-z0-9_-]+", "-", self.settings.team_name.strip()) or "team"
        self._session_id = f"{team}-{ulid.new().str}"
        return self._session_id

    def is_enabled(self) -> bool:
        return bool(
            self.settings.langfuse_public_key
            and self.settings.langfuse_secret_key
            and CallbackHandler is not None
        )

    def get_langfuse_client(self):
        return self._langfuse

    def get_callback_handler(self):
        if not self.is_enabled() or CallbackHandler is None:
            return None
        if self._handler is None:
            self._handler = CallbackHandler()
        return self._handler

    def get_langchain_config(self, dataset_name: str, task_name: str) -> dict:
        if not self._session_id:
            self.generate_session_id()
        config: dict = {
            "metadata": {
                "langfuse_session_id": self._session_id,
                "dataset_name": dataset_name,
                "task_name": task_name,
            }
        }
        handler = self.get_callback_handler()
        if handler is not None:
            config["callbacks"] = [handler]
        return config

    def flush(self) -> None:
        if self._langfuse is not None:
            self._langfuse.flush()
