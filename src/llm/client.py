from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.config import Settings
from src.llm.cache import LLMCache
from src.tracing import TracingManager


class LLMClient:
    def __init__(self, settings: Settings, tracing: TracingManager):
        self.settings = settings
        self.tracing = tracing
        self.cache = LLMCache(settings.llm_cache_dir)
        self._model = None
        if self.is_enabled():
            self._model = ChatOpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                model=settings.default_model,
                temperature=settings.default_temperature,
            )

    def is_enabled(self) -> bool:
        return self.settings.is_llm_enabled()

    def invoke(self, prompt: str, dataset_name: str, task_name: str) -> str:
        if not self.is_enabled() or self._model is None:
            return ""

        key = self.cache.make_key(self.settings.default_model, prompt)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        try:
            cfg = self.tracing.get_langchain_config(dataset_name=dataset_name, task_name=task_name)
            response = self._model.invoke([HumanMessage(content=prompt)], config=cfg)
            content = response.content if isinstance(response.content, str) else str(response.content)
            self.cache.set(key, content)
            return content
        except Exception:
            return ""
