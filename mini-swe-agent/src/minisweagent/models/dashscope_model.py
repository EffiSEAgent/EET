import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from dashscope import Generation
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("dashscope_model")


@dataclass
class DashScopeModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)


class DashScopeModel:
    def __init__(self, *, config_class: type = DashScopeModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        
        if self.config.model_name.startswith("dashscope/"):
            self.config.model_name = self.config.model_name.replace("dashscope/", "", 1)
        
        api_key = self.config.model_kwargs.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        response = Generation.call(
            model=self.config.model_name,
            messages=messages,
            **(self.config.model_kwargs | kwargs)
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"DashScope API error: {response.code} - {response.message}")
        
        return response

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(messages, **kwargs)
        
        usage = response.usage if hasattr(response, "usage") else {}
        input_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else getattr(usage, "input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else getattr(usage, "output_tokens", 0)
        
        cost_per_1k_input = 0.0002
        cost_per_1k_output = 0.0002
        
        cost = (input_tokens / 1000.0 * cost_per_1k_input + 
                output_tokens / 1000.0 * cost_per_1k_output)
        
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        
        content = ""
        if hasattr(response, "output"):
            if hasattr(response.output, "text"):
                content = response.output.text or ""
            elif isinstance(response.output, dict):
                content = response.output.get("text", "")
        
        return {
            "content": content,
            "extra": {
                "response": {
                    "usage": usage if isinstance(usage, dict) else usage.__dict__ if hasattr(usage, "__dict__") else {},
                    "request_id": getattr(response, "request_id", ""),
                },
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}

