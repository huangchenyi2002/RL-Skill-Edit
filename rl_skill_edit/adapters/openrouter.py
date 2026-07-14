"""OpenRouter adapter for RL-Skill-Edit."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Mapping
from typing import Any

import httpx
from openai import OpenAI


_PROXY_ENV_VARIABLES = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)


class OpenRouterClient:
    """OpenRouter chat client used by the RL runtime."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        openrouter_config = config.get("openrouter")
        if not isinstance(openrouter_config, Mapping):
            raise ValueError("config.openrouter must be a mapping")
        base_url = openrouter_config.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("config.openrouter.base_url must be a non-empty string")
        cost_tracking = config.get("cost_tracking")
        if not isinstance(cost_tracking, Mapping):
            raise ValueError("config.cost_tracking must be a mapping")
        if not isinstance(cost_tracking.get("enabled"), bool):
            raise ValueError("config.cost_tracking.enabled must be a boolean")
        if cost_tracking["enabled"] and not isinstance(
            cost_tracking.get("cost_per_1k_tokens"), Mapping
        ):
            raise ValueError(
                "config.cost_tracking.cost_per_1k_tokens must be a mapping"
            )

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key or not api_key.strip():
            raise ValueError("OPENROUTER_API_KEY is required")

        proxy_url = next(
            (value for name in _PROXY_ENV_VARIABLES if (value := os.environ.get(name))),
            openrouter_config.get("proxy"),
        )
        if proxy_url is not None and (
            not isinstance(proxy_url, str) or not proxy_url.strip()
        ):
            raise ValueError("OpenRouter proxy must be a non-empty string")

        http_client_kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(
                connect=10.0,
                read=45.0,
                write=10.0,
                pool=5.0,
            ),
            "limits": httpx.Limits(
                max_connections=16,
                max_keepalive_connections=0,
                keepalive_expiry=0,
            ),
            "trust_env": False,
        }
        if proxy_url is not None:
            http_client_kwargs["proxy"] = proxy_url
        http_client = httpx.Client(**http_client_kwargs)

        self.config = config
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            max_retries=0,
        )
        self.extra_headers = {
            "HTTP-Referer": "http://localhost",
            "X-Title": "RL-Skill-Edit",
        }
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.call_log: list[dict[str, Any]] = []
        self._initialize_usage_lock()

    def _initialize_usage_lock(self) -> None:
        self._usage_lock = threading.Lock()

    def _record_call(self, usage_info: dict[str, Any]) -> None:
        with self._usage_lock:
            self.total_input_tokens += int(usage_info.get("input_tokens", 0))
            self.total_output_tokens += int(usage_info.get("output_tokens", 0))
            self.total_cost_usd += float(usage_info.get("cost_usd", 0.0))
            self.call_log.append(dict(usage_info))

    @staticmethod
    def _failed_usage(
        model: str,
        call_type: str,
        error_kind: str,
        error_message: str = "",
    ) -> dict[str, Any]:
        return {
            "model": model,
            "call_type": call_type,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "ok": False,
            "error_kind": error_kind,
            "error_message": error_message[:500],
        }

    def _cost_rate(self, model: str) -> float:
        cost_tracking = self.config["cost_tracking"]
        if not cost_tracking["enabled"]:
            return 0.0
        prices = cost_tracking["cost_per_1k_tokens"]
        if model not in prices:
            raise ValueError(f"missing token price for model: {model}")
        price = prices[model]
        if (
            isinstance(price, bool)
            or not isinstance(price, (int, float))
            or not math.isfinite(float(price))
            or float(price) < 0.0
        ):
            raise ValueError(f"invalid token price for model: {model}")
        return float(price)

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        call_type: str = "unknown",
        seed: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            cost_rate = self._cost_rate(model)
        except (KeyError, TypeError, ValueError) as exc:
            failed = self._failed_usage(
                model,
                call_type,
                "configuration_error",
                str(exc),
            )
            self._record_call(failed)
            return "", failed

        full_messages: list[dict[str, Any]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        request: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_headers": self.extra_headers,
        }
        if seed is not None:
            request["seed"] = int(seed)

        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            failed = self._failed_usage(
                model,
                call_type,
                type(exc).__name__,
                str(exc),
            )
            self._record_call(failed)
            return "", failed

        provider_usage = getattr(response, "usage", None)
        if provider_usage is None:
            failed = self._failed_usage(model, call_type, "empty_response")
            self._record_call(failed)
            return "", failed

        try:
            input_tokens = provider_usage.prompt_tokens
            output_tokens = provider_usage.completion_tokens
            total_tokens = provider_usage.total_tokens
            token_counts = (input_tokens, output_tokens, total_tokens)
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in token_counts
            ):
                raise ValueError("token counts must be integers")
            if any(value < 0 for value in token_counts):
                raise ValueError("token counts must not be negative")
            if total_tokens != input_tokens + output_tokens:
                raise ValueError("total token count is inconsistent")
        except (AttributeError, ValueError):
            failed = self._failed_usage(model, call_type, "invalid_usage")
            self._record_call(failed)
            return "", failed

        choices = getattr(response, "choices", None)
        message = getattr(choices[0], "message", None) if choices else None
        text = getattr(message, "content", None)
        ok = isinstance(text, str) and bool(text.strip())
        usage_info = {
            "model": model,
            "call_type": call_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": (input_tokens + output_tokens) / 1000.0 * cost_rate,
            "ok": ok,
            "error_kind": None if ok else "empty_response",
            "error_message": "",
        }
        self._record_call(usage_info)
        return (text if ok else ""), usage_info

    def cost_summary(self) -> dict[str, Any]:
        with self._usage_lock:
            call_log = list(self.call_log)
            total_input_tokens = self.total_input_tokens
            total_output_tokens = self.total_output_tokens
            total_cost_usd = self.total_cost_usd

        breakdown: dict[str, dict[str, int | float]] = {}
        for usage_info in call_log:
            call_type = str(usage_info["call_type"])
            entry = breakdown.setdefault(
                call_type,
                {"calls": 0, "tokens": 0, "cost_usd": 0.0},
            )
            entry["calls"] += 1
            entry["tokens"] += int(usage_info["total_tokens"])
            entry["cost_usd"] += float(usage_info["cost_usd"])

        return {
            "total_calls": len(call_log),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cost_usd": round(total_cost_usd, 6),
            "breakdown_by_type": breakdown,
        }
