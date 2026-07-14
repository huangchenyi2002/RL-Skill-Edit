import importlib.util
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest

from rl_skill_edit.adapters import openrouter
from rl_skill_edit.adapters.openrouter import OpenRouterClient


PROXY_ENV_VARIABLES = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)

UNRELATED_OPENAI_ENV = {
    "OPENAI_ADMIN_KEY": "hostile-admin",
    "OPENAI_ORG_ID": "hostile-organization",
    "OPENAI_PROJECT_ID": "hostile-project",
    "OPENAI_WEBHOOK_SECRET": "hostile-webhook",
    "OPENAI_CUSTOM_HEADERS": (
        "X-Injected: hostile-header\nAuthorization: Bearer hostile"
    ),
}


class RecordingCompletions:
    def __init__(self, *, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def _bare_client(*, response=None, error: Exception | None = None, config=None):
    client = OpenRouterClient.__new__(OpenRouterClient)
    client.config = config or {
        "cost_tracking": {
            "enabled": False,
            "cost_per_1k_tokens": {},
        }
    }
    client.extra_headers = {
        "HTTP-Referer": "http://localhost",
        "X-Title": "RL-Skill-Edit",
    }
    completions = RecordingCompletions(response=response, error=error)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client.total_input_tokens = 0
    client.total_output_tokens = 0
    client.total_cost_usd = 0.0
    client.call_log = []
    client._initialize_usage_lock()
    return client, completions


def _provider_response(
    *, text: str = "answer", input_tokens: int = 8, output_tokens: int = 2
):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def _patch_recording_http_transport(monkeypatch):
    captured: dict[str, object] = {}
    real_http_client = httpx.Client

    def handle_request(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "model-a",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "answer",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "total_tokens": 10,
                },
            },
        )

    def build_http_client(**kwargs):
        captured["http_kwargs"] = kwargs
        return real_http_client(
            transport=httpx.MockTransport(handle_request),
            timeout=kwargs["timeout"],
            limits=kwargs["limits"],
            trust_env=False,
        )

    monkeypatch.setattr(
        openrouter,
        "httpx",
        SimpleNamespace(
            Client=build_http_client,
            Limits=httpx.Limits,
            Timeout=httpx.Timeout,
        ),
    )
    return captured


def _set_hostile_openai_environment(monkeypatch):
    for name in PROXY_ENV_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    for name, value in UNRELATED_OPENAI_ENV.items():
        monkeypatch.setenv(name, value)


def _client_config():
    return {
        "openrouter": {"base_url": "https://openrouter.ai/api/v1"},
        "cost_tracking": {"enabled": False},
    }


def test_openrouter_adapter_module_exists():
    assert importlib.util.find_spec("rl_skill_edit.adapters.openrouter") is not None


def test_openrouter_adapter_exposes_client():
    assert hasattr(openrouter, "OpenRouterClient")


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for name in PROXY_ENV_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterClient(
            {
                "openrouter": {
                    "base_url": "https://openrouter.ai/api/v1",
                },
                "cost_tracking": {"enabled": False},
            }
        )


def test_unrelated_openai_environment_does_not_configure_client(monkeypatch):
    _set_hostile_openai_environment(monkeypatch)
    _patch_recording_http_transport(monkeypatch)

    client = OpenRouterClient(_client_config())

    for attribute in (
        "admin_api_key",
        "organization",
        "project",
        "webhook_secret",
    ):
        assert getattr(client.client, attribute, None) not in set(
            UNRELATED_OPENAI_ENV.values()
        )
    default_headers = getattr(client.client, "default_headers", {})
    assert "X-Injected" not in default_headers
    for name, value in UNRELATED_OPENAI_ENV.items():
        assert os.environ[name] == value


def test_unrelated_openai_environment_cannot_inject_request_headers(
    monkeypatch,
):
    _set_hostile_openai_environment(monkeypatch)
    captured = _patch_recording_http_transport(monkeypatch)
    client = OpenRouterClient(_client_config())

    text, usage = client.chat(
        model="model-a",
        messages=[{"role": "user", "content": "question"}],
    )

    request = captured["request"]
    assert isinstance(request, httpx.Request)
    assert text == "answer"
    assert usage["ok"] is True
    assert request.headers["Authorization"] == "Bearer test-openrouter-key"
    assert "X-Injected" not in request.headers
    assert "OpenAI-Organization" not in request.headers
    assert "OpenAI-Project" not in request.headers
    assert json.loads(request.content) == {
        "model": "model-a",
        "messages": [{"role": "user", "content": "question"}],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    for name, value in UNRELATED_OPENAI_ENV.items():
        assert os.environ[name] == value


def test_client_uses_explicit_proxy_env_and_disables_ambient_httpx_env(
    monkeypatch,
):
    for name in PROXY_ENV_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("HTTPS_PROXY", "http://environment-proxy:8080")
    fake_http_client = object()
    captured: dict[str, dict] = {}

    def build_http_client(**kwargs):
        captured["http"] = kwargs
        return fake_http_client

    monkeypatch.setattr(httpx, "Client", build_http_client)

    client = OpenRouterClient(
        {
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "proxy": "http://configured-proxy:8080",
            },
            "cost_tracking": {"enabled": False},
        }
    )

    assert captured["http"]["proxy"] == "http://environment-proxy:8080"
    assert captured["http"]["trust_env"] is False
    assert client.extra_headers == {
        "HTTP-Referer": "http://localhost",
        "X-Title": "RL-Skill-Edit",
    }
    assert client.cost_summary()["total_calls"] == 0


def test_chat_sends_exact_request_and_records_success_usage():
    client, completions = _bare_client(
        response=_provider_response(),
        config={
            "cost_tracking": {
                "enabled": True,
                "cost_per_1k_tokens": {"model-a": 0.5},
            }
        },
    )

    text, usage = client.chat(
        model="model-a",
        messages=[{"role": "user", "content": "question"}],
        system="system rule",
        temperature=0.25,
        max_tokens=99,
        call_type="student_rollout",
        seed=11,
    )

    assert completions.calls == [
        {
            "model": "model-a",
            "messages": [
                {"role": "system", "content": "system rule"},
                {"role": "user", "content": "question"},
            ],
            "temperature": 0.25,
            "max_tokens": 99,
            "extra_headers": client.extra_headers,
            "seed": 11,
        }
    ]
    assert text == "answer"
    assert usage == {
        "model": "model-a",
        "call_type": "student_rollout",
        "input_tokens": 8,
        "output_tokens": 2,
        "total_tokens": 10,
        "cost_usd": 0.005,
        "ok": True,
        "error_kind": None,
        "error_message": "",
    }
    assert client.cost_summary() == {
        "total_calls": 1,
        "total_input_tokens": 8,
        "total_output_tokens": 2,
        "total_tokens": 10,
        "total_cost_usd": 0.005,
        "breakdown_by_type": {
            "student_rollout": {
                "calls": 1,
                "tokens": 10,
                "cost_usd": 0.005,
            }
        },
    }


def test_provider_failure_is_recorded_without_retry():
    client, completions = _bare_client(error=RuntimeError("provider down"))

    text, usage = client.chat(
        model="model-a",
        messages=[{"role": "user", "content": "question"}],
        call_type="editor",
    )

    assert len(completions.calls) == 1
    assert "seed" not in completions.calls[0]
    assert text == ""
    assert usage["ok"] is False
    assert usage["error_kind"] == "RuntimeError"
    assert usage["error_message"] == "provider down"
    assert client.cost_summary()["total_calls"] == 1


@pytest.mark.parametrize(
    "response",
    (
        None,
        SimpleNamespace(choices=[], usage=None),
        SimpleNamespace(
            choices=[SimpleNamespace(message=None)],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            usage=None,
        ),
    ),
)
def test_incomplete_provider_response_fails_closed_and_is_recorded(response):
    client, completions = _bare_client(response=response)

    text, usage = client.chat(
        model="model-a",
        messages=[{"role": "user", "content": "question"}],
    )

    assert len(completions.calls) == 1
    assert text == ""
    assert usage["ok"] is False
    assert usage["error_kind"] == "empty_response"
    assert client.cost_summary()["total_calls"] == 1


@pytest.mark.parametrize(
    "choices",
    (
        {"unexpected": "mapping"},
        7,
        [None],
        [7],
    ),
)
def test_malformed_choices_fail_closed_and_preserve_usage(choices):
    response = SimpleNamespace(
        choices=choices,
        usage=SimpleNamespace(
            prompt_tokens=8,
            completion_tokens=2,
            total_tokens=10,
        ),
    )
    client, completions = _bare_client(response=response)

    text, usage = client.chat(
        model="model-a",
        messages=[{"role": "user", "content": "question"}],
    )

    assert len(completions.calls) == 1
    assert text == ""
    assert usage["ok"] is False
    assert usage["error_kind"] == "empty_response"
    assert usage["input_tokens"] == 8
    assert usage["output_tokens"] == 2
    assert usage["total_tokens"] == 10
    assert client.cost_summary()["total_tokens"] == 10


def test_empty_text_response_preserves_provider_usage():
    client, completions = _bare_client(
        response=_provider_response(text=""),
        config={
            "cost_tracking": {
                "enabled": True,
                "cost_per_1k_tokens": {"model-a": 0.5},
            }
        },
    )

    text, usage = client.chat(
        model="model-a",
        messages=[{"role": "user", "content": "question"}],
    )

    assert len(completions.calls) == 1
    assert text == ""
    assert usage["ok"] is False
    assert usage["error_kind"] == "empty_response"
    assert usage["input_tokens"] == 8
    assert usage["output_tokens"] == 2
    assert usage["total_tokens"] == 10
    assert usage["cost_usd"] == 0.005
    assert client.cost_summary()["total_tokens"] == 10


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "total_tokens"),
    (
        (True, 2, 3),
        ("8", 2, 10),
        (1.5, 2, 3),
        (8, 2, 999),
    ),
)
def test_invalid_provider_usage_fails_closed(input_tokens, output_tokens, total_tokens):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
        ),
    )
    client, completions = _bare_client(response=response)

    text, usage = client.chat(
        model="model-a",
        messages=[{"role": "user", "content": "question"}],
    )

    assert len(completions.calls) == 1
    assert text == ""
    assert usage["ok"] is False
    assert usage["error_kind"] == "invalid_usage"
    assert client.cost_summary()["total_tokens"] == 0


def test_missing_model_price_fails_closed_before_provider_request():
    client, completions = _bare_client(
        response=_provider_response(),
        config={
            "cost_tracking": {
                "enabled": True,
                "cost_per_1k_tokens": {},
            }
        },
    )

    text, usage = client.chat(
        model="unpriced-model",
        messages=[{"role": "user", "content": "question"}],
    )

    assert completions.calls == []
    assert text == ""
    assert usage["ok"] is False
    assert usage["error_kind"] == "configuration_error"
    assert client.cost_summary()["total_calls"] == 1


def test_unrepresentable_model_price_fails_closed_before_provider_request():
    client, completions = _bare_client(
        response=_provider_response(),
        config={
            "cost_tracking": {
                "enabled": True,
                "cost_per_1k_tokens": {"model-a": 10**400},
            }
        },
    )

    text, usage = client.chat(
        model="model-a",
        messages=[{"role": "user", "content": "question"}],
    )

    assert completions.calls == []
    assert text == ""
    assert usage["ok"] is False
    assert usage["error_kind"] == "configuration_error"
    assert client.cost_summary()["total_calls"] == 1
    assert client.cost_summary()["total_cost_usd"] == 0.0


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "price"),
    (
        pytest.param(10**400, 0, 1.0, id="token-float-overflow"),
        pytest.param(1000, 1000, 1e308, id="nonfinite-product"),
    ),
)
def test_nonfinite_calculated_cost_fails_closed_and_preserves_tokens(
    input_tokens, output_tokens, price
):
    client, completions = _bare_client(
        response=_provider_response(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        config={
            "cost_tracking": {
                "enabled": True,
                "cost_per_1k_tokens": {"model-a": price},
            }
        },
    )

    text, usage = client.chat(
        model="model-a",
        messages=[{"role": "user", "content": "question"}],
    )

    assert len(completions.calls) == 1
    assert text == ""
    assert usage["ok"] is False
    assert usage["error_kind"] == "invalid_cost"
    assert usage["input_tokens"] == input_tokens
    assert usage["output_tokens"] == output_tokens
    assert usage["total_tokens"] == input_tokens + output_tokens
    assert usage["cost_usd"] == 0.0
    summary = client.cost_summary()
    assert summary["total_calls"] == 1
    assert summary["total_tokens"] == input_tokens + output_tokens
    assert summary["total_cost_usd"] == 0.0


def test_nonfinite_accumulated_cost_marks_the_current_call_failed():
    client, completions = _bare_client(
        response=_provider_response(input_tokens=1000, output_tokens=1000),
        config={
            "cost_tracking": {
                "enabled": True,
                "cost_per_1k_tokens": {"model-a": 5e307},
            }
        },
    )

    first_text, first_usage = client.chat(
        model="model-a",
        messages=[{"role": "user", "content": "first"}],
    )
    second_text, second_usage = client.chat(
        model="model-a",
        messages=[{"role": "user", "content": "second"}],
    )

    assert len(completions.calls) == 2
    assert first_text == "answer"
    assert first_usage["ok"] is True
    assert second_text == ""
    assert second_usage["ok"] is False
    assert second_usage["error_kind"] == "invalid_cost"
    assert second_usage["total_tokens"] == 2000
    assert second_usage["cost_usd"] == 0.0
    summary = client.cost_summary()
    assert summary["total_calls"] == 2
    assert summary["total_tokens"] == 4000
    assert math.isfinite(summary["total_cost_usd"])


def test_parallel_usage_accounting_is_atomic():
    client = OpenRouterClient.__new__(OpenRouterClient)
    client.total_input_tokens = 0
    client.total_output_tokens = 0
    client.total_cost_usd = 0.0
    client.call_log = []
    client._initialize_usage_lock()

    def record(index: int) -> None:
        client._record_call(
            {
                "model": "mock",
                "call_type": "student_rollout",
                "input_tokens": 2,
                "output_tokens": 1,
                "total_tokens": 3,
                "cost_usd": 0.01,
                "ok": True,
                "request_index": index,
            }
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(record, range(1000)))

    summary = client.cost_summary()
    assert summary["total_calls"] == 1000
    assert summary["total_input_tokens"] == 2000
    assert summary["total_output_tokens"] == 1000
    assert summary["total_tokens"] == 3000
    assert summary["total_cost_usd"] == 10.0
    assert summary["breakdown_by_type"] == {
        "student_rollout": {
            "calls": 1000,
            "tokens": 3000,
            "cost_usd": pytest.approx(10.0),
        }
    }
    assert {entry["request_index"] for entry in client.call_log} == set(range(1000))
