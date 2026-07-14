from concurrent.futures import ThreadPoolExecutor

from src.client import OpenRouterClient


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
