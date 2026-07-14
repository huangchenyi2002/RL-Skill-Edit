# src/client.py
"""
统一 OpenRouter API 客户端
所有模型调用（student + teacher）全部走这里
使用 OpenAI SDK，base_url 指向 OpenRouter
"""

import os
import threading
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class OpenRouterClient:
    """
    封装 OpenRouter 的所有 API 调用
    对应提案 §15 Cost Model 的成本追踪
    """

    def __init__(self, config: dict):
        self.config = config

        # ── 读取 API Key ──────────────────────────────
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "\n❌ OPENROUTER_API_KEY 未找到！\n"
                "请在项目根目录创建 .env 文件并写入：\n"
                "  OPENROUTER_API_KEY=sk-or-v1-xxxxx\n"
            )

        # ── 初始化 OpenAI 兼容客户端 ──────────────────
        # 关键：read 超时 45s；禁用 keep-alive 避免死锁
        # 每个并行 worker 用独立连接（避免连接池瓶颈）
        # 代理选择顺序：
        #   1. 环境变量 HTTPS_PROXY / HTTP_PROXY / ALL_PROXY
        #   2. config.openrouter.proxy
        #   3. 自动探测常见端口 (7890 本地, 7899 服务器)
        proxy_url = (
            os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
            or os.environ.get("ALL_PROXY")
            or os.environ.get("all_proxy")
            or config.get("openrouter", {}).get("proxy")
        )

        if not proxy_url:
            proxy_url = self._auto_detect_proxy()

        if proxy_url:
            print(f"  [Client] 使用代理: {proxy_url}")
        else:
            print("  [Client] 未使用代理 (直连)")

        http_client_kwargs = {
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
            "trust_env": True,  # 读取代理环境变量
        }
        if proxy_url:
            http_client_kwargs["proxy"] = proxy_url

        http_client = httpx.Client(**http_client_kwargs)
        self.client = OpenAI(
            api_key=api_key,
            base_url=config["openrouter"]["base_url"],
            http_client=http_client,
            max_retries=0,      # 禁用 SDK 内部重试
        )

        # ── 请求头（用于 OpenRouter 统计面板）─────────
        self.extra_headers = {
            "HTTP-Referer": os.environ.get(
                "OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.environ.get(
                "OPENROUTER_APP_NAME", "OSD-Experiment"),
        }

        # ── 成本追踪计数器 ────────────────────────────
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.total_cost_usd      = 0.0
        self.call_log            = []
        self._initialize_usage_lock()

    def _initialize_usage_lock(self) -> None:
        self._usage_lock = threading.Lock()

    def _record_call(self, usage_info: dict) -> None:
        with self._usage_lock:
            self.total_input_tokens += int(usage_info.get("input_tokens", 0))
            self.total_output_tokens += int(usage_info.get("output_tokens", 0))
            self.total_cost_usd += float(usage_info.get("cost_usd", 0.0))
            self.call_log.append(dict(usage_info))

    @staticmethod
    def _auto_detect_proxy() -> str:
        """
        自动探测常见的本地/服务器代理端口。
        返回可用的代理 URL 或 None。

        测试顺序：
          - 7890 (Clash 默认，本地)
          - 7899 (服务器上的自定义端口)
          - 10809 (V2Ray 默认)
          - 1080  (Shadowsocks 默认)
        """
        import socket
        candidates = [7890, 7899, 10809, 1080]
        for port in candidates:
            try:
                s = socket.socket(
                    socket.AF_INET,
                    socket.SOCK_STREAM)
                s.settimeout(0.3)
                r = s.connect_ex(
                    ("127.0.0.1", port))
                s.close()
                if r == 0:
                    return f"http://127.0.0.1:{port}"
            except Exception:
                continue
        return None

    # ─────────────────────────────────────────────────────────────
    # 核心调用方法
    # ─────────────────────────────────────────────────────────────

    def chat(
        self,
        model:       str,
        messages:    list[dict],
        system:      str = None,
        temperature: float = 0.0,
        max_tokens:  int = 2048,
        call_type:   str = "unknown",
        max_retries: int = 1,  # 保留参数只是为了签名兼容
        seed:         int = None,
    ) -> tuple[str, dict]:
        """
        发送 Chat 请求，返回 (response_text, usage_info)

        行为：
        - 一次调用，出错直接返回 ("", 空 usage)
        - 不重试、不等待、不打印错误日志
        - 让并行框架自己容错（继续跑其他 task）
        """
        # 构建完整 messages
        full_messages = []
        if system:
            full_messages.append(
                {"role": "system", "content": system})
        full_messages.extend(messages)

        empty_usage = {
            "model":         model,
            "call_type":     call_type,
            "input_tokens":  0,
            "output_tokens": 0,
            "total_tokens":  0,
            "cost_usd":      0.0,
            "ok":            False,
            "error_kind":    "empty_response",
            "error_message": "",
        }

        try:
            request = dict(
                model=model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=self.extra_headers,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=60.0,
                    write=10.0,
                    pool=10.0,
                ),
            )
            if seed is not None:
                request["seed"] = int(seed)
            response = self.client.chat.completions.create(**request)

            # 防御：response/choices/message/usage 都可能 None
            if not response or not response.choices:
                return "", empty_usage
            msg = response.choices[0].message
            text = (msg.content
                    if msg and msg.content
                    else "")
            usage = response.usage
            if usage is None:
                return text, empty_usage

            input_tokens  = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens  = usage.total_tokens
            cost = self._calc_cost(
                model, input_tokens, output_tokens)

            usage_info = {
                "model":         model,
                "call_type":     call_type,
                "input_tokens":  input_tokens,
                "output_tokens": output_tokens,
                "total_tokens":  total_tokens,
                "cost_usd":      cost,
                "ok":            bool(text),
                "error_kind":    None if text else "empty_response",
                "error_message": "",
            }
            self._record_call(usage_info)
            return text, usage_info

        except Exception as exc:
            failed = dict(empty_usage)
            failed["error_kind"] = type(exc).__name__
            failed["error_message"] = str(exc)[:500]
            self._record_call(failed)
            return "", failed

    # ─────────────────────────────────────────────────────────────
    # 成本计算与汇报
    # ─────────────────────────────────────────────────────────────

    def _calc_cost(
        self, model: str, input_t: int, output_t: int
    ) -> float:
        if not self.config.get(
                "cost_tracking", {}).get("enabled", False):
            return 0.0
        prices = self.config["cost_tracking"].get(
            "cost_per_1k_tokens", {})
        price  = prices.get(model, 0.005)
        return (input_t + output_t) / 1000.0 * price

    def cost_summary(self) -> dict:
        with self._usage_lock:
            call_log = list(self.call_log)
            total_input_tokens = self.total_input_tokens
            total_output_tokens = self.total_output_tokens
            total_cost_usd = self.total_cost_usd
        breakdown = {}
        for log in call_log:
            ct = log["call_type"]
            if ct not in breakdown:
                breakdown[ct] = {
                    "calls": 0, "tokens": 0, "cost_usd": 0.0}
            breakdown[ct]["calls"]    += 1
            breakdown[ct]["tokens"]   += log["total_tokens"]
            breakdown[ct]["cost_usd"] += log["cost_usd"]

        return {
            "total_calls":         len(call_log),
            "total_input_tokens":  total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens":        (total_input_tokens
                                    + total_output_tokens),
            "total_cost_usd":      round(total_cost_usd, 6),
            "breakdown_by_type":   breakdown,
        }

    def print_cost_summary(self):
        s = self.cost_summary()
        print("\n" + "─" * 45)
        print("📊  OpenRouter 成本汇总（对应提案 §15）")
        print("─" * 45)
        print(f"  总调用次数   : {s['total_calls']}")
        print(f"  总 Token 数  : {s['total_tokens']:,}")
        print(f"  估算总成本   : ${s['total_cost_usd']:.4f} USD")
        print("  分项明细：")
        for ct, info in s["breakdown_by_type"].items():
            print(f"    [{ct}]  "
                  f"调用 {info['calls']} 次  "
                  f"Token {info['tokens']:,}  "
                  f"${info['cost_usd']:.4f}")
        print("─" * 45)
