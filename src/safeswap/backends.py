"""OpenRouter chat client: text in, text + exact USD cost out, with a call cache and a spend cap.

Adapted from foveal's bench/openrouter.py (text only; no tools or images). OpenRouter reports
the exact charge per call in `usage.cost`, which is what every savings number is built on.
Cached replays report the original cost but do not count toward the spend cap.

Reads OPENROUTER_API_KEY from the environment (e.g. loaded from .env).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .callcache import CallCache, key

URL = "https://openrouter.ai/api/v1/chat/completions"


class SpendCapExceeded(RuntimeError):
    pass


@dataclass
class Completion:
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    truncated: bool = False  # stopped at max_tokens
    cached: bool = False


class OpenRouter:
    def __init__(
        self,
        api_key: str | None = None,
        cache: CallCache | None = None,
        spend_cap_usd: float = 5.0,
        timeout: float = 120.0,
        retries: int = 3,
    ):
        self._key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.cache = cache if cache is not None else CallCache()
        self.spend_cap_usd = spend_cap_usd
        self.spent_usd = 0.0
        self._spend_lock = threading.Lock()
        self.timeout = timeout
        self.retries = retries
        self._http: httpx.Client | None = None

    def _client(self) -> httpx.Client:
        with self._spend_lock:
            return self._client_locked()

    def _client_locked(self) -> httpx.Client:
        if self._http is None:
            if not self._key:
                raise RuntimeError("OPENROUTER_API_KEY is not set (add it to .env)")
            self._http = httpx.Client(
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self._key}", "X-Title": "safeswap"},
            )
        return self._http

    def complete(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **extra: Any,
    ) -> Completion:
        params = {"max_tokens": max_tokens, "temperature": temperature, **extra}
        k = key(model, messages, params)
        hit = self.cache.get(k)
        if hit is not None:
            return Completion(**{**hit, "cached": True})
        if self.spent_usd >= self.spend_cap_usd:  # checked before each call; can overshoot by
            # at most the calls already in flight
            raise SpendCapExceeded(
                f"spent ${self.spent_usd:.4f} of ${self.spend_cap_usd:.2f} cap; raise it in config"
            )
        t0 = time.perf_counter()
        data = self._post(
            {"model": model, "messages": messages, "usage": {"include": True}, **params}
        )
        latency = (time.perf_counter() - t0) * 1000
        u = data.get("usage") or {}
        if u.get("cost") is None:  # without a reported cost the spend cap can't work
            raise RuntimeError(f"OpenRouter returned no cost for {model}; stopping")
        out = Completion(
            model=data.get("model", model),
            text=data["choices"][0]["message"].get("content") or "",
            input_tokens=u.get("prompt_tokens") or 0,
            output_tokens=u.get("completion_tokens") or 0,
            cost_usd=float(u["cost"]),
            latency_ms=latency,
            truncated=data["choices"][0].get("finish_reason") == "length",
        )
        with self._spend_lock:
            self.spent_usd += out.cost_usd
        rec = out.__dict__.copy()
        rec.pop("cached")
        self.cache.put(k, rec)
        return out

    def _post(self, body: dict) -> dict:
        for attempt in range(self.retries + 1):
            r = self._client().post(URL, json=body)
            if r.status_code == 200:
                data = r.json()
                if "error" in data:  # some upstream errors arrive with HTTP 200
                    raise RuntimeError(f"OpenRouter error: {data['error']}")
                return data
            if r.status_code in (408, 429, 500, 502, 503, 504) and attempt < self.retries:
                time.sleep(2**attempt * 2)
                continue
            raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {r.text[:300]}")
        raise AssertionError("unreachable")
