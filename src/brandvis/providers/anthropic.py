"""Anthropic Claude via the Messages API (no SDK needed), optionally with the server-side web search tool.

Authentication: ``ANTHROPIC_API_KEY`` (or ``BRANDVIS_ANTHROPIC_KEY``) is sent as ``x-api-key``.
Cited web sources come from the answer's citations (the pages the answer actually relies on), reported as domains so they
line up with the Gemini surfaces.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .base import ProviderError, QuotaExhausted, Response

URL = "https://api.anthropic.com/v1/messages"
VERSION = "2023-06-01"
SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}
MAX_TOKENS = 2500


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def read_key() -> str:
    """The API key from the environment, with the whitespace and quotes that are easy to paste by accident removed."""
    key = (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("BRANDVIS_ANTHROPIC_KEY") or "").strip().strip("\"'").strip()
    if not key:
        raise ProviderError("no Anthropic API key: set ANTHROPIC_API_KEY (or BRANDVIS_ANTHROPIC_KEY)")
    return key


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, *, timeout: float = 180.0, max_retries: int = 6):
        self.model, self.timeout, self.max_retries = model, timeout, max_retries

    def _key(self) -> str:
        return read_key()

    def _post(self, body: dict) -> dict:
        headers = {"content-type": "application/json", "anthropic-version": VERSION, "x-api-key": self._key()}
        data = json.dumps(body).encode()
        last = ""
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(URL, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                last = e.read().decode(errors="replace")[:1500]
                flat = re.sub(r"\s+", " ", last)
                if e.code == 401:
                    raise ProviderError(f"Anthropic rejected the API key (401): {flat[:200]}") from None
                if e.code in (400, 402) and "credit balance" in last.lower():
                    raise QuotaExhausted(f"Anthropic credit balance too low: {flat[:200]}") from None
                if e.code not in RETRY_STATUS or attempt == self.max_retries:
                    raise ProviderError(f"HTTP {e.code}: {flat[:300]}") from None
                retry_after = e.headers.get("retry-after") if e.headers else None
                delay = float(retry_after) + 1 if retry_after and retry_after.replace(".", "").isdigit() else min(2.0 ** attempt, 60.0) * (0.75 + random.random() / 2)
                time.sleep(min(delay, 90.0))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = str(e)
                if attempt == self.max_retries:
                    raise ProviderError(f"network error: {last}") from None
                time.sleep(min(2.0 ** attempt, 60.0))
        raise ProviderError(last)

    def generate(self, prompt: str, *, temperature: float = 1.0, grounded: bool = False) -> Response:
        messages = [{"role": "user", "content": prompt}]
        body = {"model": self.model, "max_tokens": MAX_TOKENS, "temperature": temperature, "messages": messages}
        if grounded:
            body["tools"] = [SEARCH_TOOL]
        t0 = time.time()
        text_parts: list[str] = []
        urls: list[str] = []
        queries: list[str] = []
        usage = {"input_tokens": 0, "output_tokens": 0, "web_search_requests": 0}
        stop, model = None, self.model
        for _ in range(4):                                   # a long search turn may pause; resume it a few times
            d = self._post({**body, "messages": messages})
            model, stop = d.get("model", model), d.get("stop_reason")
            u = d.get("usage") or {}
            usage["input_tokens"] += u.get("input_tokens", 0)
            usage["output_tokens"] += u.get("output_tokens", 0)
            usage["web_search_requests"] += (u.get("server_tool_use") or {}).get("web_search_requests", 0)
            for block in d.get("content", []):
                kind = block.get("type")
                if kind == "text":
                    text_parts.append(block.get("text", ""))
                    for c in block.get("citations") or []:
                        if c.get("url") and c["url"] not in urls:
                            urls.append(c["url"])
                elif kind == "server_tool_use" and (block.get("input") or {}).get("query"):
                    queries.append(block["input"]["query"])
            if stop != "pause_turn":
                break
            messages = messages + [{"role": "assistant", "content": d["content"]}]
        text = "".join(text_parts).strip()
        if not text:
            raise ProviderError(f"empty answer (stop_reason={stop})")
        usage["totalTokenCount"] = usage["input_tokens"] + usage["output_tokens"]
        return Response(text=text, model=model, grounded=grounded, sources=[_domain(u) for u in urls],
                        search_queries=queries, usage=usage, finish_reason=stop, latency_s=round(time.time() - t0, 2))
