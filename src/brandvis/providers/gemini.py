"""Google Gemini via the public REST API (no SDK needed).

Authentication: if GEMINI_API_KEY is set it is sent as ``x-goog-api-key``; otherwise no key header is added, which
works in environments where a proxy injects the credential.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request

from .base import ProviderError, QuotaExhausted, Response

BASE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_STATUS = {429, 500, 502, 503, 504}


def _retry_delay(body: str, attempt: int) -> float:
    """Honour the server's suggested delay (``retryDelay: "12s"``), else exponential backoff with jitter."""
    m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', body)
    if m:
        return min(float(m.group(1)) + 1.0, 90.0)
    return min(2.0 ** attempt, 60.0) * (0.75 + random.random() / 2)


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, *, timeout: float = 120.0, max_retries: int = 6):
        self.model, self.timeout, self.max_retries = model, timeout, max_retries

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if os.environ.get("GEMINI_API_KEY"):
            headers["x-goog-api-key"] = os.environ["GEMINI_API_KEY"]
        data = json.dumps(body).encode()
        last = ""
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(BASE.format(model=self.model), data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                last = e.read().decode(errors="replace")[:1500]
                if e.code == 429 and "PerDay" in last:
                    raise QuotaExhausted(f"daily quota exhausted for {self.model}: " + re.sub(r"\s+", " ", last)[:300]) from None
                if e.code not in RETRY_STATUS or attempt == self.max_retries:
                    raise ProviderError(f"HTTP {e.code}: {last}") from None
                time.sleep(_retry_delay(last, attempt))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = str(e)
                if attempt == self.max_retries:
                    raise ProviderError(f"network error: {last}") from None
                time.sleep(_retry_delay("", attempt))
        raise ProviderError(last)

    @staticmethod
    def _text(candidate: dict) -> str:
        return "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", []) if not p.get("thought"))

    def generate(self, prompt: str, *, temperature: float = 1.0, grounded: bool = False) -> Response:
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature}}
        if grounded:
            body["tools"] = [{"google_search": {}}]
        t0 = time.time()
        d = self._post(body)
        if not d.get("candidates"):
            raise ProviderError(f"no candidates (blocked?): {json.dumps(d.get('promptFeedback', d))[:300]}")
        c = d["candidates"][0]
        gm = c.get("groundingMetadata", {})
        sources = [g.get("web", {}).get("title", "") for g in gm.get("groundingChunks", []) if g.get("web")]
        return Response(text=self._text(c), model=d.get("modelVersion", self.model), grounded=grounded, sources=sources,
                        search_queries=gm.get("webSearchQueries", []), usage=d.get("usageMetadata", {}),
                        finish_reason=c.get("finishReason"), latency_s=round(time.time() - t0, 2))

    def generate_json(self, prompt: str, schema: dict) -> dict:
        """Deterministic structured output (used by the extractor)."""
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json", "responseSchema": schema}}
        d = self._post(body)
        if not d.get("candidates"):
            raise ProviderError("no candidates")
        try:
            return json.loads(self._text(d["candidates"][0]))
        except json.JSONDecodeError as e:
            raise ProviderError(f"invalid JSON from model: {e}") from None
