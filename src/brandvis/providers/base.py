"""Provider interface: one text answer per call, with the metadata needed to reproduce and audit it."""

from __future__ import annotations

from dataclasses import dataclass, field


class ProviderError(RuntimeError):
    """Raised when a call fails permanently (after retries)."""


class QuotaExhausted(ProviderError):
    """A daily/billing quota is used up: retrying today is pointless, so callers should stop the whole run."""


@dataclass
class Response:
    text: str
    model: str                          # model version reported by the API
    grounded: bool
    sources: list[str] = field(default_factory=list)       # cited web domains, in order
    search_queries: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None
    latency_s: float = 0.0
