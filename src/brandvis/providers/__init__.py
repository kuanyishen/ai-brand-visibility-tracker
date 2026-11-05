from .base import ProviderError, QuotaExhausted, Response
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider

PROVIDERS = {"gemini": GeminiProvider, "anthropic": AnthropicProvider}


def make_provider(name: str, model: str, **kwargs):
    try:
        return PROVIDERS[name](model, **kwargs)
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; available: {sorted(PROVIDERS)}") from None


__all__ = ["ProviderError", "QuotaExhausted", "Response", "GeminiProvider", "AnthropicProvider", "PROVIDERS", "make_provider"]
