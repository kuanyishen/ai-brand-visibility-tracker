from .base import ProviderError, QuotaExhausted, Response
from .gemini import GeminiProvider

PROVIDERS = {"gemini": GeminiProvider}


def make_provider(name: str, model: str, **kwargs):
    try:
        return PROVIDERS[name](model, **kwargs)
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; available: {sorted(PROVIDERS)}") from None


__all__ = ["ProviderError", "QuotaExhausted", "Response", "GeminiProvider", "PROVIDERS", "make_provider"]
