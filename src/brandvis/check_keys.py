"""Free check that the API keys work and that the models in config/study.yaml exist. Never prints a key.

    python -m brandvis.check_keys
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .config import load_study
from .providers.anthropic import VERSION, read_key
from .providers.base import ProviderError


def shape(key: str) -> str:
    """Describe a key without revealing it: length, expected prefix, stray characters."""
    raw = os.environ.get("ANTHROPIC_API_KEY", "")
    notes = [f"length {len(key)}", "starts with sk-ant-" if key.startswith("sk-ant-") else "does NOT start with sk-ant-"]
    if raw != raw.strip():
        notes.append("secret had leading/trailing whitespace (removed)")
    if any(c in raw for c in "\"'"):
        notes.append("secret contained quote characters (removed)")
    if any(c.isspace() for c in key):
        notes.append("key contains whitespace inside: probably two things were pasted")
    if "..." in key or "…" in key:
        notes.append("key contains '...': a shortened display copy was pasted instead of the full key")
    return ", ".join(notes)


def _get(url: str, headers: dict) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except ValueError:
            return e.code, {}
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, {"error": {"message": str(e)}}


def check_anthropic(wanted: set[str]) -> bool:
    try:
        key = read_key()
    except ProviderError as e:
        print(f"Anthropic: {e}")
        return False
    print(f"Anthropic key shape: {shape(key)}")
    status, body = _get("https://api.anthropic.com/v1/models?limit=100", {"x-api-key": key, "anthropic-version": VERSION})
    if status != 200:
        print(f"Anthropic: HTTP {status}: {(body.get('error') or {}).get('message', body)}")
        return False
    ids = {m["id"] for m in body.get("data", [])}
    print(f"Anthropic: key accepted; {len(ids)} models available")
    for m in sorted(ids):
        print("   ", m)
    missing = sorted(wanted - ids)
    if missing:
        print(f"Anthropic: configured model(s) NOT available to this key: {missing}")
    return not missing


def check_gemini(wanted: set[str]) -> bool:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    headers = {"x-goog-api-key": key} if key else {}
    status, body = _get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", headers)
    if status != 200:
        print(f"Gemini: HTTP {status}: {(body.get('error') or {}).get('message', body)}")
        return False
    ids = {m["name"].split("/")[-1] for m in body.get("models", [])}
    print(f"Gemini: key accepted; {len(ids)} models available")
    missing = sorted(wanted - ids)
    if missing:
        print(f"Gemini: configured model(s) NOT available: {missing}")
    return not missing


def main() -> None:
    study = load_study()
    wanted = {"anthropic": set(), "gemini": set()}
    for s in study.surfaces:
        wanted[s.provider].add(s.model)
    wanted["gemini"] |= {m for m in (study.extractor.get("model"), study.extractor.get("fallback_model")) if m}
    results = {}
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("BRANDVIS_ANTHROPIC_KEY"):
        results["anthropic"] = check_anthropic(wanted["anthropic"])
    else:
        print("Anthropic: ANTHROPIC_API_KEY is not set (skipped)")
    if os.environ.get("GEMINI_API_KEY"):
        results["gemini"] = check_gemini(wanted["gemini"])
    else:
        print("Gemini: GEMINI_API_KEY is not set (skipped)")
    print("RESULT:", results or "no keys found")
    raise SystemExit(0 if results and all(results.values()) else 1)


if __name__ == "__main__":
    main()
