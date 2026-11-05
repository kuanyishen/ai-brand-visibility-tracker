"""Load the study design (prompts, brands, surfaces) from config/*.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    intent: str          # discovery | alternative | versus | reputation
    topic: str = ""
    competitor: str | None = None


@dataclass(frozen=True)
class Brand:
    name: str
    type: str = "brand"
    focus: bool = False
    aliases: tuple[str, ...] = ()
    cs: tuple[str, ...] = ()   # case-sensitive aliases


@dataclass(frozen=True)
class Surface:
    id: str
    provider: str
    model: str
    grounded: bool = False
    runs: int | None = None      # repeats per prompt for this surface (default: study.runs_per_prompt)
    label: str = ""              # human-readable name for charts and the dashboard


@dataclass(frozen=True)
class Study:
    category: str
    focus_brand: str
    runs_per_prompt: int
    temperature: float
    surfaces: tuple[Surface, ...]
    extractor: dict = field(default_factory=dict)


def _load(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_prompts(path: Path = CONFIG / "prompts.yaml") -> list[Prompt]:
    items = [Prompt(**p) for p in _load(path)["prompts"]]
    ids = [p.id for p in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prompt ids")
    return items


def load_brands(path: Path = CONFIG / "brands.yaml") -> list[Brand]:
    out = []
    for b in _load(path)["brands"]:
        out.append(Brand(name=b["name"], type=b.get("type", "brand"), focus=bool(b.get("focus", False)),
                         aliases=tuple(b.get("aliases", [])) or ((b["name"],) if not b.get("cs") else ()),
                         cs=tuple(b.get("cs", []))))
    names = [b.name for b in out]
    if len(names) != len(set(names)):
        raise ValueError("duplicate brand names")
    return out


def load_study(path: Path = CONFIG / "study.yaml") -> Study:
    d = _load(path)
    return Study(category=d["category"], focus_brand=d["focus_brand"], runs_per_prompt=int(d["runs_per_prompt"]),
                 temperature=float(d["temperature"]), surfaces=tuple(Surface(**s) for s in d["surfaces"]),
                 extractor=d.get("extractor", {}))
