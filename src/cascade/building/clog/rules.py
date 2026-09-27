"""Rule and threshold table for the clog module, read from rules.json next to this file."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

RULES_PATH = Path(__file__).with_name("rules.json")
TAGS = ("PUBLIC", "team-proposed", "Assumption")


@lru_cache(maxsize=1)
def table() -> Dict[str, Any]:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def rule(key: str) -> Dict[str, Any]:
    """Return one rule row; KeyError if unknown."""
    return table()["rules"][key]


def value(key: str) -> Any:
    return rule(key)["value"]


def rows() -> list:
    """Flat rows for display: key, value, unit, tag, meaning, source, url."""
    out = []
    for k, r in table()["rules"].items():
        out.append({"key": k, "value": r["value"], "unit": r["unit"], "tag": r["tag"], "meaning": r["meaning"],
                    "source": r["source"], "url": r["url"]})
    return out
