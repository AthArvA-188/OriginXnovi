"""Shared bootstrap for the clog scripts.

The simulation scripts run in the cerebro_ml env (wntr, pyswmm), which has no pydantic. ``cascade.building``'s
__init__ imports the pydantic data model, so when pydantic is missing we register ``cascade.building`` as a bare
namespace package; ``cascade.building.clog`` itself needs only numpy, pandas and scikit-learn.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "clog"
SIMS = RAW / "sims"
EVAL = ROOT / "eval" / "clog"
MODELS = ROOT / "models" / "clog"
TMP = os.environ.get("TMP", str(ROOT / "runs" / "tmp"))


def bootstrap() -> None:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    here = str(Path(__file__).resolve().parent)
    pp = os.environ.get("PYTHONPATH", "")
    if here not in pp.split(os.pathsep):  # joblib/loky workers inherit this and can bootstrap themselves
        os.environ["PYTHONPATH"] = os.pathsep.join([here, src] + ([pp] if pp else []))
    if "cascade.building" not in sys.modules and importlib.util.find_spec("pydantic") is None:
        import cascade  # noqa: F401  (tiny __init__)

        pkg = types.ModuleType("cascade.building")
        pkg.__path__ = [str(ROOT / "src" / "cascade" / "building")]
        sys.modules["cascade.building"] = pkg
    for d in (RAW, SIMS, EVAL, MODELS):
        d.mkdir(parents=True, exist_ok=True)
