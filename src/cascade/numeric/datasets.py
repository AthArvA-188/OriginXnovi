"""REAL public datasets for the numeric backtests: BDG2 hourly meters and LEAD1.0 human anomaly labels.

Raw files live under data/raw/numeric/ (git-ignored). fetch() downloads a missing file and always checks its
sha256 against the value pinned here (hashes recorded on 2026-09-26 and matched to the research verifier's).

Licences (read 2026-09-26):
- BDG2: CC BY-SA. The LICENSE header says "Attribution-ShareAlike 4.0 Unported" but the body is the 3.0 Unported
  legal code, and GitHub reports NOASSERTION. Attribution and share-alike apply either way.
- LEAD1.0: no licence stated (GitHub API licence null; Kaggle terms not verified). Internal evaluation only: we
  publish aggregate metrics, never the labelled series.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import RAW_DIR, DataCard

ACCESSED = "2026-09-26"
BDG2_REPO = "https://github.com/buds-lab/building-data-genome-project-2"
BDG2_MEDIA = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/"
BDG2_LICENCE = ("CC BY-SA (LICENSE header says 4.0 Unported; body is the 3.0 Unported legal code; "
                "GitHub reports NOASSERTION). Attribution and share-alike apply.")
BDG2_CITATION = "Building Data Genome Project 2 (Miller et al. 2020), Scientific Data 7, 368, doi:10.1038/s41597-020-00712-x"
LEAD_REPO = "https://github.com/samy101/lead-dataset"
LEAD_LICENCE = "none stated (GitHub API licence null; Kaggle terms not verified): internal evaluation only"
LEAD_CITATION = "LEAD1.0 annotated energy-anomaly dataset, arXiv:2203.17256; data from github.com/samy101/lead-dataset"

FILES: Dict[str, Dict[str, object]] = {
    "bdg2_electricity": {"path": "bdg2/electricity_cleaned.csv", "url": BDG2_MEDIA + "meters/cleaned/electricity_cleaned.csv",
                         "sha256": "b6ffc9b4dfcefe5c753594730a08ae822b0d50fec6815abb8f185591e6c630a3", "bytes": 174977911},
    "bdg2_metadata": {"path": "bdg2/metadata.csv", "url": BDG2_MEDIA + "metadata/metadata.csv",
                      "sha256": "992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02", "bytes": 272024},
    "bdg2_weather": {"path": "bdg2/weather.csv", "url": BDG2_MEDIA + "weather/weather.csv",
                     "sha256": "a8189f1c6acdf3b9933a9e6354b8e7c1278cd56a7075929623a17565d44f04bd", "bytes": 19457782},
    "bdg2_water": {"path": "bdg2/water_cleaned.csv", "url": BDG2_MEDIA + "meters/cleaned/water_cleaned.csv",
                   "sha256": "cf3474e7d3ca89b7e04674ef80ed6d6248473f5ccd523b1624ad73625d5fb220", "bytes": 15394905},
    "lead_small": {"path": "lead/lead1.0-small.zip",
                   "url": "https://raw.githubusercontent.com/samy101/lead-dataset/main/data/lead1.0-small.zip",
                   "sha256": "b0e34c66c0c4ce780b7bb19b199d33e19210675f16be102a5ac32d6b22e39d8f", "bytes": 10027589},
}

# BDG2 sites whose meters are in Salesforce/GiftEvalPretrain (folders bdg-2_bear, bdg-2_fox, bdg-2_panther, bdg-2_rat,
# bull, cockatoo, hog; tree listed 2026-09-26). The amazon/chronos-2 card metadata lists GiftEvalPretrain among its
# datasets, so these sites are excluded from the test pool to avoid scoring a model on values it may have seen.
GIFT_EVAL_PRETRAIN_SITES = ("Bear", "Bull", "Cockatoo", "Fox", "Hog", "Panther", "Rat")
GIFT_EVAL_URL = "https://huggingface.co/datasets/Salesforce/GiftEvalPretrain"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def path_of(name: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / str(FILES[name]["path"])


def fetch(name: str, raw_dir: Path = RAW_DIR, download: bool = True) -> Path:
    """Local path of a pinned file, downloading it if missing. Raises on a sha256 mismatch."""
    spec = FILES[name]
    p = path_of(name, raw_dir)
    if not p.exists():
        if not download:
            raise FileNotFoundError(f"{p} missing; run: python -m cascade.numeric.datasets --fetch {name}")
        import requests

        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        with requests.get(str(spec["url"]), stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
        tmp.replace(p)
    got = sha256_of(p)
    if got != spec["sha256"]:
        raise ValueError(f"{p}: sha256 {got} != pinned {spec['sha256']}")
    return p


def data_card(name: str) -> DataCard:
    spec = FILES[name]
    if name.startswith("bdg2"):
        return DataCard(f"BDG2 {Path(str(spec['path'])).stem}", "REAL", str(spec["url"]), BDG2_LICENCE,
                        str(spec["sha256"]), ACCESSED, BDG2_CITATION)
    return DataCard("LEAD1.0-small", "REAL", str(spec["url"]), LEAD_LICENCE, str(spec["sha256"]), ACCESSED,
                    LEAD_CITATION)


def write_manifest(raw_dir: Path = RAW_DIR) -> Dict[str, object]:
    out = {}
    for name in FILES:
        p = path_of(name, raw_dir)
        out[name] = {**FILES[name], "present": p.exists(), "licence": data_card(name).licence, "accessed": ACCESSED}
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "MANIFEST.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def load_metadata(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(fetch("bdg2_metadata", raw_dir))


def load_electricity(columns: Optional[List[str]] = None, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    cols = None if columns is None else ["timestamp", *columns]
    df = pd.read_csv(fetch("bdg2_electricity", raw_dir), usecols=cols, parse_dates=["timestamp"])
    return df.set_index("timestamp").asfreq("h")


def flat_day_stats(s_pre: pd.Series, zero_window_start: str = "2017-01-01") -> Tuple[float, int, int]:
    """(share, count, days) of flat days from zero_window_start on: a calendar day whose readings span at most 0.1% of
    the series' median (the median of all of s_pre). A day with no reading counts as not flat but stays in the
    denominator."""
    z = s_pre[s_pre.index >= pd.Timestamp(zero_window_start)]
    span = z.resample("D").agg(lambda d: d.max() - d.min() if d.notna().any() else float("nan"))
    flat = span <= 1e-3 * max(1.0, float(s_pre.median()))
    return float(flat.mean()), int(flat.sum()), int(len(flat))


def select_offices(meta: pd.DataFrame, elec: pd.DataFrame, n: int = 20, seed: int = 0,
                   test_start: str = "2017-11-01", exclude_sites=GIFT_EVAL_PRETRAIN_SITES,
                   max_missing: float = 0.01, min_median_kwh: float = 5.0, max_zero: float = 0.01,
                   max_flat_days: float = 0.05, zero_window_start: str = "2017-01-01"
                   ) -> Tuple[List[str], Dict[str, object]]:
    """Draw n eligible office meters. Every eligibility test reads only data from before test_start.

    A flat day is a calendar day whose readings span at most 0.1% of the meter's pre-test median (a stuck or
    switched-off meter; BDG2 'cleaned' files contain months of constant near-zero values for some meters)."""
    t0 = pd.Timestamp(test_start)
    offices = meta.loc[meta.primaryspaceusage == "Office", ["building_id", "site_id"]]
    offices = offices[offices.building_id.isin(elec.columns)]
    pre = elec[elec.index < t0]
    eligible_all, eligible = [], []
    for b, site in sorted(zip(offices.building_id, offices.site_id)):
        s = pre[b]
        if s.isna().mean() > max_missing or s.median() < min_median_kwh:
            continue
        z = s[s.index >= pd.Timestamp(zero_window_start)]
        if (z == 0).mean() > max_zero:
            continue
        if flat_day_stats(s, zero_window_start)[0] > max_flat_days:
            continue
        eligible_all.append(b)
        if site not in exclude_sites:
            eligible.append(b)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(eligible, size=min(n, len(eligible)), replace=False).tolist())
    info = {"n_office_meters": int(len(offices)), "n_eligible_before_exclusion": len(eligible_all),
            "n_eligible_after_exclusion": len(eligible), "excluded_sites": list(exclude_sites),
            "exclusion_reason": f"sites present in {GIFT_EVAL_URL} (possible pretraining overlap)",
            "rule": {"primaryspaceusage": "Office", "max_missing_pre_test": max_missing,
                     "min_median_kwh_pre_test": min_median_kwh, "max_zero_share": max_zero,
                     "max_flat_day_share": max_flat_days, "flat_day": "daily max-min <= 0.1% of pre-test median",
                     "zero_window": [zero_window_start, test_start], "seed": seed, "n": n},
            "chosen": chosen, "site_of": {b: s for b, s in zip(offices.building_id, offices.site_id) if b in chosen}}
    return chosen, info


def selection_diagnostics(meta: pd.DataFrame, elec: pd.DataFrame, n: int = 20, seed: int = 0,
                          test_start: str = "2017-11-01", exclude_sites=GIFT_EVAL_PRETRAIN_SITES,
                          max_missing: float = 0.01, min_median_kwh: float = 5.0, max_zero: float = 0.01,
                          max_flat_days: float = 0.05, alt_flat_limits=(0.01,), zero_window_start: str = "2017-01-01",
                          near_constant_window=("2017-04-01", "2017-10-31")) -> Dict[str, object]:
    """Numbers behind the disclosed flat-day rule, all from pre-test data: pool sizes with no flat-day rule, at the
    chosen limit and at the alternative limit(s); each meter the rule removes (flat-day share and count, and its
    min / max reading in near_constant_window); the meters an alternative limit would also remove; and the draw
    (same seed) that the selection gives without the flat-day rule."""
    t0 = pd.Timestamp(test_start)
    offices = meta.loc[meta.primaryspaceusage == "Office", ["building_id", "site_id"]]
    offices = offices[offices.building_id.isin(elec.columns)]
    pre = elec[elec.index < t0]
    rows = []
    for b, site in sorted(zip(offices.building_id, offices.site_id)):
        s = pre[b]
        z = s[s.index >= pd.Timestamp(zero_window_start)]
        other_ok = (not s.isna().mean() > max_missing) and (not s.median() < min_median_kwh)             and (not (z == 0).mean() > max_zero)
        share, cnt, days = flat_day_stats(s, zero_window_start) if other_ok else (float("nan"), 0, 0)
        rows.append({"meter": b, "site": site, "other_rules_ok": bool(other_ok), "flat_share": share,
                     "flat_days": cnt, "days": days, "excluded_site": site in exclude_sites})
    t = pd.DataFrame(rows)
    ok = t[t.other_rules_ok]

    def pool(limit):
        keep = ok if limit is None else ok[~(ok.flat_share > limit)]
        return {"before_exclusion": int(len(keep)), "after_exclusion": int((~keep.excluded_site).sum())}

    w0, w1 = pd.Timestamp(near_constant_window[0]), pd.Timestamp(near_constant_window[1]) + pd.Timedelta(hours=23)
    removed = ok[ok.flat_share > max_flat_days]
    removed_meters = {r.meter: {"site": r.site, "flat_day_share": round(float(r.flat_share), 4),
                                "flat_days": int(r.flat_days), "days": int(r.days),
                                "in_excluded_site": bool(r.excluded_site),
                                "min_kwh_in_window": float(pre[r.meter].loc[w0:w1].min()),
                                "max_kwh_in_window": float(pre[r.meter].loc[w0:w1].max())}
                      for r in removed.itertuples()}
    alt = {}
    for lim in alt_flat_limits:
        extra = ok[(ok.flat_share > lim) & ~(ok.flat_share > max_flat_days) & ~ok.excluded_site]
        alt[str(lim)] = {**pool(lim), "also_removed_after_exclusion": {
            r.meter: {"site": r.site, "flat_days": int(r.flat_days), "flat_day_share": round(float(r.flat_share), 4)}
            for r in extra.itertuples()}}
    first, _ = select_offices(meta, elec, n=n, seed=seed, test_start=test_start, exclude_sites=exclude_sites,
                              max_missing=max_missing, min_median_kwh=min_median_kwh, max_zero=max_zero,
                              max_flat_days=1.0, zero_window_start=zero_window_start)
    return {"flat_day_rule": f"share of days {zero_window_start} to {test_start} (exclusive) whose max - min <= 0.1% "
                             f"of the pre-test median; limit {max_flat_days}",
            "pool_without_flat_rule": pool(None), "pool_at_chosen_limit": pool(max_flat_days),
            "removed_by_flat_rule": removed_meters, "near_constant_window": list(near_constant_window),
            "alternative_limits": alt, "draw_without_flat_rule_same_seed": first}


def load_lead(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    import zipfile

    with zipfile.ZipFile(fetch("lead_small", raw_dir)) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(z.open(name), parse_dates=["timestamp"])
    return df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Fetch and verify the numeric datasets (BDG2, LEAD1.0-small)")
    ap.add_argument("--fetch", nargs="*", default=list(FILES), help=f"any of {list(FILES)}")
    args = ap.parse_args(argv)
    for name in args.fetch or list(FILES):
        print(name, fetch(name))
    print(json.dumps(write_manifest(), indent=1))


if __name__ == "__main__":
    main()
