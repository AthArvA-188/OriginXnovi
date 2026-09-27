"""Risk-based periodic-check schedule for the demo tower -> eval/clog/schedule.json.

The asset register is a SYNTHETIC demo (dates, backups, FOG probe readings are invented and labelled so). Strainer
and drain-down inputs are SIMULATED readings taken from this module's own held-out test scenarios
(eval/clog/riser_metrics.json strainer_demo, eval/clog/drain_test.json).

    python scripts/clog_schedule.py [--as-of 2026-09-26]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clog_common as C  # noqa: E402

C.bootstrap()

from cascade.building.clog import rules as R  # noqa: E402
from cascade.building.clog import schedule as S  # noqa: E402


def demo_assets(riser: dict, drain: dict) -> list:
    strl, strm = riser["strainer_demo"]["STR_L"], riser["strainer_demo"]["STR_M"]
    comm = drain["commissioning"]["drain_down_median_s"]
    by_open = {r["opening"]: r for r in drain["rows"]}
    clean_ratio = by_open[1.0]["drain_down_median_s"] / comm
    slow = by_open[0.2]
    slow_ratio = slow["drain_down_median_s"] / comm
    return [
        S.Asset("BF-1", "backflow", "Reduced-pressure backflow assembly, domestic water service", "2025-10-20"),
        S.Asset("GT-1", "grease_trap", "Grease trap under ground-floor bar sink", "2026-09-25", fse_permit=True),
        S.Asset("GI-1", "grease_interceptor", "Grease interceptor, ground-floor restaurant (FSE permit on file)",
                "2026-08-26", fse_permit=True,
                fog_readings=[("2026-08-29", 6.0), ("2026-09-05", 9.0), ("2026-09-12", 13.0), ("2026-09-19", 16.0),
                              ("2026-09-26", 19.0)]),
        S.Asset("STR-L", "prv_strainer", "Strainer at low-zone PRV station (F1)", "2026-05-01",
                nightly_alarm_persistent=bool(strl["act4_z_persistent_2of3"]),
                nightly_alarm_nights=int(strl["act4_z_alarm_nights"]), nightly_nights=int(strl["nights"]),
                extra_headloss_m=strl["median_extra_headloss_m_at_4lps"],
                provenance=f"SIMULATED nightly-test readings (test scenario {strl['scenario']}, true K "
                           f"{strl['true_K']:.0f})"),
        S.Asset("STR-M", "prv_strainer", "Strainer at mid-zone PRV station (F12)", "2026-06-15",
                nightly_alarm_persistent=bool(strm["act4_z_persistent_2of3"]),
                nightly_alarm_nights=int(strm["act4_z_alarm_nights"]), nightly_nights=int(strm["nights"]),
                extra_headloss_m=strm["median_extra_headloss_m_at_4lps"],
                provenance=f"SIMULATED nightly-test readings (test scenario {strm['scenario']}, true K "
                           f"{strm['true_K']:.0f})"),
        S.Asset("SB-A", "stack_base", "Stack A base cleanout (serves F2-F32)", "2026-06-01", floors_served=31,
                drain_down_ratio=clean_ratio,
                provenance="SIMULATED drain-down test (clean drain, opening 1.0)"),
        S.Asset("SB-B", "stack_base", "Stack B base cleanout (serves F2-F16)", "2026-02-10", floors_served=15,
                backups_12m=1, drain_down_ratio=slow_ratio, drain_down_censored=slow["censored_share"] > 0.5,
                provenance="SIMULATED drain-down test (partial clog, orifice opening 0.2)"),
        S.Asset("BD-1", "building_drain", "Building drain main cleanout", "2026-03-15", floors_served=32,
                backups_12m=1, last_backup="2026-09-25", provenance="SYNTHETIC backup ticket"),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2026-09-26")
    args = ap.parse_args(argv)
    riser = json.loads((C.EVAL / "riser_metrics.json").read_text(encoding="utf-8"))
    drain = json.loads((C.EVAL / "drain_test.json").read_text(encoding="utf-8"))
    assets = demo_assets(riser, drain)
    as_of = date.fromisoformat(args.as_of)
    rows = S.schedule(assets, as_of)
    out = {"as_of": args.as_of, "label": "Proposed schedule (advisory; every row requires human approval)",
           "register_label": "SYNTHETIC demo asset register; strainer and drain-down readings SIMULATED",
           "rows": rows, "register": [asdict(a) for a in assets],
           "rules": {k: R.rule(k) for k in sorted({k for r in rows for k in r["rule_keys"]})}}
    (C.EVAL / "schedule.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for r in rows:
        print(f"{r['next_check']}  {r['status']:20s} {r['asset_id']:6s} {r['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
