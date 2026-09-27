"""Build the SYNTHETIC fire layer, check every plan card against the rules, and draw the three figures (ask 5 Part A).

Outputs (all derived from the synthetic 32-floor tower in cascade.building.synthetic; nothing here is real):
  eval/fire/fire_layer.json        stairs, elevators, FCC, battery room, design occupant load, assistance, impairments
  eval/fire/planner_checks.json    rule-conformance results for every floor x zone (room, core, shaft, electrical room),
                                   plus escalated cards (east office fire + a second alarm on every other floor)
  eval/fire/plan_card_F19-E.json   example card (and .md), plus an escalated example with a second alarm on F22
  docs/figures/fire_flowchart.svg, fire_section_F19.svg, fire_swimlane.svg

Usage: python scripts/fire_build_plans.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cascade.building import fire  # noqa: E402
from cascade.building.synthetic import TowerConfig, build_tower_model  # noqa: E402

OUT = ROOT / "eval" / "fire"
FIG = ROOT / "docs" / "figures"
EXAMPLE_ZONE = "F19-E"
ESCALATION = ["F22"]
ZONE_KINDS = ("room", "core", "shaft", "electrical_room")


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = TowerConfig()
    b = build_tower_model(cfg)
    layer = fire.synthetic_fire_layer(b)
    fire.save_fire_layer(layer, OUT / fire.LAYER_FILE)
    bhash = hashlib.sha256(b.model_dump_json().encode()).hexdigest()[:16]

    checked, violations, per_kind = 0, {}, {}
    for f in fire.occupied_floors(b):
        for z in b.zones_on(f.floor_id):
            if z.kind not in ZONE_KINDS:
                continue
            card = fire.plan_for_incident(b, layer, z.zone_id)
            bad = fire.check_card(b, layer, card)
            checked += 1
            per_kind[z.kind] = per_kind.get(z.kind, 0) + 1
            if bad:
                violations[z.zone_id] = bad
    # escalated cards: fire in every floor's east office, second alarm on every other floor (trigger E1)
    esc_checked, esc_viol, esc_moved = 0, {}, 0
    floors = fire.occupied_floors(b)
    for f in floors:
        for g in floors:
            if g.floor_id == f.floor_id:
                continue
            card = fire.plan_for_incident(b, layer, f"{f.floor_id}-E", detections=[g.floor_id])
            bad = fire.check_card(b, layer, card)
            esc_checked += 1
            rule_stage = card.level - int(fire.ft("FIRE_STAGING_BELOW"))
            esc_moved += int(card.staging_floor != (rule_stage if rule_stage >= 1 else None))
            if bad:
                esc_viol[f"{f.floor_id}-E+{g.floor_id}"] = bad
    rules = [{"key": k, "value": th.value, "unit": th.unit, "tag": th.tag, "url": th.url, "meaning": th.meaning, "note": th.note}
             for k, th in fire.fire_thresholds().items()]
    checks = {"generated_at": datetime.now(timezone.utc).isoformat(), "synthetic": True, "building_id": b.building_id,
              "building_hash": bhash, "tower_config": {"floors": cfg.floors, "seed": cfg.seed, "floor_height_m": cfg.floor_height_m},
              "cards_checked": checked, "cards_passing": checked - len(violations), "by_zone_kind": per_kind,
              "violations": violations,
              "escalated_scenario": "fire in each floor's east office plus a second alarm on each other floor",
              "escalated_cards_checked": esc_checked, "escalated_cards_passing": esc_checked - len(esc_viol),
              "escalated_staging_moved": esc_moved, "escalated_violations": esc_viol,
              "escalated_rules_checked": [
                  "alert set contains the floor above and below every alarming floor",
                  "staging is the first non-alerted floor at or below FIRE_STAGING_BELOW under the fire, never an alerted floor",
                  "a staging move below the sourced rule is tagged [team-proposed, validate] in the card line's detail and basis"],
              "rules_checked": [
                  "alert set contains the fire floor, the floor above and the floor below",
                  "staging exactly FIRE_STAGING_BELOW floors below the fire (exterior when that is below floor 1)",
                  "relocation floors at least FIRE_RELOCATE_MIN_BELOW below, never alert, staging or floor 1",
                  "evacuation stair differs from the attack stair",
                  "stair enclosures at least STAIR_MIN_SEP_FT apart at their nearest points",
                  "no line executes anything; every human decision names FSD or IC; AI role is evidence only",
                  "decision-support banner and SYNTHETIC flag present",
                  "firefighting-water watch only below the fire and ranked by path score",
                  "battery-room hazard always listed; 'near' exactly when within ESS_NEAR_FLOORS",
                  "fire department elevator only when the fire is at or above FIRE_LIFT_MIN_FIRE_FLOOR, exit FIRE_LIFT_EXIT_BELOW below",
                  "elevator recall stated as conditional on a lobby or hoistway detector",
                  "an impairment on an alert or relocation floor is the first hazard"],
              "rules": rules, "height_checks": layer.height_checks, "runtime_s": None}
    card = fire.plan_for_incident(b, layer, EXAMPLE_ZONE)
    esc = fire.plan_for_incident(b, layer, EXAMPLE_ZONE, detections=ESCALATION)
    (OUT / f"plan_card_{EXAMPLE_ZONE}.json").write_text(card.model_dump_json(indent=1), encoding="utf-8")
    (OUT / f"plan_card_{EXAMPLE_ZONE}.md").write_text(fire.plan_card_markdown(card), encoding="utf-8")
    (OUT / f"plan_card_{EXAMPLE_ZONE}_escalated.json").write_text(esc.model_dump_json(indent=1), encoding="utf-8")
    (FIG / "fire_flowchart.svg").write_text(fire.svg_flowchart(card), encoding="utf-8")
    (FIG / "fire_section_F19.svg").write_text(fire.svg_section(card, layer), encoding="utf-8")
    (FIG / "fire_swimlane.svg").write_text(fire.svg_swimlane(card), encoding="utf-8")
    checks["runtime_s"] = round(time.time() - t0, 2)
    (OUT / "planner_checks.json").write_text(json.dumps(checks, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"escalated cards {esc_checked}, passing {esc_checked - len(esc_viol)}, staging moved {esc_moved}")
    print(f"cards {checked}, passing {checked - len(violations)}, violations {len(violations)}; F19-E alert {card.alert_floors} "
          f"staging {card.staging_floor} relocation {card.relocation_floors}; escalated alert {esc.alert_floors}; {checks['runtime_s']} s")


if __name__ == "__main__":
    main()
