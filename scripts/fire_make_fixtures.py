"""Write the tiny SYNTHETIC fixtures used by tests/test_firesense.py (no network, no data/raw).

They mimic the column layout of the two Mendeley files (EN54 room and Industrial Hall) with two sensor nodes and a
few hand-placed episodes, so the tests can check episode segmentation (gap OR label change), the Hall id 13 -> 14
merge, feature causality, alarm run lengths, attribution and metric arithmetic. The values are invented; they are
not real measurements and are never used for any reported number.

Usage: python scripts/fire_make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "fire"

EN54_EPISODES = [  # (start, end, scenario_label, ternary_label)
    ("2022-07-04 01:00:00", "2022-07-04 01:20:00", "Wood", "Fire"),
    ("2022-07-04 01:20:10", "2022-07-04 01:30:00", "Deodorant", "Nuisance"),  # label change, no gap
    ("2022-07-04 01:50:00", "2022-07-04 02:00:00", "Ethanol", "Nuisance"),
    ("2022-07-04 02:12:00", "2022-07-04 02:20:00", "Ethanol", "Nuisance"),  # same label, 12 min gap
]
HALL_EPISODES = [  # (start, end, scenario_label, anomaly_scenario, fire, nuisance)
    ("2023-07-10 09:40:00", "2023-07-10 09:50:00", "Deodorant", 3, 0, 1),
    ("2023-07-10 10:10:00", "2023-07-10 10:20:00", "Ethanol", 5, 0, 0),
    ("2023-07-10 10:39:57", "2023-07-10 10:39:57", "Cable", 13, 1, 0),  # 1-row fragment
    ("2023-07-10 10:40:00", "2023-07-10 10:55:00", "Cable", 14, 1, 0),
]


def _series(t: pd.DatetimeIndex, rng: np.random.Generator, episodes, fire_like) -> dict:
    n = len(t)
    pm = 20 + rng.normal(0, 2, n)
    co = 0.1 + rng.normal(0, 0.03, n)
    voc = 0.5 + rng.normal(0, 0.05, n)
    for ep in episodes:
        s, e, lab = pd.Timestamp(ep[0]), pd.Timestamp(ep[1]), ep[2]
        m = (t >= s) & (t <= e)
        ramp = np.clip((t[m] - s).total_seconds().to_numpy() / 300.0, 0, 1)
        if fire_like(lab):
            pm[m] += 300 * ramp
            co[m] += 5 * ramp
        elif lab in ("Deodorant",):
            pm[m] += 150 * ramp
            voc[m] += 2 * ramp
        else:
            voc[m] += 3 * ramp
    return {"pm": pm, "co": co, "voc": voc}


def make_en54() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    # 30 min warm-up + 30 min clean background before the first episode, 65 min after the last (guard band 60 min)
    t = pd.date_range("2022-07-03 23:30:00", "2022-07-04 03:25:00", freq="10s")
    rows = []
    for k, sid in enumerate(("Sensorknoten0001", "Sensorknoten0002")):
        tt = t + pd.Timedelta(seconds=k)
        v = _series(tt, rng, EN54_EPISODES, lambda lab: lab == "Wood")
        scen = np.array(["Background"] * len(tt), dtype=object)
        tern = np.array(["Background"] * len(tt), dtype=object)
        for s, e, lab, ter in EN54_EPISODES:
            m = (tt >= pd.Timestamp(s)) & (tt <= pd.Timestamp(e) + pd.Timedelta(seconds=k))
            scen[m] = lab
            tern[m] = ter
        df = pd.DataFrame({
            "Date": [x.strftime("%Y-%m-%d %H:%M:%S.%f") + "+00:00" for x in tt], "Sensor_ID": sid,
            "CO2_Room": np.round(600 + rng.normal(0, 5, len(tt))), "CO_Room": np.round(v["co"], 2), "H2_Room": np.round(0.1 + rng.normal(0, 0.01, len(tt)), 2),
            "Humidity_Room": np.round(50 + rng.normal(0, 0.3, len(tt)), 1), "PM05_Room": np.round(v["pm"] * 0.6, 1), "PM100_Room": 0.0,
            "PM10_Room": np.round(v["pm"] * 0.2, 1), "PM25_Room": np.round(v["pm"] * 0.1, 1), "PM40_Room": 0.0,
            "PM_Room_Typical_Size": 0.45, "PM_Total_Room": np.round(v["pm"], 1), "Temperature_Room": np.round(25 + rng.normal(0, 0.05, len(tt)), 1),
            "UV_Room": 0.0, "VOC_Room_RAW": np.round(v["voc"], 2), "scenario_label": scen,
            "anomaly_label": np.where(scen == "Background", "Normal", "Anomaly"), "ternary_label": tern})
        rows.append(df)
    return pd.concat(rows).sort_values("Date", kind="stable").reset_index(drop=True)


def make_hall() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    t = pd.date_range("2023-07-10 08:30:00", "2023-07-10 11:00:00", freq="10s", tz="UTC")
    rows = []
    for k, sid in enumerate(("sensornode0006", "sensornode0007")):
        tt = t + pd.Timedelta(seconds=k)
        eps_utc = [(pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC"), lab, aid, f, nu) for s, e, lab, aid, f, nu in HALL_EPISODES]
        v = _series(tt, rng, [(s, e, lab) for s, e, lab, *_ in eps_utc], lambda lab: lab == "Cable")
        scen = np.array(["Background"] * len(tt), dtype=object)
        aid_col = np.full(len(tt), np.nan)
        fire = np.zeros(len(tt))
        nuis = np.zeros(len(tt))
        for s, e, lab, aid, f, nu in eps_utc:
            m = (tt >= s) & (tt <= e + pd.Timedelta(seconds=k))
            if aid == 13:  # one row only, on the first sensor
                m = np.zeros(len(tt), dtype=bool)
                if k == 0:
                    m[np.argmin(np.abs((tt - s).total_seconds().to_numpy()))] = True
            scen[m] = lab
            aid_col[m] = aid
            fire[m] = f
            nuis[m] = nu
        local = tt.tz_convert("Europe/Berlin")
        df = pd.DataFrame({
            "Date": [x.strftime("%Y-%m-%d %H:%M:%S%z")[:-2] + ":" + x.strftime("%z")[-2:] for x in local], "Sensor_ID": sid,
            "CO2_Room": np.round(700 + rng.normal(0, 5, len(tt))), "scenario_label": scen, "progress_label": "Ignition",
            "experiment_number": 0.0, "H2_Room": np.round(0.1 + rng.normal(0, 0.01, len(tt)), 2),
            "PM05_Room": np.round(v["pm"] * 0.6, 1), "PM100_Room": 0.0, "PM10_Room": np.round(v["pm"] * 0.2, 1),
            "PM25_Room": np.round(v["pm"] * 0.1, 1), "PM40_Room": 0.0, "PM_Room_Typical_Size": 0.5, "PM_Total_Room": np.round(v["pm"], 1),
            "VOC_Room_RAW": np.round(v["voc"], 2), "Valid_Experiment": 1.0, "fire": fire, "nuisance": nuis,
            "Temperature_Room": np.round(28 + rng.normal(0, 0.05, len(tt)), 1), "Humidity_Room": np.round(47 + rng.normal(0, 0.3, len(tt)), 1),
            "CO_Room": np.round(v["co"], 2), "anomaly_label": (~np.isnan(aid_col)).astype(float), "anomaly_scenario": aid_col})
        rows.append(df)
    return pd.concat(rows).sort_values("Date", kind="stable").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = make_en54()
    a.to_csv(OUT / "en54_tiny.csv", index=False)
    h = make_hall()
    h.to_csv(OUT / "hall_tiny.csv", index=False)
    (OUT / "README.md").write_text(
        "# Fire verifier test fixtures (SYNTHETIC)\n\nGenerated by `scripts/fire_make_fixtures.py`. Invented values in the column layout of the "
        "Mendeley EN54 room and Industrial Hall files, two sensor nodes each. Used only by `tests/test_firesense.py` to check "
        "segmentation, feature causality, attribution and metric arithmetic. Never used for a reported number.\n", encoding="utf-8")
    print("en54_tiny", len(a), "hall_tiny", len(h))


if __name__ == "__main__":
    main()
