"""LA electricity tariff and grid-event windows used to price common-area policies (TARIFF APPROXIMATION).

LADWP Schedule A-2 Rate B base charges as posted on 2026-09-26 [ASSUMPTION: rate choice; a tower's actual schedule may
differ]. The 2026 adjustment factors (ECA, VEA, CRPSEA, VRPSEA, IRCA, ...) are listed separately by LADWP and their
applicability to A-2 is unverified, so they are EXCLUDED by default and every $ figure understates a real bill.
The LADWP TOU page does not list holidays as Base, so holidays are NOT mapped to Base for LADWP.
SCE windows are given without prices. Demand savings are a coincident-peak estimate.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

LADWP_RATES_URL = "https://www.ladwp.com/account/customer-service/electric-rates/standard-commercial-industrial-rates"
LADWP_TOU_URL = "https://www.ladwp.com/account/understanding-your-rates/commercial-electric-rates"
LADWP_ADJ_URL = "https://www.ladwp.com/account/customer-service/electric-rates/commercial-adjustment-billing-factors"
LADWP_DR_URL = "https://www.ladwp.com/commercial-services/programs-and-rebates-commercial/demand-response-program"
SCE_TOU_URL = "https://www.sce.com/business/rates-financing/rate-plans/business-time-of-use-rate-plans"
SCE_SUMMARY_URL = ("https://www.sce.com/sites/default/files/custom-files/"
                   "Summary%20of%20Available%20Residential%20and%20Nonresidential%20Rate%20Options.pdf")
FLEX_URL = "https://www.flexalert.org/what-is-flex-alert"

LADWP_A2B: Dict = {
    "name": "LADWP A-2 Rate B (TOU), base charges",
    "accessed": "2026-09-26",
    "urls": [LADWP_TOU_URL, LADWP_RATES_URL],
    "high_season_months": [6, 7, 8, 9],
    "periods_weekday": {"high_peak": [[13, 17]], "low_peak": [[10, 13], [17, 20]]},
    "weekend_period": "base",
    "holidays_as_base": False,
    "energy_usd_per_kwh": {"high": {"high_peak": 0.06322, "low_peak": 0.05595, "base": 0.03522},
                           "low": {"high_peak": 0.05688, "low_peak": 0.05688, "base": 0.03895}},
    "demand_usd_per_kw": {"high": {"high_peak": 10.00, "low_peak": 3.75}, "low": {"high_peak": 4.75, "low_peak": None}},
    "facilities_usd_per_kw": 5.36,
    "service_usd_per_month": 20.0,
    "notes": ["low-season low-peak demand charge is not listed in our source; treated as 0",
              "adjustment factors excluded (applicability to A-2 unverified): " + LADWP_ADJ_URL,
              "demand is estimated as the maximum 15-min mean kW of the simulated common-area load (coincident-peak estimate)"],
}

SCE_TOU: Dict = {
    "name": "SCE business TOU windows (no prices)", "accessed": "2026-09-26", "urls": [SCE_TOU_URL, SCE_SUMMARY_URL],
    "summer": "Jun 1 - Sep 30", "summer_months": [6, 7, 8, 9], "on_peak_weekday": [16, 21],
    "cpp": "12 events a year, weekdays 16:00-21:00",
}

LADWP_DR: Dict = {
    "name": "LADWP commercial Demand Response (2025 season terms)", "accessed": "2026-09-26", "urls": [LADWP_DR_URL],
    "season": ["06-15", "10-15"], "window": [13, 21], "event_hours": 4, "max_events": 20, "min_curtail_kw": 100,
    "pay_usd_per_kw_month": {"day_ahead": 10.0, "two_hour": 15.0}, "pay_usd_per_kwh": 0.25,
}

FLEX_ALERT: Dict = {"name": "CAISO Flex Alert (voluntary)", "urls": [FLEX_URL], "window": [16, 21],
                    "action": "turn off unnecessary lights", "accessed": "2026-09-26"}


def _idx(ts) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(ts)


def ladwp_season(ts) -> np.ndarray:
    ix = _idx(ts)
    return np.where(np.isin(ix.month, LADWP_A2B["high_season_months"]), "high", "low")


def ladwp_period(ts) -> np.ndarray:
    """'high_peak' | 'low_peak' | 'base' for local clock timestamps (weekends are base; holidays are NOT mapped)."""
    ix = _idx(ts)
    h = ix.hour
    wk = ix.dayofweek < 5
    out = np.full(len(ix), "base", dtype=object)
    for name, spans in LADWP_A2B["periods_weekday"].items():
        m = np.zeros(len(ix), dtype=bool)
        for a, b in spans:
            m |= (h >= a) & (h < b)
        out[wk & m] = name
    return out


def energy_rate(ts) -> np.ndarray:
    s, p = ladwp_season(ts), ladwp_period(ts)
    tab = LADWP_A2B["energy_usd_per_kwh"]
    return np.array([tab[a][b] for a, b in zip(s, p)], dtype=float)


def energy_cost(kwh: pd.Series) -> float:
    """$ energy charge for a kWh series indexed by local clock timestamps."""
    return float((kwh.to_numpy() * energy_rate(kwh.index)).sum())


def demand_charges(kw: pd.Series, *, interval: str = "15min") -> Dict[str, float]:
    """Monthly demand and facilities charges from a kW series (local clock index), coincident-peak estimate."""
    k = kw.resample(interval).mean().dropna()
    per = ladwp_period(k.index)
    sea = ladwp_season(k.index)
    tab = LADWP_A2B["demand_usd_per_kw"]
    total_dem = 0.0
    total_fac = 0.0
    for _, g in k.groupby(k.index.to_period("M")):
        m_per = ladwp_period(g.index)
        season = ladwp_season(g.index[:1])[0]
        for p in ("high_peak", "low_peak"):
            rate = tab[season].get(p)
            sel = g[m_per == p]
            if rate and len(sel):
                total_dem += rate * float(sel.max())
        total_fac += LADWP_A2B["facilities_usd_per_kw"] * float(g.max())
    del per, sea
    return {"demand_usd": total_dem, "facilities_usd": total_fac}


def is_summer_high_peak(ts) -> np.ndarray:
    return (ladwp_season(ts) == "high") & (ladwp_period(ts) == "high_peak")


def tou_hour_table() -> pd.DataFrame:
    """Hour-of-day windows (weekday) for the page chart: LADWP periods by season, SCE on-peak, DR and Flex Alert."""
    rows = []
    for h in range(24):
        ts_s = pd.Timestamp(2027, 7, 14, h)  # a summer Wednesday
        ts_w = pd.Timestamp(2027, 1, 13, h)  # a winter Wednesday
        rows.append({
            "hour": h,
            "ladwp_summer_period": ladwp_period([ts_s])[0], "ladwp_summer_usd_kwh": float(energy_rate([ts_s])[0]),
            "ladwp_winter_period": ladwp_period([ts_w])[0], "ladwp_winter_usd_kwh": float(energy_rate([ts_w])[0]),
            "sce_summer_on_peak": bool(SCE_TOU["on_peak_weekday"][0] <= h < SCE_TOU["on_peak_weekday"][1]),
            "ladwp_dr_window": bool(LADWP_DR["window"][0] <= h < LADWP_DR["window"][1]),
            "flex_alert_window": bool(FLEX_ALERT["window"][0] <= h < FLEX_ALERT["window"][1]),
        })
    return pd.DataFrame(rows)


def tariff_artifact() -> Dict:
    return {"ladwp_a2b": LADWP_A2B, "sce_tou": SCE_TOU, "ladwp_dr": LADWP_DR, "flex_alert": FLEX_ALERT,
            "hour_table": tou_hour_table().to_dict(orient="records"), "label": "TARIFF APPROXIMATION"}


def rate_lookup(season: str, period: str) -> Optional[float]:
    return LADWP_A2B["energy_usd_per_kwh"][season][period]
