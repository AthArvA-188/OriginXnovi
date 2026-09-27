"""Detector and evaluation helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from cascade.building.clog import detect as DT


def test_persistent_two_of_three():
    assert DT.persistent([0, 1, 0, 1, 0, 0, 0])
    assert not DT.persistent([1, 0, 0, 1, 0, 0, 1])
    assert DT.persistent([1, 1])


def test_macro_f1_from_confusion_matches_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
    cm = confusion_matrix(y, p, labels=[0, 1, 2, 3])
    assert abs(DT.macro_f1_from_cm(cm) - f1_score(y, p, average="macro")) < 1e-12


def test_zrule_locates_largest_deviation_and_threshold_from_clean():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (400, 5))
    y = np.zeros(400, int)
    X[300:, 3] += 8
    y[300:] = 2
    z = DT.ZRule().fit(X, y)
    assert (z.locate(X[300:]) == 3).all()
    thr = DT.threshold_at(z.maxz(X), y, 0.05)
    assert 0.03 < (z.maxz(X[y == 0]) > thr).mean() < 0.07


def test_alarm_rates_and_bootstrap_ci_by_scenario():
    rows = []
    for s in range(20):
        sev = 0 if s < 10 else 3
        for d in range(7):
            rows.append({"scenario": s, "day": d, "sev": sev, "flag": bool(sev and d != 0) or (s == 0 and d == 1)})
    df = pd.DataFrame(rows)
    out = DT.alarm_rates(df, "flag", n_boot=200)
    assert abs(out["3"]["rate"] - 6 / 7) < 1e-9 and out["3"]["n_scenarios"] == 10
    assert out["0"]["ci95"][0] <= out["0"]["rate"] <= out["0"]["ci95"][1]
    assert out["3"]["persistent_2of3"] == 1.0 and out["0"]["persistent_2of3"] == 0.0


def test_oof_scores_grouped_and_finite():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (240, 3))
    y = np.repeat([0, 1, 2, 3], 60)
    X[:, 0] += y
    groups = np.tile(np.arange(8), 30)
    oof = DT.oof_scores(X, y, groups, "z", n_splits=4)
    assert np.isfinite(oof).all()
