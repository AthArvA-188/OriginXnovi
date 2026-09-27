"""Riser design plausibility (every outlet <= 80 psi static) and the clog rule table."""
from cascade.building.clog import geometry as G
from cascade.building.clog import rules as R


def test_every_outlet_at_or_below_80_psi_static():
    rows = G.static_table()
    assert len(rows) == G.FLOORS
    assert R.value("UPC_608_2_PRV_80PSI") == 56.2
    assert max(r["static_m"] for r in rows) <= 56.2
    assert min(r["static_m"] for r in rows) > 14.0  # every floor keeps usable pressure


def test_prvs_feed_lower_zones_and_booster_feeds_top():
    assert G.ZONES["L"]["fed_by"] == "PRV" and G.ZONES["M"]["fed_by"] == "PRV"
    assert G.ZONES["H"]["fed_by"] == "booster"
    covered = sorted(f for z in G.ZONES.values() for f in range(z["floors"][0], z["floors"][1] + 1))
    assert covered == list(range(1, G.FLOORS + 1))


def test_location_pipes_partition_each_zone_riser():
    pipes = [p for loc in G.LOCATIONS if not loc.startswith("STR") for p in G.location_pipes(loc)]
    assert len(pipes) == len(set(pipes))
    assert G.location_pipes("STR_L") == ["STR_L"]
    assert G.location_pipes("L1") == ["P02", "P03", "P04", "P05", "P06"]


def test_k_bands_are_contiguous():
    b = R.value("K_BANDS")
    assert b["mild"][1] == b["moderate"][0] and b["moderate"][1] == b["severe"][0]
    assert G.severity_of_k(0.0, b) == 0
    assert G.severity_of_k(b["mild"][0], b) == 1
    assert G.severity_of_k(b["moderate"][0], b) == 2
    assert G.severity_of_k(b["severe"][0], b) == 3


def test_rule_table_tags_and_urls():
    for row in R.rows():
        assert row["tag"] in R.TAGS, row["key"]
        if row["tag"] == "PUBLIC":
            assert row["url"].startswith("https://"), row["key"]
    assert "40 mm" in R.rule("UPC_608_2_PRV_80PSI")["meaning"]  # the strainer exception is recorded
    assert R.value("BACKFLOW_ANNUAL") == 365
    assert "never fire" in R.rule("ACTIVE_TEST_LPS")["meaning"]
