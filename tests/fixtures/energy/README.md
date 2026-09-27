Tiny fixtures for tests/test_energy_*.py, cut from the real downloads in data/raw/energy/.
Rebuild and check them with `python scripts/train_energy_models.py fixtures` (writes to data/raw/energy/fixtures_check
and asserts the content equals these files; pass `--out tests/fixtures/energy` to overwrite them).
- combined_Room2/5.csv.gz: first 10 dates of ROBOD rooms 2 and 5, the 9 columns the tests use (CC BY 4.0, figshare 19234530)
- noaa_normals_*.csv: byte copies of the NOAA NCEI 1991-2020 normals for USW00093134 and USW00023174 (US government data)
- openmeteo_la_2025_gmt_slice.json: Open-Meteo archive 2025 (requested timezone=GMT), hours of the UTC dates 2025-01-01/02 and 2025-03-09/10 only (CC BY 4.0)
