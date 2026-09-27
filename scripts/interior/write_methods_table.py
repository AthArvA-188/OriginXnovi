"""Write eval/interior/methods_table.json and eval/interior/sources.json for the interior page.

The methods table summarises what each interior wall test captures, the data it produces, whether
we found a public labelled dataset, and how Cerebro uses it. Statements come from the cited sources
(read 2026-09-26 by the research and verify agents); the page reads this file verbatim.
    python scripts/interior/write_methods_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACC = "2026-09-26"

SOURCES = {
    "matthews_db": {"title": "Matthews, Allaix, Wijte, Vullings: Non-destructive Estimation of Concrete Compressive Strength: Databases (Zenodo 15392443)",
                    "url": "https://zenodo.org/records/15392443", "licence": "CC BY 4.0", "accessed": ACC},
    "matthews_paper": {"title": "Matthews et al. 2026, Advancing non-destructive concrete compressive strength estimation, NDT&E International 158, 103549 (TNO full text)",
                       "url": "https://publications.tno.nl/publication/34645091/4b25FibC/matthews-2026-advancing.pdf", "licence": "CC BY 4.0", "accessed": ACC},
    "bam_data": {"title": "Gebauer et al., Interrelated Data Set from Nondestructive and Destructive Material Testing of Concrete Compressive Strength Specimens (Harvard Dataverse)",
                 "url": "https://doi.org/10.7910/DVN/AFCITK", "licence": "CC0 1.0 (Dataverse metadata)", "accessed": ACC},
    "bam_paper": {"title": "Gebauer et al. 2023, Data in Brief (PMC10196957)", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10196957/",
                  "licence": "CC BY 4.0 article", "accessed": ACC},
    "tampere": {"title": "Mikkonen and Vinha, Hygrothermal measurements of timber-framed wall structures (Zenodo 17778563)",
                "url": "https://zenodo.org/records/17778563", "licence": "CC BY 4.0", "accessed": ACC},
    "irt_moisture": {"title": "Comparison of IRT and other techniques to assess moisture content of wall specimens (Sensors, PMC9101659)",
                     "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9101659/", "licence": "CC BY 4.0 article", "accessed": ACC},
    "irt_gpr": {"title": "IRT and GPR techniques for moisture detection in buildings (Sensors, PMC7696806)",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7696806/", "licence": "CC BY 4.0 article", "accessed": ACC},
    "protimeter": {"title": "Protimeter: comparing pinless vs pin moisture meters",
                   "url": "https://blog.protimeter.com/blog/comparing-pinless-moisture-meters-vs.-pin-moisture-meters", "licence": "vendor page", "accessed": ACC},
    "epa_sampling": {"title": "EPA: Mold testing or sampling", "url": "https://www.epa.gov/mold/mold-testing-or-sampling", "licence": "US government page", "accessed": ACC},
    "epa_guide": {"title": "EPA: A brief guide to mold, moisture and your home", "url": "https://www.epa.gov/mold/brief-guide-mold-moisture-and-your-home",
                  "licence": "US government page", "accessed": ACC},
    "fhwa_ndt": {"title": "FHWA SHRP2 R06A: Nondestructive Testing for Concrete Bridge Decks",
                 "url": "https://www.fhwa.dot.gov/goshrp2/Solutions/Construction/R06A/Nondestructive_Testing_for_Concrete_Bridge_Decks",
                 "licence": "US government page", "accessed": ACC},
    "usda_wood": {"title": "USDA FPL Wood Handbook chapter 14 (source of WOOD_MC_MAX in the shared threshold table)",
                  "url": "https://www.fpl.fs.usda.gov/documnts/fplgtr/fplgtr282/chapter_14_fpl_gtr282.pdf", "licence": "US government", "accessed": ACC},
}

METHODS = [
    {"method": "Pin moisture meter", "captures": "electrical resistance between two pins pushed into the material",
     "data": "one number on a meter-specific scale (%MC for wood, WME, or a relative 0-100 scale)",
     "public_labelled_data": "none found", "cerebro": "rule-graded: wood %MC against WOOD_MC_MAX; other scales need a dry reference on the same meter, else U",
     "sources": ["protimeter", "usda_wood"]},
    {"method": "Pinless (capacitance / RF) meter", "captures": "dielectric response a short depth under the surface",
     "data": "relative reading; not comparable across meters or materials",
     "public_labelled_data": "none found", "cerebro": "rule-graded as U unless compared with a dry reference on the same meter",
     "sources": ["protimeter"]},
    {"method": "Infrared thermography", "captures": "surface temperature; wet areas can show as cooler patches",
     "data": "thermograms and temperature differences; depends on the indoor-outdoor gradient; does not measure moisture content directly",
     "public_labelled_data": "no open interior-moisture thermal set found (BFDD IR is exterior)", "cerebro": "photo grading with the interior_water rubric (image rows); ML not trained",
     "sources": ["irt_moisture", "irt_gpr"]},
    {"method": "RH / temperature probes and sensors", "captures": "air or in-wall relative humidity and temperature over time",
     "data": "time series (%RH, degC)", "public_labelled_data": "Tampere timber-wall lab measurements (CC BY 4.0), not LA high-rise walls",
     "cerebro": "rule-graded by interior_water.json (RH rows) and the building water module; no ML forecaster shipped",
     "sources": ["tampere", "epa_guide"]},
    {"method": "Rebound hammer", "captures": "surface hardness of concrete (rebound number, median of many impacts)",
     "data": "rebound number RN", "public_labelled_data": "Matthews et al. database (CC BY 4.0), paired with core/cube strength",
     "cerebro": "ML strength estimate (RN-only power law, or SonReb with UPV) with a 90% interval; calibrate with cores",
     "sources": ["matthews_db", "bam_data"]},
    {"method": "Ultrasonic pulse velocity (UPV)", "captures": "speed of a sound pulse through concrete; lower speed suggests weaker or cracked concrete",
     "data": "velocity Vp (m/s) from path length and transit time; direct, semi-direct or indirect", "public_labelled_data": "Matthews et al. database (CC BY 4.0)",
     "cerebro": "ML strength estimate (Vp-only power law, or SonReb with rebound) with a 90% interval", "sources": ["matthews_db", "bam_data"]},
    {"method": "Ground-penetrating radar (GPR)", "captures": "reflections of radio waves from rebar, voids and moisture below the surface",
     "data": "radargrams", "public_labelled_data": "none found for interior walls", "cerebro": "not modelled; record the reading and photos",
     "sources": ["irt_gpr", "fhwa_ndt"]},
    {"method": "Impact echo", "captures": "resonant frequency after a small impact; finds delamination and thickness",
     "data": "frequency spectra", "public_labelled_data": "none found for interior walls", "cerebro": "not modelled; record only",
     "sources": ["fhwa_ndt"]},
    {"method": "Tap / hammer sounding", "captures": "hollow sound over delaminated plaster, render or concrete",
     "data": "inspector's marked areas (qualitative)", "public_labelled_data": "none found", "cerebro": "record only (method 'sounding' in the readings template)",
     "sources": ["fhwa_ndt"]},
    {"method": "Mold sampling", "captures": "spores or surface growth in samples",
     "data": "lab counts; EPA says sampling is usually unnecessary when growth is visible and there are no federal limits",
     "public_labelled_data": "none", "cerebro": "record only; visible mold is graded from photos by the interior_water rubric",
     "sources": ["epa_sampling", "epa_guide"]},
]


def main() -> int:
    d = ROOT / "eval" / "interior"
    d.mkdir(parents=True, exist_ok=True)
    (d / "methods_table.json").write_text(json.dumps({"methods": METHODS, "accessed": ACC}, indent=2), encoding="utf-8")
    (d / "sources.json").write_text(json.dumps({"sources": SOURCES}, indent=2), encoding="utf-8")
    print("wrote", len(METHODS), "methods and", len(SOURCES), "sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
