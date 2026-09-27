"""Write eval/facade/sources.json: sources and licences shown on the exterior page footer.

Every entry was opened by the research/verify agents or by our download scripts on 2026-09-26
(licence read at source). The page reads this file; nothing is typed into the page itself.
    python scripts/facade/write_sources.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACC = "2026-09-26"

SOURCES = [
    {"id": "fisp", "title": "1 RCNY 103-04, Periodic Inspection of Exterior Walls and Appurtenances (NYC FISP rule)",
     "url": "https://www.nyc.gov/assets/buildings/rules/1_RCNY_103-04.pdf", "licence": "public rule text", "accessed": ACC,
     "used_for": "Safe / SWARMP / Unsafe classes in the facade_ll11 rubric; drones and photos do not eliminate close-up inspection"},
    {"id": "sb721", "title": "California Health and Safety Code 17973 (SB 721, exterior elevated elements)",
     "url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=HSC&sectionNum=17973", "licence": "public law", "accessed": ACC,
     "used_for": "California / LA context: wood-supported balconies, decks and walkways"},
    {"id": "sb326", "title": "California Civil Code 5551 (SB 326, condominium balconies)",
     "url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=CIV&sectionNum=5551", "licence": "public law", "accessed": ACC,
     "used_for": "California / LA context"},
    {"id": "sf5f", "title": "SF DBI Facade Inspection and Maintenance Program (SFEBC Chapter 5F)",
     "url": "https://www.sf.gov/reports--june-2024--dbi-facade-inspection-and-maintenance-program", "licence": "public page", "accessed": ACC,
     "used_for": "California context; we found no City of LA periodic high-rise facade ordinance (absence of evidence, not proof)"},
    {"id": "ozgenel", "title": "Ozgenel, Concrete Crack Images for Classification (Mendeley 5y9wdsg2zt v2)",
     "url": "https://data.mendeley.com/datasets/5y9wdsg2zt/2", "licence": "CC BY 4.0", "accessed": ACC,
     "used_for": "training only (no source-photo ids, so it cannot be split without leakage)"},
    {"id": "sdnet", "title": "SDNET2018, Maguire, Dorafshan, Thomas (Utah State University)",
     "url": "https://digitalcommons.usu.edu/all_datasets/48/", "licence": "CC BY 4.0", "accessed": ACC,
     "used_for": "walls subset: training, validation and the held-out test, split by source photo"},
    {"id": "sdnet_paper", "title": "Dorafshan, Thomas, Maguire 2018, SDNET2018 (Data in Brief, PMC6247444)",
     "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6247444/", "licence": "CC BY 4.0 article", "accessed": ACC,
     "used_for": "dataset description"},
    {"id": "bfdd", "title": "BFDD: Pixel-Level Aligned RGB-IR Image Dataset for Building Facade Defect Segmentation (Mendeley 9ych7czvyg v1)",
     "url": "https://data.mendeley.com/datasets/9ych7czvyg/1", "licence": "CC BY 4.0", "accessed": ACC,
     "used_for": "out-of-domain test (real drone facade photos) and the bundled sample photos; label value 1 = crack is inferred by overlay, not documented"},
    {"id": "timm_resnet18", "title": "timm/resnet18.a1_in1k (Hugging Face)",
     "url": "https://huggingface.co/timm/resnet18.a1_in1k", "licence": "Apache-2.0 tag (ImageNet-1k pretraining)", "accessed": ACC,
     "used_for": "backbone weights before fine-tuning"},
    {"id": "onnxruntime", "title": "onnxruntime (PyPI)", "url": "https://pypi.org/project/onnxruntime/", "licence": "MIT", "accessed": ACC,
     "used_for": "CPU inference in the app"},
    {"id": "dima806", "title": "dima806/surface_crack_image_detection (Hugging Face)", "url": "https://huggingface.co/dima806/surface_crack_image_detection",
     "licence": "Apache-2.0", "accessed": ACC,
     "used_for": "NOT used or scored here. Its card reports accuracy on its own test set (likely an Ozgenel re-host, inferred); that is not our result"},
]


def main() -> int:
    out = ROOT / "eval" / "facade" / "sources.json"
    out.write_text(json.dumps({"sources": SOURCES}, indent=2), encoding="utf-8")
    print(out, len(SOURCES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
