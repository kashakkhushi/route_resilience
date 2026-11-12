"""
Build OSM ground-truth masks for every GeoTIFF in a directory.

    python scripts/prepare_groundtruth.py --raster data/raw/sentinel2 --sensor sentinel2

Requires requirements-geo.txt and internet access on the first run only; results
are cached under data/osm, so reruns are offline and reproducible.

Always sanity-check the output. The most common failure is a CRS mismatch
producing a near-empty mask, which then trains the model to predict nothing and
looks like a modelling problem for hours. The road-pixel fraction printed per
scene is the check: for an urban tile it should land roughly in 2-15%.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.osm_groundtruth import build_ground_truth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", required=True, help="GeoTIFF file or directory")
    ap.add_argument("--sensor", default="sentinel2",
                    choices=["sentinel2", "liss4", "cartosat3"])
    ap.add_argument("--out", default="data/interim/masks")
    ap.add_argument("--network-type", default="drive",
                    help="OSM filter: drive | drive_service | all")
    ap.add_argument("--cache", default="data/osm")
    args = ap.parse_args()

    src = Path(args.raster)
    if src.is_dir():
        rasters = sorted(src.glob("*.tif")) + sorted(src.glob("*.tiff"))
    else:
        rasters = [src]
    if not rasters:
        raise SystemExit(f"no GeoTIFFs found under {src}")

    reports = []
    for r in rasters:
        print(f"[osm] {r.name} ...")
        try:
            rep = build_ground_truth(r, args.out, args.sensor,
                                     args.network_type, args.cache)
            frac = rep["road_pixel_fraction"]
            kept, dropped = rep["kept"], rep["dropped"]
            flag = "   <-- SUSPICIOUS: check CRS / bbox" if frac < 0.005 else ""
            print(f"      roads={frac:.3%}  kept={kept}  dropped={dropped}{flag}")
            if rep.get("dropped_classes"):
                print(f"      dropped as unresolvable at this GSD: {rep['dropped_classes']}")
            reports.append(rep)
        except Exception as e:
            print(f"      FAILED: {e}")
            reports.append({"source_raster": str(r), "error": str(e)})

    out = Path(args.out) / "groundtruth_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))
    ok = sum(1 for r in reports if "error" not in r)
    print(f"\n{ok}/{len(reports)} succeeded -> {out}")


if __name__ == "__main__":
    main()
