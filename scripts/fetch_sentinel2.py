"""
Download a cloud-free Sentinel-2 L2A RGB window around a point, no login needed.

    python scripts/fetch_sentinel2.py --center 17.385,78.487 --size-km 24 --out data/s2/hyderabad.tif

Uses the public Earth Search STAC catalogue (Element 84, AWS open data) and
reads only the requested window from the Cloud-Optimised GeoTIFFs, so a 24 km
window costs a few MB instead of a full 110 km tile.
Output: one 3-band GeoTIFF (R, G, B = B04, B03, B02) at 10 m in the tile's UTM CRS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAC = "https://earth-search.aws.element84.com/v1"


def main() -> None:
    import rasterio
    from pystac_client import Client
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    ap = argparse.ArgumentParser()
    ap.add_argument("--center", default="17.385,78.487", help="lat,lon (default Hyderabad)")
    ap.add_argument("--size-km", type=float, default=24.0)
    ap.add_argument("--start", default="2025-11-01")
    ap.add_argument("--end", default="2026-04-30")
    ap.add_argument("--max-cloud", type=float, default=10.0)
    ap.add_argument("--out", default="data/s2/scene.tif")
    a = ap.parse_args()

    lat, lon = (float(v) for v in a.center.split(","))
    dlat = a.size_km / 2 / 111.32
    dlon = dlat / np.cos(np.radians(lat))
    bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

    items = list(Client.open(STAC).search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{a.start}/{a.end}",
        query={"eo:cloud_cover": {"lt": a.max_cloud}}, max_items=60).items())
    if not items:
        raise SystemExit("No Sentinel-2 scene below the cloud limit - raise --max-cloud.")

    def coverage(it):
        from shapely.geometry import box, shape
        g = shape(it.geometry)
        return g.intersection(box(*bbox)).area / box(*bbox).area

    items.sort(key=lambda it: (-round(coverage(it), 2), it.properties["eo:cloud_cover"]))
    item = items[0]
    print(f"[s2] {item.id}  cloud {item.properties['eo:cloud_cover']:.1f}%  "
          f"coverage {coverage(item):.0%}  date {item.datetime:%Y-%m-%d}")

    bands, meta = [], None
    for key in ("red", "green", "blue"):
        href = item.assets[key].href
        with rasterio.open(href) as src:
            b = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            bands.append(src.read(1, window=win))
            if meta is None:
                meta = {"crs": src.crs, "transform": src.window_transform(win)}
    arr = np.stack(bands)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=3, dtype=arr.dtype, crs=meta["crs"],
                       transform=meta["transform"], compress="deflate") as d:
        d.write(arr)
    (out.with_suffix(".json")).write_text(json.dumps({
        "item": item.id, "date": str(item.datetime), "cloud_cover": item.properties["eo:cloud_cover"],
        "bbox_wgs84": bbox, "bands": ["B04 red", "B03 green", "B02 blue"],
        "source": "Copernicus Sentinel-2 L2A via Earth Search (AWS open data)"}, indent=2))
    print(f"[s2] wrote {out}  {arr.shape[2]}x{arr.shape[1]} px")


if __name__ == "__main__":
    main()
