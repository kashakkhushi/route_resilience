"""
Build a LISS-IV (or Sentinel-2) road dataset with OpenStreetMap labels.

    python scripts/prepare_liss4.py --scene <folder or .tif> --out data/processed_liss4 \
        --center 17.385,78.487 --size-km 24

ResourceSat-2/2A LISS-IV (5.8 m, bands G/R/NIR, no blue) as delivered by
Bhoonidhi: either one multi-band GeoTIFF or one GeoTIFF per band (BAND2/3/4).

Steps
  1. find and stack the three bands, cut a square window around --center
     (falls back to the scene centre if the point is outside the scene);
  2. upsample by --upsample (default 2 -> 2.9 m/px) so 1-2 px roads become
     learnable by a model with a stride-4 encoder;
  3. fetch OSM roads for the window and rasterise them on the same grid
     (src/data/osm_groundtruth.py); roads narrower than --min-width-m are
     dropped as unresolvable at this resolution;
  4. cut tiles and split them SPATIALLY (west -> train, then val, east -> test)
     so test tiles never neighbour a training tile;
  5. write the same layout as DeepGlobe preparation, so kaggle_train.py and
     evaluate_checkpoint.py run on it unchanged.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def find_bands(scene: Path) -> list[Path]:
    """Return [G, R, NIR] GeoTIFF paths, or [single multi-band file]."""
    tifs = sorted([p for p in (scene.rglob("*") if scene.is_dir() else [scene])
                   if p.suffix.lower() in (".tif", ".tiff")])
    if not tifs:
        raise SystemExit(f"No GeoTIFF found under {scene}")
    by_band = {}
    for p in tifs:
        m = re.search(r"(?:BAND|_B)(\d)", p.stem.upper())
        if m:
            by_band[int(m.group(1))] = p
    if all(b in by_band for b in (2, 3, 4)):
        return [by_band[2], by_band[3], by_band[4]]           # G, R, NIR
    import rasterio

    for p in sorted(tifs, key=lambda q: -q.stat().st_size):
        with rasterio.open(p) as s:
            if s.count >= 3:
                return [p]
    raise SystemExit("Could not identify LISS-IV bands. Files found:\n  "
                     + "\n  ".join(str(t) for t in tifs))


def _open_projected(path: Path, center: tuple | None):
    """Open a raster; if it is in lat/lon, wrap it in a UTM VRT so every
    distance downstream (GSD, road widths, window size) is in metres."""
    import rasterio
    from rasterio.vrt import WarpedVRT

    src = rasterio.open(path)
    if src.crs is None or not src.crs.is_geographic:
        return src
    lon = center[1] if center else (src.bounds.left + src.bounds.right) / 2
    lat = center[0] if center else (src.bounds.top + src.bounds.bottom) / 2
    epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
    print(f"[liss4] {path.name} is in lat/lon - reprojecting on the fly to EPSG:{epsg}")
    return WarpedVRT(src, crs=f"EPSG:{epsg}")


def read_window(bands: list[Path], center: tuple | None, size_km: float, upsample: float):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform as warp_xy
    from rasterio.windows import Window

    with _open_projected(bands[0], center) as src:
        crs, tr, W, H = src.crs, src.transform, src.width, src.height
        gsd = abs(tr.a)
        if center is not None:
            xs, ys = warp_xy("EPSG:4326", crs, [center[1]], [center[0]])
            col, row = ~tr * (xs[0], ys[0])
        else:
            col, row = W / 2, H / 2
        inside = 0 <= col < W and 0 <= row < H
        if not inside:
            print(f"[liss4] centre {center} is outside this scene - using the scene centre")
            col, row = W / 2, H / 2
        half = int(size_km * 1000 / gsd / 2)
        c0, r0 = int(max(col - half, 0)), int(max(row - half, 0))
        c1, r1 = int(min(col + half, W)), int(min(row + half, H))
        win = Window(c0, r0, c1 - c0, r1 - r0)
        out_shape = (int(win.height * upsample), int(win.width * upsample))
        wtr = rasterio.windows.transform(win, tr) * rasterio.Affine.scale(1 / upsample)

    arrs = []
    if len(bands) == 1:
        with _open_projected(bands[0], center) as src:
            a = src.read([1, 2, 3], window=win, out_shape=(3, *out_shape),
                         resampling=Resampling.bilinear)
            arrs = list(a)
    else:
        for b in bands:
            with _open_projected(b, center) as src:
                arrs.append(src.read(1, window=win, out_shape=out_shape,
                                     resampling=Resampling.bilinear))
    stack = np.stack(arrs, -1).astype(np.float32)               # G, R, NIR
    return {"array": stack, "transform": wtr, "crs": crs, "gsd": gsd / upsample,
            "native_gsd": gsd, "window": [int(c0), int(r0), int(c1), int(r1)]}


def to_display_rgb(stack: np.ndarray, sensor: str = "liss4") -> tuple[np.ndarray, np.ndarray]:
    """Per-band 1-99 percentile stretch (ignoring nodata zeros). LISS-IV bands
    (G, R, NIR) go through the project's pseudo-RGB mapping (no blue band);
    Sentinel-2 input is already R, G, B."""
    from src.data.geo_io import select_rgb

    valid = stack.sum(-1) > 0
    out = np.zeros_like(stack)
    for i in range(3):
        v = stack[..., i][valid]
        lo, hi = (np.percentile(v, [1, 99]) if v.size else (0, 1))
        out[..., i] = np.clip((stack[..., i] - lo) / max(hi - lo, 1e-6), 0, 1)
    return (select_rgb(out, "liss4") if sensor == "liss4" else out), valid


def osm_mask(win: dict, cache: Path, min_width_m: float, sensor: str = "liss4") -> dict:
    import rasterio
    from rasterio.warp import transform_bounds

    from src.data.osm_groundtruth import download_osm_roads, rasterize_roads

    h, w = win["array"].shape[:2]
    tmp = cache / "window_grid.tif"
    cache.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tmp, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="uint8", crs=win["crs"], transform=win["transform"]) as d:
        d.write(np.zeros((1, h, w), np.uint8))
        b = transform_bounds(win["crs"], "EPSG:4326", *d.bounds, densify_pts=21)
    edges = download_osm_roads(b, network_type="drive", cache_dir=cache)
    rep = rasterize_roads(edges, tmp, sensor=sensor, min_resolvable_width_m=min_width_m)
    rep["bounds_wgs84"] = list(b)
    return rep


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="folder with the LISS-IV product, or a .tif")
    ap.add_argument("--sensor", default="liss4", choices=["liss4", "sentinel2"],
                    help="liss4: bands G,R,NIR (5.8 m). sentinel2: one RGB GeoTIFF (10 m) "
                         "from scripts/fetch_sentinel2.py")
    ap.add_argument("--out", default="data/processed_liss4")
    ap.add_argument("--center", default="17.385,78.487",
                    help="lat,lon of the study area (default: Hyderabad)")
    ap.add_argument("--size-km", type=float, default=24.0)
    ap.add_argument("--upsample", type=float, default=2.0)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--min-width-m", type=float, default=6.0,
                    help="drop OSM roads narrower than this (unresolvable at 5.8 m)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--cache", default="data/osm")
    args = ap.parse_args()

    center = tuple(float(v) for v in args.center.split(",")) if args.center else None
    bands = find_bands(Path(args.scene))
    print("[liss4] bands:", [b.name for b in bands])
    win = read_window(bands, center, args.size_km, args.upsample)
    rgb, valid = to_display_rgb(win["array"], args.sensor)
    print(f"[liss4] window {rgb.shape[1]}x{rgb.shape[0]} px at {win['gsd']:.2f} m/px "
          f"(native {win['native_gsd']:.2f} m)")

    lab = osm_mask(win, Path(args.cache), args.min_width_m, args.sensor)
    mask = lab["mask"] & valid
    print(f"[liss4] OSM roads kept={lab['kept']} dropped={lab['dropped']} "
          f"road pixels={mask.mean():.2%}")
    if mask.mean() < 0.003:
        raise SystemExit("Almost no OSM roads in this window - check --center / CRS.")

    out = Path(args.out)
    T = args.tile
    H, W = mask.shape
    # Keep only tiles that are almost entirely inside the scene. Scene edges are
    # often slanted, so a whole edge column can be empty - the split is therefore
    # made over the VALID tiles, not over raw columns.
    cand = [(r, ci, c) for r in range(0, H - T + 1, T)
            for ci, c in enumerate(range(0, W - T + 1, T))
            if valid[r:r + T, c:c + T].mean() >= 0.9]
    if len(cand) < 10:
        raise SystemExit(f"Only {len(cand)} usable tiles - the window is mostly outside the scene.")
    # Spatial split by whole columns, east -> test, then val, rest train, sized by
    # tile count, so no test tile shares a column with a training tile.
    per_col = {}
    for r, ci, c in cand:
        per_col[ci] = per_col.get(ci, 0) + 1
    split_of, acc = {}, 0
    n = len(cand)
    for ci in sorted(per_col, reverse=True):
        split_of[ci] = ("test" if acc < args.test_frac * n else
                        "val" if acc < (args.test_frac + args.val_frac) * n else "train")
        acc += per_col[ci]
    counts = {"train": 0, "val": 0, "test": 0}
    for r, ci, c in cand:
        split = split_of[ci]
        stem = f"{args.sensor}_{r // T:02d}_{ci:02d}"
        di, dm = out / split / "images", out / split / "masks"
        di.mkdir(parents=True, exist_ok=True)
        dm.mkdir(parents=True, exist_ok=True)
        im = (rgb[r:r + T, c:c + T] * 255).astype(np.uint8)
        cv2.imwrite(str(di / f"{stem}.png"), im[..., ::-1])
        cv2.imwrite(str(dm / f"{stem}.png"), mask[r:r + T, c:c + T].astype(np.uint8) * 255)
        counts[split] += 1
    print("[liss4] tiles:", counts)

    prev = (rgb * 255).astype(np.uint8).copy()
    prev[mask] = (0.4 * prev[mask] + 0.6 * np.array([255, 210, 0])).astype(np.uint8)
    s = 1600 / max(prev.shape[:2])
    cv2.imwrite(str(out / "preview_osm_overlay.jpg"),
                cv2.resize(prev, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)[..., ::-1])

    (out / "manifest.json").write_text(json.dumps({
        "source": ("ResourceSat-2/2A LISS-IV (Bhoonidhi, NRSC/ISRO)" if args.sensor == "liss4"
                   else "Copernicus Sentinel-2 L2A (10 m RGB)"),
        "sensor": args.sensor,
        "bands": [str(b) for b in bands], "center_latlon": center,
        "size_km": args.size_km, "upsample": args.upsample,
        "gsd_m": win["gsd"], "native_gsd_m": win["native_gsd"],
        "labels": "OpenStreetMap drive network, buffered by road class",
        "min_width_m": args.min_width_m, "osm": {k: v for k, v in lab.items() if k != "mask"},
        "split": "spatial columns: west=train, then val, east=test", "counts": counts,
    }, indent=2, default=str))
    print(f"[liss4] wrote {out}")


if __name__ == "__main__":
    main()
