"""
Get a real flooded satellite image + road masks for the dashboard demo.

Source: SpaceNet 8 (Maxar, ~0.3-0.8 m), public on AWS Open Data. Floods:
  Germany        - heavy rain, July 2021 (Ahr valley)
  Louisiana-East - Hurricane Ida, August 2021
Each area has a PRE-flood image, a POST-flood image (same place) and
hand-drawn roads with a "flooded" flag.

    python scripts\\flood_demo_image.py                    (Germany, best tile)
    python scripts\\flood_demo_image.py --area Louisiana-East
    python scripts\\flood_demo_image.py --top 3            (3 best tiles)

Output folder data\\flood_demo\\<tile>\\ :
  post_flood.tif / .png     image during the flood  -> upload this
  pre_flood.tif  / .png     same place before the flood
  road_mask.png             all roads, white = road  -> ground-truth mask
  flooded_road_mask.png     only flooded roads
  preview.jpg               side-by-side check picture
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/tarballs/"
TARS = {"Germany": "Germany_Training_Public.tar.gz",
        "Louisiana-East": "Louisiana-East_Training_Public.tar.gz"}


def download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"  already downloaded: {dst}")
        return
    print(f"  downloading {url}\n  (a few GB - this takes a while, it is done only once)")
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0)); done = 0
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b); done += len(b)
            if total:
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.rename(dst)


def extract(tgz: Path, out: Path) -> Path:
    marker = out / ".extracted"
    if not marker.exists():
        print(f"  extracting {tgz.name} ...")
        with tarfile.open(tgz) as t:
            t.extractall(out)
        marker.write_text("ok")
    return out


def find_mapping(root: Path) -> list[dict]:
    csvs = sorted(root.rglob("*label_image_mapping*.csv"))
    if not csvs:
        raise SystemExit(f"label_image_mapping.csv not found under {root}")
    with open(csvs[0], newline="") as f:
        return list(csv.DictReader(f))


def find_file(root: Path, name: str) -> Path | None:
    if not name:
        return None
    hits = list(root.rglob(Path(name).name))
    return hits[0] if hits else None


def is_flooded(v) -> bool:
    return str(v).strip().lower() in ("yes", "true", "1", "flooded")


def roads_of(geojson: Path):
    import geopandas as gpd
    g = gpd.read_file(geojson)
    if g.empty:
        return g
    g = g[g.geometry.notna() & g.geom_type.isin(["LineString", "MultiLineString"])]
    g = g.copy()
    g["is_flooded"] = g["flooded"].apply(is_flooded) if "flooded" in g.columns else False
    return g


def flooded_length_m(g) -> tuple[float, float]:
    if g.empty:
        return 0.0, 0.0
    m = g.to_crs(g.estimate_utm_crs())
    L = m.length
    return float(L[g["is_flooded"].values].sum()), float(L.sum())


def burn(g, ref_tif: Path, half_width_m: float):
    """Rasterise road lines, buffered by half_width_m, on the grid of ref_tif."""
    import rasterio
    from rasterio.features import rasterize
    with rasterio.open(ref_tif) as src:
        shape, transform, crs = (src.height, src.width), src.transform, src.crs
    if g.empty:
        return np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    utm = g.estimate_utm_crs()
    buf = g.to_crs(utm).buffer(half_width_m).to_crs(crs)
    allm = rasterize([(geom, 255) for geom in buf if not geom.is_empty],
                     out_shape=shape, transform=transform, fill=0, dtype="uint8")
    fl = [geom for geom, f in zip(buf, g["is_flooded"]) if f and not geom.is_empty]
    flm = rasterize([(geom, 255) for geom in fl], out_shape=shape, transform=transform,
                    fill=0, dtype="uint8") if fl else np.zeros(shape, np.uint8)
    return allm, flm


def to_grid(src_tif: Path, ref_tif: Path, dst_tif: Path) -> None:
    """Write src_tif resampled onto ref_tif's grid (pre/post can differ slightly)."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    with rasterio.open(ref_tif) as ref, rasterio.open(src_tif) as src:
        prof = ref.profile.copy()
        n = min(src.count, 3)
        prof.update(count=n, dtype="uint8", compress="deflate")
        out = np.zeros((n, ref.height, ref.width), np.float32)
        for b in range(n):
            reproject(rasterio.band(src, b + 1), out[b], src_transform=src.transform,
                      src_crs=src.crs, dst_transform=ref.transform, dst_crs=ref.crs,
                      resampling=Resampling.bilinear)
        out = stretch(out)
        with rasterio.open(dst_tif, "w", **prof) as d:
            d.write(out)


def stretch(a: np.ndarray) -> np.ndarray:
    """2-98 % per-band stretch to uint8 (skips no-data zeros)."""
    o = np.zeros(a.shape, np.uint8)
    for b in range(a.shape[0]):
        v = a[b][a[b] > 0]
        if v.size == 0:
            continue
        lo, hi = np.percentile(v, (2, 98))
        o[b] = np.clip((a[b] - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return o


def to_png(tif: Path, png: Path) -> np.ndarray:
    import rasterio
    from PIL import Image
    with rasterio.open(tif) as s:
        a = s.read()
    rgb = np.moveaxis(a[:3], 0, -1) if a.shape[0] >= 3 else np.repeat(a[0][..., None], 3, -1)
    Image.fromarray(rgb).save(png)
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="Germany", choices=list(TARS))
    ap.add_argument("--top", type=int, default=1, help="how many tiles to export")
    ap.add_argument("--road-width-m", type=float, default=6.0,
                    help="road width burnt into the mask (DeepGlobe-like)")
    ap.add_argument("--data", default="data/sn8")
    ap.add_argument("--out", default="data/flood_demo")
    ap.add_argument("--keep-tar", action="store_true")
    a = ap.parse_args()

    from PIL import Image

    data = Path(a.data); data.mkdir(parents=True, exist_ok=True)
    print("[1/4] download SpaceNet 8", a.area)
    tgz = data / TARS[a.area]
    download(BASE + TARS[a.area], tgz)
    root = extract(tgz, data / a.area)
    if not a.keep_tar:
        tgz.unlink(missing_ok=True)

    print("[2/4] rank tiles by flooded road length")
    rows = find_mapping(root)
    scored = []
    for r in rows:
        lab = find_file(root, r.get("label", ""))
        pre = find_file(root, r.get("pre-event image", "") or r.get("pre-event", ""))
        post = find_file(root, r.get("post-event image 1", "") or r.get("post-event 1", ""))
        if not (lab and pre and post):
            continue
        g = roads_of(lab)
        fl, tot = flooded_length_m(g)
        if fl > 0:
            scored.append((fl, tot, lab, pre, post, g))
    if not scored:
        raise SystemExit("no tile with flooded roads found - check the extracted folder")
    scored.sort(key=lambda s: -s[0])
    print(f"  {len(scored)} tiles have flooded roads; best: {scored[0][0]:.0f} m flooded")

    out_all = Path(a.out)
    for k, (fl, tot, lab, pre, post, g) in enumerate(scored[: a.top]):
        name = f"{a.area}_{pre.stem}"
        o = out_all / name; o.mkdir(parents=True, exist_ok=True)
        print(f"[3/4] tile {k + 1}: {name}  flooded {fl:.0f} m of {tot:.0f} m road")
        to_grid(pre, pre, o / "pre_flood.tif")
        to_grid(post, pre, o / "post_flood.tif")
        pre_rgb = to_png(o / "pre_flood.tif", o / "pre_flood.png")
        post_rgb = to_png(o / "post_flood.tif", o / "post_flood.png")
        allm, flm = burn(g, pre, a.road_width_m / 2)
        Image.fromarray(allm).save(o / "road_mask.png")
        Image.fromarray(flm).save(o / "flooded_road_mask.png")

        print("[4/4] preview")
        ov = post_rgb.copy().astype(np.float32)
        ov[allm > 0] = ov[allm > 0] * 0.4 + np.array([0, 230, 120]) * 0.6
        ov[flm > 0] = ov[flm > 0] * 0.3 + np.array([255, 40, 60]) * 0.7
        h = 700
        panels = [Image.fromarray(x.astype(np.uint8)).resize((int(x.shape[1] * h / x.shape[0]), h))
                  for x in (pre_rgb, post_rgb, ov)]
        sheet = Image.new("RGB", (sum(p.width for p in panels) + 20, h), "white")
        x = 0
        for p in panels:
            sheet.paste(p, (x, 0)); x += p.width + 10
        sheet.save(o / "preview.jpg", quality=90)
        import rasterio
        with rasterio.open(o / "post_flood.tif") as s_:
            px = abs(s_.transform.a); cy = (s_.bounds.top + s_.bounds.bottom) / 2
            px_m = px * 111320 * np.cos(np.radians(cy)) if s_.crs and s_.crs.is_geographic else px
        print(f"  pixel size ~{px_m:.2f} m  (enter this in the dashboard if you upload the PNG)")
        (o / "README.txt").write_text(f"Pixel size: ~{px_m:.2f} m per pixel\n" +
            f"SpaceNet 8 {a.area}, tile {pre.stem}\n"
            f"Roads: {tot:.0f} m, flooded: {fl:.0f} m ({100 * fl / max(tot, 1):.1f} %)\n"
            "preview.jpg: before | during flood | roads (green) and flooded roads (red)\n"
            "Upload post_flood.tif (keeps pixel size) or post_flood.png, with road_mask.png as ground truth.\n"
            "Source: SpaceNet 8 (Maxar imagery, CC BY-SA 4.0), Hansch et al., CVPRW 2022.\n")
        print(f"  -> {o}")
    print("\nDone. Open preview.jpg first to check the tile.")


if __name__ == "__main__":
    main()
