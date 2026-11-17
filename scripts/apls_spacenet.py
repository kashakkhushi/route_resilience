"""
Topological evaluation (APLS) on real, georeferenced imagery.

    python scripts/apls_spacenet.py --checkpoint best.pt --out outputs/apls --n-tiles 60

Data: SpaceNet 3 roads, Las Vegas (AOI_2_Vegas): WorldView-3 pan-sharpened RGB
GeoTIFFs at 0.3 m with hand-labelled road centrelines (GeoJSON). Downloaded
anonymously from the public `spacenet-dataset` S3 bucket. APLS (Average Path
Length Similarity) was introduced with this dataset, so it is the natural place
to measure whether the extracted graph is routable, not just pixel-accurate.

Per tile:
  1. read PS-RGB, resample to the training GSD (0.5 m, DeepGlobe) so roads have
     the width the model learned;
  2. segment with the trained checkpoint, build the raw graph, heal it;
  3. rasterise the reference centrelines on the same grid and build the
     reference graph the SAME way (skeleton -> graph), so both graphs are
     compared like-for-like;
  4. APLS (symmetric, both directions) for the raw and the healed graph.

Optionally (--osm) the same tiles are scored against OpenStreetMap roads
fetched for the tile bounds, which exercises the project's OSM ground-truth
pipeline (src/data/osm_groundtruth.py) on real imagery.

Honest caveats, also written into the report:
  * Las Vegas is out of domain for a DeepGlobe-trained model (different
    continent, sensor and road style), so this measures generalisation;
  * centreline rasterisation uses a fixed lane width, so pixel metrics against
    it are indicative only - APLS is the headline number here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUCKET = "spacenet-dataset"
PREFIXES = [
    "spacenet/SN3_roads/train/{aoi}/",
    "spacenet/SN3_roads/test_public/{aoi}/",
]
TARBALLS = [
    "spacenet/SN3_roads/tarballs/SN3_roads_train_{aoi}.tar.gz",
]


# --------------------------------------------------------------------- data
def _s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _tile_id(key: str) -> str | None:
    """'.../SN3_roads_train_AOI_2_Vegas_PS-RGB_img123.tif' -> 'img123'."""
    name = key.rsplit("/", 1)[-1]
    if "_img" not in name:
        return None
    return "img" + name.split("_img", 1)[1].split(".")[0]


def fetch_spacenet(aoi: str, n_tiles: int, dest: Path, seed: int = 0) -> list[dict]:
    """Download n PS-RGB tiles with matching road GeoJSON. Tries the unpacked
    bucket layout first, then streams the tarball and stops once enough pairs
    are extracted."""
    dest.mkdir(parents=True, exist_ok=True)
    s3 = _s3()

    for pref in PREFIXES:
        pref = pref.format(aoi=aoi)
        imgs, labs = {}, {}
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=pref):
            for o in page.get("Contents", []):
                k = o["Key"]
                tid = _tile_id(k)
                if tid is None:
                    continue
                if "/PS-RGB/" in k and k.endswith(".tif"):
                    imgs[tid] = k
                elif "/geojson_roads/" in k and k.endswith(".geojson"):
                    labs[tid] = k
        both = sorted(set(imgs) & set(labs), key=lambda t: int(t[3:]))
        print(f"[spacenet] {pref}: {len(imgs)} images, {len(labs)} labels, {len(both)} pairs")
        if not both:
            continue
        rng = np.random.default_rng(seed)
        pick = sorted(rng.choice(both, size=min(n_tiles, len(both)), replace=False),
                      key=lambda t: int(t[3:]))
        out = []
        for tid in pick:
            ip, lp = dest / f"{tid}.tif", dest / f"{tid}.geojson"
            if not ip.exists():
                s3.download_file(BUCKET, imgs[tid], str(ip))
            if not lp.exists():
                s3.download_file(BUCKET, labs[tid], str(lp))
            out.append({"id": tid, "image": ip, "label": lp})
        return out

    # Fallback: stream the tarball, keep only PS-RGB + geojson_roads members.
    import tarfile

    for tb in TARBALLS:
        key = tb.format(aoi=aoi)
        try:
            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"]
        except Exception as e:  # noqa: BLE001
            print(f"[spacenet] tarball {key} unavailable: {e}")
            continue
        print(f"[spacenet] streaming {key} ...")
        imgs, labs, out = {}, {}, []
        with tarfile.open(fileobj=body, mode="r|gz") as tf:
            for m in tf:
                tid = _tile_id(m.name)
                if tid is None or not m.isfile():
                    continue
                if "PS-RGB" in m.name and m.name.endswith(".tif"):
                    p = dest / f"{tid}.tif"
                    p.write_bytes(tf.extractfile(m).read())
                    imgs[tid] = p
                elif "geojson_roads" in m.name and "speed" not in m.name \
                        and m.name.endswith(".geojson"):
                    p = dest / f"{tid}.geojson"
                    p.write_bytes(tf.extractfile(m).read())
                    labs[tid] = p
                if tid in imgs and tid in labs and all(o["id"] != tid for o in out):
                    out.append({"id": tid, "image": imgs[tid], "label": labs[tid]})
                    if len(out) >= n_tiles:
                        break
        if out:
            return out

    # Nothing worked: print the bucket layout so the paths can be fixed.
    print("[spacenet] could not locate tiles. Bucket layout under spacenet/SN3_roads/:")
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix="spacenet/SN3_roads/", Delimiter="/")
    for p in r.get("CommonPrefixes", []):
        print("   ", p["Prefix"])
    raise SystemExit("SpaceNet download failed - send this log.")


def read_tile(path: Path, target_gsd: float) -> dict:
    """Read a PS-RGB GeoTIFF, resample to target_gsd. Returns the image and the
    affine mapping from lon/lat to the resampled pixel grid."""
    import cv2
    import rasterio

    with rasterio.open(path) as src:
        arr = np.transpose(src.read([1, 2, 3]), (1, 2, 0)).astype(np.float32)
        tr, crs = src.transform, src.crs
        bounds = src.bounds
    h, w = arr.shape[:2]
    lat = (bounds.top + bounds.bottom) / 2.0
    if crs is not None and crs.is_geographic:
        gsd = abs(tr.a) * 111_320.0 * np.cos(np.radians(lat))
    else:
        gsd = abs(tr.a)
    scale = gsd / target_gsd
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
    nodata = float((arr.sum(-1) == 0).mean())
    return {"image": img, "transform": tr, "crs": crs, "scale": scale,
            "src_gsd": float(gsd), "bounds": tuple(bounds), "nodata": nodata,
            "shape": (nh, nw)}


def lines_to_pixels(geoms, transform, scale) -> list[np.ndarray]:
    """Shapely (Multi)LineStrings in the raster CRS -> pixel polylines (x, y)."""
    inv = ~transform
    out = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        parts = getattr(g, "geoms", [g])
        for ln in parts:
            if ln.geom_type != "LineString":
                continue
            xs, ys = ln.xy
            cols, rows = inv * (np.asarray(xs), np.asarray(ys))
            out.append(np.stack([np.asarray(cols) * scale,
                                 np.asarray(rows) * scale], 1))
    return out


def rasterise(polys, shape, width_px: int) -> np.ndarray:
    import cv2

    m = np.zeros(shape, np.uint8)
    for p in polys:
        if len(p) >= 2:
            cv2.polylines(m, [np.round(p).astype(np.int32)], False, 1,
                          thickness=max(int(width_px), 1))
    return m.astype(bool)


def spacenet_lines(label_path: Path):
    import geopandas as gpd

    g = gpd.read_file(label_path)
    return list(g.geometry) if len(g) else []


def osm_lines(bounds_wgs84, cache: Path):
    from src.data.osm_groundtruth import download_osm_roads

    edges = download_osm_roads(bounds_wgs84, network_type="drive", cache_dir=cache)
    return list(edges.to_crs("EPSG:4326").geometry)


# --------------------------------------------------------------------- eval
def evaluate_tile(model, t: dict, ref_polys, args) -> dict:
    from src.eval.metrics import evaluate_mask
    from src.eval.topology import apls
    from src.graph.healing import HealingConfig, heal_graph
    from src.graph.skeletonize import build_raw_graph
    from src.models.inference import evidence_map_from_prediction, predict_scene

    pred = t.get("_pred")
    if pred is None:
        pred = predict_scene(model, t["image"], tile_size=512, overlap=128,
                             device=args.device, tta=False, threshold=0.5)
        pred["evidence"] = evidence_map_from_prediction(pred)
        t["_pred"] = pred

    gsd = args.target_gsd
    lane_px = args.road_width_m / gsd
    ref = rasterise(ref_polys, t["shape"], lane_px)
    if ref.sum() < 200:
        return {"skipped": "too little reference road"}

    G_ref = build_raw_graph(ref, pixel_size_m=gsd)
    if G_ref.number_of_nodes() < 4:
        return {"skipped": "reference graph too small"}

    raw = t.get("_raw")
    if raw is None:
        raw = build_raw_graph(pred["mask"], pixel_size_m=gsd)
        healed, rep = heal_graph(raw, HealingConfig(max_gap_px=args.max_gap_m / gsd),
                                 pred["evidence"])
        t["_raw"], t["_healed"], t["_heal_rep"] = raw, healed, rep
    raw, healed = t["_raw"], t["_healed"]

    tol = args.tolerance_m / gsd
    out = {"ref_nodes": G_ref.number_of_nodes(), "pred_nodes_raw": raw.number_of_nodes(),
           "pred_nodes_healed": healed.number_of_nodes(),
           "bridges_added": t["_heal_rep"].get("bridges_added")}
    if raw.number_of_nodes() < 2:
        out.update(apls_raw=0.0, apls_healed=0.0)
    else:
        out["apls_raw"] = apls(G_ref, raw, n_pairs=args.pairs, tolerance_px=tol)["apls"]
        out["apls_healed"] = apls(G_ref, healed, n_pairs=args.pairs, tolerance_px=tol)["apls"]
    m = evaluate_mask(pred["mask"], ref, None, buffer_px=int(round(2.0 / gsd)))
    out.update({k: m.get(k) for k in ("iou", "precision", "recall", "relaxed_iou", "cldice")})
    t["_ref_" + args._ref_name] = ref
    return out


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if "skipped" not in r]
    if not ok:
        return {"n_tiles": 0}

    def mean(k):
        v = [r[k] for r in ok if r.get(k) is not None]
        return round(float(np.mean(v)), 4) if v else None

    def std(k):
        v = [r[k] for r in ok if r.get(k) is not None]
        return round(float(np.std(v)), 4) if v else None

    return {"n_tiles": len(ok), "n_skipped": len(rows) - len(ok),
            "apls_raw": mean("apls_raw"), "apls_healed": mean("apls_healed"),
            "apls_healed_std": std("apls_healed"),
            "healing_gain": (round(mean("apls_healed") - mean("apls_raw"), 4)
                             if mean("apls_raw") is not None else None),
            "iou": mean("iou"), "relaxed_iou": mean("relaxed_iou"),
            "precision": mean("precision"), "recall": mean("recall"),
            "cldice": mean("cldice")}


def save_figure(t: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.viz.plots import plot_network

    img = t["image"]
    lo, hi = np.percentile(img, [2, 98])
    disp = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    panels = [("Satellite (SpaceNet, Las Vegas)", disp),
              ("Reference roads (SpaceNet labels)", t.get("_ref_spacenet")),
              ("Model road mask", t["_pred"]["mask"])]
    if t.get("_ref_osm") is not None:
        panels.append(("OpenStreetMap roads", t["_ref_osm"]))
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(5 * (len(panels) + 1), 5))
    for ax, (title, a) in zip(axes, panels):
        ax.imshow(a, cmap=None if a.ndim == 3 else "gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plot_network(t["_healed"], ax=axes[-1], title="Healed road graph")
    axes[-1].set_xlim(0, img.shape[1]); axes[-1].set_ylim(img.shape[0], 0)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="outputs/apls")
    ap.add_argument("--aoi", default="AOI_2_Vegas")
    ap.add_argument("--n-tiles", type=int, default=60)
    ap.add_argument("--local-dir", default=None,
                    help="use tiles already on disk (<id>.tif + <id>.geojson) instead of S3")
    ap.add_argument("--target-gsd", type=float, default=0.5,
                    help="metres/pixel the model was trained at (DeepGlobe 0.5 m)")
    ap.add_argument("--road-width-m", type=float, default=7.0,
                    help="width used to rasterise reference centrelines")
    ap.add_argument("--tolerance-m", type=float, default=8.0,
                    help="APLS node-snapping tolerance")
    ap.add_argument("--max-gap-m", type=float, default=30.0, help="healing max bridge length")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--osm", action="store_true", help="also score against OpenStreetMap")
    ap.add_argument("--figures", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from scripts.evaluate_checkpoint import load_model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    t0 = time.time()
    if args.local_dir:
        d = Path(args.local_dir)
        tiles = [{"id": p.stem, "image": p, "label": p.with_suffix(".geojson")}
                 for p in sorted(d.glob("*.tif")) if p.with_suffix(".geojson").exists()]
        tiles = tiles[: args.n_tiles]
    else:
        tiles = fetch_spacenet(args.aoi, args.n_tiles, out / "tiles", args.seed)
    print(f"[apls] {len(tiles)} tiles  ({time.time() - t0:.0f}s)")

    model, ck, name = load_model(Path(args.checkpoint))
    model = model.to(args.device)
    print(f"[apls] model {name} (epoch {ck.get('epoch')}) on {args.device}")

    rows_sn, rows_osm, done = [], [], []
    for i, rec in enumerate(tiles):
        t = read_tile(rec["image"], args.target_gsd)
        if t["nodata"] > 0.2:
            rows_sn.append({"id": rec["id"], "skipped": f"{t['nodata']:.0%} nodata"})
            continue
        args._ref_name = "spacenet"
        r = {"id": rec["id"], **evaluate_tile(model, t, lines_to_pixels(
            spacenet_lines(rec["label"]), t["transform"], t["scale"]), args)}
        rows_sn.append(r)

        if args.osm:
            from rasterio.warp import transform_bounds
            try:
                b = (t["bounds"] if (t["crs"] is None or t["crs"].is_geographic)
                     else transform_bounds(t["crs"], "EPSG:4326", *t["bounds"]))
                geoms = osm_lines(b, out / "osm_cache")
                if t["crs"] is not None and not t["crs"].is_geographic:
                    import geopandas as gpd
                    geoms = list(gpd.GeoSeries(geoms, crs="EPSG:4326").to_crs(t["crs"]))
                args._ref_name = "osm"
                ro = {"id": rec["id"], **evaluate_tile(model, t, lines_to_pixels(
                    geoms, t["transform"], t["scale"]), args)}
            except Exception as e:  # noqa: BLE001
                ro = {"id": rec["id"], "skipped": f"OSM failed: {e}"}
            rows_osm.append(ro)

        if "skipped" not in r and len(done) < args.figures:
            save_figure(t, out / f"apls_{rec['id']}.png")
            done.append(rec["id"])
        msg = r.get("skipped") or (f"APLS raw {r['apls_raw']:.3f} -> healed "
                                   f"{r['apls_healed']:.3f}  relaxedIoU {r['relaxed_iou']:.3f}")
        print(f"  [{i + 1}/{len(tiles)}] {rec['id']}: {msg}")

    report = {
        "dataset": f"SpaceNet 3 roads, {args.aoi} (WorldView-3, 0.3 m, resampled to "
                   f"{args.target_gsd} m)",
        "checkpoint": str(args.checkpoint), "architecture": name, "epoch": ck.get("epoch"),
        "settings": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "spacenet": {"summary": summarise(rows_sn), "tiles": rows_sn},
        "osm": ({"summary": summarise(rows_osm), "tiles": rows_osm} if args.osm else None),
        "caveats": [
            "Out-of-domain test: the model was trained on DeepGlobe (0.5 m, mostly Asia); "
            "Las Vegas imagery is a different sensor, continent and road style.",
            "Reference centrelines are rasterised at a fixed width, so pixel metrics are "
            "indicative; APLS is the headline topology number.",
            f"APLS node tolerance {args.tolerance_m} m; the same graph construction "
            "(skeleton -> graph) is applied to prediction and reference.",
        ],
        "runtime_s": round(time.time() - t0, 1),
    }
    (out / "apls_report.json").write_text(json.dumps(report, indent=2, default=str))

    def line(name, s):
        if not s or not s.get("n_tiles"):
            return f"| {name} | 0 | - | - | - | - |"
        return (f"| {name} | {s['n_tiles']} | {s['apls_raw']} | **{s['apls_healed']}** "
                f"(±{s['apls_healed_std']}) | {s['healing_gain']:+} | {s['relaxed_iou']} |")

    md = [f"# APLS on real georeferenced imagery: {report['dataset']}", "",
          f"Model: {name}, epoch {ck.get('epoch')}. Tolerance {args.tolerance_m} m.", "",
          "| Reference | Tiles | APLS (raw graph) | APLS (healed graph) | Healing gain | Relaxed IoU |",
          "|---|---|---|---|---|---|",
          line("SpaceNet labels", report["spacenet"]["summary"])]
    if args.osm:
        md.append(line("OpenStreetMap", report["osm"]["summary"]))
    md += ["", "## Caveats", *[f"- {c}" for c in report["caveats"]], "",
           f"Figures: {', '.join('apls_' + d + '.png' for d in done)}",
           f"Runtime: {report['runtime_s']} s"]
    (out / "apls_report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
