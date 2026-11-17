"""
Data sources for the coordinate route planner.

    roads      OpenStreetMap drive network for the trip's bounding box
               (src/data/osm_groundtruth.download_osm_roads, cached on disk)
    imagery    the most recent Sentinel-2 L2A scene over the box that is
               actually cloud-free *there* (checked with the scene's own cloud
               mask, SCL), read from the public Earth Search catalogue - no login
    roads seen the fine-tuned Sentinel-2 SegFormer run on that image, prepared
               exactly as in training (x4 upsampling to 2.5 m/px, per-band
               1-99 % stretch, uint8), see scripts/prepare_liss4.py

Everything that needs the network is in fetch_* functions; the rest is pure
array work and is unit-tested offline. Results are cached under
data/plan_cache so a trip that was analysed once opens instantly (and offline).
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

STAC = "https://earth-search.aws.element84.com/v1"
UPSAMPLE = 4                     # 10 m -> 2.5 m/px, as in training
# Sentinel-2 scene classification (SCL) values that make a pixel unusable:
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8/9 cloud, 10 thin cirrus.
SCL_BAD = (0, 1, 3, 8, 9, 10)
CACHE = Path("data/plan_cache")


# -------------------------------------------------------------- bbox helpers
def snap_bbox(bbox: tuple, grid: float = 0.005) -> tuple:
    """Round a (w, s, e, n) box outwards to a ~500 m grid so nearby trips share
    one cached download and one model run."""
    w, s, e, n = bbox
    f = lambda v, up: (math.ceil if up else math.floor)(v / grid) * grid
    return (round(f(w, False), 6), round(f(s, False), 6),
            round(f(e, True), 6), round(f(n, True), 6))


def bbox_key(bbox: tuple) -> str:
    return "_".join(f"{v:.4f}" for v in bbox)


# ----------------------------------------------------------- pure array work
def cloud_fraction(scl: np.ndarray) -> float:
    """Share of unusable pixels (cloud, shadow, cirrus, no data) in an SCL window."""
    return float(np.isin(scl, SCL_BAD).mean()) if scl.size else 1.0


def prepare_rgb(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Raw Sentinel-2 R, G, B (H, W, 3) -> uint8 image prepared like the training
    tiles: per-band 1-99 percentile stretch over usable pixels, scaled to 0-255.
    Unusable pixels are set to 0."""
    stack = stack.astype(np.float32)
    ok = valid & (stack.sum(-1) > 0)
    out = np.zeros_like(stack)
    for i in range(3):
        v = stack[..., i][ok]
        lo, hi = (np.percentile(v, [1, 99]) if v.size else (0.0, 1.0))
        out[..., i] = np.clip((stack[..., i] - lo) / max(hi - lo, 1e-6), 0, 1)
    out[~ok] = 0
    return (out * 255).astype(np.uint8)


def to_wgs84_overlay(arr: np.ndarray, transform, crs, *, max_px: int = 1400,
                     resampling: str = "bilinear") -> tuple[np.ndarray, list]:
    """Reproject a (H, W) or (H, W, C) raster to lon/lat for a web-map image
    overlay. Returns (array, [[south, west], [north, east]])."""
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    a = arr if arr.ndim == 3 else arr[..., None]
    H, W, C = a.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (W, H)
    dst_tr, dw, dh = calculate_default_transform(crs, "EPSG:4326", W, H,
                                                 left, bottom, right, top)
    k = max(dw, dh) / max_px
    if k > 1:
        dw, dh = int(dw / k), int(dh / k)
        dst_tr = dst_tr * rasterio.Affine.scale(k)
    out = np.zeros((C, dh, dw), a.dtype)
    for c in range(C):
        reproject(a[..., c], out[c], src_transform=transform, src_crs=crs,
                  dst_transform=dst_tr, dst_crs="EPSG:4326",
                  resampling=getattr(Resampling, resampling))
    w, n = dst_tr * (0, 0)
    e, s = dst_tr * (dw, dh)
    res = out.transpose(1, 2, 0)
    return (res if arr.ndim == 3 else res[..., 0]), [[s, w], [n, e]]


def png_data_uri(rgb_or_rgba: np.ndarray) -> str:
    import base64

    import cv2

    a = rgb_or_rgba
    a = cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA if a.shape[-1] == 4 else cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", a)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def roads_overlay(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """RGBA: model-detected road pixels in magenta, transparent elsewhere."""
    rgba = np.zeros((*prob.shape, 4), np.uint8)
    m = prob >= threshold
    rgba[m] = (236, 72, 153, 230)
    return rgba


# ------------------------------------------------------------- network work
def fetch_osm_edges(bbox: tuple, cache_dir: Path = Path("data/osm")):
    """OSM drive network for the box (lon/lat GeoDataFrame with u, v, highway,
    name, geometry). Cached, so a repeated trip works offline."""
    from src.data.osm_groundtruth import download_osm_roads

    return download_osm_roads(bbox, network_type="drive", cache_dir=cache_dir)


def _coverage(item, bbox) -> float:
    from shapely.geometry import box, shape

    g = shape(item.geometry)
    b = box(*bbox)
    return g.intersection(b).area / b.area


def fetch_latest_scene(bbox: tuple, *, days: int = 120, max_scene_cloud: float = 60.0,
                       max_box_cloud: float = 0.05, max_checks: int = 10,
                       today: dt.date | None = None) -> dict:
    """Newest Sentinel-2 L2A scene that fully covers the box and is clear over it.

    Scene-level cloud cover says nothing about one small box, so each candidate
    (newest first) is checked with its own SCL cloud mask over the box only
    (a ~30 KB read). The first with <= max_box_cloud unusable pixels wins; if
    none qualifies, the clearest one checked is returned with a warning."""
    import rasterio
    from pystac_client import Client
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    today = today or dt.date.today()
    start = today - dt.timedelta(days=days)
    items = list(Client.open(STAC).search(
        collections=["sentinel-2-l2a"], bbox=bbox,
        datetime=f"{start.isoformat()}/{today.isoformat()}",
        query={"eo:cloud_cover": {"lt": max_scene_cloud}}, max_items=100).items())
    items = [it for it in items if _coverage(it, bbox) > 0.99]
    if not items:
        raise LookupError(f"No Sentinel-2 scene covers this area in the last {days} days.")
    items.sort(key=lambda it: it.datetime, reverse=True)

    checked = []
    for it in items[:max_checks]:
        with rasterio.open(it.assets["scl"].href) as src:
            b = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            scl = src.read(1, window=win)
        cf = cloud_fraction(scl)
        checked.append((cf, it))
        if cf <= max_box_cloud:
            break
    cf, it = min(checked, key=lambda t: t[0]) if checked[-1][0] > max_box_cloud else checked[-1]
    return {
        "item": it, "id": it.id, "date": it.datetime.date().isoformat(),
        "scene_cloud_pct": round(float(it.properties.get("eo:cloud_cover", float("nan"))), 1),
        "box_unusable_pct": round(100 * cf, 1),
        "candidates_checked": len(checked),
        "warning": (None if cf <= max_box_cloud else
                    f"No clear scene in the last {days} days; using the clearest one "
                    f"({100 * cf:.0f}% of the area under cloud or shadow). Roads under "
                    "cloud are marked not checkable."),
    }


def read_scene_window(item, bbox: tuple, upsample: int = UPSAMPLE) -> dict:
    """R, G, B at 10 m upsampled x`upsample` (bilinear, as in training) plus a
    usability mask from SCL on the same grid (nearest)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    bands, meta = [], None
    for key in ("red", "green", "blue"):
        with rasterio.open(item.assets[key].href) as src:
            b = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            shape = (int(win.height * upsample), int(win.width * upsample))
            bands.append(src.read(1, window=win, out_shape=shape,
                                  resampling=Resampling.bilinear))
            if meta is None:
                meta = {"crs": src.crs,
                        "transform": src.window_transform(win) * rasterio.Affine.scale(1 / upsample),
                        "shape": shape, "bounds_native": b}
    with rasterio.open(item.assets["scl"].href) as src:
        win = from_bounds(*meta["bounds_native"], transform=src.transform)
        scl = src.read(1, window=win, out_shape=meta["shape"], resampling=Resampling.nearest)
    return {"stack": np.stack(bands, -1), "valid": ~np.isin(scl, SCL_BAD),
            "transform": meta["transform"], "crs": str(meta["crs"]),
            "gsd_m": 10.0 / upsample}


# ------------------------------------------------------------------ model
_MODEL: dict = {}


def s2_checkpoint() -> Path | None:
    """The fine-tuned Sentinel-2 checkpoint: ROUTE_RESILIENCE_S2_CHECKPOINT, or the
    checkpoint whose folder name says Sentinel-2 (e.g. checkpoints/segformer_s2)."""
    import os

    env = os.getenv("ROUTE_RESILIENCE_S2_CHECKPOINT")
    if env and Path(env).is_file():
        return Path(env)
    root = Path("checkpoints")
    for p in sorted(root.rglob("*.pt")) if root.is_dir() else []:
        f = p.parent.name.lower()
        if "s2" in f or "sentinel" in f:
            return p
    return None


def load_model(ckpt: Path):
    """Load once per server process; a SegFormer-B3 load takes seconds."""
    key = str(ckpt)
    if key not in _MODEL:
        import torch

        from src.models.baselines import build_model

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        args = ck.get("args", {})
        name = args.get("model", "segformer_b3")
        try:
            model = build_model(name, args.get("in_channels", 3), encoder_weights=None,
                                hf_config=ck.get("hf_config"))
        except TypeError:
            model = build_model(name, args.get("in_channels", 3))
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint {ckpt} does not match '{name}' "
                               f"({len(missing)} missing, {len(unexpected)} unexpected tensors). "
                               'For SegFormer install transformers>=5.0,<6.0.')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL.clear()
        _MODEL[key] = (model.to(device).eval(), device, name, ck.get("epoch"))
    return _MODEL[key]


def segment(rgb_u8: np.ndarray, ckpt: Path) -> dict:
    """Road probability with the same tiling as evaluate_checkpoint.py
    (512 px tiles, 128 px overlap, no TTA)."""
    from src.models.inference import predict_scene

    model, device, name, epoch = load_model(ckpt)
    prob = predict_scene(model, rgb_u8, tile_size=512, overlap=128, device=device,
                         threshold=0.5, tta=False)["prob"].astype(np.float32)
    return {"prob": prob, "architecture": name, "epoch": epoch, "checkpoint": str(ckpt),
            "device": device}


# ------------------------------------------------------------------ cache
def cache_load(key: str) -> dict | None:
    d = CACHE / key
    if not (d / "meta.json").exists():
        return None
    z = np.load(d / "arrays.npz")
    meta = json.loads((d / "meta.json").read_text())
    from rasterio import Affine

    meta["transform"] = Affine(*meta["transform"])
    return {**meta, "rgb": z["rgb"], "valid": z["valid"],
            "prob": z["prob"].astype(np.float32) / 255.0}


def cache_save(key: str, rec: dict) -> None:
    d = CACHE / key
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "arrays.npz", rgb=rec["rgb"], valid=rec["valid"],
                        prob=np.round(rec["prob"] * 255).astype(np.uint8))
    meta = {k: v for k, v in rec.items() if k not in ("rgb", "valid", "prob", "transform")}
    meta["transform"] = list(rec["transform"])[:6]
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
