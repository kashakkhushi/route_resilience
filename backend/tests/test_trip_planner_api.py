"""
Coordinate trip planner API (POST /api/plan, /api/plan/{id}/reroute).

The internet-facing pieces (OSM download, Sentinel-2 search/read, model run)
are replaced by synthetic stand-ins with the same signatures; everything else -
array preparation, reprojection, caching, road scoring, routing - is the real code.

    T1  offline pieces: bbox snapping, cloud fraction, RGB preparation, overlay
        reprojection, cache round trip
    T2  plan: route, every road scored, overlays, A/B snapped, timings
    T3  a second plan for the same area comes from the cache (no download)
    T4  reroute: closing a route road -> detour; closing a cut road -> unreachable;
        flood; avoid unverified
    T5  degraded modes: no checkpoint -> plan on OSM only with a reason;
        OSM unreachable -> 503; far-apart points -> 422; unknown plan -> 404

    python backend/tests/test_trip_planner_api.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name if ok else (name, detail))
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"   [{detail}]"))


def main() -> int:
    import cv2
    from pyproj import Transformer
    from rasterio.transform import from_origin

    from test_route_planner import CRS, STEP, make_edges, node_xy
    from src.routing import sources as S

    work = Path(tempfile.mkdtemp())
    os.chdir(work)
    S.CACHE = work / "plan_cache"

    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    to_m = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    edges = make_edges()
    # tertiary (8 m) is the narrowest class a 10 m sensor is asked to confirm
    edges.loc[edges["highway"] == "residential", "highway"] = "tertiary"
    calls = {"osm": 0, "scene": 0, "read": 0, "model": 0}

    def ll(i, j):
        lon, lat = to_ll.transform(*node_xy(i, j))
        return lat, lon

    MISSING = "r11-12"                       # the model does not see this road

    def fake_osm(bbox, cache_dir=None):
        calls["osm"] += 1
        return edges

    def fake_scene(bbox, **k):
        calls["scene"] += 1
        return {"item": object(), "id": "S2X_TEST_20261110", "date": "2026-11-10",
                "scene_cloud_pct": 3.0, "box_unusable_pct": 0.0,
                "candidates_checked": 1, "warning": None}

    def fake_read(item, bbox, upsample=4):
        calls["read"] += 1
        w, s, e, n = bbox
        x0, y1 = to_m.transform(w, n)
        x1, y0 = to_m.transform(e, s)
        tr = from_origin(x0, y1, 2.5, 2.5)
        H, W = int((y1 - y0) / 2.5), int((x1 - x0) / 2.5)
        stack = np.random.default_rng(0).integers(500, 3000, (H, W, 3)).astype(np.uint16)
        fake_read.tr, fake_read.shape = tr, (H, W)
        return {"stack": stack, "valid": np.ones((H, W), bool), "transform": tr,
                "crs": CRS, "gsd_m": 2.5}

    def fake_segment(rgb, ckpt):
        calls["model"] += 1
        prob = np.zeros(fake_read.shape, np.float32)
        inv = ~fake_read.tr
        g = edges.to_crs(CRS)
        for _, row in g.iterrows():
            if row["name"] == MISSING:
                continue
            a, b = row.geometry.coords[0], row.geometry.coords[-1]
            pa = tuple(int(round(v)) for v in inv * a)
            pb = tuple(int(round(v)) for v in inv * b)
            cv2.line(prob, pa, pb, 0.9, 4)
        # one road the satellite sees that OSM does not have (600 m, south of the grid)
        ya = node_xy(3, 0)[1] - 150
        pa = tuple(int(round(v)) for v in inv * (node_xy(0, 0)[0], ya))
        pb = tuple(int(round(v)) for v in inv * (node_xy(0, 3)[0], ya))
        cv2.line(prob, pa, pb, 0.9, 4)
        return {"prob": prob, "architecture": "segformer_b3", "epoch": 24,
                "checkpoint": "checkpoints/segformer_s2/best.pt", "device": "cpu"}

    print("\n[T0] real window reader on local GeoTIFFs shaped like Sentinel-2 assets")
    import rasterio
    from types import SimpleNamespace
    rng = np.random.default_rng(3)
    assets = {}
    for key, res, val in (("red", 10, None), ("green", 10, None), ("blue", 10, None),
                          ("scl", 20, 4)):
        n = 1000 if res == 10 else 500
        arr = (np.full((n, n), val, np.uint8) if val is not None
               else rng.integers(100, 4000, (n, n)).astype(np.uint16))
        if key == "scl":
            arr[:50, :] = 9                                   # cloud band along the top
        path = work / f"{key}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1,
                           dtype=arr.dtype, crs=CRS,
                           transform=from_origin(225000, 1930000, res, res)) as d:
            d.write(arr[None])
        assets[key] = SimpleNamespace(href=str(path))
    item = SimpleNamespace(assets=assets)
    w_, n_ = to_ll.transform(226000, 1929000)
    e_, s_ = to_ll.transform(229000, 1926000)
    win = S.read_scene_window(item, (w_, s_, e_, n_))
    H, W = win["valid"].shape
    check("x4 upsampled grid (~3 km -> ~1200 px)", 1150 < H < 1300 and 1150 < W < 1300, (H, W))
    check("stack matches mask", win["stack"].shape[:2] == (H, W))
    check("2.5 m transform", abs(win["transform"].a - 2.5) < 1e-6, win["transform"])
    check("clear area valid", win["valid"].mean() > 0.95, win["valid"].mean())
    with rasterio.open(assets["scl"].href) as src:
        from rasterio.windows import from_bounds as fb
        scl_top = src.read(1, window=fb(225000, 1929000, 230000, 1930000, src.transform))
    check("cloud band detected by SCL", S.cloud_fraction(scl_top) == 1.0)

    S.fetch_osm_edges = fake_osm
    S.fetch_latest_scene = fake_scene
    S.read_scene_window = fake_read
    S.segment = fake_segment
    S.s2_checkpoint = lambda: Path("checkpoints/segformer_s2/best.pt")

    print("\n[T1] offline pieces")
    bb = S.snap_bbox((78.40123, 17.40123, 78.41987, 17.41987))
    check("snap grows outward", bb[0] <= 78.40123 and bb[2] >= 78.41987 and bb[1] <= 17.40123)
    scl = np.array([[4, 4, 8, 9], [5, 6, 3, 4]])
    check("cloud fraction counts cloud+shadow", abs(S.cloud_fraction(scl) - 3 / 8) < 1e-9)
    st = np.random.default_rng(1).integers(0, 4000, (50, 60, 3)).astype(np.uint16)
    v = np.ones((50, 60), bool); v[:10] = False
    rgb = S.prepare_rgb(st, v)
    check("prepared uint8 RGB", rgb.dtype == np.uint8 and rgb.shape == (50, 60, 3))
    check("unusable pixels zeroed", rgb[:10].max() == 0)
    img, bounds = S.to_wgs84_overlay(rgb, from_origin(230000, 1925000, 10, 10), CRS)
    check("overlay bounds are lat/lon", 17 < bounds[0][0] < bounds[1][0] < 18
          and 78 < bounds[0][1] < bounds[1][1] < 79, bounds)
    check("overlay png", S.png_data_uri(img).startswith("data:image/png;base64,"))

    from fastapi.testclient import TestClient

    from backend.app.main import app

    c = TestClient(app)
    A, B = ll(1, 0), ll(1, 3)
    body = {"a": {"lat": A[0], "lon": A[1]}, "b": {"lat": B[0], "lon": B[1]},
            "optimise": "distance"}

    print("\n[T2] plan")
    r = c.post("/api/plan", json=body)
    j = r.json()
    check("status 200", r.status_code == 200, (r.status_code, str(j)[:300]))
    check("route reachable", j["route"]["reachable"])
    check("3 blocks long", abs(j["route"]["length_m"] - 3 * STEP) < 5, j["route"].get("length_m"))
    check("A snapped to its junction", j["a"]["snap_m"] < 1, j["a"]["snap_m"])
    check("satellite available", j["satellite"]["available"], j["satellite"])
    cnt = j["satellite"]["counts"]
    check("every road scored", sum(cnt.values()) == j["road_count"], cnt)
    status = {f["properties"]["name"]: f["properties"]["status"] for f in j["roads"]["features"]}
    check("unseen road unverified", status[MISSING] == "unverified", status[MISSING])
    check("seen primary road verified", status["r30-31"] == "verified", status["r30-31"])
    check("narrow service road not checkable at 10 m", status["r01-02"] == "not_checkable",
          status["r01-02"])
    check("unverified length on route reported", j["route"]["unverified_length_m"] > 0)
    nr = j["satellite"]["new_roads"]
    check("satellite-only road found (not in OSM)", nr["count"] == 1, nr.get("count"))
    check("its length is about 600 m", nr["count"] == 1
          and 450 < nr["geojson"]["features"][0]["properties"]["length_m"] < 700,
          nr["geojson"]["features"][0]["properties"] if nr["count"] else None)
    lon0, lat0 = nr["geojson"]["features"][0]["geometry"]["coordinates"][0][0] if nr["count"] else (0, 0)
    exp_lon, exp_lat = to_ll.transform(node_xy(0, 1)[0], node_xy(3, 0)[1] - 150)
    check("it is where it was drawn", abs(lat0 - exp_lat) < 2e-4, (lat0, exp_lat))
    check("OSM roads the model sees are not reported as new", nr["count"] <= 1)
    ov = j["satellite"]["overlays"]
    check("satellite + model overlays", ov["satellite"]["image"].startswith("data:image/png")
          and ov["model_roads"]["image"].startswith("data:image/png"))
    check("critical roads listed", 0 < len(j["critical_roads"]) <= 5)
    check("timings reported", "total_s" in j["timings"])
    check("scene date reported", j["satellite"]["scene"]["date"] == "2026-11-10")

    print("\n[T3] cache")
    before = dict(calls)
    j2 = c.post("/api/plan", json=body).json()
    check("second plan from cache", j2["satellite"]["from_cache"] is True)
    check("no new scene/model work", calls["scene"] == before["scene"]
          and calls["model"] == before["model"], (before, calls))
    check("same statuses from cache", {f["properties"]["name"]: f["properties"]["status"]
                                      for f in j2["roads"]["features"]} == status)

    print("\n[T3b] healing on the satellite mask")
    hs = j["satellite"].get("healing") or {}
    check("healing summary returned", "gaps_joined" in hs and hs.get("max_gap_m") == 100.0, hs.keys())
    check("bridges as map lines", hs.get("geojson", {}).get("type") == "FeatureCollection"
          and len(hs["geojson"]["features"]) == hs["gaps_joined"])
    check("'bridged' counted", "bridged" in j["satellite"]["counts"], j["satellite"]["counts"])
    check("route reports bridged length", "bridged_length_m" in j["route"])
    import json as _json
    metas = list(S.CACHE.glob("*/meta.json"))
    check("bridges cached with the image", metas and "healing" in _json.loads(metas[0].read_text()))
    # an image cached before healing existed is upgraded on first use
    m = _json.loads(metas[0].read_text()); m.pop("healing", None)
    metas[0].write_text(_json.dumps(m))
    j_old = c.post("/api/plan", json=body).json()
    check("old cache upgraded, not re-downloaded", j_old["satellite"]["from_cache"] is True
          and "healing_s" in j_old["timings"]
          and "healing" in _json.loads(metas[0].read_text()), j_old["timings"])

    print("\n[T4] reroute")
    pid = j["plan_id"]
    on_route = j["route"]["edges"]
    rr = c.post(f"/api/plan/{pid}/reroute", json={"closed": [on_route[1]]}).json()
    check("closing a route road -> detour", rr["route"]["reachable"]
          and rr["compared_to_normal"]["extra_m"] > 0, rr.get("compared_to_normal"))
    check("detour avoids it", on_route[1] not in rr["route"]["edges"])
    rr2 = c.post(f"/api/plan/{pid}/reroute", json={"avoid_unverified": True}).json()
    eid_missing = [f["properties"]["eid"] for f in j["roads"]["features"]
                   if f["properties"]["name"] == MISSING][0]
    check("avoid unverified works", rr2["route"]["reachable"]
          and eid_missing not in rr2["route"]["edges"])
    lat, lon = ll(1, 1.5)
    rr3 = c.post(f"/api/plan/{pid}/reroute",
                 json={"flood": {"lat": lat, "lon": lon, "radius_m": 60}}).json()
    check("flood blocks the road it covers", eid_missing in rr3["blocked_edges"]
          and rr3["route"]["reachable"])
    # every road around B -> cut off
    def touches_b(name):
        if "-" not in name:
            return False
        u, v = name[1:].split("-")
        return "13" in (u, v)
    b_roads = [f["properties"]["eid"] for f in j["roads"]["features"]
               if touches_b(f["properties"]["name"])]
    rr4 = c.post(f"/api/plan/{pid}/reroute", json={"closed": b_roads}).json()
    check("B cut off -> unreachable", rr4["route"]["reachable"] is False, b_roads)
    check("unknown road -> 422", c.post(f"/api/plan/{pid}/reroute",
                                        json={"closed": [99999]}).status_code == 422)

    print("\n[T4b] a failing satellite step must not lose the plan")
    from src.routing import planner as PP
    real_support, real_build = PP.satellite_support, PP.build_network
    def sat_boom(*a, **k):
        raise ValueError("bad raster")
    PP.satellite_support = sat_boom
    r = c.post("/api/plan", json=body)
    j5 = r.json()
    check("satellite step crashes -> still 200", r.status_code == 200, r.status_code)
    check("...route still found", j5.get("route", {}).get("reachable") is True)
    check("...and the reason is shown", j5.get("satellite", {}).get("available") is False
          and "bad raster" in j5["satellite"].get("reason", ""), j5.get("satellite"))
    PP.satellite_support = real_support
    def net_boom(*a, **k):
        raise KeyError("u")
    PP.build_network = net_boom
    r = c.post("/api/plan", json=body)
    check("unexpected error -> 500 that names it", r.status_code == 500
          and "KeyError" in r.json().get("detail", ""), r.text[:200])
    PP.build_network = real_build

    print("\n[T5] degraded modes")
    S.s2_checkpoint = lambda: None
    S.CACHE = work / "cache2"
    j3 = c.post("/api/plan", json=body).json()
    check("no checkpoint -> still routes", j3["route"]["reachable"])
    check("...and says why", j3["satellite"]["available"] is False
          and "checkpoint" in j3["satellite"]["reason"], j3["satellite"])
    check("...roads not checkable", all(f["properties"]["status"] == "not_checkable"
                                        for f in j3["roads"]["features"]))

    def boom(*a, **k):
        raise ConnectionError("offline")
    S.fetch_osm_edges = boom
    r = c.post("/api/plan", json=body)
    check("OSM unreachable -> 503", r.status_code == 503, r.status_code)
    far = {"a": {"lat": 17.30, "lon": 78.30}, "b": {"lat": 17.50, "lon": 78.55}}
    check("far apart -> 422", c.post("/api/plan", json=far).status_code == 422)
    check("unknown plan -> 404", c.post("/api/plan/nope/reroute", json={}).status_code == 404)

    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for item in FAIL:
        name, detail = item if isinstance(item, tuple) else (item, "")
        print(f"  FAILED: {name}  {detail}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
