"""
Coordinate trip planner service: A, B (lat, lon) -> route, satellite check of
every road, the roads this trip depends on, and re-planning under closures.

    plan()      download (or load cached) OSM roads + latest clear Sentinel-2,
                run the S2 model, score every road, route A -> B
    reroute()   same trip under closures / a flood circle / "avoid roads the
                satellite could not confirm"

Satellite steps are best effort: if there is no internet, no S2 checkpoint or
no usable scene, the plan still runs on OpenStreetMap alone and says why the
satellite check is missing. Nothing is invented in its place.
"""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict

import numpy as np

from src.routing import planner as P
from src.routing import sources as S

PLANS: "OrderedDict[str, dict]" = OrderedDict()
MAX_PLANS = 20
CHECKABLE_MIN_WIDTH_M = 8.0      # S2 training labels dropped roads narrower than 8 m
TOLERANCE_PX = 4                 # 10 m at 2.5 m/px: OSM vs image misalignment
SNAP_WARN_M = 300

NOTE = ("Road network: OpenStreetMap. Travel time = length / typical speed for the road "
        "class (no live traffic); one-way restrictions are not modelled. 'Seen' means the "
        "Sentinel-2 model saw the road in the latest clear image. 'Hidden, joined by "
        "healing' means the model saw the road on both sides of a short gap (up to "
        f"{P.HEAL_MAX_GAP_M:.0f} m - typically a tree crown or a shadow) and the healing "
        "step joined it. 'Not seen' means it was not seen even after healing - worth "
        "checking (a closure, construction, a long tree-covered stretch or a model miss); "
        "one image cannot prove a road is closed. Roads narrower than ~8 m cannot be "
        "resolved at 10 m and are 'not checkable'.")

WEIGHTS = {"time": "time_s", "distance": "length_m"}


class PlanError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


# ------------------------------------------------------------------ helpers
def _satellite(bbox: tuple, refresh: bool, timings: dict) -> dict:
    """Cached or fresh: scene, prepared RGB, usable mask, model probability."""
    key = S.bbox_key(bbox)
    if not refresh:
        rec = S.cache_load(key)
        if rec is not None:
            if "healing" not in rec:          # cached before healing was added
                t = time.time()
                rec["healing"] = _heal(rec)
                timings["healing_s"] = round(time.time() - t, 1)
                S.cache_save(key, {k: v for k, v in rec.items() if k != "from_cache"})
            rec["from_cache"] = True
            return rec
    ckpt = S.s2_checkpoint()
    if ckpt is None:
        raise LookupError("No Sentinel-2 checkpoint found (expected checkpoints/segformer_s2/best.pt).")
    t = time.time()
    scene = S.fetch_latest_scene(bbox)
    timings["find_scene_s"] = round(time.time() - t, 1)
    t = time.time()
    win = S.read_scene_window(scene["item"], bbox)
    timings["download_image_s"] = round(time.time() - t, 1)
    rgb = S.prepare_rgb(win["stack"], win["valid"])
    t = time.time()
    seg = S.segment(rgb, ckpt)
    timings["model_s"] = round(time.time() - t, 1)
    rec = {
        "scene": {k: v for k, v in scene.items() if k != "item"},
        "model": {k: v for k, v in seg.items() if k != "prob"},
        "rgb": rgb, "valid": win["valid"], "prob": seg["prob"],
        "transform": win["transform"], "crs": win["crs"], "gsd_m": win["gsd_m"],
        "fetched_on": time.strftime("%Y-%m-%d"),
    }
    t = time.time()
    rec["healing"] = _heal(rec)
    timings["healing_s"] = round(time.time() - t, 1)
    S.cache_save(key, rec)
    rec["from_cache"] = False
    return rec


def _heal(rec: dict) -> dict:
    """Healing on the satellite mask; never allowed to break the plan."""
    try:
        return P.healing_bridges(rec["prob"], rec["transform"], valid=rec["valid"])
    except Exception as e:                           # pragma: no cover
        return {"bridges": [], "report": None, "max_gap_m": P.HEAL_MAX_GAP_M,
                "error": f"{type(e).__name__}: {e}"}


def _overlays(rec: dict) -> dict:
    rgb = rec["rgb"][::S.UPSAMPLE, ::S.UPSAMPLE]               # back to ~10 m for display
    tr = rec["transform"] * __import__("rasterio").Affine.scale(S.UPSAMPLE)
    img, bounds = S.to_wgs84_overlay(rgb, tr, rec["crs"], max_px=900)
    roads, rb = S.to_wgs84_overlay(S.roads_overlay(rec["prob"]), rec["transform"], rec["crs"],
                                   max_px=1400, resampling="nearest")
    return {"satellite": {"image": S.png_data_uri(img), "bounds": bounds},
            "model_roads": {"image": S.png_data_uri(roads), "bounds": rb}}


def _healing_summary(sat: dict, crs: str) -> dict:
    h = sat.get("healing") or {}
    br = h.get("bridges") or []
    out = {"gaps_joined": len(br), "max_gap_m": h.get("max_gap_m", P.HEAL_MAX_GAP_M)}
    if h.get("error"):
        out["error"] = h["error"]
    try:
        out["geojson"] = P.bridges_geojson(br, sat["transform"], crs)
    except Exception:                                # pragma: no cover
        out["geojson"] = {"type": "FeatureCollection", "features": []}
    return out


def _route_summary(net, r: dict) -> dict:
    if not r.get("reachable"):
        return {"reachable": False}
    return {k: r[k] for k in ("reachable", "length_m", "time_min", "coords", "edges",
                              "verified_length_m", "unverified_length_m",
                              "bridged_length_m")}


def _trip(plan: dict, closed: set, flood: dict | None, avoid_unverified: bool) -> dict:
    net, a, b, w = plan["net"], plan["src"], plan["dst"], plan["weight"]
    r = P.route(net, a, b, weight=w, closed=closed, flood=flood,
                avoid_unverified=avoid_unverified)
    crit = P.critical_roads_on_route(net, r, a, b, weight=w, top=5, closed=closed,
                                     flood=flood, avoid_unverified=avoid_unverified)
    out = {"route": _route_summary(net, r), "blocked_edges": r["blocked_edges"],
           "critical_roads": [{k: v for k, v in c.items() if k != "alt_coords"} for c in crit]}
    if crit:
        worst = crit[0]
        out["worst_case"] = {"edge": worst["edge"], "cuts_trip": worst["cuts_trip"],
                             "alternate_coords": worst.get("alt_coords")}
    return out


def _store(plan: dict) -> str:
    pid = uuid.uuid4().hex[:12]
    PLANS[pid] = plan
    while len(PLANS) > MAX_PLANS:
        PLANS.popitem(last=False)
    return pid


# --------------------------------------------------------------------- API
def plan(a: tuple, b: tuple, *, satellite: bool = True, refresh: bool = False,
         optimise: str = "time") -> dict:
    t0 = time.time()
    timings: dict = {}
    try:
        bbox = S.snap_bbox(P.bbox_for(a, b))
    except ValueError as e:
        raise PlanError(422, str(e))

    sat, sat_reason = None, None
    if satellite:
        try:
            sat = _satellite(bbox, refresh, timings)
        except Exception as e:                       # no internet / checkpoint / scene
            sat_reason = f"Satellite check skipped: {e}"
    else:
        sat_reason = "Satellite check not requested."

    t = time.time()
    try:
        edges = S.fetch_osm_edges(bbox)
    except Exception as e:
        raise PlanError(503, f"Could not download the OpenStreetMap road network ({e}). "
                             "An internet connection is needed the first time an area is used.")
    timings["osm_s"] = round(time.time() - t, 1)
    if edges is None or len(edges) == 0:
        raise PlanError(422, "OpenStreetMap has no drivable roads in this area.")

    crs = sat["crs"] if sat else P.utm_crs((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    net = P.build_network(edges, crs)
    counts = None
    new_roads = None
    overlays = None
    if sat:
        try:        # a failure here must not lose the OSM plan
            counts = P.satellite_support(net, sat["prob"], sat["transform"],
                                         checkable_min_width_m=CHECKABLE_MIN_WIDTH_M,
                                         tolerance_px=TOLERANCE_PX, valid=sat["valid"],
                                         bridges=(sat.get("healing") or {}).get("bridges"))
            overlays = _overlays(sat)
        except Exception as e:
            sat_reason = f"Satellite check failed ({type(e).__name__}: {e})"
            sat, counts = None, None
            for eid in net.edges:                     # undo any partial scoring
                net.support[eid], net.status[eid] = None, "not_checkable"
    if sat:
        try:        # display-only extra; it must never break a plan
            new_roads = P.satellite_only_roads(net, sat["prob"], sat["transform"],
                                               valid=sat["valid"])
        except Exception as e:                       # pragma: no cover
            new_roads = {"geojson": {"type": "FeatureCollection", "features": []},
                         "count": 0, "length_km": 0.0, "error": str(e)}

    # Snap to the main connected road network, not to an isolated fragment
    # (a car park or a gated lane OSM lists without a link to the street).
    main = max(__import__("networkx").connected_components(net.G), key=len)
    src, da = P.nearest_node(net, *a, among=main)
    dst, db = P.nearest_node(net, *b, among=main)
    if src == dst:
        raise PlanError(422, "A and B snap to the same junction - choose points further apart.")
    warnings = []
    for lbl, d in (("A", da), ("B", db)):
        if d > SNAP_WARN_M:
            warnings.append(f"{lbl} is {d:.0f} m from the nearest mapped road.")
    if sat and sat["scene"].get("warning"):
        warnings.append(sat["scene"]["warning"])

    weight = WEIGHTS.get(optimise, "time_s")
    p = {"net": net, "src": src, "dst": dst, "weight": weight, "bbox": bbox}
    trip = _trip(p, set(), None, False)
    if not trip["route"]["reachable"]:
        warnings.append("A and B are not connected in the OpenStreetMap network of this area.")
    pid = _store(p)
    timings["total_s"] = round(time.time() - t0, 1)

    nodes = net.G.nodes
    return {
        "plan_id": pid,
        "bbox": list(bbox),
        "a": {"lat": a[0], "lon": a[1], "snapped": [nodes[src]["lat"], nodes[src]["lon"]],
              "snap_m": round(da, 1)},
        "b": {"lat": b[0], "lon": b[1], "snapped": [nodes[dst]["lat"], nodes[dst]["lon"]],
              "snap_m": round(db, 1)},
        "optimise": optimise,
        "roads": P.roads_geojson(net),
        "road_count": len(net.edges),
        "satellite": ({
            "available": True, "scene": sat["scene"], "model": sat.get("model"),
            "from_cache": sat["from_cache"], "fetched_on": sat.get("fetched_on"),
            "counts": counts, "overlays": overlays,
            "checkable_min_width_m": CHECKABLE_MIN_WIDTH_M,
            "new_roads": new_roads,
            "healing": _healing_summary(sat, net.crs),
        } if sat else {"available": False, "reason": sat_reason}),
        **trip,
        "warnings": warnings,
        "timings": timings,
        "note": NOTE,
    }


def reroute(pid: str, *, closed: list[int] | None = None, flood: dict | None = None,
            avoid_unverified: bool = False) -> dict:
    p = PLANS.get(pid)
    if p is None:
        raise PlanError(404, "Unknown or expired plan - plan the trip again.")
    net = p["net"]
    bad = [e for e in (closed or []) if e not in net.edges]
    if bad:
        raise PlanError(422, f"Unknown road id(s): {bad[:5]}")
    normal = P.route(net, p["src"], p["dst"], weight=p["weight"])
    trip = _trip(p, set(closed or []), flood, avoid_unverified)
    r = trip["route"]
    cmp = None
    if normal.get("reachable") and r.get("reachable"):
        cmp = {"extra_min": round(r["time_min"] - normal["time_min"], 1),
               "extra_m": round(r["length_m"] - normal["length_m"], 1)}
    return {"plan_id": pid, "normal": _route_summary(net, normal), **trip,
            "compared_to_normal": cmp,
            "conditions": {"closed": sorted(set(closed or [])), "flood": flood,
                           "avoid_unverified": avoid_unverified}}
