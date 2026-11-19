#!/usr/bin/env python
"""
Real-flood demonstration on real roads (India).

Two modes, same analysis:

  extent     a published flood-extent map (shapefile / geojson / kml / zip, e.g.
             UNOSAT "flood waters over Chennai, Dec 2015" from data.humdata.org)
             is laid over the OpenStreetMap drive network. Roads with at least
             --min-frac of their length inside the flood polygons are closed.

  satellite  two Sentinel-2 images of the same place, one BEFORE and one DURING
             a flood (e.g. Silchar, Assam). Evidence per road:
               * water    the image's own scene classification (SCL class 6 =
                          water) shows NEW water on the road (water during the
                          flood that was not water before)
               * lost     our Sentinel-2 road model saw the road before and does
                          not see it during the flood, on cloud-free pixels
             Two scenarios are reported: STRICT (water only) and BROAD
             (water + lost). Neither is a confirmed closure: they are
             satellite evidence of possible flooding.

For each scenario: share of sampled trips cut, Resilience Index (NER), mean
detour, how many pieces the network breaks into, and the flooded roads that
cost the most when closed alone. Also checks how many of the network's most
critical road segments (pre-flood betweenness) actually flooded.

Usage
-----
  python scripts/flood_roads_demo.py extent --flood data/flood/chennai_2015.zip \
      --center 13.04,80.22 --size-km 10 --name "Chennai, Dec 2015 floods"

  python scripts/flood_roads_demo.py satellite --center 24.825,92.79 --size-km 8 \
      --pre 2025-01-01:2025-04-30 --during 2025-06-01:2025-07-31 \
      --name "Silchar, Assam - 2025 monsoon floods"

Outputs in --out: summary.md, results.json, map.png (and before_after.png in
satellite mode).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics.criticality import edge_betweenness, node_betweenness  # noqa: E402
from src.analytics.simulation import (Scenario, _path_lengths,  # noqa: E402
                                      evaluate_scenario, sample_od_pairs)
from src.routing.planner import build_network, utm_crs  # noqa: E402

WATER_SCL = 6                 # Sentinel-2 scene classification: water
CHECKABLE_MIN_WIDTH_M = 8.0   # as in the Trip Planner (S2 labels dropped narrower roads)
TOLERANCE_PX = 4


# ------------------------------------------------------------------ helpers
def bbox_around(lat: float, lon: float, size_km: float) -> tuple:
    h = size_km / 2
    dlat = h / 111.32
    dlon = h / (111.32 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def pct(v) -> str:
    return "-" if v is None else f"{v:.1f}%"


EDGES_FILE = None


def osm_network(bbox, crs):
    if EDGES_FILE:
        import geopandas as gpd
        print(f"  loading OSM edges from {EDGES_FILE}")
        edges = gpd.read_file(EDGES_FILE, layer="edges")
    else:
        from src.data.osm_groundtruth import download_osm_roads
        print(f"  OSM drive network for {tuple(round(v, 4) for v in bbox)} (cached in data/osm) ...")
        edges = download_osm_roads(bbox, network_type="drive", cache_dir="data/osm")
    if edges is None or len(edges) == 0:
        sys.exit("OpenStreetMap has no drivable roads in this box.")
    net = build_network(edges, crs)
    giant = max(nx.connected_components(net.G), key=len)
    G = net.G.subgraph(giant).copy()
    print(f"  {net.G.number_of_nodes():,} junctions, {len(net.edges):,} road segments; "
          f"largest connected part {G.number_of_nodes():,} junctions")
    return net, G


def edge_samples(geom, step_m=10.0):
    import shapely
    n = max(2, int(geom.length // step_m) + 1)
    return shapely.get_coordinates(
        shapely.line_interpolate_point(geom, np.linspace(0, 1, n), normalized=True))


# ------------------------------------------------------------ flood: extent
SKIP_WORDS = ("extent", "aoi", "analysis", "reference", "permanent", "normal", "pre_",
              "boundary", "admin", "sea", "ocean", "coast", "footprint", "frame")


def list_layers(path: Path) -> list[Path]:
    """All vector layers in a file or zip (UNOSAT zips hold several shapefiles:
    flood water, reference/permanent water, analysis extent ...)."""
    import tempfile
    import zipfile
    if path.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp())
        zipfile.ZipFile(path).extractall(tmp)
        return sorted(p for ext in ("*.shp", "*.geojson", "*.json", "*.kml", "*.gpkg")
                      for p in tmp.rglob(ext))
    return [path]


def read_flood_layer(path: Path, layer: str = "", quiet: bool = False):
    import geopandas as gpd
    layers = list_layers(path)
    if not layers:
        sys.exit(f"No shapefile/geojson/kml found in {path}")
    if not quiet:
        print("  layers found:")
    for p in ([] if quiet else layers):
        try:
            g = gpd.read_file(p)
            kinds = ",".join(sorted(set(g.geometry.geom_type.dropna())))
            print(f"    - {p.name}: {len(g)} features ({kinds})")
        except Exception as e:
            print(f"    - {p.name}: unreadable ({e})")
    if layer:
        pick = [p for p in layers if layer.lower() in p.name.lower()]
        pick.sort(key=lambda p: (p.stem.lower() != layer.lower(), len(p.name)))
        if not pick:
            sys.exit(f"No layer name contains '{layer}'.")
    else:
        def score(p):
            n = p.stem.lower()
            return (("flood" in n) * 2 + ("water" in n) - 5 * any(w in n for w in SKIP_WORDS)
                    - 5 * ("preflood" in n or "pre_flood" in n), n)
        pick = sorted(layers, key=score, reverse=True)
        if len(layers) > 1 and not quiet:
            print("  (several layers - pass --layer <part of the name> to choose another)")
    print(f"  USING LAYER: {pick[0].name}")
    return gpd.read_file(pick[0])


def flooded_from_extent(net, path: Path, bbox, min_frac: float, layer: str = "",
                        minus_layer: str = "") -> tuple[set, object]:
    import geopandas as gpd
    from shapely.geometry import box

    print(f"  reading flood extent {path} ...")
    fl = read_flood_layer(path, layer)
    if fl.crs is None:
        fl = fl.set_crs("EPSG:4326")
    fl = fl[fl.geometry.notna()]
    fl = fl[fl.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    fl_ll = fl.to_crs("EPSG:4326").clip(box(*bbox))
    if fl_ll.empty:
        sys.exit("The flood extent does not overlap this box - check --center / --size-km "
                 f"(flood file bounds: {fl.to_crs('EPSG:4326').total_bounds.round(4).tolist()}).")
    def union(g):
        return g.union_all() if hasattr(g, "union_all") else g.unary_union
    water = union(fl_ll.to_crs(net.crs).buffer(0))
    if minus_layer:
        pre = read_flood_layer(path, minus_layer, quiet=True)
        if pre.crs is None:
            pre = pre.set_crs("EPSG:4326")
        pre = pre[pre.geometry.notna()]
        pre_ll = pre.to_crs("EPSG:4326").clip(box(*bbox))
        if not pre_ll.empty:
            normal = union(pre_ll.to_crs(net.crs).buffer(0))
            before = water.area
            water = water.difference(normal)
            print(f"  minus normal (pre-flood) water: {before / 1e6:.2f} -> "
                  f"{water.area / 1e6:.2f} km2 of NEW flood water in the box")
        import geopandas as gpd
        fl_ll = gpd.GeoDataFrame(geometry=[water], crs=net.crs).to_crs("EPSG:4326")
    hit = set()
    for eid, e in net.edges.items():
        g = e["geom_m"]
        if g.length > 0 and g.intersects(water):
            if g.intersection(water).length / g.length >= min_frac:
                hit.add(eid)
    print(f"  {len(hit):,} road segments with >= {min_frac:.0%} of their length in the flood extent")
    return hit, fl_ll


# ---------------------------------------------------------- flood: satellite
def best_scene(bbox, start, end, epsg=None, max_checks=15):
    """Clearest Sentinel-2 L2A scene over the box between start and end."""
    import rasterio
    from pystac_client import Client
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    from src.routing.sources import STAC, _coverage, cloud_fraction

    items = list(Client.open(STAC).search(collections=["sentinel-2-l2a"], bbox=bbox,
                                          datetime=f"{start}/{end}", max_items=200).items())
    items = [it for it in items if _coverage(it, bbox) > 0.99]
    if epsg is not None:
        items = [it for it in items if it.properties.get("proj:epsg") in (None, epsg)]
    if not items:
        sys.exit(f"No Sentinel-2 scene fully covers the box between {start} and {end}.")
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    best = None
    for it in items[:max_checks]:
        with rasterio.open(it.assets["scl"].href) as src:
            b = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            cf = cloud_fraction(src.read(1, window=win))
        print(f"    {it.datetime.date()}  scene cloud {it.properties.get('eo:cloud_cover', 0):5.1f}%"
              f"  unusable over the box {100 * cf:5.1f}%")
        if best is None or cf < best[0]:
            best = (cf, it)
        if cf <= 0.02:
            break
    return best[1], best[0]


def read_scene(item, bbox):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    from src.routing.sources import prepare_rgb, read_scene_window

    w = read_scene_window(item, bbox)
    with rasterio.open(item.assets["scl"].href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*b, transform=src.transform)
        scl = src.read(1, window=win, out_shape=w["stack"].shape[:2],
                       resampling=Resampling.nearest)
    w["scl"] = scl
    w["rgb"] = prepare_rgb(w["stack"], w["valid"])
    w["epsg"] = int(str(w["crs"]).split(":")[-1]) if "EPSG" in str(w["crs"]).upper() else None
    return w


def road_status(net, prob, transform, valid):
    from src.routing.planner import satellite_support
    for eid in net.edges:
        net.support[eid], net.status[eid] = None, "not_checkable"
    satellite_support(net, prob, transform, checkable_min_width_m=CHECKABLE_MIN_WIDTH_M,
                      tolerance_px=TOLERANCE_PX, valid=valid)
    return dict(net.status)


def water_fraction(net, transform, new_water, ok):
    inv = ~transform
    H, W = new_water.shape
    out = {}
    for eid, e in net.edges.items():
        xy = edge_samples(e["geom_m"])
        cc, rr = inv * (xy[:, 0], xy[:, 1])
        cc, rr = np.round(cc).astype(int), np.round(rr).astype(int)
        m = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        m[m] &= ok[rr[m], cc[m]]
        out[eid] = float(new_water[rr[m], cc[m]].mean()) if m.sum() >= 2 else None
    return out


# ------------------------------------------------------------ impact
def impact(G, net, flooded: set, pairs, base, label: str) -> dict:
    rem = [(net.edges[e]["u"], net.edges[e]["v"]) for e in flooded
           if G.has_edge(net.edges[e]["u"], net.edges[e]["v"])]
    sc = Scenario(label, removed_edges=rem)
    r = evaluate_scenario(G, sc, pairs, base)
    H = sc.apply(G)
    comps = sorted((len(c) for c in nx.connected_components(H)), reverse=True)
    km = sum(net.edges[e]["length_m"] for e in flooded) / 1000
    return {
        "scenario": label, "roads_closed": len(rem), "km_closed": round(km, 2),
        "trips_cut_pct": round(100 * r["disconnection_rate"], 1),
        "resilience_index_ner": r["efficiency_retained"],
        "mean_detour_pct": (round(100 * (r["mean_detour_ratio"] - 1), 1)
                            if r.get("mean_detour_ratio") else None),
        "pieces_after": len(comps),
        "largest_piece_pct": round(100 * comps[0] / max(G.number_of_nodes(), 1), 1) if comps else 0,
    }


def worst_single_roads(G, net, flooded, pairs, base, eb, top=8):
    cand = [e for e in flooded if G.has_edge(net.edges[e]["u"], net.edges[e]["v"])]
    key = lambda e: eb.get((net.edges[e]["u"], net.edges[e]["v"]),
                           eb.get((net.edges[e]["v"], net.edges[e]["u"]), 0.0))
    cand = sorted(cand, key=key, reverse=True)[:25]
    rows = []
    for e in cand:
        ed = net.edges[e]
        r = evaluate_scenario(G, Scenario("one", removed_edges=[(ed["u"], ed["v"])]), pairs, base)
        rows.append({"road": ed["name"] or "(unnamed)", "class": ed["cls"],
                     "length_m": round(ed["length_m"]), "betweenness": round(key(e), 4),
                     "trips_cut_pct": round(100 * r["disconnection_rate"], 2),
                     "ner": r["efficiency_retained"],
                     "mean_detour_pct": (round(100 * (r["mean_detour_ratio"] - 1), 2)
                                         if r.get("mean_detour_ratio") else None)})
    rows.sort(key=lambda r: (r["ner"], -r["trips_cut_pct"]))
    return rows[:top]


def critical_overlap(G, net, flooded, eb, top=20):
    flooded_pairs = {tuple(sorted((net.edges[e]["u"], net.edges[e]["v"]))) for e in flooded}
    ranked = sorted(eb, key=lambda k: -eb[k])[:top]
    hit = sum(1 for k in ranked if tuple(sorted(k)) in flooded_pairs)
    share_all = len([k for k in flooded_pairs if G.has_edge(*k)]) / max(G.number_of_edges(), 1)
    return {"top_n": top, "flooded_among_top": hit,
            "flooded_share_overall_pct": round(100 * share_all, 1)}


# -------------------------------------------------------------------- plots
def plot_map(net, G, flooded_main, flooded_extra, out, title, flood_ll=None, lost=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10), dpi=130)
    if flood_ll is not None and not flood_ll.empty:
        flood_ll.plot(ax=ax, color="#3b82f6", alpha=0.35, zorder=0, linewidth=0)
    nodes = set(G.nodes)
    for eid, e in net.edges.items():
        if e["u"] not in nodes:
            continue
        xs, ys = e["geom_ll"].xy
        if eid in flooded_main:
            ax.plot(xs, ys, color="#dc2626", lw=1.8, zorder=3)
        elif flooded_extra and eid in flooded_extra:
            ax.plot(xs, ys, color="#f59e0b", lw=1.4, zorder=2)
        else:
            ax.plot(xs, ys, color="#94a3b8", lw=0.5, zorder=1)
    ax.plot([], [], color="#dc2626", lw=2, label="flooded / closed")
    if flooded_extra:
        ax.plot([], [], color="#f59e0b", lw=2, label="road lost from view (possible flood)")
    if flood_ll is not None:
        ax.fill([], [], color="#3b82f6", alpha=0.35, label="flood extent")
    lat = np.mean([d["lat"] for _, d in G.nodes(data=True)])
    ax.set_aspect(1 / math.cos(math.radians(lat)))
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_before_after(pre, dur, new_water, out, dates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = 4  # back to ~10 m for display
    a, b = pre["rgb"][::s, ::s], dur["rgb"][::s, ::s].copy()
    nw = new_water[::s, ::s]
    c = b.copy()
    c[nw] = (0.35 * c[nw] + 0.65 * np.array([40, 120, 255])).astype(np.uint8)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6), dpi=130)
    for x, im, t in zip(ax, (a, b, c), (f"Before ({dates[0]})", f"During flood ({dates[1]})",
                                        "New water during the flood (blue)")):
        x.imshow(im); x.set_title(t, fontsize=10); x.axis("off")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["extent", "satellite"])
    ap.add_argument("--center", required=True, help="lat,lon")
    ap.add_argument("--size-km", type=float, default=8.0)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--pivots", type=int, default=300)
    ap.add_argument("--edges", type=Path, default=None, help="cached OSM edges .gpkg (offline)")
    # extent mode
    ap.add_argument("--flood", type=Path, help="flood extent file (.shp/.geojson/.kml/.zip)")
    ap.add_argument("--layer", default="", help="part of the layer name to use inside the zip")
    ap.add_argument("--minus-layer", default="",
                    help="pre-flood (normal) water layer to subtract, so rivers, lakes and the sea "
                         "are not counted as flood")
    ap.add_argument("--list-layers", action="store_true", help="only list the layers in --flood")
    ap.add_argument("--min-frac", type=float, default=0.3,
                    help="a road is flooded if at least this share of its length is in the extent")
    # satellite mode
    ap.add_argument("--pre", default="", help="YYYY-MM-DD:YYYY-MM-DD (before the flood)")
    ap.add_argument("--during", default="", help="YYYY-MM-DD:YYYY-MM-DD (during the flood)")
    ap.add_argument("--water-frac", type=float, default=0.3,
                    help="road counted as under water if this share of it is NEW water")
    ap.add_argument("--model", type=Path, default=None, help="Sentinel-2 checkpoint (default: auto)")
    a = ap.parse_args()

    global EDGES_FILE
    EDGES_FILE = a.edges
    t0 = time.time()
    lat, lon = (float(v) for v in a.center.split(","))
    bbox = bbox_around(lat, lon, a.size_km)
    out = a.out or Path("reports") / f"flood_{a.mode}"
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or f"{a.size_km:g} km box around {lat:.4f},{lon:.4f}"
    extra, flood_ll, sat_info = set(), None, None

    if a.mode == "extent":
        if not a.flood or not a.flood.exists():
            ap.error("extent mode needs --flood <file> that exists")
        if a.list_layers:
            read_flood_layer(a.flood, a.layer)
            return
        print("[1/4] road network")
        net, G = osm_network(bbox, utm_crs(lat, lon))
        print("[2/4] flood extent -> flooded roads")
        flooded, flood_ll = flooded_from_extent(net, a.flood, bbox, a.min_frac, a.layer, a.minus_layer)
        scenarios = [("flood extent", flooded)]
        main_set = flooded
    else:
        if not (a.pre and a.during):
            ap.error("satellite mode needs --pre and --during date ranges")
        from src.routing.sources import s2_checkpoint, segment
        ckpt = a.model or s2_checkpoint()
        if ckpt is None or not Path(ckpt).exists():
            sys.exit("No Sentinel-2 checkpoint found (checkpoints/segformer_s2/best.pt) - pass --model")
        print("[1/4] Sentinel-2 scenes")
        print("  BEFORE:")
        it_pre, cf_pre = best_scene(bbox, *a.pre.split(":"))
        pre = read_scene(it_pre, bbox)
        print("  DURING:")
        it_dur, cf_dur = best_scene(bbox, *a.during.split(":"), epsg=pre["epsg"])
        dur = read_scene(it_dur, bbox)
        if dur["rgb"].shape != pre["rgb"].shape or str(dur["crs"]) != str(pre["crs"]):
            sys.exit("The two scenes are on different grids - choose dates on the same tile.")
        dates = (str(it_pre.datetime.date()), str(it_dur.datetime.date()))
        print(f"  using {dates[0]} (unusable {100 * cf_pre:.0f}%) and {dates[1]} "
              f"(unusable {100 * cf_dur:.0f}%)")
        if cf_dur > 0.5:
            print("  WARNING: more than half the box is cloudy during the flood; "
                  "most roads will be 'not checkable'.")
        print("[2/4] road model on both images + OSM network")
        net, G = osm_network(bbox, str(pre["crs"]))
        p_pre = segment(pre["rgb"], Path(ckpt))["prob"]
        p_dur = segment(dur["rgb"], Path(ckpt))["prob"]
        st_pre = road_status(net, p_pre, pre["transform"], pre["valid"])
        st_dur = road_status(net, p_dur, dur["transform"], dur["valid"])
        ok = pre["valid"] & dur["valid"]
        new_water = (dur["scl"] == WATER_SCL) & (pre["scl"] != WATER_SCL) & ok
        wf = water_fraction(net, pre["transform"], new_water, ok)
        water = {e for e, f in wf.items() if f is not None and f >= a.water_frac}
        lost = {e for e in net.edges if st_pre.get(e) in ("verified", "bridged")
                and st_dur.get(e) == "unverified"}
        lost_only = lost - water
        print(f"  {len(water):,} roads with new water on them; {len(lost_only):,} more roads "
              "seen before but not during the flood")
        scenarios = [("STRICT: new water on road", water),
                     ("BROAD: water + road lost from view", water | lost)]
        main_set, extra = water, lost_only
        plot_before_after(pre, dur, new_water, out / "before_after.png", dates)
        sat_info = {"before": {"id": it_pre.id, "date": dates[0],
                               "unusable_pct": round(100 * cf_pre, 1)},
                    "during": {"id": it_dur.id, "date": dates[1],
                               "unusable_pct": round(100 * cf_dur, 1)},
                    "new_water_km2": round(float(new_water.sum()) * (2.5 ** 2) / 1e6, 2),
                    "checkable_before": sum(1 for s in st_pre.values()
                                            if s in ("verified", "bridged", "unverified")),
                    "checkable_during": sum(1 for s in st_dur.values()
                                            if s in ("verified", "bridged", "unverified")),
                    "model": str(ckpt)}

    print("[3/4] impact on routing")
    pairs = sample_od_pairs(G, a.pairs, seed=42)
    base = _path_lengths(G, pairs, "length_m")
    k = a.pivots or None
    eb = edge_betweenness(G, k=k)
    results = [impact(G, net, s, pairs, base, lbl) for lbl, s in scenarios]
    worst = worst_single_roads(G, net, main_set, pairs, base, eb)
    overlap = critical_overlap(G, net, main_set, eb)
    bc = node_betweenness(G, k=k)

    print("[4/4] figures + report")
    plot_map(net, G, main_set, extra, out / "map.png", name, flood_ll=flood_ll)
    res = {"area": name, "mode": a.mode, "bbox_wgs84": [round(v, 5) for v in bbox],
           "network": {"junctions": G.number_of_nodes(), "segments": G.number_of_edges(),
                       "km": round(sum(d["length_m"] for *_, d in G.edges(data=True)) / 1000, 1)},
           "satellite": sat_info, "flood_file": str(a.flood) if a.flood else None,
           "scenarios": results, "worst_single_flooded_roads": worst,
           "critical_roads_that_flooded": overlap, "od_pairs": len(pairs),
           "runtime_s": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str))

    n = res["network"]
    L = [f"# Flood impact on roads: {name}", "",
         f"Network: OpenStreetMap drive roads, largest connected part - {n['junctions']:,} "
         f"junctions, {n['segments']:,} segments, {n['km']} km. {len(pairs)} fixed trips "
         "(origin-destination pairs); distances by road length.", ""]
    if a.mode == "extent":
        L += [f"Flood evidence: published flood-extent map `{a.flood.name}`"
              + (f" (layer '{a.layer}'" + (f" minus normal water '{a.minus_layer}'" if a.minus_layer else "")
                 + ")" if a.layer else "") + ". A road counts as "
              f"flooded if at least {a.min_frac:.0%} of its length lies inside the extent. "
              "This is an extent map, not a road-by-road survey.", ""]
    else:
        s = sat_info
        L += [f"Flood evidence: Sentinel-2 before ({s['before']['date']}, "
              f"{s['before']['unusable_pct']}% of the box unusable) and during "
              f"({s['during']['date']}, {s['during']['unusable_pct']}% unusable). "
              f"New water during the flood: {s['new_water_km2']} km2 (Sentinel-2 scene "
              "classification). Roads checkable by the 10 m model: "
              f"{s['checkable_before']} before, {s['checkable_during']} during.", "",
              "STRICT = roads with new water on them. BROAD = STRICT plus roads our model saw "
              "before but cannot see during the flood (cloud-free pixels only). Both are "
              "satellite evidence of possible flooding, not confirmed closures.", ""]
    L += ["## Impact", "",
          "| scenario | roads closed | km closed | trips cut | Resilience Index (NER) | "
          "mean detour | pieces | largest piece |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['scenario']} | {r['roads_closed']} | {r['km_closed']} | "
                 f"{r['trips_cut_pct']}% | {r['resilience_index_ner']:.3f} | "
                 f"{pct(r['mean_detour_pct'])} | {r['pieces_after']} | {r['largest_piece_pct']}% |")
    L += ["", "## Flooded roads that cost the most when closed alone", "",
          "| road | class | length | trips cut | NER | mean detour |", "|---|---|---|---|---|---|"]
    for w in worst:
        L.append(f"| {w['road']} | {w['class']} | {w['length_m']} m | {w['trips_cut_pct']}% | "
                 f"{w['ner']:.3f} | {pct(w['mean_detour_pct'])} |")
    o = overlap
    L += ["", "## Did the critical roads flood?", "",
          f"Of the {o['top_n']} most critical road segments before the flood (highest "
          f"betweenness), **{o['flooded_among_top']}** were flooded, against "
          f"{o['flooded_share_overall_pct']}% of all segments.", "",
          f"Runtime {res['runtime_s']} s. Figures: map.png"
          + (", before_after.png" if a.mode == "satellite" else "") + "."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
