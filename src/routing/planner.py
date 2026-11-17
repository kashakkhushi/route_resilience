"""
Coordinate route planner: OpenStreetMap network + satellite verification.

Input is two points (lat, lon). The road network comes from OpenStreetMap, which
is reliable for *where roads were mapped*. A recent satellite image is then used
to check, road by road, whether the road is actually visible today. The routing
graph is the OSM graph; the satellite only adds evidence per road:

    support(e) = fraction of sample points along road e where the segmentation
                 model sees road within a small tolerance
    status(e)  = "verified"     support >= VERIFIED_AT
                 "bridged"      seen only after healing: the model saw the road on
                                both sides of a short gap (tree, shadow) and the
                                project's healing step (Layer 4b) joined it
                 "unverified"   not seen even after healing
                 "not_checkable" road too narrow for the sensor's resolution

"Unverified" deliberately does NOT mean "blocked": a road can be missed because
of tree cover, shadow, a dark surface or simply the model. Blocked-road detection
needs a before/after pair of images (change detection) and is out of scope; the
UI states this. What the planner offers is routing that can *avoid* roads the
satellite could not confirm, plus explicit "what if this road is closed" tests.

Everything here is pure computation on in-memory objects so it can be tested
without network access; downloading OSM and imagery lives in the API layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import shapely

# Free-flow speeds (km/h) by OSM class, for Indian urban roads. Disclosed in every
# response: travel time is distance / assumed speed, not live traffic.
SPEED_KMH = {
    "motorway": 80, "motorway_link": 50, "trunk": 60, "trunk_link": 40,
    "primary": 45, "primary_link": 35, "secondary": 40, "secondary_link": 30,
    "tertiary": 30, "tertiary_link": 25, "residential": 20, "unclassified": 20,
    "living_street": 10, "service": 15, "road": 20,
}
DEFAULT_SPEED_KMH = 20
VERIFIED_AT = 0.5            # support at or above this -> "verified"
# Longest gap the healing step may join on the satellite mask: 100 m, i.e. 10
# pixels of the 10 m sensor - a row of tree crowns or a building's shadow over a
# street, not a missing road. The image pipeline allows up to 140 m at this
# resolution; this is kept tighter because here a joined gap changes a road's
# status. Direction and the model's own evidence along the gap still decide
# whether a bridge is accepted (same cost function as the image pipeline).
HEAL_MAX_GAP_M = 100.0
MAX_SIDE_KM = 10.0           # keep downloads and CPU inference bounded (~10 km box)


@dataclass
class Network:
    """Undirected routing graph in a metric CRS plus per-edge attributes."""
    G: nx.Graph
    crs: str
    # edge id -> attributes (geometry in metric CRS and in lon/lat)
    edges: dict = field(default_factory=dict)
    support: dict = field(default_factory=dict)       # eid -> float | None
    status: dict = field(default_factory=dict)        # eid -> str


# ----------------------------------------------------------------- geometry
def bbox_for(a: tuple, b: tuple, pad_km: float = 0.6,
             max_side_km: float = MAX_SIDE_KM) -> tuple:
    """(west, south, east, north) around two (lat, lon) points, padded.

    Raises ValueError when the points are too far apart for this prototype:
    the satellite step runs a neural network on CPU over the whole box."""
    (la1, lo1), (la2, lo2) = a, b
    for la, lo in (a, b):
        if not (-85 <= la <= 85 and -180 <= lo <= 180):
            raise ValueError(f"Not a valid coordinate: {la}, {lo}")
    lat_c = (la1 + la2) / 2
    km_lat = 111.32
    km_lon = 111.32 * math.cos(math.radians(lat_c))
    side_ns = abs(la1 - la2) * km_lat + 2 * pad_km
    side_ew = abs(lo1 - lo2) * km_lon + 2 * pad_km
    if max(side_ns, side_ew) > max_side_km:
        raise ValueError(
            f"A and B are too far apart for one analysis window "
            f"({max(side_ns, side_ew):.1f} km; limit {max_side_km:.0f} km including padding). "
            "Choose points closer together.")
    dlat = pad_km / km_lat
    dlon = pad_km / km_lon
    return (min(lo1, lo2) - dlon, min(la1, la2) - dlat,
            max(lo1, lo2) + dlon, max(la1, la2) + dlat)


def utm_crs(lat: float, lon: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _road_class(h) -> str:
    if isinstance(h, (list, tuple)):
        h = h[0] if h else "road"
    if h is None or (isinstance(h, float) and math.isnan(h)) or h == "":
        return "road"
    return str(h)


# ------------------------------------------------------------------ network
def _with_node_ids(edges_gdf):
    """Make sure the table has integer `u`/`v` node ids.

    osmnx keeps u, v, key in the index; older cached files were saved without
    them. In both cases the ids are recovered: from the index if present,
    otherwise from the line endpoints (identical coordinates = same junction)."""
    g = edges_gdf
    if "u" in g.columns and "v" in g.columns:
        return g
    if g.index.names and "u" in g.index.names and "v" in g.index.names:
        return g.reset_index()
    g = g.copy()
    ids: dict = {}

    def nid(pt):
        k = (round(pt[0], 7), round(pt[1], 7))
        if k not in ids:
            ids[k] = len(ids)
        return ids[k]

    us, vs = [], []
    for geom in g.geometry:
        if geom is None or geom.is_empty or geom.geom_type != "LineString":
            us.append(-1); vs.append(-1)
            continue
        cs = list(geom.coords)
        us.append(nid(cs[0])); vs.append(nid(cs[-1]))
    g["u"], g["v"] = us, vs
    return g


def build_network(edges_gdf, crs: str) -> Network:
    """Undirected routing graph from an OSM edge table (osmnx graph_to_gdfs).

    OSM drive networks list two-way streets twice (u->v and v->u); they are
    merged into one undirected edge. Node positions come from the geometry
    endpoints, so the graph needs nothing but the edge table."""
    edges_gdf = _with_node_ids(edges_gdf)
    g_m = edges_gdf.to_crs(crs)
    g_ll = edges_gdf.to_crs("EPSG:4326")
    G = nx.Graph()
    net = Network(G=G, crs=crs)
    seen = {}
    for i in range(len(g_m)):
        row = g_m.iloc[i]
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != "LineString":
            continue
        u, v = int(row["u"]), int(row["v"])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        length = float(geom.length)
        if key in seen and net.edges[seen[key]]["length_m"] <= length:
            continue
        cls = _road_class(row.get("highway"))
        eid = seen.get(key, len(net.edges))
        seen[key] = eid
        ll = g_ll.iloc[i].geometry
        name = row.get("name")
        if isinstance(name, (list, tuple)):
            name = name[0] if name else None
        if isinstance(name, float) and math.isnan(name):
            name = None
        speed = SPEED_KMH.get(cls, DEFAULT_SPEED_KMH)
        net.edges[eid] = {
            "u": u, "v": v, "cls": cls, "name": name, "length_m": length,
            "speed_kmh": speed, "time_s": length / (speed / 3.6),
            "geom_m": geom, "geom_ll": ll,
        }
        (x0, y0), (x1, y1) = geom.coords[0], geom.coords[-1]
        (lo0, la0), (lo1, la1) = ll.coords[0], ll.coords[-1]
        # The geometry direction follows u->v in osmnx output.
        G.add_node(u, x=x0, y=y0, lon=lo0, lat=la0)
        G.add_node(v, x=x1, y=y1, lon=lo1, lat=la1)
        G.add_edge(u, v, eid=eid, length_m=length, time_s=net.edges[eid]["time_s"])
    for eid in net.edges:
        net.support[eid] = None
        net.status[eid] = "not_checkable"
    return net


# ------------------------------------------------------- satellite evidence
def satellite_support(net: Network, prob: np.ndarray, transform, *,
                      checkable_min_width_m: float, step_m: float = 10.0,
                      tolerance_px: int = 3, threshold: float = 0.5,
                      half_width_m: dict | None = None,
                      valid: np.ndarray | None = None,
                      bridges: list | None = None, bridge_width_px: int = 3) -> dict:
    """Score each road by how much of it the segmentation model sees.

    `prob` is the model's road probability on a raster whose `transform` maps
    pixel -> the network's metric CRS. A sample point counts as seen when the
    maximum probability within `tolerance_px` exceeds `threshold` (OSM
    centrelines and image roads are never pixel-aligned). Roads narrower than
    the sensor can resolve are left "not_checkable" rather than counted as
    missing - punishing a road for being invisible to a 10 m sensor would be
    the model's fault, not the road's.

    `valid` (optional, same shape as `prob`) marks pixels the image can be
    trusted on; cloud, cloud shadow and no-data pixels are False. Sample points
    there are skipped, so a road under a cloud is "not_checkable", never
    "unverified".

    `bridges` (optional) are the gaps the healing step joined on this mask, as
    [[row, col], [row, col]] pixel pairs (see `healing_bridges`). A road that is
    not verified on the raw mask but is once those bridges are drawn in is
    "bridged": occluded, recovered by healing."""
    from src.data.osm_groundtruth import DEFAULT_HALF_WIDTH_M, ROAD_HALF_WIDTH_M

    widths = half_width_m or ROAD_HALF_WIDTH_M
    inv = ~transform
    H, W = prob.shape
    import cv2

    raw = prob >= threshold
    healed = None
    if bridges:
        bm = np.zeros(raw.shape, np.uint8)
        for (r0, c0), (r1, c1) in bridges:
            cv2.line(bm, (int(round(c0)), int(round(r0))), (int(round(c1)), int(round(r1))),
                     1, thickness=bridge_width_px)
        healed = raw | (bm > 0)

    def widen(m):
        if tolerance_px <= 0:
            return m
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * tolerance_px + 1, 2 * tolerance_px + 1))
        return cv2.dilate(m.astype(np.uint8), k) > 0

    seen = widen(raw)
    seen_h = widen(healed) if healed is not None else None

    counts = {"verified": 0, "bridged": 0, "unverified": 0, "not_checkable": 0,
              "outside_image": 0}
    ok = valid if valid is not None else np.ones_like(seen, bool)
    for eid, e in net.edges.items():
        full_w = 2 * widths.get(e["cls"], DEFAULT_HALF_WIDTH_M)
        if full_w < checkable_min_width_m:
            net.support[eid], net.status[eid] = None, "not_checkable"
            counts["not_checkable"] += 1
            continue
        geom = e["geom_m"]
        n = max(2, int(geom.length // step_m) + 1)
        xy = shapely.get_coordinates(
            shapely.line_interpolate_point(geom, np.linspace(0, 1, n), normalized=True))
        cc, rr = inv * (xy[:, 0], xy[:, 1])
        cc, rr = np.round(cc).astype(int), np.round(rr).astype(int)
        inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        inside[inside] &= ok[rr[inside], cc[inside]]
        total = int(inside.sum())
        hits = int(seen[rr[inside], cc[inside]].sum())
        if total < max(2, n // 2):
            net.support[eid], net.status[eid] = None, "not_checkable"
            counts["outside_image"] += 1
            continue
        s = hits / total
        net.support[eid] = round(s, 3)
        if s >= VERIFIED_AT:
            net.status[eid] = "verified"
        elif seen_h is not None and \
                seen_h[rr[inside], cc[inside]].sum() / total >= VERIFIED_AT:
            net.status[eid] = "bridged"
        else:
            net.status[eid] = "unverified"
        counts[net.status[eid]] += 1
    return counts


# ----------------------------------------------------------------- routing
def nearest_node(net: Network, lat: float, lon: float,
                 among: set | None = None) -> tuple[int, float]:
    """Closest graph node to a point (optionally only among `among`), and its
    distance in metres."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", net.crs, always_xy=True)
    x, y = t.transform(lon, lat)
    best, bd = None, float("inf")
    for n, d in net.G.nodes(data=True):
        if among is not None and n not in among:
            continue
        dd = (d["x"] - x) ** 2 + (d["y"] - y) ** 2
        if dd < bd:
            best, bd = n, dd
    return best, math.sqrt(bd)


def _blocked_edges(net: Network, closed: set, avoid_unverified: bool,
                   flood: dict | None) -> set:
    blocked = set(closed)
    if avoid_unverified:
        blocked |= {eid for eid, s in net.status.items() if s == "unverified"}
    if flood:
        from pyproj import Transformer
        from shapely.geometry import Point

        t = Transformer.from_crs("EPSG:4326", net.crs, always_xy=True)
        x, y = t.transform(flood["lon"], flood["lat"])
        circle = Point(x, y).buffer(float(flood["radius_m"]))
        blocked |= {eid for eid, e in net.edges.items() if e["geom_m"].intersects(circle)}
    return blocked


def route(net: Network, src: int, dst: int, *, weight: str = "time_s",
          closed: set | None = None, avoid_unverified: bool = False,
          flood: dict | None = None) -> dict:
    """Fastest (or shortest) path between two nodes on the open roads.

    Blocked roads are hidden through the weight function (returning None), and
    the search is bidirectional Dijkstra - both matter on city-sized graphs,
    where this runs once per road on the route for the criticality check."""
    blocked = _blocked_edges(net, closed or set(), avoid_unverified, flood)
    out = {"blocked_edges": sorted(blocked), "reachable": False}
    wfun = (lambda u, v, d: None if d["eid"] in blocked else d[weight])
    try:
        _, nodes = nx.bidirectional_dijkstra(net.G, src, dst, weight=wfun)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return out
    eids = [net.G[a][b]["eid"] for a, b in zip(nodes[:-1], nodes[1:])]
    length = sum(net.edges[e]["length_m"] for e in eids)
    time_s = sum(net.edges[e]["time_s"] for e in eids)
    statuses = [net.status[e] for e in eids]
    out.update(
        reachable=True, nodes=nodes, edges=eids,
        length_m=round(length, 1), time_min=round(time_s / 60, 1),
        coords=_path_coords(net, nodes),
        unverified_length_m=round(sum(net.edges[e]["length_m"]
                                      for e, s in zip(eids, statuses) if s == "unverified"), 1),
        verified_length_m=round(sum(net.edges[e]["length_m"]
                                    for e, s in zip(eids, statuses) if s == "verified"), 1),
        bridged_length_m=round(sum(net.edges[e]["length_m"]
                                   for e, s in zip(eids, statuses) if s == "bridged"), 1),
    )
    return out


def _path_coords(net: Network, nodes: list) -> list:
    """[[lat, lon], ...] following each road's real geometry, in path order."""
    coords: list = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        e = net.edges[net.G[a][b]["eid"]]
        seg = list(e["geom_ll"].coords)
        if e["u"] != a:            # geometry stored u->v; walk it the right way
            seg = seg[::-1]
        pts = [[round(la, 6), round(lo, 6)] for lo, la in seg]
        coords.extend(pts if not coords else pts[1:])
    return coords


def critical_roads_on_route(net: Network, base: dict, src: int, dst: int, *,
                            weight: str = "time_s", top: int = 3,
                            closed: set | None = None, avoid_unverified: bool = False,
                            flood: dict | None = None) -> list:
    """For each road on the route: what happens to this trip if it is closed?

    Each road is removed on its own and the trip re-planned. The ones whose
    closure cuts A from B, or lengthens the trip most, are the trip's single
    points of failure. Exact (one shortest-path per road on the route), which is
    affordable because a route has tens of roads, not thousands."""
    if not base.get("reachable"):
        return []
    found = []
    # the conditions are fixed for the whole check: resolve them once
    fixed = _blocked_edges(net, closed or set(), avoid_unverified, flood)
    for eid in dict.fromkeys(base["edges"]):
        r = route(net, src, dst, weight=weight, closed=fixed | {eid})
        e = net.edges[eid]
        item = {"edge": eid, "name": e["name"], "class": e["cls"],
                "length_m": round(e["length_m"], 1),
                "midpoint": _midpoint(e["geom_ll"])}
        if not r["reachable"]:
            item.update(cuts_trip=True, extra_min=None, extra_m=None, detour_pct=None)
        else:
            # Detour measured in the unit the route was optimised for, so the
            # ranking can never show a "faster" detour around a closed road.
            if weight == "length_m":
                extra = r["length_m"] - base["length_m"]
                pct = 100 * extra / max(base["length_m"], 1e-6)
            else:
                extra = r["time_min"] - base["time_min"]
                pct = 100 * extra / max(base["time_min"], 1e-6)
            item.update(cuts_trip=False,
                        extra_min=round(r["time_min"] - base["time_min"], 1),
                        extra_m=round(r["length_m"] - base["length_m"], 1),
                        detour_pct=round(pct, 1),
                        alt_coords=r["coords"])
        found.append(item)
    # cutting roads first; then the biggest detour in the optimised unit
    found.sort(key=lambda i: (not i["cuts_trip"], -(i["detour_pct"] or 0)))
    return found[:top]


def _midpoint(ll) -> list:
    p = ll.interpolate(0.5, normalized=True)
    return [round(p.y, 6), round(p.x, 6)]


def roads_geojson(net: Network) -> dict:
    """All roads, lon/lat, with class, name, satellite support and status."""
    feats = []
    for eid, e in net.edges.items():
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[round(x, 6), round(y, 6)] for x, y in e["geom_ll"].coords]},
            "properties": {"eid": eid, "class": e["cls"], "name": e["name"],
                           "length_m": round(e["length_m"], 1),
                           "support": net.support.get(eid), "status": net.status.get(eid)},
        })
    return {"type": "FeatureCollection", "features": feats}


def healing_bridges(prob: np.ndarray, transform, *, valid: np.ndarray | None = None,
                    threshold: float = 0.5, max_gap_m: float = HEAL_MAX_GAP_M) -> dict:
    """Run the project's healing step (Layer 4b) on the satellite road mask.

    Same algorithm as the image pipeline: skeleton -> graph -> MST bridges
    between dead ends (union-find) + loop restoration, scored by distance,
    direction and the model's own probability along the gap. Here it answers
    one question per road: was a road missed only because something sat on it
    for a short stretch? Returns the accepted bridges as pixel pairs
    [[row, col], [row, col]] (small; cached with the image) and the report."""
    from src.graph.healing import HealingConfig, heal_graph
    from src.graph.skeletonize import build_raw_graph

    px = abs(transform.a)
    mask = prob >= threshold
    if valid is not None:
        mask &= valid
    if not mask.any():
        return {"bridges": [], "report": None, "max_gap_m": max_gap_m}
    gap_px = max_gap_m / px
    # The image pipeline's defaults are tuned for a 60 px gap; scale the cost
    # limits with the gap so the same rule applies at this resolution.
    cfg = HealingConfig(max_gap_px=gap_px, max_cost=gap_px * 140 / 60,
                        loop_max_cost=gap_px * 90 / 60,
                        tangent_window=max(4, int(round(30 / px))))
    G = build_raw_graph(mask, pixel_size_m=px, min_spur_len=40.0 / px)
    H, report = heal_graph(G, cfg, evidence_map=prob)
    bridges = [[[float(x) for x in H.nodes[u]["pos"]], [float(x) for x in H.nodes[v]["pos"]]]
               for u, v, d in H.edges(data=True) if d.get("kind") == "healed"]
    return {"bridges": bridges, "report": report, "max_gap_m": max_gap_m}


def bridges_geojson(bridges: list, transform, crs: str) -> dict:
    """Healing bridges as lon/lat lines, for drawing on the map."""
    from pyproj import Transformer

    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    feats = []
    for (r0, c0), (r1, c1) in bridges:
        x, y = transform * (np.array([c0, c1]) + 0.5, np.array([r0, r1]) + 0.5)
        lon, lat = to_ll.transform(x, y)
        feats.append({"type": "Feature", "properties": {},
                      "geometry": {"type": "LineString",
                                   "coordinates": [[round(float(a), 6), round(float(b), 6)]
                                                   for a, b in zip(lon, lat)]}})
    return {"type": "FeatureCollection", "features": feats}


def satellite_only_roads(net: Network, prob: np.ndarray, transform, *,
                         valid: np.ndarray | None = None, threshold: float = 0.5,
                         margin_m: float = 15.0, min_length_m: float = 200.0,
                         max_roads: int = 150) -> dict:
    """Road-like lines the satellite model sees where OpenStreetMap has no road.

    The opposite check to `satellite_support`: every OSM road is masked out
    (its assumed width plus `margin_m` for misalignment), and what remains of
    the model's road mask is thinned to centrelines. Lines shorter than
    `min_length_m` are dropped - at 10 m resolution short blobs are mostly
    noise (field edges, roofs, canals).

    These are *candidates* for unmapped roads (new tracks, roads OSM lacks),
    not confirmed roads: the Sentinel-2 model has test IoU 0.46. They are
    returned for display only and are never used for routing."""
    import cv2
    from skimage.morphology import remove_small_objects

    from src.data.osm_groundtruth import DEFAULT_HALF_WIDTH_M, ROAD_HALF_WIDTH_M
    from src.graph.skeletonize import build_raw_graph

    H, W = prob.shape
    px = abs(transform.a)
    inv = ~transform
    osm = np.zeros((H, W), np.uint8)
    for e in net.edges.values():
        xy = np.asarray(e["geom_m"].coords)
        c, r = inv * (xy[:, 0], xy[:, 1])
        pts = np.round(np.stack([c, r], 1)).astype(np.int32)
        half = ROAD_HALF_WIDTH_M.get(e["cls"], DEFAULT_HALF_WIDTH_M) + margin_m
        cv2.polylines(osm, [pts], False, 1, thickness=max(1, int(round(2 * half / px))))
    cand = (prob >= threshold) & (osm == 0)
    if valid is not None:
        cand &= valid
    cand = remove_small_objects(cand, min_size=int((min_length_m / px) * 2))
    out = {"type": "FeatureCollection", "features": []}
    if not cand.any():
        return {"geojson": out, "count": 0, "length_km": 0.0}

    G = build_raw_graph(cand, pixel_size_m=px, min_spur_len=40.0 / px)
    from pyproj import Transformer

    to_ll = Transformer.from_crs(net.crs, "EPSG:4326", always_xy=True)
    feats = []
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        length = sum(d.get("length_m", 0.0) for _, _, d in sub.edges(data=True))
        if length < min_length_m:
            continue
        lines = []
        for _, _, d in sub.edges(data=True):
            rc = np.asarray(d["pts"], float)
            x, y = transform * (rc[:, 1] + 0.5, rc[:, 0] + 0.5)
            lon, lat = to_ll.transform(np.asarray(x), np.asarray(y))
            ll = [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)]
            lines.append(ll[::3] + ([ll[-1]] if (len(ll) - 1) % 3 else []))
        lines = [l for l in lines if len(l) >= 2]
        if lines:
            feats.append((length, lines))
    feats.sort(key=lambda t: -t[0])
    for length, lines in feats[:max_roads]:
        out["features"].append({"type": "Feature",
                                "geometry": {"type": "MultiLineString", "coordinates": lines},
                                "properties": {"length_m": round(length, 1)}})
    return {"geojson": out, "count": len(out["features"]),
            "length_km": round(sum(f[0] for f in feats[:max_roads]) / 1000, 2)}
