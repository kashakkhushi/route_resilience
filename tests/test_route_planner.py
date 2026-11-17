"""
Coordinate route planner core (src/routing/planner.py), no network needed.

A synthetic 4x4 street grid in UTM 44N (Hyderabad) plus a synthetic model
probability raster in which some roads are drawn and some are not.

    P1  network: 24 undirected edges, duplicates (u->v, v->u) merged
    P2  satellite support: drawn roads verified, missing road unverified,
        narrow road not_checkable, road under cloud not_checkable
    P3  routing: fastest A->B; closing a road on it gives a longer detour;
        closing a cut road reports unreachable
    P4  avoid_unverified routes around the unverified road
    P5  flood circle blocks every road it touches
    P6  critical roads: cutting roads ranked first; result is the true effect
    P7  bbox guard: far-apart points are refused
    python tests/test_route_planner.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name if ok else (name, detail))
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"   [{detail}]"))


CRS = "EPSG:32644"
X0, Y0, STEP = 230000.0, 1925000.0, 200.0        # grid origin (m), block size


def node_xy(i, j):
    return X0 + j * STEP, Y0 - i * STEP


def make_edges(extra_spur=True):
    import geopandas as gpd
    from shapely.geometry import LineString

    rows, nid = [], lambda i, j: 100 + i * 4 + j
    for i in range(4):
        for j in range(4):
            for di, dj in ((0, 1), (1, 0)):
                ii, jj = i + di, j + dj
                if ii < 4 and jj < 4:
                    cls = "residential" if (i, j, ii, jj) != (0, 1, 0, 2) else "service"
                    g = LineString([node_xy(i, j), node_xy(ii, jj)])
                    rows.append({"u": nid(i, j), "v": nid(ii, jj), "key": 0,
                                 "highway": "primary" if i == 3 else cls,
                                 "name": f"r{i}{j}-{ii}{jj}", "geometry": g})
                    # osmnx lists two-way streets twice
                    rows.append({"u": nid(ii, jj), "v": nid(i, j), "key": 0,
                                 "highway": "primary" if i == 3 else cls,
                                 "name": f"r{i}{j}-{ii}{jj}",
                                 "geometry": LineString(list(g.coords)[::-1])})
    if extra_spur:                                # dead-end spur hanging off node (0,3)
        rows.append({"u": nid(0, 3), "v": 999, "key": 0, "highway": "residential",
                     "name": "spur", "geometry": LineString([node_xy(0, 3),
                                                             (X0 + 3 * STEP + 300, Y0)])})
    return gpd.GeoDataFrame(rows, crs=CRS).to_crs("EPSG:4326")   # osmnx gives lon/lat


def raster(net, draw_all_but=(), res=2.5, pad=100):
    """Model probability raster: every road drawn except those named."""
    import cv2
    from rasterio.transform import from_origin

    W = int((3 * STEP + 2 * pad + 300) / res)
    H = int((3 * STEP + 2 * pad) / res)
    tr = from_origin(X0 - pad, Y0 + pad, res, res)
    prob = np.zeros((H, W), np.float32)
    inv = ~tr
    for eid, e in net.edges.items():
        if e["name"] in draw_all_but:
            continue
        pts = [tuple(int(round(v)) for v in inv * xy) for xy in e["geom_m"].coords]
        cv2.line(prob, pts[0], pts[-1], 0.9, 3)
    return prob, tr


def _ll(x, y):
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y)
    return lat, lon


def main() -> int:
    from src.routing import planner as P

    edges = make_edges()
    net = P.build_network(edges, CRS)
    nid = lambda i, j: 100 + i * 4 + j

    print("\n[P1] network")
    check("24 grid edges + 1 spur", len(net.edges) == 25, len(net.edges))
    check("17 nodes", net.G.number_of_nodes() == 17, net.G.number_of_nodes())
    e0 = next(iter(net.edges.values()))
    check("length in metres", abs(e0["length_m"] - STEP) < 1.0, e0["length_m"])
    check("speed from class", net.edges[[k for k, e in net.edges.items()
                                         if e["cls"] == "primary"][0]]["speed_kmh"] == 45)

    print("\n[P1b] OSM table shapes (osmnx index / old cache without u, v)")
    for lbl, g in (("u,v,key in index", edges.set_index(["u", "v", "key"])),
                   ("no u,v columns", edges.drop(columns=["u", "v", "key"]))):
        n2 = P.build_network(g, CRS)
        check(f"{lbl}: same roads and junctions",
              len(n2.edges) == 25 and n2.G.number_of_nodes() == 17,
              (len(n2.edges), n2.G.number_of_nodes()))
    a2, _ = P.nearest_node(n2, 17.39, 78.40); b2, _ = P.nearest_node(n2, 17.38, 78.41)
    check("routing works on recovered junction ids", P.route(n2, a2, b2)["reachable"])

    print("\n[P2] satellite support")
    prob, tr = raster(net, draw_all_but=("r11-12",))
    valid = np.ones_like(prob, bool)
    inv = ~tr
    cx, cy = (int(v) for v in inv * node_xy(2.5, 0))          # cloud over road (2,0)-(3,0)
    valid[cy - 60: cy + 60, cx - 30: cx + 30] = False
    counts = P.satellite_support(net, prob, tr, checkable_min_width_m=6.0, valid=valid)
    st = {e["name"]: net.status[k] for k, e in net.edges.items()}
    check("drawn road verified", st["r00-01"] == "verified", st["r00-01"])
    check("missing road unverified", st["r11-12"] == "unverified", st["r11-12"])
    check("narrow service road not_checkable", st["r01-02"] == "not_checkable", st["r01-02"])
    check("road under cloud not_checkable", st["r20-30"] == "not_checkable", st["r20-30"])
    check("counts add up", sum(counts.values()) == len(net.edges), counts)

    print("\n[P2b] occlusion: a short hidden stretch is joined by healing")
    import cv2
    import geopandas as gpd
    from rasterio.transform import from_origin
    from shapely.geometry import LineString
    # Two parallel east-west roads, each split into west (300 m) / middle (60 m) / east (300 m).
    xs = [0, 300, 360, 660]
    rows = []
    for k, (y, tag) in enumerate(((0, "s"), (-400, "n"))):
        for a, b, part in zip(xs[:-1], xs[1:], ("w", "m", "e")):
            rows.append({"u": 10 * k + xs.index(a), "v": 10 * k + xs.index(b), "key": 0,
                         "highway": "primary", "name": f"{tag}{part}",
                         "geometry": LineString([(X0 + a, Y0 + y), (X0 + b, Y0 + y)])})
    occ_net = P.build_network(gpd.GeoDataFrame(rows, crs=CRS).to_crs("EPSG:4326"), CRS)
    res, pad = 2.5, 100
    tr2 = from_origin(X0 - pad, Y0 + pad, res, res)
    prob2 = np.zeros((int((400 + 2 * pad) / res), int((660 + 2 * pad) / res)), np.float32)
    inv2 = ~tr2
    def seg(x0, x1, y):
        p0 = tuple(int(round(v)) for v in inv2 * (X0 + x0, Y0 + y))
        p1 = tuple(int(round(v)) for v in inv2 * (X0 + x1, Y0 + y))
        cv2.line(prob2, p0, p1, 0.9, 3)
    seg(0, 295, 0); seg(365, 660, 0)       # south road: 70 m hidden (a tree row)
    seg(0, 150, -400)                      # north road: gone after 150 m
    raw_counts = P.satellite_support(occ_net, prob2, tr2, checkable_min_width_m=6.0)
    st0 = {e["name"]: occ_net.status[k] for k, e in occ_net.edges.items()}
    check("without healing the hidden stretch is 'not seen'", st0["sm"] == "unverified", st0)
    h = P.healing_bridges(prob2, tr2)
    check("healing joins the 70 m gap", len(h["bridges"]) >= 1, h["report"])
    cnt2 = P.satellite_support(occ_net, prob2, tr2, checkable_min_width_m=6.0,
                               bridges=h["bridges"])
    st2 = {e["name"]: occ_net.status[k] for k, e in occ_net.edges.items()}
    check("hidden stretch becomes 'bridged'", st2["sm"] == "bridged", st2)
    check("roads seen directly stay 'verified'", st2["sw"] == st2["se"] == "verified", st2)
    check("a road that is really gone stays 'not seen'",
          st2["nm"] == "unverified" and st2["ne"] == "unverified", st2)
    check("bridged counted", cnt2["bridged"] == 1 and sum(cnt2.values()) == len(occ_net.edges), cnt2)
    rt = P.route(occ_net, P.nearest_node(occ_net, *_ll(X0, Y0))[0],
                 P.nearest_node(occ_net, *_ll(X0 + 660, Y0))[0], weight="length_m")
    check("route reports bridged length", abs(rt["bridged_length_m"] - 60) < 2, rt.get("bridged_length_m"))
    rt2 = P.route(occ_net, P.nearest_node(occ_net, *_ll(X0, Y0))[0],
                  P.nearest_node(occ_net, *_ll(X0 + 660, Y0))[0], weight="length_m",
                  avoid_unverified=True)
    check("'avoid not seen' does not avoid bridged roads", rt2["reachable"])
    gj = P.bridges_geojson(h["bridges"], tr2, CRS)
    check("bridges as map lines", len(gj["features"]) == len(h["bridges"]))

    print("\n[P3] routing and closures")
    A, B = nid(1, 0), nid(1, 3)
    base = P.route(net, A, B, weight="length_m")
    check("reachable", base["reachable"])
    check("shortest is 3 blocks", abs(base["length_m"] - 3 * STEP) < 3, base["length_m"])
    check("coords start at A", np.allclose(base["coords"][0],
                                           [net.G.nodes[A]["lat"], net.G.nodes[A]["lon"]], atol=1e-5))
    check("coords end at B", np.allclose(base["coords"][-1],
                                         [net.G.nodes[B]["lat"], net.G.nodes[B]["lon"]], atol=1e-5))
    e_mid = base["edges"][1]
    r2 = P.route(net, A, B, weight="length_m", closed={e_mid})
    check("closing a road -> detour", r2["reachable"] and r2["length_m"] > base["length_m"],
          r2.get("length_m"))
    check("detour avoids the closed road", e_mid not in r2["edges"])
    spur = [k for k, e in net.edges.items() if e["name"] == "spur"][0]
    r3 = P.route(net, A, 999, closed={spur})
    check("closing the only road to a dead end -> unreachable", r3["reachable"] is False)

    print("\n[P4] avoid unverified")
    r4 = P.route(net, A, B, weight="length_m", avoid_unverified=True)
    bad = [k for k, e in net.edges.items() if e["name"] == "r11-12"][0]
    check("unverified road was on the shortest route", bad in base["edges"])
    check("avoided when asked", r4["reachable"] and bad not in r4["edges"])
    check("unverified length reported", base["unverified_length_m"] > 0)

    print("\n[P5] flood")
    from pyproj import Transformer
    x, y = node_xy(1, 1.5)
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y)
    r5 = P.route(net, A, B, weight="length_m", flood={"lat": lat, "lon": lon, "radius_m": 60})
    check("flooded road blocked", bad in r5["blocked_edges"] and bad not in r5["edges"])
    check("only nearby roads blocked", len(r5["blocked_edges"]) == 1, r5["blocked_edges"])

    print("\n[P6] critical roads on a trip")
    base2 = P.route(net, nid(1, 0), 999, weight="length_m")
    crit = P.critical_roads_on_route(net, base2, nid(1, 0), 999, weight="length_m", top=10)
    check("every road on the route checked", len(crit) == len(set(base2["edges"])))
    check("spur cuts the trip and ranks first", crit[0]["edge"] == spur and crit[0]["cuts_trip"])
    others = [c for c in crit if not c["cuts_trip"]]
    check("others have a real detour", all(c["extra_m"] is not None and c["extra_m"] >= 0
                                           for c in others) and len(others) > 0, others)
    pcts = [c["detour_pct"] for c in others]
    check("ranked by detour", pcts == sorted(pcts, reverse=True), pcts)

    print("\n[P6b] OpenStreetMap download falls back to another server when one is busy")
    import osmnx as ox
    from src.data import osm_groundtruth as O
    waits, O.OVERPASS_WAITS_S = O.OVERPASS_WAITS_S, (0, 0)
    real_g = ox.graph_from_bbox
    before = (ox.settings.overpass_url, ox.settings.overpass_rate_limit)
    tried = []
    def busy_main(**k):
        tried.append(ox.settings.overpass_url)
        if "overpass-api.de" in ox.settings.overpass_url:
            raise ValueError("'overpass-api.de' responded: 502 Bad Gateway - server at capacity")
        return "graph"
    ox.graph_from_bbox = busy_main
    try:
        got = O._graph_from_bbox_any_server((78.3, 17.4, 78.4, 17.5), "drive")
        check("main server busy -> mirror answers", got == "graph" and "kumi" in tried[-1], tried)
        check("osmnx settings restored afterwards",
              (ox.settings.overpass_url, ox.settings.overpass_rate_limit) == before)
        def down(**k):
            raise ConnectionError("unreachable")
        ox.graph_from_bbox = down
        try:
            O._graph_from_bbox_any_server((78.3, 17.4, 78.4, 17.5), "drive")
            check("all servers down -> clear error", False)
        except ConnectionError as e:
            check("all servers down -> clear error", "every OpenStreetMap" in str(e), str(e)[:80])
        def empty(**k):
            raise ValueError("No data elements in server response")
        ox.graph_from_bbox = empty
        try:
            O._graph_from_bbox_any_server((78.3, 17.4, 78.4, 17.5), "drive")
            check("empty area is not retried", False)
        except ValueError:
            check("empty area is not retried", True)
    finally:
        ox.graph_from_bbox, O.OVERPASS_WAITS_S = real_g, waits

    print("\n[P7] bbox guard")
    try:
        P.bbox_for((17.40, 78.40), (17.50, 78.50))
        check("far points refused", False)
    except ValueError:
        check("far points refused", True)
    bb = P.bbox_for((17.40, 78.40), (17.42, 78.42))
    check("near points padded box", bb[0] < 78.40 and bb[3] > 17.42, bb)
    gj = P.roads_geojson(net)
    check("geojson has every road", len(gj["features"]) == len(net.edges))

    print("\n[P8] OpenStreetMap server busy -> next server")
    import osmnx as ox
    from osmnx._errors import ResponseStatusCodeError
    from src.data import osm_groundtruth as O
    real_dl, real_waits = ox.graph_from_bbox, O.OVERPASS_WAITS_S
    O.OVERPASS_WAITS_S = (0, 0)
    tried = []
    def busy_twice(**k):
        tried.append(ox.settings.overpass_url)
        if len(tried) < 3:
            raise ResponseStatusCodeError("502 Bad Gateway: server at capacity")
        return "graph"
    ox.graph_from_bbox = busy_twice
    try:
        got = O._graph_from_bbox_any_server((78.3, 17.4, 78.4, 17.5), "drive")
        check("busy main server -> mirror answers", got == "graph" and tried[-1] != tried[0], tried)
        check("server setting restored", ox.settings.overpass_url == O.OVERPASS_SERVERS[0])
        ox.graph_from_bbox = lambda **k: (_ for _ in ()).throw(ResponseStatusCodeError("502"))
        try:
            O._graph_from_bbox_any_server((78.3, 17.4, 78.4, 17.5), "drive")
            check("all busy -> clear error", False)
        except ConnectionError as e:
            check("all busy -> clear error", "try again" in str(e), str(e)[:80])
    finally:
        ox.graph_from_bbox, O.OVERPASS_WAITS_S = real_dl, real_waits

    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for item in FAIL:
        name, detail = item if isinstance(item, tuple) else (item, "")
        print(f"  FAILED: {name}  {detail}")
    return 1 if FAIL else 0




if __name__ == "__main__":
    raise SystemExit(main())
