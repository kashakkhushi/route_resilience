"""
Georeferencing the graph - turns pixel coordinates into real-world outputs.

Everything upstream works in (row, col) pixel space because that is what the
skeletonizer produces. Nothing is deliverable to a planner in pixel space, so
this module is the bridge: it attaches a CRS, converts lengths to true metres,
and exports GeoJSON/GraphML that opens directly in QGIS or renders on Leaflet.

The length correction is not cosmetic. Pixel lengths multiplied by a single GSD
constant are wrong in any projected CRS away from the reference meridian, and
wrong by ~15% at the edge of a UTM zone. Since every centrality and detour number
is weighted by length, that error propagates into the criticality ranking itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np


def georeference_graph(G: nx.Graph, transform, crs, compute_true_length: bool = True):
    """Attach projected coordinates and WGS84 lon/lat to every node and edge."""
    from .. data.geo_io import geo_to_wgs84, pixel_to_geo

    H = G.copy()
    nodes = list(H.nodes)
    if not nodes:
        return H

    rows = np.array([H.nodes[n]["pos"][0] for n in nodes])
    cols = np.array([H.nodes[n]["pos"][1] for n in nodes])
    xs, ys = pixel_to_geo(rows, cols, transform)
    lons, lats = geo_to_wgs84(xs, ys, crs)

    for n, x, y, lon, lat in zip(nodes, xs, ys, lons, lats):
        H.nodes[n].update(x=float(x), y=float(y), lon=float(lon), lat=float(lat))

    for u, v, d in H.edges(data=True):
        pts = d.get("pts")
        if pts is None or len(pts) < 2:
            continue
        pr = np.array([p[0] for p in pts], float)
        pc = np.array([p[1] for p in pts], float)
        ex, ey = pixel_to_geo(pr, pc, transform)
        d["geo_pts"] = list(zip(map(float, ex), map(float, ey)))
        if compute_true_length:
            # True polyline length in CRS units, not pixel_count * gsd.
            seg = np.hypot(np.diff(ex), np.diff(ey))
            d["length_m"] = float(seg.sum())

    H.graph.update(crs=str(crs), georeferenced=True)
    return H


def graph_to_geojson(G: nx.Graph, path: str | Path | None = None,
                     scores: dict | None = None,
                     gatekeepers=None) -> dict:
    """Export edges (LineString) and nodes (Point) as a WGS84 FeatureCollection.

    Criticality is written onto the features themselves so QGIS can style by it
    and Leaflet can colour segments without a second data join.
    """
    from .. data.geo_io import geo_to_wgs84

    crs = G.graph.get("crs")
    gates = set(gatekeepers or [])
    features = []

    for u, v, d in G.edges(data=True):
        geo = d.get("geo_pts")
        if geo:
            xs = np.array([p[0] for p in geo])
            ys = np.array([p[1] for p in geo])
            lon, lat = geo_to_wgs84(xs, ys, crs) if crs else (xs, ys)
            coords = [[float(a), float(b)] for a, b in zip(lon, lat)]
        elif "lon" in G.nodes[u]:
            coords = [[G.nodes[u]["lon"], G.nodes[u]["lat"]],
                      [G.nodes[v]["lon"], G.nodes[v]["lat"]]]
        else:
            continue
        edge_score = (0.5 * (scores.get(u, 0) + scores.get(v, 0))) if scores else None
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "u": int(u), "v": int(v),
                "length_m": round(d.get("length_m", 0.0), 2),
                "kind": d.get("kind", "observed"),
                "healed": d.get("kind") == "healed",
                "criticality": round(edge_score, 4) if edge_score is not None else None,
            },
        })

    for n, d in G.nodes(data=True):
        if "lon" not in d:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
            "properties": {
                "node": int(n),
                "degree": G.degree(n),
                "criticality": round(scores.get(n, 0.0), 4) if scores else None,
                "gatekeeper": n in gates,
                "kind": d.get("kind", "junction"),
            },
        })

    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
          "features": features}
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(fc))
    return fc


def save_graphml(G: nx.Graph, path: str | Path) -> Path:
    """GraphML export. Nested lists (pts/geo_pts) are not representable in
    GraphML, so they are stripped - use GeoJSON when geometry must be preserved."""
    export = G.copy()
    for _, _, d in export.edges(data=True):
        d.pop("pts", None)
        d.pop("geo_pts", None)
        for k, v in list(d.items()):
            if isinstance(v, (list, tuple, dict)):
                d.pop(k)
    for _, d in export.nodes(data=True):
        pos = d.pop("pos", None)
        if pos is not None:
            d["row"], d["col"] = float(pos[0]), float(pos[1])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(export, path)
    return Path(path)


def load_graphml(path: str | Path) -> nx.Graph:
    """Reload a GraphML graph and restore the 'pos' attribute the pipeline needs."""
    G = nx.read_graphml(path)
    H = nx.Graph()
    for n, d in G.nodes(data=True):
        nid = int(n)
        attrs = dict(d)
        r, c = float(attrs.pop("row", 0)), float(attrs.pop("col", 0))
        H.add_node(nid, pos=(r, c), **attrs)
    for u, v, d in G.edges(data=True):
        attrs = dict(d)
        attrs.setdefault("length_m", 1.0)
        attrs.setdefault("kind", "observed")
        pu, pv = H.nodes[int(u)]["pos"], H.nodes[int(v)]["pos"]
        attrs["pts"] = [pu, pv]
        H.add_edge(int(u), int(v), **attrs)
    return H
