#!/usr/bin/env python
"""
Criticality and resilience analysis on a REAL city road network (OpenStreetMap).

Why this exists
---------------
The flood / failure scenarios in run_demo.py run on the synthetic city. This
script runs exactly the same analytics (src/analytics) on the real OSM drive
network of an area of Hyderabad (or anywhere), so the resilience numbers in the
report can come from a real city instead of an illustrative one.

What it computes
----------------
  1. network statistics            nodes, edges, km of road, junctions, dead ends,
                                   articulation points ("gatekeepers"), bridge
                                   edges, alpha (meshedness) index
  2. top junctions                 by betweenness centrality (k-pivot estimate),
                                   with the road names that meet there
  3. single-junction closures      for each top junction: remove it alone and
                                   measure trips cut, efficiency retained (NER)
                                   and mean detour, over a fixed OD sample
  4. targeted vs random failure    sequential removal of the most central
                                   junction (re-ranked each step) vs random
                                   removal averaged over several seeds
  5. optional circular "flood"     --flood lat,lon,radius_m removes every junction
                                   inside the circle (a hypothetical scenario, not
                                   a real inundation map)

Distances are road length in metres (no traffic, no one-way streets).

Usage
-----
    python scripts/real_city_resilience.py --center 17.4435,78.3772 --size-km 4
    python scripts/real_city_resilience.py --center 17.4435,78.3772 --size-km 4 \
        --flood 17.4400,78.3800,600
    python scripts/real_city_resilience.py --edges data/osm/osm_<...>.gpkg   # offline

Outputs (in --out): summary.md, results.json, map.png, robustness.png
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

from src.analytics.criticality import (bridge_edges, edge_betweenness,  # noqa: E402
                                       gatekeeper_nodes, network_statistics,
                                       node_betweenness)
from src.analytics.resilience import node_ablation_study  # noqa: E402
from src.analytics.simulation import (Scenario, _path_lengths,  # noqa: E402
                                      evaluate_scenario, sample_od_pairs)
from src.routing.planner import build_network, utm_crs  # noqa: E402


def bbox_around(lat: float, lon: float, size_km: float) -> tuple:
    h = size_km / 2
    dlat = h / 111.32
    dlon = h / (111.32 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def junction_names(G, net, n) -> str:
    names = []
    for _, _, d in G.edges(n, data=True):
        nm = net.edges[d["eid"]].get("name")
        if nm and nm not in names:
            names.append(nm)
    return " / ".join(names[:3]) if names else "(unnamed roads)"


def pct(v) -> str:
    return "-" if v is None else f"+{v:.1f}%"


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def closure(G, removed, pairs, base, name) -> dict:
    """Scenario impact over trips that do not start or end at a closed junction
    (a trip from a closed junction is not 'cut by' the closure in any useful sense)."""
    gone = set(removed)
    keep = [p for p in pairs if p[0] not in gone and p[1] not in gone]
    return strip(evaluate_scenario(G, Scenario(name, removed_nodes=list(removed)), keep,
                                   {p: base[p] for p in keep}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--center", default="17.4435,78.3772",
                    help="lat,lon of the area centre (default: Gachibowli / HITEC City)")
    ap.add_argument("--size-km", type=float, default=4.0, help="side of the square area")
    ap.add_argument("--edges", type=Path, default=None,
                    help="use a cached OSM edges .gpkg instead of downloading")
    ap.add_argument("--name", default="", help="area name for the report")
    ap.add_argument("--pivots", type=int, default=300,
                    help="betweenness pivots (k); 0 = exact (slow on big graphs)")
    ap.add_argument("--pairs", type=int, default=300, help="origin-destination pairs")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--steps", type=int, default=10, help="junctions removed in the failure curves")
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--flood", default="", help="lat,lon,radius_m (hypothetical circle)")
    ap.add_argument("--out", type=Path, default=Path("reports/real_city"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lat, lon = (float(x) for x in a.center.split(","))
    bbox = bbox_around(lat, lon, a.size_km)

    # ---- 1. road network
    import geopandas as gpd

    if a.edges:
        print(f"[1/5] loading OSM edges from {a.edges}")
        edges = gpd.read_file(a.edges, layer="edges")
        b = edges.to_crs("EPSG:4326").total_bounds
        lat, lon = (b[1] + b[3]) / 2, (b[0] + b[2]) / 2
    else:
        from src.data.osm_groundtruth import download_osm_roads
        print(f"[1/5] downloading OSM drive network for {a.size_km:g} km box around "
              f"{lat:.4f},{lon:.4f} (cached in data/osm) ...")
        edges = download_osm_roads(bbox, network_type="drive", cache_dir="data/osm")
    net = build_network(edges, utm_crs(lat, lon))
    G_all = net.G
    giant = max(nx.connected_components(G_all), key=len)
    G = G_all.subgraph(giant).copy()
    for n, d in G.nodes(data=True):
        d["pos"] = (d["x"], d["y"])
    print(f"      {G_all.number_of_nodes()} junctions, {G_all.number_of_edges()} road segments; "
          f"largest connected part {G.number_of_nodes()} junctions "
          f"({G.number_of_nodes() / max(G_all.number_of_nodes(), 1):.0%})")

    k = a.pivots or None
    stats = network_statistics(G)

    # ---- 2. top junctions by betweenness
    print(f"[2/5] betweenness centrality ({'exact' if k is None else f'{k} pivots'}) ...")
    bc = node_betweenness(G, k=k)
    gates = set(gatekeeper_nodes(G))
    bridges = bridge_edges(G)
    top = sorted(bc, key=lambda n: -bc[n])[: a.top]
    eb = edge_betweenness(G, k=k)
    top_e = sorted(eb, key=lambda e: -eb[e])[: a.top]

    # ---- 3. single-junction closures
    print(f"[3/5] closing each of the top {a.top} junctions one at a time ...")
    pairs = sample_od_pairs(G, a.pairs, seed=42)
    base = _path_lengths(G, pairs, "length_m")
    junctions = []
    for rank, n in enumerate(top, 1):
        r = closure(G, [n], pairs, base, f"close_{n}")
        nd = G.nodes[n]
        junctions.append({
            "rank": rank, "node": int(n), "lat": round(nd["lat"], 6), "lon": round(nd["lon"], 6),
            "roads": junction_names(G, net, n), "degree": G.degree(n),
            "betweenness": round(bc[n], 4), "is_articulation_point": n in gates,
            "trips_cut_pct": round(100 * r["disconnection_rate"], 2),
            "efficiency_retained": r["efficiency_retained"],
            "mean_detour_pct": (round(100 * (r["mean_detour_ratio"] - 1), 2)
                                if r["mean_detour_ratio"] else None),
            "worst_detour_pct": (round(100 * (r["worst_detour_ratio"] - 1), 1)
                                 if r["worst_detour_ratio"] else None),
        })
    top_roads = []
    for (u, v) in top_e:
        e = net.edges[G[u][v]["eid"]]
        top_roads.append({"name": e.get("name") or "(unnamed)", "class": e["cls"],
                          "length_m": round(e["length_m"]), "betweenness": round(eb[(u, v)], 4),
                          "is_bridge_edge": tuple(sorted((u, v))) in set(bridges)})

    # ---- 4. targeted vs random failure
    print(f"[4/5] failure curves: targeted (re-ranked each step) vs random x{a.random_seeds} ...")
    tgt = node_ablation_study(G, max_nodes=a.steps, strategy="betweenness",
                              n_pairs=a.pairs, seed=0, k=min(k or 10**9, 200))
    rnd_runs = [node_ablation_study(G, max_nodes=a.steps, strategy="random",
                                    n_pairs=a.pairs, seed=s) for s in range(a.random_seeds)]
    n_steps = min(len(tgt["steps"]), *(len(r["steps"]) for r in rnd_runs))
    curve = []
    for i in range(n_steps):
        curve.append({
            "removed": i + 1,
            "targeted_efficiency": tgt["steps"][i]["efficiency_ratio"],
            "targeted_trips_cut_pct": round(100 * tgt["steps"][i]["disconnection_rate"], 2),
            "random_efficiency": round(float(np.mean(
                [r["steps"][i]["efficiency_ratio"] for r in rnd_runs])), 4),
            "random_trips_cut_pct": round(100 * float(np.mean(
                [r["steps"][i]["disconnection_rate"] for r in rnd_runs])), 2),
        })

    # ---- 5. optional hypothetical flood circle
    flood = None
    if a.flood:
        fl, fo, fr = (float(x) for x in a.flood.split(","))
        print(f"[5/5] hypothetical flood: {fr:.0f} m circle at {fl},{fo}")
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", net.crs, always_xy=True)
        fx, fy = tr.transform(fo, fl)
        hit = [n for n, d in G.nodes(data=True) if math.hypot(d["x"] - fx, d["y"] - fy) <= fr]
        r = closure(G, hit, pairs, base, "flood_circle")
        flood = {"lat": fl, "lon": fo, "radius_m": fr, "junctions_removed": len(hit),
                 "trips_cut_pct": round(100 * r["disconnection_rate"], 2),
                 "efficiency_retained": r["efficiency_retained"],
                 "mean_detour_pct": (round(100 * (r["mean_detour_ratio"] - 1), 2)
                                     if r["mean_detour_ratio"] else None)}
    else:
        print("[5/5] no --flood given, skipped")

    area = a.name or (f"OSM area from {a.edges.name}" if a.edges
                      else f"{a.size_km:g} km box around {lat:.4f},{lon:.4f}")
    res = {"area": area,
           "source": "OpenStreetMap drive network (osmnx), largest connected part",
           "weight": "road length (m); no traffic, no one-way",
           "pivots": k, "od_pairs": len(pairs), "network": stats,
           "top_junctions": junctions, "top_roads": top_roads,
           "failure_curve": curve, "flood": flood,
           "runtime_s": round(time.time() - t0, 1)}
    (a.out / "results.json").write_text(json.dumps(res, indent=2, default=str))

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10), dpi=130)
    for u, v, d in G.edges(data=True):
        xs, ys = net.edges[d["eid"]]["geom_ll"].xy
        ax.plot(xs, ys, color="#9aa3b2", lw=0.6, zorder=1)
    for u, v in bridges:
        xs, ys = net.edges[G[u][v]["eid"]]["geom_ll"].xy
        ax.plot(xs, ys, color="#f59e0b", lw=1.6, zorder=2)
    gp = [G.nodes[n] for n in gates]
    ax.scatter([p["lon"] for p in gp], [p["lat"] for p in gp], s=7, c="#f59e0b", zorder=3,
               label=f"articulation points ({len(gates)})")
    for j in junctions:
        ax.scatter(j["lon"], j["lat"], s=140, c="#dc2626", edgecolors="white", zorder=4)
        ax.annotate(str(j["rank"]), (j["lon"], j["lat"]), color="white", fontsize=7,
                    ha="center", va="center", zorder=5, weight="bold")
    if flood:
        r_deg = flood["radius_m"] / 111320
        ax.add_patch(plt.Circle((flood["lon"], flood["lat"]), r_deg, color="#2563eb",
                                alpha=0.25, zorder=0))
    ax.plot([], [], color="#f59e0b", lw=1.6, label=f"bridge edges ({len(bridges)})")
    ax.scatter([], [], s=80, c="#dc2626", label=f"top {len(junctions)} junctions by betweenness")
    ax.set_aspect(1 / math.cos(math.radians(lat)))
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"{res['area']} - OpenStreetMap drive network", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(a.out / "map.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    x = [c["removed"] for c in curve]
    ax[0].plot(x, [c["targeted_efficiency"] for c in curve], "o-", c="#dc2626", label="targeted")
    ax[0].plot(x, [c["random_efficiency"] for c in curve], "o-", c="#64748b",
               label=f"random (mean of {a.random_seeds})")
    ax[0].set_xlabel("junctions removed"); ax[0].set_ylabel("network efficiency retained (NER)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(x, [c["targeted_trips_cut_pct"] for c in curve], "o-", c="#dc2626")
    ax[1].plot(x, [c["random_trips_cut_pct"] for c in curve], "o-", c="#64748b")
    ax[1].set_xlabel("junctions removed"); ax[1].set_ylabel("trips disconnected (%)")
    ax[1].grid(alpha=0.3)
    fig.suptitle(res["area"], fontsize=10)
    fig.tight_layout(); fig.savefig(a.out / "robustness.png"); plt.close(fig)

    # ---- summary
    s = stats
    L = [f"# Real-city resilience: {res['area']}", "",
         f"Source: {res['source']}. Weight: {res['weight']}. "
         f"Betweenness: {'exact' if k is None else f'{k}-pivot estimate'}. "
         f"{len(pairs)} fixed origin-destination pairs.", "",
         "## Network", "",
         f"- {s['nodes']:,} junctions, {s['edges']:,} road segments, {s['total_length_km']} km of road",
         f"- {s['intersections']:,} intersections (3+ roads), {s['dead_ends']:,} dead ends",
         f"- {s['gatekeepers']:,} articulation points (junctions whose loss splits the network)",
         f"- {s['bridge_edges']:,} bridge edges (road segments whose loss splits the network)",
         f"- alpha (meshedness) index {s['alpha_index']}  (0 = tree, 1 = fully meshed)", "",
         f"## Top {len(junctions)} junctions by betweenness, each closed on its own", "",
         "| # | roads meeting there | lat, lon | betweenness | articulation pt | trips cut | NER | mean detour |",
         "|---|---|---|---|---|---|---|---|"]
    for j in junctions:
        L.append(f"| {j['rank']} | {j['roads']} | {j['lat']:.5f}, {j['lon']:.5f} | "
                 f"{j['betweenness']:.3f} | {'yes' if j['is_articulation_point'] else 'no'} | "
                 f"{j['trips_cut_pct']:.1f}% | {j['efficiency_retained']:.3f} | "
                 f"{pct(j['mean_detour_pct'])} |")
    L += ["", f"## Top {len(top_roads)} road segments by betweenness", "",
          "| road | class | length | betweenness | bridge edge |", "|---|---|---|---|---|"]
    for r in top_roads:
        L.append(f"| {r['name']} | {r['class']} | {r['length_m']} m | {r['betweenness']:.3f} | "
                 f"{'yes' if r['is_bridge_edge'] else 'no'} |")
    L += ["", "## Targeted vs random junction failure", "",
          "Trips that start or end at a removed junction count as cut in these curves.", "",
          "| removed | NER targeted | NER random | trips cut targeted | trips cut random |",
          "|---|---|---|---|---|"]
    for c in curve:
        L.append(f"| {c['removed']} | {c['targeted_efficiency']:.3f} | {c['random_efficiency']:.3f} | "
                 f"{c['targeted_trips_cut_pct']:.1f}% | {c['random_trips_cut_pct']:.1f}% |")
    if flood:
        L += ["", "## Hypothetical flood circle (not a real inundation map)", "",
              f"- {flood['radius_m']:.0f} m around {flood['lat']}, {flood['lon']}: "
              f"{flood['junctions_removed']} junctions closed",
              f"- trips cut {flood['trips_cut_pct']:.1f}%, NER {flood['efficiency_retained']:.3f}, "
              f"mean detour {pct(flood['mean_detour_pct'])}"]
    L += ["", f"Runtime {res['runtime_s']} s. Figures: map.png, robustness.png."]
    (a.out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
