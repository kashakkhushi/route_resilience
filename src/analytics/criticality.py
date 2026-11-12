"""
Layer 5 - Criticality analysis.

Four complementary notions of "important", because any one alone is misleading:

  betweenness centrality  - how many shortest paths run through this node.
                            High betweenness = a load-bearing corridor.
  articulation point      - removing it *disconnects* the network outright.
                            This is the true "gatekeeper": binary, not gradual.
  bridge edge             - the edge equivalent; the only link between two regions.
  criticality score       - a blended, normalized ranking for the dashboard.

Betweenness is O(V*E) exactly (Brandes). On a healed city graph of ~10^4 nodes
that is a few seconds; on ~10^5 it is minutes, so we sample k pivots and say so
in the output rather than silently changing the definition.

Edge weights are `length_m`, so shortest paths are geometric distance. Swap in a
travel-time weight (length / free-flow speed) when road class data is available -
the code path is identical, only the weight key changes.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np


def _weight_key(G: nx.Graph) -> str:
    sample = next(iter(G.edges(data=True)), None)
    if sample and "travel_time_s" in sample[2]:
        return "travel_time_s"
    return "length_m"


def node_betweenness(G: nx.Graph, k: int | None = None, seed: int = 42) -> dict:
    """Brandes betweenness. `k` pivots gives an unbiased estimate; None = exact."""
    w = _weight_key(G)
    n = G.number_of_nodes()
    if k is not None and k >= n:
        k = None
    return nx.betweenness_centrality(G, k=k, weight=w, normalized=True, seed=seed)


def edge_betweenness(G: nx.Graph, k: int | None = None, seed: int = 42) -> dict:
    w = _weight_key(G)
    n = G.number_of_nodes()
    if k is not None and k >= n:
        k = None
    return nx.edge_betweenness_centrality(G, k=k, weight=w, normalized=True, seed=seed)


def gatekeeper_nodes(G: nx.Graph) -> list[int]:
    """Articulation points: nodes whose removal increases the component count.

    These are the nodes the pitch calls Gatekeepers. Unlike high-betweenness
    nodes, which are merely busy, these are single points of failure - losing
    one strands an entire neighbourhood."""
    return sorted(nx.articulation_points(G))


def bridge_edges(G: nx.Graph) -> list[tuple]:
    """Edges whose removal disconnects the graph. The road-segment analogue of
    a gatekeeper - typically literal bridges, flyovers and level crossings."""
    return sorted(tuple(sorted(e)) for e in nx.bridges(G))


def _normalize(d: dict) -> dict:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {k: 0.0 for k in d}
    return {k: float((v - lo) / (hi - lo)) for k, v in d.items()}


def criticality_scores(
    G: nx.Graph,
    k: int | None = None,
    w_betweenness: float = 0.5,
    w_gatekeeper: float = 0.3,
    w_degree: float = 0.1,
    w_load: float = 0.1,
) -> dict:
    """Blended 0-1 criticality per node, plus the components that produced it.

    The gatekeeper term is deliberately weighted high: a node that disconnects
    the city matters more than one that is merely busy, and the two rarely
    coincide - busy nodes usually have alternatives, which is why they are busy.
    """
    if G.number_of_nodes() == 0:
        return {"scores": {}, "components": {}, "meta": {"exact": True}}

    bc = node_betweenness(G, k=k)
    gates = set(gatekeeper_nodes(G))
    deg = dict(G.degree())

    # "Load": sum of incident road length. A junction joining six long arterials
    # carries more consequence than one joining six residential stubs.
    load = {
        n: sum(d.get("length_m", 0.0) for _, _, d in G.edges(n, data=True))
        for n in G.nodes
    }

    nb, nd, nl = _normalize(bc), _normalize(deg), _normalize(load)
    scores = {
        n: w_betweenness * nb.get(n, 0.0)
        + w_gatekeeper * (1.0 if n in gates else 0.0)
        + w_degree * nd.get(n, 0.0)
        + w_load * nl.get(n, 0.0)
        for n in G.nodes
    }
    return {
        "scores": _normalize(scores),
        "components": {"betweenness": bc, "gatekeeper": gates,
                       "degree": deg, "load": load},
        "meta": {"exact": k is None, "pivots": k, "weight": _weight_key(G)},
    }


def rank_critical_nodes(G: nx.Graph, top: int = 20, k: int | None = None) -> list[dict]:
    """Ranked table for the dashboard and the report."""
    res = criticality_scores(G, k=k)
    bc, gates = res["components"]["betweenness"], res["components"]["gatekeeper"]
    rows = [
        {
            "node": n,
            "score": round(s, 4),
            "betweenness": round(bc.get(n, 0.0), 5),
            "is_gatekeeper": n in gates,
            "degree": G.degree(n),
            "pos": G.nodes[n]["pos"],
        }
        for n, s in res["scores"].items()
    ]
    rows.sort(key=lambda r: (-r["score"], r["node"]))
    return rows[:top]


def rank_critical_edges(G: nx.Graph, top: int = 20, k: int | None = None) -> list[dict]:
    eb = edge_betweenness(G, k=k)
    brs = set(bridge_edges(G))
    rows = []
    for (u, v), b in eb.items():
        key = tuple(sorted((u, v)))
        d = G[u][v]
        rows.append({
            "edge": key,
            "betweenness": round(b, 5),
            "is_bridge": key in brs,
            "length_m": round(d.get("length_m", 0.0), 1),
            "kind": d.get("kind", "observed"),
        })
    rows.sort(key=lambda r: (-int(r["is_bridge"]), -r["betweenness"]))
    return rows[:top]


def network_statistics(G: nx.Graph) -> dict:
    """Descriptive statistics reported alongside every scenario."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n == 0:
        return {"nodes": 0, "edges": 0}

    total_len = sum(d.get("length_m", 0.0) for _, _, d in G.edges(data=True))
    healed_len = sum(d.get("length_m", 0.0) for _, _, d in G.edges(data=True)
                     if d.get("kind") == "healed")
    comps = list(nx.connected_components(G))
    giant = max(comps, key=len)
    Gg = G.subgraph(giant)

    stats = {
        "nodes": n,
        "edges": m,
        "components": len(comps),
        "giant_component_fraction": round(len(giant) / n, 4),
        "total_length_km": round(total_len / 1000.0, 3),
        "healed_length_km": round(healed_len / 1000.0, 3),
        "mean_degree": round(2 * m / n, 3),
        "dead_ends": sum(1 for _, d in G.degree() if d == 1),
        "intersections": sum(1 for _, d in G.degree() if d >= 3),
        "gatekeepers": len(gatekeeper_nodes(G)),
        "bridge_edges": len(bridge_edges(G)),
        # Meshedness / alpha index: 0 = tree (no alternatives), 1 = fully meshed.
        # This is the single best one-number summary of route redundancy.
        "alpha_index": round(
            (m - n + len(comps)) / max(2 * n - 5 * len(comps), 1), 4
        ),
    }
    if Gg.number_of_nodes() > 1 and Gg.number_of_nodes() <= 3000:
        stats["avg_shortest_path_m"] = round(
            nx.average_shortest_path_length(Gg, weight=_weight_key(G)), 1
        )
    return stats
