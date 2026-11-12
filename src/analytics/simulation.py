"""
Layer 5b - Disaster simulation and what-if analysis.

A scenario removes nodes and/or edges, then we re-measure the network. The
honest way to report impact is on a *fixed* set of origin-destination pairs
sampled once from the intact network, so that before/after numbers are
comparable. Sampling fresh pairs per scenario silently changes the question.

Three impact families are reported, and they answer different questions:

  connectivity  - what fraction of OD pairs became unreachable?
                  This is the emergency-response question: can the ambulance
                  get there at all?
  detour        - for pairs still reachable, how much longer is the trip?
                  This is the daily-mobility question.
  efficiency    - global efficiency (mean inverse path length), the standard
                  network-science summary that degrades smoothly and so is
                  suitable for plotting robustness curves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import networkx as nx
import numpy as np

from .criticality import _weight_key, node_betweenness


@dataclass
class Scenario:
    name: str
    removed_nodes: list = field(default_factory=list)
    removed_edges: list = field(default_factory=list)
    description: str = ""

    def apply(self, G: nx.Graph) -> nx.Graph:
        H = G.copy()
        H.remove_edges_from([e for e in self.removed_edges if H.has_edge(*e)])
        H.remove_nodes_from([n for n in self.removed_nodes if H.has_node(n)])
        return H


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------
def flood_scenario(G: nx.Graph, center: tuple[float, float], radius_px: float,
                   name: str | None = None) -> Scenario:
    """Everything inside a radius is impassable. The simplest flood proxy;
    replace with a DEM-derived inundation polygon when elevation data exists."""
    cy, cx = center
    hit = [n for n, d in G.nodes(data=True)
           if np.hypot(d["pos"][0] - cy, d["pos"][1] - cx) <= radius_px]
    return Scenario(
        name or f"flood_r{int(radius_px)}",
        removed_nodes=hit,
        description=f"Inundation centred at ({cy:.0f}, {cx:.0f}), radius {radius_px:.0f}px",
    )


def polygon_flood_scenario(G: nx.Graph, mask: np.ndarray, name: str = "flood_mask") -> Scenario:
    """Flood defined by a raster mask (e.g. from a hydrology model or SAR water map)."""
    hit = []
    for n, d in G.nodes(data=True):
        r, c = int(round(d["pos"][0])), int(round(d["pos"][1]))
        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c]:
            hit.append(n)
    return Scenario(name, removed_nodes=hit, description="Raster-defined inundation extent")


def closure_scenario(edges: Sequence[tuple], name: str = "road_closure") -> Scenario:
    """Planned works, accident or bridge failure on specific segments."""
    return Scenario(name, removed_edges=[tuple(e) for e in edges],
                    description=f"{len(edges)} segment(s) closed")


def targeted_attack(G: nx.Graph, n_nodes: int = 5, k: int | None = None) -> Scenario:
    """Remove the highest-betweenness nodes - the worst realistic case.

    Recomputed sequentially: after removing the top node the network reorganizes,
    and the next-most-critical node is usually *not* the original number two.
    Sequential removal is materially more damaging than batch removal, and it is
    the honest upper bound on damage."""
    H, removed = G.copy(), []
    for _ in range(n_nodes):
        if H.number_of_nodes() == 0:
            break
        bc = node_betweenness(H, k=k)
        top = max(bc, key=bc.get)
        removed.append(top)
        H.remove_node(top)
    return Scenario(f"targeted_top{n_nodes}", removed_nodes=removed,
                    description="Sequential removal of highest-betweenness nodes")


def random_failure(G: nx.Graph, n_nodes: int = 5, seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes)
    chosen = rng.choice(nodes, size=min(n_nodes, len(nodes)), replace=False)
    return Scenario(f"random_{n_nodes}", removed_nodes=[int(x) for x in chosen],
                    description="Uniformly random node failures (baseline)")


# ---------------------------------------------------------------------------
# Impact measurement
# ---------------------------------------------------------------------------
def sample_od_pairs(G: nx.Graph, n_pairs: int = 300, seed: int = 42) -> list[tuple]:
    """Fixed OD set drawn from the giant component of the intact network."""
    rng = np.random.default_rng(seed)
    if G.number_of_nodes() < 2:
        return []
    giant = list(max(nx.connected_components(G), key=len))
    if len(giant) < 2:
        return []
    pairs = set()
    guard = 0
    while len(pairs) < n_pairs and guard < n_pairs * 20:
        a, b = rng.choice(len(giant), size=2, replace=False)
        pairs.add((giant[a], giant[b]))
        guard += 1
    return sorted(pairs)


def _path_lengths(G: nx.Graph, pairs, weight: str) -> dict:
    """Shortest-path length per OD pair; inf when unreachable.

    Grouped by source so we run one Dijkstra per distinct origin instead of one
    per pair - typically a 10-50x saving on realistic OD sets."""
    by_src: dict = {}
    for s, t in pairs:
        by_src.setdefault(s, []).append(t)

    out = {}
    for s, targets in by_src.items():
        if s not in G:
            for t in targets:
                out[(s, t)] = float("inf")
            continue
        dist = nx.single_source_dijkstra_path_length(G, s, weight=weight)
        for t in targets:
            out[(s, t)] = dist.get(t, float("inf"))
    return out


def global_efficiency(G: nx.Graph, pairs, weight: str) -> float:
    """Mean inverse shortest-path length over the fixed OD set."""
    if not pairs:
        return 0.0
    d = _path_lengths(G, pairs, weight)
    return float(np.mean([1.0 / v if np.isfinite(v) and v > 0 else 0.0 for v in d.values()]))


def evaluate_scenario(G: nx.Graph, scenario: Scenario, pairs=None,
                      baseline: dict | None = None) -> dict:
    """Run one scenario and quantify the damage."""
    weight = _weight_key(G)
    pairs = pairs if pairs is not None else sample_od_pairs(G)
    base = baseline if baseline is not None else _path_lengths(G, pairs, weight)

    H = scenario.apply(G)
    after = _path_lengths(H, pairs, weight)

    disconnected, detours = 0, []
    for p in pairs:
        b, a = base[p], after[p]
        if not np.isfinite(b):
            continue  # unreachable even before: not this scenario's fault
        if not np.isfinite(a):
            disconnected += 1
        elif b > 0:
            detours.append(a / b)

    n_valid = sum(1 for p in pairs if np.isfinite(base[p]))
    eff_before = float(np.mean([1 / v if np.isfinite(v) and v > 0 else 0 for v in base.values()]))
    eff_after = float(np.mean([1 / v if np.isfinite(v) and v > 0 else 0 for v in after.values()]))

    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "removed_nodes": len(scenario.removed_nodes),
        "removed_edges": len(scenario.removed_edges),
        "nodes_remaining": H.number_of_nodes(),
        "components_after": nx.number_connected_components(H) if H.number_of_nodes() else 0,
        "giant_fraction_after": (
            round(len(max(nx.connected_components(H), key=len)) / G.number_of_nodes(), 4)
            if H.number_of_nodes() else 0.0
        ),
        "od_pairs": n_valid,
        "od_disconnected": disconnected,
        "disconnection_rate": round(disconnected / max(n_valid, 1), 4),
        "mean_detour_ratio": round(float(np.mean(detours)), 4) if detours else None,
        "p95_detour_ratio": round(float(np.percentile(detours, 95)), 4) if detours else None,
        "worst_detour_ratio": round(float(np.max(detours)), 4) if detours else None,
        "efficiency_before": round(eff_before, 6),
        "efficiency_after": round(eff_after, 6),
        "efficiency_retained": round(eff_after / eff_before, 4) if eff_before > 0 else 0.0,
        "_graph": H,
    }


def alternate_route(G: nx.Graph, scenario: Scenario, source, target) -> dict:
    """Before/after route geometry for the dashboard's rerouting view."""
    weight = _weight_key(G)
    H = scenario.apply(G)

    def route(graph):
        try:
            path = nx.shortest_path(graph, source, target, weight=weight)
            length = nx.shortest_path_length(graph, source, target, weight=weight)
            return path, length
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, float("inf")

    p0, l0 = route(G)
    p1, l1 = route(H)
    return {
        "source": source, "target": target,
        "original_path": p0, "original_length_m": l0,
        "alternate_path": p1, "alternate_length_m": l1,
        "reachable": p1 is not None,
        "detour_ratio": round(l1 / l0, 4) if np.isfinite(l1) and l0 > 0 else None,
        "extra_distance_m": round(l1 - l0, 1) if np.isfinite(l1) else None,
    }


def run_scenario_suite(G: nx.Graph, scenarios: Iterable[Scenario],
                       n_pairs: int = 300) -> list[dict]:
    """Evaluate many scenarios against one shared baseline (and one OD sample)."""
    weight = _weight_key(G)
    pairs = sample_od_pairs(G, n_pairs)
    baseline = _path_lengths(G, pairs, weight)
    results = []
    for sc in scenarios:
        r = evaluate_scenario(G, sc, pairs, baseline)
        r.pop("_graph", None)
        results.append(r)
    return results
