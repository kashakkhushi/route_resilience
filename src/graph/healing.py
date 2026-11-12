"""
Layer 4b - Topology healing.

Problem: the segmentation output, however good, is fragmented wherever a tree,
shadow or cloud sat on the road. The graph therefore has many connected
components and is unusable for routing.

Solution, in the exact terms of the pitch:

    Disjoint Set  -> track which fragments are already merged, in near-O(1)
    MST (Kruskal) -> choose the cheapest set of bridges that connects everything

Why MST rather than "join every endpoint pair within N pixels": greedy pairing
creates redundant parallel bridges and ladder artefacts, and it will happily
connect two roads that merely pass close to each other. A spanning tree adds
exactly (components - 1) edges - the minimum number that can possibly restore
connectivity - and Kruskal's union-find guarantees no bridge is added between
two fragments already linked by a cheaper chain.

The cost function is the part that decides whether healing is smart or reckless:

    cost = distance * (1 + w_ang * angular_penalty) * (1 + w_ev * (1 - evidence))

    angular_penalty  a bridge should continue the road's existing direction;
                     a 90-degree hook between two endpoints is almost always wrong
    evidence         mean road probability (or occlusion probability) along the
                     proposed bridge. Bridging *through a detected tree canopy* is
                     cheap and correct; bridging across an open field is expensive.

Bridges above `max_cost` are refused outright: leaving a genuine gap is better
than inventing a road that does not exist, because a hallucinated bridge silently
corrupts every downstream centrality and rerouting number.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


class DisjointSet:
    """Union-Find with path compression and union by rank."""

    def __init__(self, items=()):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}
        self.n_sets = len(self.parent)

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.n_sets += 1
        return x

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> bool:
        """Return True if a merge happened (i.e. they were in different sets)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.n_sets -= 1
        return True

    def connected(self, a, b) -> bool:
        return self.find(a) == self.find(b)


@dataclass
class HealingConfig:
    max_gap_px: float = 60.0
    w_angle: float = 2.0
    w_evidence: float = 1.5
    max_cost: float = 140.0
    tangent_window: int = 12  # pixels used to estimate a road's local direction
    # --- redundancy (loop) restoration ---
    # A break in a ring road leaves both halves connected via the long way round,
    # so the MST pass skips it. These control the second pass that repairs such
    # gaps: bridge an intra-component pair only if going around costs
    # `detour_threshold` times more than the direct gap.
    restore_loops: bool = True
    detour_threshold: float = 6.0
    loop_max_cost: float = 90.0


def _tangent(G: nx.Graph, node: int, window: int) -> np.ndarray:
    """Unit vector pointing *outward* from the road at this endpoint."""
    pos = np.array(G.nodes[node]["pos"], dtype=float)
    edges = list(G.edges(node, data=True))
    if not edges:
        return np.array([0.0, 0.0])
    pts = np.asarray(edges[0][2]["pts"], dtype=float)
    # Orient the chain so index 0 is at our node.
    if np.linalg.norm(pts[0] - pos) > np.linalg.norm(pts[-1] - pos):
        pts = pts[::-1]
    far = pts[min(window, len(pts) - 1)]
    v = pos - far
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([0.0, 0.0])


def _line_evidence(evidence_map: np.ndarray | None, p0, p1, n_samples: int = 24) -> float:
    """Mean evidence along the straight bridge. 1.0 when no map is supplied."""
    if evidence_map is None:
        return 1.0
    ts = np.linspace(0, 1, n_samples)[:, None]
    pts = np.asarray(p0)[None] * (1 - ts) + np.asarray(p1)[None] * ts
    r = np.clip(pts[:, 0].round().astype(int), 0, evidence_map.shape[0] - 1)
    c = np.clip(pts[:, 1].round().astype(int), 0, evidence_map.shape[1] - 1)
    return float(evidence_map[r, c].mean())


def candidate_bridges(G: nx.Graph, cfg: HealingConfig,
                      evidence_map: np.ndarray | None = None,
                      same_component: bool = False) -> list[tuple]:
    """Score plausible bridges between degree-1 endpoints.

    same_component=False -> inter-fragment bridges, fed to Kruskal (pass 1)
    same_component=True  -> intra-fragment gaps, for loop restoration (pass 2)

    Returns a list of (cost, u, v, meta) sorted ascending.
    """
    from scipy.spatial import cKDTree

    endpoints = [n for n in G.nodes if G.degree(n) == 1]
    if len(endpoints) < 2:
        return []

    comp_of = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            comp_of[n] = i

    pos = np.array([G.nodes[n]["pos"] for n in endpoints])
    tree = cKDTree(pos)
    pairs = tree.query_pairs(cfg.max_gap_px, output_type="ndarray")

    out = []
    for i, j in pairs:
        u, v = endpoints[i], endpoints[j]
        if (comp_of[u] == comp_of[v]) != same_component:
            continue
        pu, pv = pos[i], pos[j]
        d = float(np.linalg.norm(pv - pu))
        if d < 1e-6:
            continue

        direction = (pv - pu) / d
        tu, tv = _tangent(G, u, cfg.tangent_window), _tangent(G, v, cfg.tangent_window)
        # Each endpoint's outward tangent should point along the bridge.
        align = 0.5 * (float(tu @ direction) + float(tv @ -direction))
        ang_pen = (1.0 - align) / 2.0  # 0 = perfect continuation, 1 = doubling back

        ev = _line_evidence(evidence_map, pu, pv)
        cost = d * (1 + cfg.w_angle * ang_pen) * (1 + cfg.w_evidence * (1 - ev))
        if cost > cfg.max_cost:
            continue
        out.append((cost, u, v, {"gap_px": d, "alignment": align, "evidence": ev}))

    out.sort(key=lambda t: t[0])
    return out


def heal_graph(G: nx.Graph, cfg: HealingConfig | None = None,
               evidence_map: np.ndarray | None = None) -> tuple[nx.Graph, dict]:
    """Kruskal over candidate bridges, using Disjoint Set to merge fragments.

    Returns the healed graph and a report suitable for the dashboard.
    """
    cfg = cfg or HealingConfig()
    H = G.copy()
    before = nx.number_connected_components(H)

    dsu = DisjointSet(H.nodes)
    for u, v in H.edges:
        dsu.union(u, v)

    candidates = candidate_bridges(H, cfg, evidence_map)
    added, rejected_cycle = [], 0

    for cost, u, v, meta in candidates:
        # This single check is what makes it an MST rather than greedy pairing:
        # if the two fragments are already linked (possibly via earlier, cheaper
        # bridges), adding this one would create a redundant cycle.
        if not dsu.union(u, v):
            rejected_cycle += 1
            continue
        pu, pv = H.nodes[u]["pos"], H.nodes[v]["pos"]
        px = H.graph.get("pixel_size_m", 1.0)
        H.add_edge(u, v,
                   pts=[tuple(pu), tuple(pv)],
                   length_px=meta["gap_px"],
                   length_m=meta["gap_px"] * px,
                   kind="healed",
                   cost=cost,
                   **meta)
        added.append((u, v, cost))

    # --- pass 2: loop restoration -----------------------------------------
    # Pass 1 is a spanning tree by construction, so it cannot repair a gap whose
    # two sides are already linked the long way round - a break in a ring road,
    # or in any grid block. Left alone, these turn a meshed city into a tree and
    # make every detour metric wildly pessimistic. Here we re-examine exactly
    # those pairs and bridge the ones where the topological distance is
    # disproportionate to the geometric gap.
    loops_added = 0
    if cfg.restore_loops:
        weight = "length_px"
        for cost, u, v, meta in candidate_bridges(H, cfg, evidence_map, same_component=True):
            if cost > cfg.loop_max_cost or H.has_edge(u, v):
                continue
            try:
                around = nx.shortest_path_length(H, u, v, weight=weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            # If the graph already offers a comparable route, the gap is probably
            # not a road - two cul-de-sacs backing onto each other, say.
            if around < cfg.detour_threshold * max(meta["gap_px"], 1e-6):
                continue
            px = H.graph.get("pixel_size_m", 1.0)
            H.add_edge(u, v,
                       pts=[tuple(H.nodes[u]["pos"]), tuple(H.nodes[v]["pos"])],
                       length_px=meta["gap_px"],
                       length_m=meta["gap_px"] * px,
                       kind="healed",
                       cost=cost,
                       detour_ratio_before=round(around / max(meta["gap_px"], 1e-6), 2),
                       **meta)
            loops_added += 1

    after = nx.number_connected_components(H)
    healed_len = sum(d["length_px"] for _, _, d in H.edges(data=True) if d["kind"] == "healed")
    total_len = sum(d["length_px"] for _, _, d in H.edges(data=True))

    report = {
        "components_before": before,
        "components_after": after,
        "components_merged": before - after,
        "bridges_added": len(added),
        "loops_restored": loops_added,
        "candidates_considered": len(candidates),
        "candidates_rejected_as_cycle": rejected_cycle,
        "healed_length_px": round(healed_len, 1),
        "healed_length_fraction": round(healed_len / max(total_len, 1e-6), 4),
        "mean_bridge_cost": round(float(np.mean([c for _, _, c in added])), 2) if added else 0.0,
    }
    return H, report


def largest_component(G: nx.Graph) -> nx.Graph:
    """Routing and centrality are only meaningful on a connected graph.
    After healing this should contain the overwhelming majority of the network;
    if it does not, that is a signal the segmentation stage underperformed."""
    if G.number_of_nodes() == 0:
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()
