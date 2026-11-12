"""
Graph-level topology metrics - "Topological Accuracy" from the evaluation table.

The brief asks for: shortest paths between random point pairs on the ground-truth
OSM graph versus the predicted graph, and the resulting error. That is exactly
APLS (Average Path Length Similarity), the standard SpaceNet road metric, so we
implement it properly rather than approximating it.

Why APLS and not IoU: a prediction can score 0.75 IoU and still be worthless if
the missing 25% happens to be the only bridge across a river - every route across
town breaks. APLS measures precisely that, because it compares *routes*, not pixels.

The subtlety that trips people up: the two graphs have different node sets. A node
at (412, 118) in the prediction has no counterpart in OSM. So every ground-truth
node must first be snapped to its nearest location on the predicted graph, within
a tolerance. Nodes with no match within tolerance contribute a score of 0 - which
is correct, because an unreachable destination is a total routing failure.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

EPS = 1e-9


def _positions(G: nx.Graph) -> tuple[list, np.ndarray]:
    nodes = list(G.nodes)
    pos = np.array([G.nodes[n]["pos"] for n in nodes], dtype=float)
    return nodes, pos


def snap_nodes(G_src: nx.Graph, G_dst: nx.Graph, tolerance_px: float = 12.0) -> dict:
    """Map each node of G_src to the nearest node of G_dst within tolerance.

    Returns {src_node: dst_node or None}. Unmatched nodes are kept explicitly
    rather than dropped, because ignoring them would inflate the score by
    silently excluding exactly the regions the model failed on.
    """
    from scipy.spatial import cKDTree

    if G_dst.number_of_nodes() == 0:
        return {n: None for n in G_src.nodes}

    dst_nodes, dst_pos = _positions(G_dst)
    tree = cKDTree(dst_pos)
    src_nodes, src_pos = _positions(G_src)

    dist, idx = tree.query(src_pos, distance_upper_bound=tolerance_px)
    return {
        n: (dst_nodes[i] if np.isfinite(d) and i < len(dst_nodes) else None)
        for n, d, i in zip(src_nodes, dist, idx)
    }


def _sample_pairs(G: nx.Graph, n_pairs: int, seed: int) -> list[tuple]:
    rng = np.random.default_rng(seed)
    if G.number_of_nodes() < 2:
        return []
    nodes = list(max(nx.connected_components(G), key=len))
    if len(nodes) < 2:
        return []
    pairs, guard = set(), 0
    while len(pairs) < n_pairs and guard < n_pairs * 25:
        a, b = rng.choice(len(nodes), 2, replace=False)
        pairs.add((nodes[a], nodes[b]))
        guard += 1
    return sorted(pairs)


def _apls_directional(G_gt: nx.Graph, G_pred: nx.Graph, n_pairs: int,
                      tolerance_px: float, weight: str, seed: int) -> dict:
    """One direction of APLS: how well G_pred reproduces routes measured on G_gt."""
    mapping = snap_nodes(G_gt, G_pred, tolerance_px)
    pairs = _sample_pairs(G_gt, n_pairs, seed)
    if not pairs:
        return {"apls": 0.0, "pairs": 0, "unmatched": 0, "unreachable": 0}

    # One Dijkstra per distinct source instead of one per pair.
    gt_len, pr_len = {}, {}
    by_src: dict = {}
    for s, t in pairs:
        by_src.setdefault(s, []).append(t)

    for s, targets in by_src.items():
        d_gt = nx.single_source_dijkstra_path_length(G_gt, s, weight=weight)
        for t in targets:
            gt_len[(s, t)] = d_gt.get(t, float("inf"))

        sp = mapping.get(s)
        if sp is None or sp not in G_pred:
            for t in targets:
                pr_len[(s, t)] = float("inf")
            continue
        d_pr = nx.single_source_dijkstra_path_length(G_pred, sp, weight=weight)
        for t in targets:
            tp = mapping.get(t)
            pr_len[(s, t)] = d_pr.get(tp, float("inf")) if tp is not None else float("inf")

    scores, unmatched, unreachable = [], 0, 0
    for p in pairs:
        L, Lp = gt_len[p], pr_len[p]
        if not np.isfinite(L) or L <= 0:
            continue  # not routable in ground truth either; not the model's fault
        if mapping.get(p[0]) is None or mapping.get(p[1]) is None:
            unmatched += 1
            scores.append(0.0)
            continue
        if not np.isfinite(Lp):
            unreachable += 1
            scores.append(0.0)
            continue
        # Relative path-length error, clipped so a 3x detour scores 0 rather than
        # going negative and dragging the mean below zero.
        scores.append(max(0.0, 1.0 - abs(L - Lp) / L))

    return {
        "apls": float(np.mean(scores)) if scores else 0.0,
        "pairs": len(scores),
        "unmatched": unmatched,
        "unreachable": unreachable,
    }


def apls(G_gt: nx.Graph, G_pred: nx.Graph, n_pairs: int = 500,
         tolerance_px: float = 12.0, weight: str = "length_m",
         seed: int = 42) -> dict:
    """Symmetric APLS - the harmonic mean of both directions.

    Both directions are required. GT->pred alone rewards a model that omits roads
    (fewer routes to get wrong); pred->GT alone rewards one that hallucinates them.
    The harmonic mean punishes either failure, which is why SpaceNet uses it.
    """
    a = _apls_directional(G_gt, G_pred, n_pairs, tolerance_px, weight, seed)
    b = _apls_directional(G_pred, G_gt, n_pairs, tolerance_px, weight, seed)
    h = 2 * a["apls"] * b["apls"] / (a["apls"] + b["apls"] + EPS)
    return {
        "apls": round(float(h), 5),
        "apls_gt_to_pred": round(a["apls"], 5),
        "apls_pred_to_gt": round(b["apls"], 5),
        "pairs_evaluated": a["pairs"],
        "nodes_unmatched": a["unmatched"],
        "routes_unreachable": a["unreachable"],
        "tolerance_px": tolerance_px,
    }


def path_length_error(G_gt: nx.Graph, G_pred: nx.Graph, n_pairs: int = 300,
                      tolerance_px: float = 12.0, weight: str = "length_m",
                      seed: int = 42) -> dict:
    """Average Path Length error, reported in metres and percent.

    This is the literal reading of the brief's "run shortest-path between random
    point pairs on ground-truth OSM vs. model graph and calculate error", and it
    is more interpretable to a planner than APLS: "routes are on average 8% longer
    than reality" means something to a non-specialist.
    """
    mapping = snap_nodes(G_gt, G_pred, tolerance_px)
    pairs = _sample_pairs(G_gt, n_pairs, seed)

    abs_err, rel_err, failed = [], [], 0
    for s, t in pairs:
        try:
            L = nx.shortest_path_length(G_gt, s, t, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        sp, tp = mapping.get(s), mapping.get(t)
        if sp is None or tp is None:
            failed += 1
            continue
        try:
            Lp = nx.shortest_path_length(G_pred, sp, tp, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            failed += 1
            continue
        abs_err.append(abs(L - Lp))
        rel_err.append(abs(L - Lp) / max(L, EPS))

    n = len(abs_err) + failed
    return {
        "mean_abs_path_error_m": round(float(np.mean(abs_err)), 2) if abs_err else None,
        "median_abs_path_error_m": round(float(np.median(abs_err)), 2) if abs_err else None,
        "mean_relative_error": round(float(np.mean(rel_err)), 5) if rel_err else None,
        "p90_relative_error": round(float(np.percentile(rel_err, 90)), 5) if rel_err else None,
        "routes_compared": len(abs_err),
        "routes_failed": failed,
        "route_failure_rate": round(failed / max(n, 1), 4),
    }


def connectivity_ratio(G_before: nx.Graph, G_after: nx.Graph) -> dict:
    """Percentage increase in the largest connected component after MST healing.

    This is the direct measure of what the healing stage bought, and the number
    to put on a slide. Reported both as node count and network length, because
    healing that reconnects many tiny stubs looks impressive by node count while
    adding almost no routable kilometres.
    """
    def stats(G):
        if G.number_of_nodes() == 0:
            return 0, 0.0, 0
        comps = list(nx.connected_components(G))
        giant = max(comps, key=len)
        Gg = G.subgraph(giant)
        length = sum(d.get("length_m", 0.0) for _, _, d in Gg.edges(data=True))
        return len(giant), length, len(comps)

    n0, l0, c0 = stats(G_before)
    n1, l1, c1 = stats(G_after)
    total0 = max(G_before.number_of_nodes(), 1)

    return {
        "components_before": c0,
        "components_after": c1,
        "lcc_nodes_before": n0,
        "lcc_nodes_after": n1,
        "lcc_node_increase_pct": round(100.0 * (n1 - n0) / max(n0, 1), 2),
        "lcc_length_km_before": round(l0 / 1000, 3),
        "lcc_length_km_after": round(l1 / 1000, 3),
        "lcc_length_increase_pct": round(100.0 * (l1 - l0) / max(l0, EPS), 2),
        "lcc_coverage_before": round(n0 / total0, 4),
        "lcc_coverage_after": round(n1 / total0, 4),
    }


def evaluate_graph(G_gt: nx.Graph, G_pred: nx.Graph, G_unhealed: nx.Graph | None = None,
                   tolerance_px: float = 12.0, n_pairs: int = 300) -> dict:
    """Every graph-level number in the evaluation table, in one call."""
    out = {
        **apls(G_gt, G_pred, n_pairs, tolerance_px),
        **path_length_error(G_gt, G_pred, n_pairs, tolerance_px),
    }
    if G_unhealed is not None:
        out["healing"] = connectivity_ratio(G_unhealed, G_pred)
    return out
