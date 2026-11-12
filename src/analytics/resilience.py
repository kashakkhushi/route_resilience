"""
Resilience index and node ablation - Phase III of the brief.

The brief defines:

    R = avg_shortest_path_length(baseline) / avg_shortest_path_length(perturbed)

    "A lower R indicates a highly vulnerable network."

This is implemented as `resilience_index_aspl` and is the primary reported number.

**The trap in that definition, and how we handle it.** Average shortest path
length is undefined the moment a single OD pair becomes disconnected - it goes to
infinity, so R collapses to exactly 0. On a real road network the first serious
flood disconnects *something*, so a naive implementation returns 0.0 for every
meaningful scenario and the metric carries no information at all.

Three standard repairs, all computed and reported:

1. `aspl_connected_only`  - average over pairs still connected. Interpretable, but
   optimistic: it silently discards the worst-affected trips, so a network that
   severs half the city can score well.
2. `efficiency_ratio`     - global efficiency uses the mean of 1/d, so a
   disconnected pair contributes 0 instead of infinity. Degrades smoothly, never
   undefined. This is the network-science standard (Latora & Marchiori).
3. `disconnection_rate`   - the fraction that became unreachable, reported
   alongside, so the information discarded by (1) is never actually lost.

Reporting all three is the honest choice: (1) satisfies the brief literally,
(2) is what should drive plots and rankings, (3) is what an emergency planner
actually cares about.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .criticality import _weight_key, network_statistics, node_betweenness
from .simulation import Scenario, _path_lengths, sample_od_pairs

EPS = 1e-12


def structural_redundancy(G: nx.Graph) -> float:
    """Alpha index (meshedness), clipped to [0, 1].

    0 = tree: every edge is a single point of failure. Grid cities ~0.2-0.4."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n < 3:
        return 0.0
    c = nx.number_connected_components(G)
    return float(np.clip((m - n + c) / max(2 * n - 5 * c, 1), 0.0, 1.0))


def _aspl_stats(lengths: dict) -> dict:
    """Average shortest path length over a fixed OD set, three ways."""
    vals = np.array(list(lengths.values()), dtype=float)
    finite = vals[np.isfinite(vals)]
    return {
        "aspl_all": float(vals.mean()) if len(vals) else float("inf"),
        "aspl_connected": float(finite.mean()) if len(finite) else float("inf"),
        "efficiency": float(np.mean([1 / v if np.isfinite(v) and v > 0 else 0.0
                                     for v in vals])) if len(vals) else 0.0,
        "n_finite": int(len(finite)),
        "n_total": int(len(vals)),
    }


def resilience_index_aspl(G: nx.Graph, scenario: Scenario, pairs=None,
                          baseline: dict | None = None) -> dict:
    """The brief's Resilience Index, with the disconnection caveat made explicit."""
    weight = _weight_key(G)
    pairs = pairs if pairs is not None else sample_od_pairs(G)
    base = baseline if baseline is not None else _path_lengths(G, pairs, weight)

    H = scenario.apply(G)
    after = _path_lengths(H, pairs, weight)

    # Restrict to pairs that were routable BEFORE: anything already broken is
    # not this scenario's doing and would bias every comparison.
    valid = [p for p in pairs if np.isfinite(base[p])]
    b = _aspl_stats({p: base[p] for p in valid})
    a = _aspl_stats({p: after[p] for p in valid})

    disconnected = sum(1 for p in valid if not np.isfinite(after[p]))
    disc_rate = disconnected / max(len(valid), 1)

    # ---------------------------------------------------------------- #
    # PRIMARY metric: Network Efficiency Resilience (NER).
    #
    #     R = E(G') / E(G),   E(H) = (1/|P|) * sum_{(s,t) in P} 1 / d_H(s,t)
    #
    # with 1/inf := 0 and P a fixed origin-destination sample drawn once from
    # the intact network.
    #
    # Why this is the right primary index, rather than a patched one:
    #
    #   Bounded.     Removing nodes or edges can never create a shorter path,
    #                so d_G'(s,t) >= d_G(s,t) for every pair, hence
    #                1/d_G' <= 1/d_G termwise and 0 <= R <= 1 by construction.
    #                No clamping is applied or needed - the bound is a theorem
    #                about the construction, not a guard against bad values.
    #   Identity.    R = 1 exactly when no sampled shortest path changed, so an
    #                empty scenario scores 1 by definition.
    #   Monotone.    For nested removals G'' subset G' subset G, R(G'') <= R(G'):
    #                more damage can never raise the score.
    #   Total.       Disconnected pairs contribute 0 rather than infinity, so
    #                severing trips lowers R instead of making it undefined.
    #                Detour and disconnection are captured by one number.
    #   Readable.    R is the fraction of baseline routing efficiency retained;
    #                1 - R is the fraction lost.
    #
    # Contrast with the two ratios below, which are reported for completeness
    # but are NOT the resilience index:
    #
    #   literal ASPL ratio       -> 0 the instant any pair disconnects, because
    #                               mean path length becomes infinite.
    #   connected-only ASPL ratio-> survivorship-biased: when most trips are
    #                               severed the survivors are the short ones, so
    #                               it can exceed 1 and imply a collapsed network
    #                               improved. Kept strictly as a diagnostic.
    # ---------------------------------------------------------------- #
    r_primary = (a["efficiency"] / b["efficiency"]) if b["efficiency"] > 0 else 0.0
    # Floating-point guard only: the mathematical bound is proved above, so any
    # excursion here is round-off at the 1e-12 level, never a real value.
    r_primary = float(min(max(r_primary, 0.0), 1.0))

    r_literal = (b["aspl_all"] / a["aspl_all"]) if np.isfinite(a["aspl_all"]) else 0.0

    r_conn = (b["aspl_connected"] / a["aspl_connected"]
              if np.isfinite(a["aspl_connected"]) and a["aspl_connected"] > 0 else 0.0)
    conn_biased = bool(r_conn > 1.0 and disc_rate > 0)

    return {
        "scenario": scenario.name,

        # --- PRIMARY ---------------------------------------------------
        "resilience_index": round(r_primary, 4),
        "resilience_metric": "NER",
        "resilience_formula": (
            "NER = E(G_perturbed) / E(G_baseline), where "
            "E(H) = mean over a fixed OD sample of 1/d_H(s,t) with 1/inf = 0. "
            "Bounded in [0,1] because removing infrastructure cannot shorten any "
            "path. 1 = no routing capability lost, 0 = all sampled trips severed."
        ),
        "resilience_interpretation": (
            f"{r_primary:.0%} of baseline routing efficiency retained; "
            f"{1 - r_primary:.0%} lost."
        ),
        # Kept as an alias so existing clients keep working; identical value.
        "normalized_resilience": round(r_primary, 4),
        "efficiency_ratio": round(r_primary, 4),

        # --- COMPLIANCE: the brief's literal definition ------------------
        "resilience_index_literal_aspl": round(float(r_literal), 4),
        "literal_aspl_is_degenerate": bool(disconnected > 0),
        "literal_aspl_note": (
            "The project brief defines R = ASPL_baseline / ASPL_perturbed. It "
            "reads exactly 0 here because the scenario disconnected at least one "
            "OD pair, making mean shortest path length infinite. Reported for "
            "compliance; NER is the primary index."
            if disconnected else
            "Network stayed fully connected, so the literal ASPL ratio is "
            "well-defined for this scenario."
        ),

        # --- DIAGNOSTIC: explicitly NOT a resilience score ---------------
        "diagnostic_aspl_ratio_connected_only": round(float(r_conn), 4),
        "diagnostic_is_resilience_score": False,
        "diagnostic_survivorship_biased": conn_biased,
        "diagnostic_note": (
            f"Diagnostic only. Averages over surviving pairs, so with "
            f"{disc_rate:.0%} of trips severed the survivors are the short ones "
            "and this ratio rises above 1 - which would wrongly read as the "
            "collapsed network improving. Never present this as resilience."
            if conn_biased else
            "Diagnostic only: ratio of mean path length over pairs that remained "
            "connected. Not bounded and not a resilience score."
        ),

        # --- supporting detail ------------------------------------------
        "disconnection_rate": round(disc_rate, 4),
        "aspl_baseline_m": round(b["aspl_connected"], 1),
        "aspl_perturbed_m": (round(a["aspl_connected"], 1)
                             if np.isfinite(a["aspl_connected"]) else None),
        "aspl_increase_pct": (
            round(100 * (a["aspl_connected"] - b["aspl_connected"]) / b["aspl_connected"], 2)
            if np.isfinite(a["aspl_connected"]) and b["aspl_connected"] > 0 else None
        ),
        "network_fully_connected_after": disconnected == 0,
        "od_pairs": len(valid),
        "removed_nodes": len(scenario.removed_nodes),
        "removed_edges": len(scenario.removed_edges),
        "grade": _grade(r_primary),
        "note": (
            "Primary index is NER (bounded [0,1]). The literal ASPL ratio and the "
            "connected-only ratio are reported alongside it as compliance and "
            "diagnostic values respectively, and must not be used for ranking."
        ),
    }


def _grade(r: float) -> str:
    for cut, label in [(0.85, "A - highly resilient"), (0.70, "B - resilient"),
                       (0.55, "C - moderate"), (0.40, "D - vulnerable")]:
        if r >= cut:
            return label
    return "E - critically vulnerable"


def composite_resilience(G: nx.Graph, scenario: Scenario, pairs=None,
                         w_connectivity=0.4, w_efficiency=0.3,
                         w_detour=0.2, w_redundancy=0.1) -> dict:
    """Blended 0-1 score for the dashboard gauge.

    A single number is what planners ask for, but it hides the *shape* of the
    failure, so the components are always returned. A network can score 0.8
    because it detours gracefully everywhere, or because it is perfect for 90%
    of trips and severs the other 10%. Those demand different interventions.
    """
    base = resilience_index_aspl(G, scenario, pairs)
    H = scenario.apply(G)

    connectivity = 1.0 - base["disconnection_rate"]
    efficiency = base["efficiency_ratio"]
    inc = base["aspl_increase_pct"]
    detour = 1.0 / (1.0 + (inc or 0.0) / 100.0)
    redundancy = structural_redundancy(H)

    R = (w_connectivity * connectivity + w_efficiency * efficiency
         + w_detour * detour + w_redundancy * redundancy)
    return {
        "scenario": scenario.name,
        "composite_resilience": round(float(np.clip(R, 0, 1)), 4),
        "components": {
            "connectivity_retention": round(connectivity, 4),
            "efficiency_retention": round(efficiency, 4),
            "detour_score": round(detour, 4),
            "structural_redundancy": round(redundancy, 4),
        },
        "grade": _grade(R),
        "aspl": base,
    }


# ---------------------------------------------------------------- ablation
def node_ablation_study(G: nx.Graph, max_nodes: int = 20, strategy: str = "betweenness",
                        recompute: bool = True, n_pairs: int = 200,
                        seed: int = 0, k: int | None = 128) -> dict:
    """Phase III node ablation: remove nodes one at a time and track collapse.

    `recompute=True` re-ranks after every removal. This matters more than it
    sounds: after the top intersection is gone the network reorganizes, and the
    next most critical node is usually NOT the original number two. Batch removal
    of the original top-N systematically *understates* damage, so sequential
    removal is the honest upper bound - at the cost of one betweenness pass per
    step, which is why `k` pivots are used here by default.
    """
    weight = _weight_key(G)
    pairs = sample_od_pairs(G, n_pairs, seed)
    base = _path_lengths(G, pairs, weight)
    b = _aspl_stats(base)
    valid = [p for p in pairs if np.isfinite(base[p])]
    rng = np.random.default_rng(seed)

    H = G.copy()
    steps, removed = [], []

    def rank(graph):
        if strategy == "betweenness":
            kk = min(k, graph.number_of_nodes() - 1) if k else None
            return node_betweenness(graph, k=kk)
        if strategy == "degree":
            return dict(graph.degree())
        return {n: rng.random() for n in graph.nodes}

    scores = rank(H)
    for step in range(1, max_nodes + 1):
        if H.number_of_nodes() <= 2:
            break
        alive = [n for n in scores if n in H]
        if not alive:
            break
        victim = max(alive, key=lambda n: scores[n])
        was_gatekeeper = victim in set(nx.articulation_points(H))
        H.remove_node(victim)
        removed.append(victim)

        d = _path_lengths(H, pairs, weight)
        a = _aspl_stats({p: d[p] for p in valid})
        disc = sum(1 for p in valid if not np.isfinite(d[p]))

        steps.append({
            "step": step,
            "removed_node": int(victim),
            "was_gatekeeper": bool(was_gatekeeper),
            "fraction_removed": round(step / G.number_of_nodes(), 5),
            "resilience_index": round(
                b["aspl_all"] / a["aspl_all"] if np.isfinite(a["aspl_all"]) else 0.0, 4),
            "efficiency_ratio": round(
                a["efficiency"] / b["efficiency"] if b["efficiency"] > 0 else 0.0, 4),
            "aspl_m": round(a["aspl_connected"], 1) if np.isfinite(a["aspl_connected"]) else None,
            "aspl_increase_pct": (
                round(100 * (a["aspl_connected"] - b["aspl_connected"]) / b["aspl_connected"], 2)
                if np.isfinite(a["aspl_connected"]) and b["aspl_connected"] > 0 else None),
            "disconnection_rate": round(disc / max(len(valid), 1), 4),
            "components": nx.number_connected_components(H),
        })
        if recompute:
            scores = rank(H)

    first_break = next((s["step"] for s in steps if s["disconnection_rate"] > 0), None)
    return {
        "strategy": strategy,
        "recompute": recompute,
        "baseline_aspl_m": round(b["aspl_connected"], 1),
        "removed_nodes": removed,
        "steps": steps,
        "first_disconnection_step": first_break,
        "efficiency_auc": round(
            float(np.trapezoid([s["efficiency_ratio"] for s in steps],
                               [s["fraction_removed"] for s in steps])
                  / max(steps[-1]["fraction_removed"], EPS)), 4) if len(steps) > 1 else None,
    }


def robustness_comparison(G: nx.Graph, max_nodes: int = 15, n_pairs: int = 150,
                          seed: int = 0) -> dict:
    """Targeted vs random vs degree-based ablation on one figure.

    The gap between the random and targeted curves is the scale-free vulnerability
    signature, and it is the quantified argument for hardening specific
    intersections rather than upgrading the network uniformly.
    """
    curves = {
        s: node_ablation_study(G, max_nodes, s, n_pairs=n_pairs, seed=seed)
        for s in ("betweenness", "random", "degree")
    }
    gap = (curves["random"]["efficiency_auc"] or 0) - (curves["betweenness"]["efficiency_auc"] or 0)
    return {"curves": curves, "vulnerability_gap": round(gap, 4)}


def full_resilience_report(G: nx.Graph, scenarios=(), n_pairs: int = 200,
                           ablation_nodes: int = 12) -> dict:
    pairs = sample_od_pairs(G, n_pairs)
    report = {
        "network": network_statistics(G),
        "baseline_redundancy": round(structural_redundancy(G), 4),
        "scenarios": [composite_resilience(G, sc, pairs) for sc in scenarios],
        "ablation": robustness_comparison(G, ablation_nodes, n_pairs),
    }
    report["vulnerability_gap"] = report["ablation"]["vulnerability_gap"]
    report["interpretation"] = _interpret(report)
    return report


def _interpret(report: dict) -> list[str]:
    """Plain-language decision support - the actionable output for planners."""
    out, net = [], report["network"]
    if net.get("gatekeepers", 0):
        out.append(
            f"{net['gatekeepers']} gatekeeper intersection(s) and {net.get('bridge_edges', 0)} "
            "bridge segment(s) are single points of failure - each disconnects part of the "
            "network on its own. Highest-return hardening targets."
        )
    if report["baseline_redundancy"] < 0.15:
        out.append(
            f"Structural redundancy is low ({report['baseline_redundancy']:.2f}); the network "
            "is close to a tree, so closures force long detours rather than short ones. New "
            "link roads raise resilience more cheaply than widening existing ones."
        )
    if report["vulnerability_gap"] > 0.15:
        out.append(
            f"Targeted disruption is far more damaging than random damage (gap "
            f"{report['vulnerability_gap']:.2f}) - load is concentrated on a few corridors."
        )
    ab = report["ablation"]["curves"]["betweenness"]
    if ab.get("first_disconnection_step"):
        out.append(
            f"Removing just {ab['first_disconnection_step']} high-centrality intersection(s) "
            "already severs part of the network - pre-planned emergency routing is warranted."
        )
    for sc in report["scenarios"]:
        if sc["composite_resilience"] < 0.55:
            out.append(
                f"Scenario '{sc['scenario']}' scores {sc['composite_resilience']:.2f} "
                f"({sc['grade']}): {sc['aspl']['disconnection_rate']:.0%} of sampled trips "
                "become impossible."
            )
    return out or ["Network shows good redundancy; no single-point failures detected."]
