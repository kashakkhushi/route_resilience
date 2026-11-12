"""
Resilience metric regression tests.

Pins the mathematical contract of the PRIMARY index, Network Efficiency
Resilience (NER):

    NER = E(G') / E(G),   E(H) = mean over a fixed OD sample of 1/d_H(s,t),
                          with 1/inf := 0

Properties asserted here, each of which is a theorem about the construction
rather than an empirical observation:

    P1  Bounded        0 <= NER <= 1 for any removal set.
    P2  Identity       NER = 1 exactly when nothing was removed.
    P3  Monotone       Removing more can never raise NER.
    P4  No improvement A genuine failure cannot make the score go up.
    P5  Severance      Cutting the only bridge drives NER strictly down.
    P6  Separation     The diagnostic ratio may exceed 1, but is never used as
                       the index and is flagged as not-a-resilience-score.

P4 and P6 are the ones that caught the original defect: the old primary metric
could read above 1 after a catastrophic failure because it averaged only over
surviving OD pairs.

    python backend/tests/test_resilience_metric.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import networkx as nx  # noqa: E402

from src.analytics.resilience import resilience_index_aspl  # noqa: E402
from src.analytics.simulation import Scenario, sample_od_pairs  # noqa: E402

PASS: list[str] = []
FAIL: list[tuple[str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    (PASS if cond else FAIL).append(name if cond else (name, detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    return cond


# ------------------------------------------------------------------ fixtures
def _decorate(G: nx.Graph, coords) -> nx.Graph:
    for n, (r, c) in coords.items():
        G.nodes[n]["pos"] = (float(r), float(c))
    for u, v in G.edges:
        pu, pv = G.nodes[u]["pos"], G.nodes[v]["pos"]
        d = ((pu[0] - pv[0]) ** 2 + (pu[1] - pv[1]) ** 2) ** 0.5
        G[u][v].update(length_m=d, length_px=d, kind="observed", pts=[pu, pv])
    return G


def path_network(n: int = 14) -> nx.Graph:
    """A chain: every interior node is an articulation point."""
    G = nx.path_graph(n)
    return _decorate(G, {i: (0, i * 10) for i in G.nodes})


def grid_network(k: int = 6) -> nx.Graph:
    """A grid: highly redundant, so single failures cause detours not severance."""
    G = nx.grid_2d_graph(k, k)
    G = nx.convert_node_labels_to_integers(G, label_attribute="rc")
    return _decorate(G, {n: (d["rc"][0] * 10, d["rc"][1] * 10)
                         for n, d in G.nodes(data=True)})


def barbell_network() -> nx.Graph:
    """Two dense clusters joined by a single bridge edge."""
    G = nx.barbell_graph(6, 0)
    coords = {}
    for n in G.nodes:
        coords[n] = (0 if n < 6 else 40, (n % 6) * 10)
    return _decorate(G, coords)


# ------------------------------------------------------------------ tests
def main() -> int:
    print("\n[P2] Identity: no failure => resilience exactly 1")
    for label, G in (("path", path_network()), ("grid", grid_network()),
                     ("barbell", barbell_network())):
        r = resilience_index_aspl(G, Scenario("noop"))
        check(f"{label}: empty scenario scores exactly 1.0",
              r["resilience_index"] == 1.0, f"R={r['resilience_index']}")
        check(f"{label}: no trips disconnected",
              r["disconnection_rate"] == 0.0)
        check(f"{label}: reported as fully connected",
              r["network_fully_connected_after"] is True)

    # Removing an isolated leaf that carries no sampled traffic must also be 1.
    G = grid_network()
    G.add_node(999, pos=(500.0, 500.0))
    pairs = sample_od_pairs(G, 150, seed=7)
    r = resilience_index_aspl(G, Scenario("drop_isolated", removed_nodes=[999]), pairs)
    check("removing a node carrying no sampled traffic leaves R = 1",
          r["resilience_index"] == 1.0, f"R={r['resilience_index']}")

    print("\n[P1] Bounded: 0 <= R <= 1 over many random removals")
    import random

    rng = random.Random(0)
    G = grid_network(7)
    pairs = sample_od_pairs(G, 200, seed=11)
    nodes = list(G.nodes)
    out_of_range = []
    for trial in range(40):
        k = rng.randint(1, max(len(nodes) // 3, 2))
        victims = rng.sample(nodes, k)
        r = resilience_index_aspl(G, Scenario(f"rand{trial}", removed_nodes=victims), pairs)
        v = r["resilience_index"]
        if not (0.0 <= v <= 1.0):
            out_of_range.append((trial, k, v))
    check("R stayed within [0,1] across 40 random removals",
          not out_of_range, f"{len(out_of_range)} violations")

    print("\n[P3/P4] Monotone: more damage can never raise R")
    G = grid_network(7)
    pairs = sample_od_pairs(G, 200, seed=13)
    # Nested removal sets: each strictly contains the previous.
    order = sorted(G.nodes, key=lambda n: -G.degree(n))
    prev, violations, series = 1.0, [], []
    for k in (0, 1, 2, 4, 6, 9, 12, 16, 20):
        r = resilience_index_aspl(G, Scenario(f"cut{k}", removed_nodes=order[:k]), pairs)
        v = r["resilience_index"]
        series.append((k, v))
        if v > prev + 1e-9:
            violations.append((k, prev, v))
        prev = v
    check("R is non-increasing under nested removals", not violations,
          " -> ".join(f"{k}:{v:.3f}" for k, v in series))

    print("\n[P5] Severance: cutting the only bridge drives R down")
    G = barbell_network()
    bridges = list(nx.bridges(G))
    pairs = sample_od_pairs(G, 150, seed=17)
    r_before = resilience_index_aspl(G, Scenario("noop"), pairs)
    r_cut = resilience_index_aspl(
        G, Scenario("cut_bridge", removed_edges=[bridges[0]]), pairs)
    check("bridge exists in the barbell fixture", bool(bridges), f"{len(bridges)} bridge(s)")
    check("cutting the bridge strictly lowers R",
          r_cut["resilience_index"] < r_before["resilience_index"],
          f"{r_before['resilience_index']:.4f} -> {r_cut['resilience_index']:.4f}")
    check("cutting the bridge disconnects trips",
          r_cut["disconnection_rate"] > 0,
          f"{r_cut['disconnection_rate']:.2%} severed")

    print("\n[P4] A catastrophic failure cannot read as an improvement")
    G = path_network(20)
    pairs = sample_od_pairs(G, 200, seed=19)
    base = resilience_index_aspl(G, Scenario("noop"), pairs)
    # Sever the chain in several places: most OD pairs become unreachable and
    # the survivors are short neighbours - exactly the case that broke the old
    # metric.
    catastrophe = Scenario("shatter", removed_nodes=[4, 8, 12, 16])
    r = resilience_index_aspl(G, catastrophe, pairs)
    check("catastrophic failure lowers the PRIMARY index",
          r["resilience_index"] < base["resilience_index"],
          f"{base['resilience_index']:.4f} -> {r['resilience_index']:.4f}")
    check("catastrophic failure severs most trips",
          r["disconnection_rate"] > 0.3, f"{r['disconnection_rate']:.2%}")
    check("PRIMARY index never exceeds 1 after catastrophe",
          r["resilience_index"] <= 1.0, f"R={r['resilience_index']}")

    print("\n[P6] Diagnostic separation")
    check("diagnostic ratio does exceed 1 here (the artefact is real)",
          r["diagnostic_aspl_ratio_connected_only"] > 1.0,
          f"{r['diagnostic_aspl_ratio_connected_only']:.4f}")
    check("diagnostic is explicitly flagged as not a resilience score",
          r["diagnostic_is_resilience_score"] is False)
    check("survivorship bias is flagged when it occurs",
          r["diagnostic_survivorship_biased"] is True)
    check("diagnostic note explains the bias",
          "survivor" in r["diagnostic_note"].lower()
          or "short" in r["diagnostic_note"].lower())
    check("primary index and diagnostic are different fields",
          r["resilience_index"] != r["diagnostic_aspl_ratio_connected_only"])
    check("literal ASPL ratio is reported and flagged degenerate",
          r["resilience_index_literal_aspl"] == 0.0
          and r["literal_aspl_is_degenerate"] is True)

    print("\n[docs] Formula is carried in the payload")
    check("primary metric is named", r["resilience_metric"] == "NER", r["resilience_metric"])
    check("formula states the bound",
          "[0,1]" in r["resilience_formula"] or "0,1" in r["resilience_formula"])
    check("interpretation is plain language",
          "retained" in r["resilience_interpretation"].lower(),
          r["resilience_interpretation"])

    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for item in FAIL:
        name, detail = item if isinstance(item, tuple) else (item, "")
        print(f"  FAILED: {name}  {detail}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
