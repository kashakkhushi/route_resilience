"""
Test suite. Runs in seconds, no GPU or dataset required:

    pytest tests/ -v

The tests that matter most are the ones asserting *algorithmic properties*
rather than output values - notably that healing adds exactly
(components - 1) bridges, which is what proves the union-find is really
producing a spanning tree rather than greedily pairing endpoints.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.criticality import (
    bridge_edges,
    criticality_scores,
    gatekeeper_nodes,
    network_statistics,
)
from src.analytics.resilience import (
    composite_resilience,
    node_ablation_study,
    structural_redundancy,
)
from src.analytics.simulation import (
    Scenario,
    closure_scenario,
    evaluate_scenario,
    flood_scenario,
    sample_od_pairs,
    targeted_attack,
)
from src.data.occlusion import OcclusionConfig, apply_occlusions, occlusion_report
from src.data.preprocessing import percentile_stretch, stitch_tiles, tile_image
from src.data.synthetic import fragment_mask, generate_city
from src.graph.healing import DisjointSet, HealingConfig, heal_graph, largest_component
from src.graph.skeletonize import build_raw_graph, mask_to_skeleton, skeleton_to_graph


# --------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def city():
    return generate_city(size=512, seed=7)


@pytest.fixture(scope="module")
def healed(city):
    frag = fragment_mask(city["mask"], n_gaps=40, max_gap=18, seed=8)
    raw = build_raw_graph(frag, pixel_size_m=2.0)
    H, rep = heal_graph(raw, HealingConfig(max_gap_px=60, max_cost=160))
    return raw, largest_component(H), rep


# --------------------------------------------------------------- DisjointSet
class TestDisjointSet:
    def test_union_and_find(self):
        d = DisjointSet(range(6))
        assert d.n_sets == 6
        assert d.union(0, 1) is True
        assert d.union(0, 1) is False  # already merged
        assert d.connected(0, 1)
        assert not d.connected(0, 2)
        assert d.n_sets == 5

    def test_transitive_merge(self):
        d = DisjointSet(range(4))
        d.union(0, 1)
        d.union(2, 3)
        d.union(1, 2)
        assert d.connected(0, 3)
        assert d.n_sets == 1

    def test_path_compression_keeps_correctness(self):
        d = DisjointSet(range(200))
        for i in range(199):
            d.union(i, i + 1)
        assert d.n_sets == 1
        assert all(d.find(i) == d.find(0) for i in range(200))


# --------------------------------------------------------------- skeleton
class TestSkeletonization:
    def test_straight_line_gives_two_endpoints(self):
        m = np.zeros((64, 64), bool)
        m[30:34, 8:56] = True
        G = skeleton_to_graph(mask_to_skeleton(m, min_object_px=0))
        assert G.number_of_nodes() == 2
        assert all(d == 1 for _, d in G.degree())

    def test_cross_gives_junction(self):
        m = np.zeros((80, 80), bool)
        m[38:42, 6:74] = True
        m[6:74, 38:42] = True
        G = skeleton_to_graph(mask_to_skeleton(m, min_object_px=0))
        assert max(d for _, d in G.degree()) >= 4  # one 4-way junction
        assert sum(1 for _, d in G.degree() if d == 1) == 4  # four arms

    def test_edges_carry_geometry_and_length(self, city):
        G = build_raw_graph(city["mask"], pixel_size_m=2.0)
        for _, _, d in G.edges(data=True):
            assert d["length_px"] > 0
            assert d["length_m"] == pytest.approx(d["length_px"] * 2.0)
            assert len(d["pts"]) >= 2

    def test_node_ids_are_plain_ints(self, city):
        """GraphML and JSON export both break on numpy integer types."""
        G = build_raw_graph(city["mask"])
        assert all(isinstance(n, int) for n in G.nodes)


# --------------------------------------------------------------- healing
class TestHealing:
    def test_reduces_component_count(self, healed):
        raw, H, rep = healed
        assert rep["components_after"] < rep["components_before"]
        assert rep["components_after"] <= 3

    def test_mst_property_exact_bridge_count(self):
        """The defining property: connecting k components needs exactly k-1
        bridges. More would mean redundant cycles slipped through union-find."""
        m = np.zeros((256, 256), bool)
        for i, y in enumerate([60, 120, 180]):        # 3 parallel bars ...
            m[y : y + 4, 20:236] = True
        m[60:184, 120:124] = True                      # ... joined by a spine
        G = build_raw_graph(m, min_spur_len=0)
        n_before = nx.number_connected_components(G)

        # Break the spine to create separate fragments.
        m2 = m.copy()
        m2[95:105, :] = False
        m2[145:155, :] = False
        G2 = build_raw_graph(m2, min_spur_len=0)
        k = nx.number_connected_components(G2)
        H, rep = heal_graph(G2, HealingConfig(max_gap_px=40, max_cost=300,
                                              restore_loops=False))
        assert rep["bridges_added"] == k - rep["components_after"]
        assert rep["components_after"] == 1

    def test_healed_edges_are_labelled(self, healed):
        _, H, _ = healed
        kinds = {d["kind"] for _, _, d in H.edges(data=True)}
        assert "healed" in kinds
        for u, v, d in H.edges(data=True):
            if d["kind"] == "healed":
                assert "gap_px" in d and "alignment" in d

    def test_max_cost_refuses_absurd_bridges(self, city):
        frag = fragment_mask(city["mask"], n_gaps=40, max_gap=18, seed=8)
        raw = build_raw_graph(frag)
        strict, r_strict = heal_graph(raw, HealingConfig(max_cost=1.0))
        loose, r_loose = heal_graph(raw, HealingConfig(max_cost=300.0))
        assert r_strict["bridges_added"] == 0
        assert r_loose["bridges_added"] > 0

    def test_loop_restoration_increases_redundancy(self, city):
        """A break inside a cycle is invisible to MST; the second pass must
        recover it, otherwise the whole city degenerates into a tree."""
        frag = fragment_mask(city["mask"], n_gaps=50, max_gap=18, seed=11)
        raw = build_raw_graph(frag)
        no_loop, _ = heal_graph(raw, HealingConfig(restore_loops=False))
        with_loop, rep = heal_graph(raw, HealingConfig(restore_loops=True))
        assert rep["loops_restored"] > 0
        assert structural_redundancy(with_loop) > structural_redundancy(no_loop)

    def test_evidence_map_biases_bridging(self, city):
        """Bridging should be cheaper where the evidence raster is high."""
        frag = fragment_mask(city["mask"], n_gaps=40, max_gap=18, seed=8)
        raw = build_raw_graph(frag)
        cfg = HealingConfig(max_cost=90.0)
        hi = np.ones(city["mask"].shape, np.float32)
        lo = np.zeros(city["mask"].shape, np.float32)
        _, r_hi = heal_graph(raw, cfg, hi)
        _, r_lo = heal_graph(raw, cfg, lo)
        assert r_hi["bridges_added"] >= r_lo["bridges_added"]


# --------------------------------------------------------------- analytics
class TestCriticality:
    def test_barbell_articulation_point(self):
        """Two cliques joined by one node: that node must be the gatekeeper."""
        G = nx.barbell_graph(5, 1)
        for n in G.nodes:
            G.nodes[n]["pos"] = (float(n), 0.0)
        for u, v in G.edges:
            G[u][v].update(length_m=1.0, length_px=1.0, kind="observed",
                           pts=[(u, 0), (v, 0)])
        gates = gatekeeper_nodes(G)
        assert 5 in gates  # the connecting node
        assert len(bridge_edges(G)) == 2

    def test_scores_normalized(self, healed):
        _, H, _ = healed
        s = criticality_scores(H)["scores"]
        assert abs(max(s.values()) - 1.0) < 1e-9
        assert min(s.values()) >= 0.0

    def test_statistics_shape(self, healed):
        _, H, _ = healed
        st = network_statistics(H)
        assert st["nodes"] > 0 and st["edges"] > 0
        assert 0 <= st["giant_component_fraction"] <= 1
        assert st["total_length_km"] > 0


class TestSimulation:
    def test_removing_nothing_changes_nothing(self, healed):
        _, H, _ = healed
        r = evaluate_scenario(H, Scenario("noop"))
        assert r["disconnection_rate"] == 0.0
        assert r["efficiency_retained"] == pytest.approx(1.0, abs=1e-6)

    def test_targeted_beats_random_damage(self, healed):
        _, H, _ = healed
        pairs = sample_od_pairs(H, 150)
        from src.analytics.simulation import random_failure

        t = evaluate_scenario(H, targeted_attack(H, 5, k=64), pairs)
        r = evaluate_scenario(H, random_failure(H, 5, seed=3), pairs)
        assert t["efficiency_retained"] <= r["efficiency_retained"]

    def test_flood_removes_nodes_inside_radius(self, healed):
        _, H, _ = healed
        sc = flood_scenario(H, (256, 256), 100)
        for n in sc.removed_nodes:
            r, c = H.nodes[n]["pos"]
            assert np.hypot(r - 256, c - 256) <= 100 + 1e-6

    def test_closure_of_bridge_edge_disconnects(self, healed):
        _, H, _ = healed
        brs = bridge_edges(H)
        if not brs:
            pytest.skip("no bridge edges in this sample")
        r = evaluate_scenario(H, closure_scenario([brs[0]]))
        assert r["components_after"] >= 2


class TestResilience:
    def test_composite_bounded_with_components(self, healed):
        _, H, _ = healed
        r = composite_resilience(H, flood_scenario(H, (256, 256), 90))
        assert 0.0 <= r["composite_resilience"] <= 1.0
        assert set(r["components"]) == {
            "connectivity_retention", "efficiency_retention",
            "detour_score", "structural_redundancy",
        }

    def test_ablation_curve_degrades(self, healed):
        _, H, _ = healed
        c = node_ablation_study(H, max_nodes=5, n_pairs=80, k=32)
        eff = [s["efficiency_ratio"] for s in c["steps"]]
        assert len(eff) > 0
        assert eff[-1] <= 1.0

    def test_tree_has_zero_redundancy(self):
        T = nx.balanced_tree(2, 4)
        assert structural_redundancy(T) == 0.0


# --------------------------------------------------------------- data layer
class TestDataLayer:
    def test_occlusion_never_touches_the_label(self, city):
        before = city["mask"].copy()
        img, alpha = apply_occlusions(city["image"], city["mask"],
                                      OcclusionConfig(rng_seed=3))
        assert np.array_equal(before, city["mask"])  # label untouched
        assert img.shape == city["image"].shape
        assert 0.0 <= alpha.min() and alpha.max() <= 1.0

    def test_occlusion_actually_hides_road(self, city):
        _, alpha = apply_occlusions(city["image"], city["mask"],
                                    OcclusionConfig(rng_seed=5, cloud_prob=1.0))
        rep = occlusion_report(alpha, city["mask"])
        assert rep["road_hidden_px"] > 0
        assert rep["road_occluded_frac"] <= 0.5 + 1e-6  # respects the safety valve

    def test_tiling_round_trip_covers_scene(self):
        img = np.random.rand(700, 900, 3).astype(np.float32)
        tiles = list(tile_image(img, 256, 64))
        out = stitch_tiles([(t, np.ones((256, 256), np.float32)) for t in tiles],
                           (700, 900), 256)
        assert out.shape == (700, 900)
        assert np.all(out > 0.99)  # every pixel covered by at least one tile

    def test_percentile_stretch_range(self):
        x = np.random.rand(32, 32, 3) * 4000 + 500
        out = percentile_stretch(x)
        assert out.min() >= 0.0 and out.max() <= 1.0
