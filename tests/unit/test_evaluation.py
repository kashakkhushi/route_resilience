"""
Tests for the evaluation suite and the resilience index.

The most important test here is `test_resilience_index_zero_on_disconnection`:
it pins the documented failure mode of the brief's ASPL-ratio definition. If
someone later "fixes" that zero by quietly filtering disconnected pairs, this
test fails and the discussion happens before the number reaches a report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.resilience import (
    node_ablation_study,
    resilience_index_aspl,
    structural_redundancy,
)
from src.analytics.simulation import Scenario, sample_od_pairs
from src.data.synthetic import fragment_mask, generate_city
from src.eval.metrics import (
    aggregate,
    cl_dice,
    connectivity_metrics,
    dice,
    evaluate_mask,
    iou,
    occlusion_recall,
    relaxed_metrics,
)
from src.eval.topology import (
    apls,
    connectivity_ratio,
    path_length_error,
    snap_nodes,
)
from src.graph.healing import HealingConfig, heal_graph, largest_component
from src.graph.skeletonize import build_raw_graph


@pytest.fixture(scope="module")
def city():
    return generate_city(size=448, seed=17)


@pytest.fixture(scope="module")
def graphs(city):
    """(ground truth, fragmented, healed) built from the same city."""
    gt = build_raw_graph(city["mask"], pixel_size_m=2.0)
    frag = fragment_mask(city["mask"], n_gaps=35, max_gap=18, seed=18)
    raw = build_raw_graph(frag, pixel_size_m=2.0)
    healed, _ = heal_graph(raw, HealingConfig(max_gap_px=60, max_cost=160))
    return gt, raw, largest_component(healed)


class TestPixelMetrics:
    def test_perfect_prediction(self, city):
        m = city["mask"]
        assert iou(m, m) == pytest.approx(1.0, abs=1e-6)
        assert dice(m, m) == pytest.approx(1.0, abs=1e-6)
        assert cl_dice(m, m)["cldice"] == pytest.approx(1.0, abs=1e-3)

    def test_empty_prediction(self, city):
        empty = np.zeros_like(city["mask"])
        assert iou(empty, city["mask"]) == pytest.approx(0.0, abs=1e-6)
        assert cl_dice(empty, city["mask"])["cldice"] == 0.0

    def test_cldice_and_iou_rank_errors_in_opposite_orders(self):
        """The whole justification for using clDice as the selection metric.

        Two predictions of the same road:
          A - thinned: loses MORE pixels, but the road stays connected.
          B - broken:  loses FEWER pixels, but the road is severed.

        IoU prefers B. clDice prefers A. For routing, A is fine and B is
        useless - two disconnected graph components. If a change ever makes
        these two metrics agree here, the test fails and the selection metric
        needs re-examining.
        """
        gt = np.zeros((160, 160), bool)
        gt[60:68, 10:118] = True                    # 8 px tall bar

        thinned = gt.copy()
        for r in (60, 61, 66, 67):                  # erode top and bottom
            thinned[r, 10:118] = False

        broken = gt.copy()
        for x in range(25, 110, 20):                # sever full width
            broken[60:68, x:x + 7] = False

        assert (~thinned & gt).sum() > (~broken & gt).sum()   # A loses more pixels
        assert iou(broken, gt) > iou(thinned, gt)             # ... yet IoU prefers B
        assert cl_dice(thinned, gt)["cldice"] > cl_dice(broken, gt)["cldice"]
        assert cl_dice(thinned, gt)["cldice"] > 0.95          # topology intact

    def test_relaxed_iou_forgives_small_shift(self, city):
        """A 2 px georeferencing shift should cost a lot of strict IoU and
        almost no relaxed IoU - that is the entire point of the buffer."""
        gt = city["mask"]
        shifted = np.roll(gt, 2, axis=0)
        assert iou(shifted, gt) < 0.75
        assert relaxed_metrics(shifted, gt, buffer_px=4)["relaxed_iou"] > 0.9

    def test_relaxed_buffer_monotone(self, city):
        gt = city["mask"]
        shifted = np.roll(gt, 3, axis=1)
        small = relaxed_metrics(shifted, gt, 2)["relaxed_iou"]
        large = relaxed_metrics(shifted, gt, 6)["relaxed_iou"]
        assert large >= small

    def test_occlusion_recall_splits_correctly(self, city):
        gt = city["mask"]
        occ = np.zeros_like(gt, float)
        occ[: gt.shape[0] // 2] = 1.0          # occlude the top half
        pred = gt.copy()
        pred[: gt.shape[0] // 2] = False       # model sees nothing under occlusion

        r = occlusion_recall(pred, gt, occ)
        assert r["recall_visible"] == pytest.approx(1.0, abs=1e-3)
        assert r["recall_occluded"] == pytest.approx(0.0, abs=1e-3)
        assert r["occlusion_gap"] == pytest.approx(1.0, abs=1e-3)

    def test_connectivity_metrics_detect_fragmentation(self, city):
        frag = fragment_mask(city["mask"], n_gaps=40, max_gap=20, seed=3)
        c = connectivity_metrics(frag, city["mask"])
        assert c["components_pred"] >= c["components_true"]
        assert c["component_ratio"] >= 1.0

    def test_evaluate_mask_returns_all_rubric_keys(self, city):
        occ = np.zeros_like(city["mask"], float)
        occ[:100] = 1.0
        m = evaluate_mask(city["mask"], city["mask"], occ)
        for k in ("iou", "dice", "cldice", "relaxed_iou",
                  "recall_visible", "recall_occluded", "occlusion_gap"):
            assert k in m

    def test_aggregate_reports_std(self):
        rows = [{"iou": 0.5}, {"iou": 0.7}, {"iou": 0.6}]
        a = aggregate(rows)
        assert a["iou"] == pytest.approx(0.6, abs=1e-6)
        assert a["iou_std"] > 0
        assert a["n_tiles"] == 3


class TestTopologyMetrics:
    def test_apls_identical_graphs_is_one(self, graphs):
        gt, _, _ = graphs
        r = apls(gt, gt, n_pairs=80, tolerance_px=5)
        assert r["apls"] > 0.95

    def test_apls_penalises_missing_roads(self, graphs):
        gt, raw, healed = graphs
        # Fragmented graph should score worse than the healed one.
        a_frag = apls(gt, largest_component(raw), n_pairs=120, tolerance_px=15)["apls"]
        a_heal = apls(gt, healed, n_pairs=120, tolerance_px=15)["apls"]
        assert a_heal >= a_frag

    def test_apls_is_symmetric(self, graphs):
        gt, _, healed = graphs
        a = apls(gt, healed, n_pairs=80, tolerance_px=15)
        b = apls(healed, gt, n_pairs=80, tolerance_px=15)
        assert a["apls"] == pytest.approx(b["apls"], abs=0.12)

    def test_snap_nodes_respects_tolerance(self, graphs):
        gt, _, healed = graphs
        tight = snap_nodes(gt, healed, tolerance_px=0.5)
        loose = snap_nodes(gt, healed, tolerance_px=40.0)
        assert sum(v is not None for v in loose.values()) >= \
               sum(v is not None for v in tight.values())

    def test_path_length_error_zero_for_self(self, graphs):
        gt, _, _ = graphs
        e = path_length_error(gt, gt, n_pairs=60, tolerance_px=5)
        assert e["mean_relative_error"] == pytest.approx(0.0, abs=1e-6)

    def test_connectivity_ratio_positive_after_healing(self, graphs):
        _, raw, healed = graphs
        c = connectivity_ratio(raw, healed)
        assert c["components_after"] <= c["components_before"]
        assert c["lcc_node_increase_pct"] > 0


class TestResilienceIndex:
    def _line_graph(self, n=12):
        """Path graph: removing any interior node necessarily disconnects it."""
        G = nx.path_graph(n)
        for i in G.nodes:
            G.nodes[i]["pos"] = (0.0, float(i * 10))
        for u, v in G.edges:
            G[u][v].update(length_m=10.0, length_px=10.0, kind="observed",
                           pts=[(0, u * 10), (0, v * 10)])
        return G

    def test_noop_scenario_is_one(self, graphs):
        _, _, G = graphs
        r = resilience_index_aspl(G, Scenario("noop"))
        assert r["resilience_index"] == pytest.approx(1.0, abs=1e-6)
        assert r["efficiency_ratio"] == pytest.approx(1.0, abs=1e-6)
        assert r["network_fully_connected_after"]

    def test_primary_index_stays_bounded_on_disconnection(self):
        """The primary index (NER) must remain informative when the network
        fragments, while the brief's literal ASPL ratio degenerates to 0.

        This is the contract that replaced the original defect: the primary
        slot now holds a bounded, monotone metric, and the two problematic
        ratios are reported beside it under names that cannot be misread."""
        G = self._line_graph()
        sc = Scenario("cut", removed_nodes=[6])
        r = resilience_index_aspl(G, sc)

        # Primary: bounded, non-degenerate, still ranks the scenario.
        assert 0.0 < r["resilience_index"] < 1.0
        assert r["resilience_metric"] == "NER"
        assert r["disconnection_rate"] > 0
        assert not r["network_fully_connected_after"]

        # Compliance value: the brief's definition, flagged as degenerate.
        assert r["resilience_index_literal_aspl"] == 0.0
        assert r["literal_aspl_is_degenerate"] is True

        # Diagnostic: never presented as resilience.
        assert r["diagnostic_is_resilience_score"] is False

    def test_diagnostic_ratio_never_occupies_the_primary_slot(self):
        """A catastrophic failure must not raise the primary index, even though
        the connected-only ratio rises above 1 under survivorship bias."""
        G = self._line_graph(20)
        pairs = sample_od_pairs(G, 120, seed=3)
        base = resilience_index_aspl(G, Scenario("noop"), pairs)
        r = resilience_index_aspl(
            G, Scenario("shatter", removed_nodes=[4, 8, 12, 16]), pairs)

        assert r["resilience_index"] < base["resilience_index"]
        assert 0.0 <= r["resilience_index"] <= 1.0
        assert r["diagnostic_aspl_ratio_connected_only"] > 1.0
        assert r["diagnostic_survivorship_biased"] is True
        assert r["resilience_index"] != r["diagnostic_aspl_ratio_connected_only"]

    def test_efficiency_ratio_degrades_monotonically(self):
        G = self._line_graph(16)
        pairs = sample_od_pairs(G, 60, seed=1)
        r1 = resilience_index_aspl(G, Scenario("a", removed_nodes=[8]), pairs)
        r2 = resilience_index_aspl(G, Scenario("b", removed_nodes=[4, 8, 12]), pairs)
        assert r2["efficiency_ratio"] <= r1["efficiency_ratio"]

    def test_tree_has_zero_redundancy(self):
        assert structural_redundancy(nx.balanced_tree(2, 4)) == 0.0

    def test_ablation_records_gatekeepers(self, graphs):
        _, _, G = graphs
        r = node_ablation_study(G, max_nodes=4, n_pairs=60, k=32)
        assert len(r["steps"]) > 0
        assert all("was_gatekeeper" in s for s in r["steps"])
        assert r["baseline_aspl_m"] > 0

    def test_sequential_ablation_at_least_as_damaging(self, graphs):
        """Re-ranking after each removal should be at least as damaging as
        removing the originally-ranked top-N; this is why we recompute."""
        _, _, G = graphs
        seq = node_ablation_study(G, 5, recompute=True, n_pairs=80, k=32)
        batch = node_ablation_study(G, 5, recompute=False, n_pairs=80, k=32)
        assert seq["steps"][-1]["efficiency_ratio"] <= \
               batch["steps"][-1]["efficiency_ratio"] + 0.05
