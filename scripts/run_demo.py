"""
End-to-end demo: zero external data, zero GPU, zero network.

    python scripts/run_demo.py --out outputs/demo

Generates a synthetic city, occludes it, fragments the mask the way a real
occlusion-damaged segmentation would, then runs the whole graph + analytics
stack and writes figures and a JSON report.

Because ground-truth topology is known, this also reports how much of it healing
recovered - the validation table worth putting in the report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.criticality import criticality_scores, network_statistics  # noqa: E402
from src.analytics.resilience import full_resilience_report  # noqa: E402
from src.analytics.simulation import (  # noqa: E402
    alternate_route,
    closure_scenario,
    flood_scenario,
    polygon_flood_scenario,
    random_failure,
    run_scenario_suite,
    targeted_attack,
)
from src.data.occlusion import OcclusionConfig, apply_occlusions, occlusion_report  # noqa: E402
from src.data.synthetic import fragment_mask, generate_city  # noqa: E402
from src.eval.metrics import evaluate_mask  # noqa: E402
from src.eval.topology import apls, connectivity_ratio, path_length_error  # noqa: E402
from src.graph.healing import HealingConfig, heal_graph, largest_component  # noqa: E402
from src.graph.skeletonize import build_raw_graph  # noqa: E402
from src.viz.plots import (  # noqa: E402
    plot_criticality_heatmap,
    plot_network,
    plot_occlusion_demo,
    plot_reroute,
    plot_scenario,
)


def plot_ablation(curves: dict, ax):
    styles = {"betweenness": ("#d62728", "-s"), "random": ("#2ca02c", "-o"),
              "degree": ("#9467bd", "-^")}
    for name, c in curves.items():
        color, st = styles.get(name, ("#333333", "-o"))
        xs = [s["fraction_removed"] for s in c["steps"]]
        ys = [s["efficiency_ratio"] for s in c["steps"]]
        ax.plot([0] + xs, [1.0] + ys, st, color=color, markersize=4,
                label=f"{name} (AUC {c['efficiency_auc']})")
    ax.set_xlabel("Fraction of intersections removed")
    ax.set_ylabel("Network efficiency retained")
    ax.set_title("Node ablation - targeted vs random failure")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend()
    return ax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--gaps", type=int, default=70)
    ap.add_argument("--pixel-size-m", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log, t0 = {}, time.time()

    # ---- 1. scene + occlusion -----------------------------------------
    print("[1/7] synthetic city + occlusion ...")
    city = generate_city(size=args.size, seed=args.seed)
    occluded, alpha = apply_occlusions(city["image"], city["mask"],
                                       OcclusionConfig(rng_seed=args.seed))
    log["occlusion"] = occlusion_report(alpha, city["mask"])
    plot_occlusion_demo(city["image"], occluded, city["mask"], alpha,
                        out / "01_occlusion.png")
    plt.close("all")

    # ---- 2. occlusion-damaged segmentation proxy ----------------------
    print("[2/7] fragmenting mask (proxy for occluded segmentation) ...")
    frag = fragment_mask(city["mask"], n_gaps=args.gaps, max_gap=22, seed=args.seed + 1)
    evidence = np.clip(0.25 + 0.75 * alpha, 0, 1)

    log["pixel_metrics"] = evaluate_mask(frag, city["mask"], alpha, buffer_px=4)
    print("      ", {k: log["pixel_metrics"][k]
                     for k in ("iou", "dice", "cldice", "relaxed_iou")})

    # ---- 3. graph + healing -------------------------------------------
    print("[3/7] graph construction + MST/union-find healing ...")
    raw = build_raw_graph(frag, pixel_size_m=args.pixel_size_m)
    healed, hrep = heal_graph(raw, HealingConfig(max_gap_px=70, max_cost=180), evidence)
    G = largest_component(healed)
    gt = build_raw_graph(city["mask"], pixel_size_m=args.pixel_size_m)

    log["healing"] = hrep
    log["connectivity_ratio"] = connectivity_ratio(raw, healed)
    log["topology_recovery"] = {
        "ground_truth": network_statistics(gt),
        "fragmented": network_statistics(raw),
        "healed": network_statistics(healed),
    }
    print("      ", json.dumps(hrep))

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    plot_network(raw, ax=axes[0], title=f"Fragmented - {hrep['components_before']} components")
    plot_network(healed, ax=axes[1], title=f"Healed - {hrep['components_after']} component(s)")
    plot_network(gt, ax=axes[2], show_healed=False, title="Ground truth")
    fig.tight_layout()
    fig.savefig(out / "02_healing.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    # ---- 4. topological accuracy --------------------------------------
    print("[4/7] APLS / path-length error vs ground truth ...")
    log["apls"] = apls(gt, G, n_pairs=300, tolerance_px=15)
    log["path_error"] = path_length_error(gt, G, n_pairs=250, tolerance_px=15)
    print("      APLS", log["apls"]["apls"],
          "| mean rel. path error", log["path_error"]["mean_relative_error"])

    # ---- 5. criticality ------------------------------------------------
    print("[5/7] criticality analysis ...")
    crit = criticality_scores(G)
    log["statistics"] = network_statistics(G)
    log["top_nodes"] = [
        {"node": int(n), "score": round(s, 4),
         "gatekeeper": n in crit["components"]["gatekeeper"]}
        for n, s in sorted(crit["scores"].items(), key=lambda t: -t[1])[:15]
    ]
    fig, ax = plt.subplots(figsize=(11, 11))
    plot_criticality_heatmap(G, crit["scores"], ax=ax,
                             gatekeepers=crit["components"]["gatekeeper"])
    fig.savefig(out / "03_criticality.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    # ---- 6. disaster scenarios ----------------------------------------
    print("[6/7] disaster simulation ...")
    c = args.size // 2
    top = sorted(G.edges(data=True), key=lambda e: -e[2].get("length_m", 0))[:2]
    scenarios = [
        flood_scenario(G, (c, c), args.size * 0.16, "flood_city_centre"),
        polygon_flood_scenario(G, city["river"], "river_overflow"),
        closure_scenario([(u, v) for u, v, _ in top], "arterial_closure"),
        random_failure(G, 6, seed=args.seed),
        targeted_attack(G, 6, k=128),
    ]
    log["scenarios"] = run_scenario_suite(G, scenarios, n_pairs=250)
    for s in log["scenarios"]:
        print(f"       {s['scenario']:<22} disconnect={s['disconnection_rate']:.1%} "
              f"detour={s['mean_detour_ratio']} eff={s['efficiency_retained']}")

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, sc in zip(axes, scenarios[:3]):
        plot_scenario(G, sc, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "04_scenarios.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    worst = max(zip(log["scenarios"], scenarios), key=lambda t: t[0]["disconnection_rate"])[1]
    nodes = sorted(G.nodes, key=lambda n: G.nodes[n]["pos"][0])
    rr = alternate_route(G, worst, nodes[0], nodes[-1])
    log["reroute_example"] = {k: v for k, v in rr.items() if "path" not in k}
    fig, ax = plt.subplots(figsize=(11, 11))
    plot_reroute(G, rr, ax=ax)
    fig.savefig(out / "05_reroute.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    # ---- 7. resilience + ablation --------------------------------------
    print("[7/7] resilience index + node ablation ...")
    rep = full_resilience_report(G, scenarios[:3], n_pairs=200, ablation_nodes=12)
    log["resilience"] = rep
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_ablation(rep["ablation"]["curves"], ax)
    fig.tight_layout()
    fig.savefig(out / "06_ablation.png", dpi=130, bbox_inches="tight")
    plt.close("all")

    log["runtime_seconds"] = round(time.time() - t0, 2)
    (out / "report.json").write_text(json.dumps(log, indent=2, default=str))

    print("\n=== RESILIENCE INDEX (three readings) ===")
    print("  NER is the primary index, bounded in [0,1] by construction.")
    print("  literal-ASPL is the brief's definition; it reads 0 the instant any")
    print("  OD pair disconnects. connected-only is a DIAGNOSTIC, not a score:")
    print("  it is survivorship-biased and can exceed 1. See docs/GRAPH_THEORY.md.\n")
    for s in rep["scenarios"]:
        a = s["aspl"]
        flag = " <-- biased" if a["diagnostic_survivorship_biased"] else ""
        print(f"  {s['scenario']:<22} "
              f"NER={a['resilience_index']:.2f}  "
              f"literal-ASPL={a['resilience_index_literal_aspl']:.2f}  "
              f"connected-only={a['diagnostic_aspl_ratio_connected_only']:.2f}{flag}  "
              f"disconnected={a['disconnection_rate']:.0%}")
    print("\n=== FINDINGS ===")
    for line in rep["interpretation"]:
        print(" -", line)
    print(f"\nWrote figures + report.json to {out}  ({log['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
