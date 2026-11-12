"""
Benchmark runner - produces the model comparison table for the report.

Evaluates any set of checkpoints against the same test tiles and the same OSM
reference graph, so numbers are directly comparable. The output table is the
single most important artefact in the final report:

    model            IoU    clDice  relIoU  rec_vis  rec_occ  gap    APLS
    unet_r34         ...      ...     ...      ...      ...    ...    ...
    deeplabv3plus    ...      ...     ...      ...      ...    ...    ...
    segformer_b3     ...      ...     ...      ...      ...    ...    ...

`rec_occ` (recall on occluded road pixels) and `gap` (rec_vis - rec_occ) are the
columns that test this project's actual claim. A model can top the IoU column and
still lose on `gap`, which would mean it is good at seeing roads and bad at
inferring them - the opposite of what is being built here.

Also runs the **occlusion ablation**: the same architecture trained with and
without synthetic occluders. That is the cleanest single experiment in the
project, because it isolates one variable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import aggregate, evaluate_mask
from .topology import apls, connectivity_ratio, path_length_error


def evaluate_checkpoint(checkpoint: str | Path, tiles: list[dict],
                        device: str = "cuda", threshold: float = 0.5) -> dict:
    """Score one model over a list of {'image', 'mask', 'occlusion'} tiles."""
    import torch

    from ..models.baselines import build_model
    from ..models.inference import predict_scene

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    name = ckpt.get("args", {}).get("model", "unknown")
    model = build_model(name, in_channels=ckpt.get("args", {}).get("in_channels", 3))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    per_tile = []
    for t in tiles:
        pred = predict_scene(model, t["image"], device=device, threshold=threshold)
        per_tile.append(evaluate_mask(pred["mask"], t["mask"],
                                      t.get("occlusion"), buffer_px=4))
    return {"model": name, "checkpoint": str(checkpoint), **aggregate(per_tile)}


def evaluate_graph_quality(mask, gt_graph, pixel_size_m: float,
                           healing_cfg=None, tolerance_px: float = 12.0) -> dict:
    """Graph-level scoring: build, heal, then compare topology to ground truth."""
    from ..graph.healing import HealingConfig, heal_graph, largest_component
    from ..graph.skeletonize import build_raw_graph

    raw = build_raw_graph(mask, pixel_size_m=pixel_size_m)
    healed, hrep = heal_graph(raw, healing_cfg or HealingConfig())
    G = largest_component(healed)
    return {
        "healing": hrep,
        "connectivity_ratio": connectivity_ratio(raw, healed),
        **apls(gt_graph, G, tolerance_px=tolerance_px),
        **path_length_error(gt_graph, G, tolerance_px=tolerance_px),
    }


def occlusion_ablation(with_occ: dict, without_occ: dict) -> dict:
    """Compare models trained with and without synthetic occluders.

    The claim is supported iff `recall_occluded` improves substantially while
    `recall_visible` holds roughly steady. If occluded recall improves only
    because the model started over-predicting everywhere, precision will have
    collapsed - which is why precision is reported in the same table.
    """
    def delta(k):
        a, b = with_occ.get(k), without_occ.get(k)
        return round(a - b, 5) if a is not None and b is not None else None

    return {
        "delta_iou": delta("iou"),
        "delta_cldice": delta("cldice"),
        "delta_recall_visible": delta("recall_visible"),
        "delta_recall_occluded": delta("recall_occluded"),
        "delta_precision": delta("precision"),
        "delta_occlusion_gap": delta("occlusion_gap"),
        "claim_supported": bool(
            (delta("recall_occluded") or 0) > 0.05
            and (delta("precision") or 0) > -0.05
        ),
        "with_occlusion_aug": with_occ,
        "without_occlusion_aug": without_occ,
    }


def format_table(rows: list[dict], columns=None) -> str:
    """Markdown table, ready to paste into the report."""
    columns = columns or [
        "model", "iou", "dice", "cldice", "relaxed_iou",
        "recall_visible", "recall_occluded", "occlusion_gap", "apls",
    ]
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c)
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v if v is not None else "-"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_benchmark(rows: list[dict], out_dir: str | Path, name: str = "benchmark") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(rows, indent=2, default=str))
    (out_dir / f"{name}.md").write_text(format_table(rows))
    return out_dir


def synthetic_benchmark(n_scenes: int = 5, size: int = 512, seed: int = 0) -> dict:
    """Model-free benchmark of the graph/healing stack alone.

    Runs without torch, so the analytics sub-team can validate and tune healing
    parameters on day one while the segmentation model is still training. Uses
    known ground-truth topology, so healing quality is measured, not guessed.
    """
    from ..data.synthetic import fragment_mask, generate_city
    from ..graph.healing import HealingConfig, heal_graph, largest_component
    from ..graph.skeletonize import build_raw_graph

    rows = []
    for i in range(n_scenes):
        city = generate_city(size=size, seed=seed + i)
        frag = fragment_mask(city["mask"], n_gaps=50, max_gap=20, seed=seed + 100 + i)

        gt = build_raw_graph(city["mask"], pixel_size_m=2.0)
        raw = build_raw_graph(frag, pixel_size_m=2.0)
        no_loop, _ = heal_graph(raw, HealingConfig(restore_loops=False))
        healed, hrep = heal_graph(raw, HealingConfig(restore_loops=True))

        rows.append({
            "scene": i,
            **{f"pix_{k}": v for k, v in evaluate_mask(frag, city["mask"]).items()
               if k in ("iou", "cldice", "relaxed_iou")},
            "components_before": hrep["components_before"],
            "components_after": hrep["components_after"],
            "bridges": hrep["bridges_added"],
            "loops_restored": hrep["loops_restored"],
            "apls_mst_only": apls(gt, largest_component(no_loop), n_pairs=150,
                                  tolerance_px=15)["apls"],
            "apls_with_loops": apls(gt, largest_component(healed), n_pairs=150,
                                    tolerance_px=15)["apls"],
        })

    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    mean = {k: round(float(np.mean([r[k] for r in rows])), 5) for k in keys}
    return {"per_scene": rows, "mean": mean,
            "loop_pass_apls_gain": round(mean["apls_with_loops"] - mean["apls_mst_only"], 5)}
