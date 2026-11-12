"""
Occlusion-robustness evaluation.

Builds a **stratified evaluation subset** in which every test tile is rendered
four ways from the same source image and the same unmodified label:

    normal        no occluders                      - the control
    canopy        tree canopy + building shadow     - soft, textured, organic
    partial       vehicles + hard partial blockage  - occluders sitting ON the road
    clutter       cloud haze + all of the above     - difficult urban clutter

Two properties make this a fair test rather than a flattering one:

1. **The label never changes.** Only the image is occluded, so recall under
   occlusion measures genuine inference, not a relabelled easier task.
2. **The same tiles appear in every condition.** Metrics are paired, so the
   normal/occluded difference is attributable to occlusion and not to one
   condition happening to draw easier tiles.

Reported per condition: IoU, Dice, precision, recall, clDice, and - split by
whether each ground-truth road pixel was occluded - Occlusion Recall.

    python scripts/evaluate_occlusion.py --checkpoint checkpoints/unet_r34/best.pt \\
                                         --data data/processed/test

Every number is computed from model output. Nothing here is estimated, and if no
checkpoint is supplied the script refuses to run rather than scoring a fallback
and presenting it as model performance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.occlusion import OcclusionConfig, apply_occlusions  # noqa: E402
from src.eval.metrics import aggregate, cl_dice, dice, iou, precision_recall  # noqa: E402

# Each condition is a fixed OcclusionConfig, so the subset is reproducible.
CONDITIONS: dict[str, OcclusionConfig | None] = {
    "normal": None,
    "canopy": OcclusionConfig(
        tree_prob=1.0, shadow_prob=1.0, cloud_prob=0.0, vehicle_prob=0.0,
        max_road_occlusion=0.45),
    "partial": OcclusionConfig(
        tree_prob=0.3, shadow_prob=0.2, cloud_prob=0.0, vehicle_prob=1.0,
        max_road_occlusion=0.45),
    "clutter": OcclusionConfig(
        tree_prob=1.0, shadow_prob=1.0, cloud_prob=1.0, vehicle_prob=1.0,
        max_road_occlusion=0.45),
}

CONDITION_NOTE = {
    "normal": "control - no occluders applied",
    "canopy": "tree canopy and directional building shadow",
    "partial": "vehicles and hard partial blockage sitting on the carriageway",
    "clutter": "cloud haze combined with canopy, shadow and vehicles",
}


def occlusion_split_recall(pred: np.ndarray, gt: np.ndarray,
                           alpha: np.ndarray | None) -> dict:
    """Recall over visible vs occluded ground-truth road pixels.

    `occlusion_recall` is the headline: recall restricted to road the model
    could not directly see. The gap against visible recall is the honest
    measure of how much is inferred rather than observed.
    """
    gt = gt.astype(bool)
    if alpha is None or not gt.any():
        r = float((pred & gt).sum() / max(gt.sum(), 1))
        return {"recall_visible": r, "recall_occluded": None,
                "occlusion_gap": None, "road_px_occluded": 0}

    occ = alpha > 0.5
    vis_m, occ_m = gt & ~occ, gt & occ
    r_vis = float((pred & vis_m).sum() / vis_m.sum()) if vis_m.any() else None
    r_occ = float((pred & occ_m).sum() / occ_m.sum()) if occ_m.any() else None
    return {
        "recall_visible": r_vis,
        "recall_occluded": r_occ,
        "occlusion_gap": (r_vis - r_occ) if (r_vis is not None and r_occ is not None) else None,
        "road_px_occluded": int(occ_m.sum()),
        "occluded_road_fraction": float(occ_m.sum() / max(gt.sum(), 1)),
    }


def load_tiles(data_dir: Path, limit: int = 0) -> list[dict]:
    """Load (image, mask) pairs directly - no augmentation, no random crops."""
    import cv2

    IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    img_dir, mask_dir = data_dir / "images", data_dir / "masks"
    if not img_dir.is_dir():
        raise SystemExit(f"No images under {img_dir}")

    out = []
    for ip in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        mp = next((mask_dir / (ip.stem + e) for e in IMG_EXT
                   if (mask_dir / (ip.stem + e)).exists()), None)
        if mp is None:
            continue
        im = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        mk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if im is None or mk is None:
            continue
        out.append({
            "id": ip.stem,
            "image": cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
            "mask": mk > 127,
        })
        if limit and len(out) >= limit:
            break
    if not out:
        raise SystemExit(f"No image/mask pairs found under {data_dir}")
    return out


def run(checkpoint: Path, data_dir: Path, out_dir: Path, limit: int,
        threshold: float, seed: int, device: str) -> dict:
    import torch

    from src.models.baselines import build_model
    from src.models.inference import predict_scene

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    name = args.get("model", "unet_r34")
    try:
        model = build_model(name, args.get("in_channels", 3), encoder_weights=None)
    except TypeError:
        model = build_model(name, args.get("in_channels", 3))
    model.load_state_dict(ck["model"])
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = model.to(dev).eval()

    tiles = load_tiles(data_dir, limit)
    print(f"Model      : {name}  ({checkpoint})")
    print(f"Device     : {dev}")
    print(f"Test tiles : {len(tiles)}  from {data_dir}")
    print(f"Threshold  : {threshold}\n")

    results: dict[str, list[dict]] = {k: [] for k in CONDITIONS}
    per_tile_rows = []

    for t_i, t in enumerate(tiles):
        for cond, cfg in CONDITIONS.items():
            if cfg is None:
                img, alpha = t["image"], None
            else:
                # Seed per tile AND per condition so the subset is reproducible.
                c = OcclusionConfig(**{**vars(cfg), "_rng": None,
                                       "rng_seed": seed + t_i})
                img, alpha = apply_occlusions(t["image"], t["mask"], c)

            pred = predict_scene(model, img, tile_size=512, overlap=128,
                                 device=dev, threshold=threshold, tta=False)
            m = pred["mask"]
            gt = t["mask"]
            p_, r_ = precision_recall(m, gt)
            row = {
                "iou": iou(m, gt), "dice": dice(m, gt),
                "precision": p_, "recall": r_,
                **{k: v for k, v in cl_dice(m, gt).items() if k == "cldice"},
                **occlusion_split_recall(m, gt, alpha),
            }
            results[cond].append(row)
            per_tile_rows.append({"tile": t["id"], "condition": cond,
                                  **{k: (round(v, 5) if isinstance(v, float) else v)
                                     for k, v in row.items()}})
        if (t_i + 1) % 10 == 0:
            print(f"  ...{t_i + 1}/{len(tiles)} tiles")

    summary = {c: aggregate(rows) for c, rows in results.items()}

    # Paired comparison: same tiles, so the delta is attributable to occlusion.
    normal = summary["normal"]
    for cond in CONDITIONS:
        if cond == "normal":
            continue
        summary[cond]["delta_iou_vs_normal"] = round(
            summary[cond].get("iou", 0) - normal.get("iou", 0), 5)
        summary[cond]["delta_dice_vs_normal"] = round(
            summary[cond].get("dice", 0) - normal.get("dice", 0), 5)

    report = {
        "checkpoint": str(checkpoint),
        "architecture": name,
        "data": str(data_dir),
        "n_tiles": len(tiles),
        "threshold": threshold,
        "seed": seed,
        "training_val_metrics": ck.get("metrics"),
        "conditions": {c: CONDITION_NOTE[c] for c in CONDITIONS},
        "summary": summary,
        "per_tile": per_tile_rows,
        "method_note": (
            "Occluders are applied to the image only; the label is never modified. "
            "All four conditions use the same tiles, so differences are paired."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "occlusion_evaluation.json").write_text(json.dumps(report, indent=2))
    (out_dir / "occlusion_evaluation.md").write_text(format_table(report))
    print("\n" + format_table(report))
    print(f"\nWrote {out_dir}/occlusion_evaluation.json and .md")
    return report


def format_table(report: dict) -> str:
    s = report["summary"]
    cols = ["iou", "dice", "precision", "recall", "cldice",
            "recall_visible", "recall_occluded", "occlusion_gap"]
    head = "| condition | " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    lines = [
        f"# Occlusion-robustness evaluation",
        "",
        f"- model: `{report['architecture']}`  ({report['checkpoint']})",
        f"- tiles: {report['n_tiles']}  |  threshold: {report['threshold']}  |  seed: {report['seed']}",
        f"- {report['method_note']}",
        "",
        head, sep,
    ]
    for cond in ("normal", "canopy", "partial", "clutter"):
        row = s.get(cond, {})
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "\u2014")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["", "Deltas against the normal control (paired, same tiles):", ""]
    for cond in ("canopy", "partial", "clutter"):
        d = s.get(cond, {})
        lines.append(f"- **{cond}**: IoU {d.get('delta_iou_vs_normal', 0):+.4f}, "
                     f"Dice {d.get('delta_dice_vs_normal', 0):+.4f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/processed/test")
    ap.add_argument("--out", default="outputs/reports")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ck = Path(a.checkpoint)
    if not ck.exists():
        raise SystemExit(
            f"No checkpoint at {ck}.\n"
            "This script scores a TRAINED MODEL only. It will not score the "
            "classical baseline or demo data and present the result as model "
            "performance. Train first:\n"
            "  python -m src.models.train --model unet_r34 --data data/processed"
        )
    run(ck, Path(a.data), Path(a.out), a.limit, a.threshold, a.seed, a.device)


if __name__ == "__main__":
    main()
