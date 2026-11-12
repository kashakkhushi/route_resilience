"""
Post-training evaluation.

Produces the full report in one run:

    1  validation evaluation      (the split used for checkpoint selection)
    2  held-out test evaluation   (never seen during training or selection)
    3  occlusion evaluation       (normal vs canopy / partial / clutter)
    4  classical baseline comparison on the identical tiles

    python scripts/evaluate_checkpoint.py --checkpoint checkpoints/unet_r34/best.pt \\
                                          --data data/processed

Principles enforced here rather than trusted:

  * The script refuses to run without a trained checkpoint. It will not score
    the classical baseline and label the result as model performance.
  * Test and validation are scored identically, so the val/test gap is a real
    generalisation signal rather than an artefact of different preprocessing.
  * Occlusion conditions are PAIRED - the same tiles in every condition - so a
    difference is attributable to occlusion and not to a lucky sample.
  * Occluders are applied to the image only; the label is never modified, so
    occluded recall measures inference rather than a relabelled easier task.
  * The classical baseline is run on exactly the same tiles, so the comparison
    is like-for-like.
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

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

CONDITIONS: dict[str, OcclusionConfig | None] = {
    "normal": None,
    "canopy": OcclusionConfig(tree_prob=1.0, shadow_prob=1.0, cloud_prob=0.0,
                              vehicle_prob=0.0, max_road_occlusion=0.45),
    "partial": OcclusionConfig(tree_prob=0.3, shadow_prob=0.2, cloud_prob=0.0,
                               vehicle_prob=1.0, max_road_occlusion=0.45),
    "clutter": OcclusionConfig(tree_prob=1.0, shadow_prob=1.0, cloud_prob=1.0,
                               vehicle_prob=1.0, max_road_occlusion=0.45),
}

CONDITION_NOTE = {
    "normal": "control - no occluders",
    "canopy": "tree canopy + directional building shadow",
    "partial": "vehicles + hard partial blockage on the carriageway",
    "clutter": "cloud haze + canopy + shadow + vehicles",
}


def load_tiles(split_dir: Path, limit: int = 0) -> list[dict]:
    import cv2

    img_dir, mask_dir = split_dir / "images", split_dir / "masks"
    if not img_dir.is_dir():
        return []
    tiles = []
    for ip in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        mp = next((mask_dir / (ip.stem + e) for e in IMG_EXT
                   if (mask_dir / (ip.stem + e)).exists()), None)
        if mp is None:
            continue
        im = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        mk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if im is None or mk is None:
            continue
        tiles.append({"id": ip.stem,
                      "image": cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
                      "mask": mk > 127})
        if limit and len(tiles) >= limit:
            break
    return tiles


def score(pred: np.ndarray, gt: np.ndarray, alpha: np.ndarray | None) -> dict:
    """Whole-tile metrics plus diagnostics restricted to the occluded region.

    The occluded region is taken from `alpha` - the synthetic occluder geometry -
    never from the prediction. That matters: defining it from the prediction
    would make the diagnostic circular.

    occlusion_precision is the metric that separates genuine inference from
    flooding. A model that simply paints every ambiguous region as road scores
    high occlusion_recall and LOW occlusion_precision. A model that actually
    infers continuity scores well on both.
    """
    p_, r_ = precision_recall(pred, gt)
    row = {"iou": iou(pred, gt), "dice": dice(pred, gt),
           "precision": p_, "recall": r_, "cldice": cl_dice(pred, gt)["cldice"]}

    gt_b, pred_b = gt.astype(bool), pred.astype(bool)
    if alpha is None:
        return row

    occ = alpha > 0.5
    row["occluded_scene_fraction"] = float(occ.mean())
    row["occluded_road_fraction"] = (float((gt_b & occ).sum() / gt_b.sum())
                                     if gt_b.any() else None)
    row["occluded_background_fraction"] = (float((occ & ~gt_b).sum() / (~gt_b).sum())
                                           if (~gt_b).any() else None)
    # How label-correlated the occluder is. 1.0 means every occluder pixel sits
    # on road, i.e. the occluder marks the label instead of hiding it.
    row["occluder_on_road_ratio"] = (float((occ & gt_b).sum() / occ.sum())
                                     if occ.any() else None)

    tp_occ = float((pred_b & gt_b & occ).sum())
    gt_occ = float((gt_b & occ).sum())
    pr_occ = float((pred_b & occ).sum())
    union_occ = float(((pred_b | gt_b) & occ).sum())

    row["occlusion_recall"] = tp_occ / gt_occ if gt_occ > 0 else None
    row["occlusion_precision"] = tp_occ / pr_occ if pr_occ > 0 else None
    row["occlusion_iou"] = tp_occ / union_occ if union_occ > 0 else None

    vis_m = gt_b & ~occ
    row["recall_visible"] = (float((pred_b & vis_m).sum() / vis_m.sum())
                             if vis_m.any() else None)
    if row["recall_visible"] is not None and row["occlusion_recall"] is not None:
        row["occlusion_gap"] = row["recall_visible"] - row["occlusion_recall"]
    return row


def save_debug(out_dir: Path, tile_id: str, cond: str, image: np.ndarray,
               occluded: np.ndarray, gt: np.ndarray, alpha: np.ndarray | None,
               pred: np.ndarray) -> None:
    """Write the panels needed to audit the occluder by eye."""
    import cv2

    d = out_dir / "debug" / cond
    d.mkdir(parents=True, exist_ok=True)

    def u8(x):
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def bgr(x):
        return cv2.cvtColor(u8(x), cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(d / f"{tile_id}_1_original.png"), bgr(image))
    cv2.imwrite(str(d / f"{tile_id}_2_occluded.png"), bgr(occluded))
    cv2.imwrite(str(d / f"{tile_id}_3_gt_mask.png"), u8(gt.astype(np.float32)))
    if alpha is not None:
        cv2.imwrite(str(d / f"{tile_id}_4_occlusion_mask.png"), u8(alpha))

    # Overlay: green = ground-truth road, red = occluder, blue = prediction.
    ov = u8(occluded).astype(np.float32)
    ov[gt.astype(bool)] = 0.55 * ov[gt.astype(bool)] + 0.45 * np.array([0, 255, 0])
    if alpha is not None:
        o = alpha > 0.5
        ov[o] = 0.55 * ov[o] + 0.45 * np.array([255, 0, 0])
    pb = pred.astype(bool)
    ov[pb] = 0.70 * ov[pb] + 0.30 * np.array([0, 0, 255])
    cv2.imwrite(str(d / f"{tile_id}_5_overlay.png"),
                cv2.cvtColor(ov.astype(np.uint8), cv2.COLOR_RGB2BGR))


def load_model(ckpt_path: Path):
    import torch

    from src.models.baselines import build_model

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    name = a.get("model", "unet_r34")
    try:
        m = build_model(name, a.get("in_channels", 3), encoder_weights=None)
    except TypeError:
        m = build_model(name, a.get("in_channels", 3))
    m.load_state_dict(ck["model"], strict=True)
    return m.eval(), ck, name


def predict(model, image: np.ndarray, device: str, threshold: float,
            tile: int) -> np.ndarray:
    from src.models.inference import predict_scene

    return predict_scene(model, image, tile_size=tile, overlap=tile // 4,
                         device=device, threshold=threshold, tta=False)["mask"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="outputs/reports")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-occlusion", action="store_true")
    ap.add_argument("--debug-viz", type=int, default=0,
                    help="save original/occluded/GT/occlusion/overlay panels for "
                         "the first N tiles of each condition")
    ap.add_argument("--harness-version", default="v2-corrected-occluder",
                    help="recorded in the report so superseded runs stay traceable")
    a = ap.parse_args()

    ck_path = Path(a.checkpoint)
    if not ck_path.exists():
        raise SystemExit(
            f"No checkpoint at {ck_path}.\n"
            "This script evaluates a TRAINED MODEL. It will not score the "
            "classical baseline and present it as model performance.\n"
            "Train first:  python scripts/kaggle_train.py --data data/processed"
        )

    import torch

    from backend.app.services.model_registry import classical_extract

    model, ck, arch = load_model(ck_path)
    inf = ck.get("inference", {})
    threshold = a.threshold if a.threshold is not None else inf.get("threshold", 0.5)
    tile = a.tile if a.tile is not None else inf.get("image_size", 512)
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = model.to(device)

    data = Path(a.data)
    print("=" * 66)
    print("  Post-training evaluation")
    print("=" * 66)
    print(f"  checkpoint : {ck_path}")
    print(f"  arch       : {arch}   epoch {ck.get('epoch')}   "
          f"best {ck.get('best_metric')}={ck.get('best_value')}")
    print(f"  device     : {device}   threshold {threshold}   tile {tile}")
    print(f"  git        : {inf.get('git_commit')}")

    report: dict = {
        "harness_version": a.harness_version,
        "harness_note": (
            "Occluder corrected: vehicles are sampled from the road corridor "
            "(on-road, edge-straddling and adjacent) with per-vehicle colour, "
            "size, orientation and cast shadow. The previous harness sampled "
            "only road pixels and used one flat colour per image, which marked "
            "the road instead of hiding it and inflated occlusion recall. "
            "Model weights, architecture, threshold, splits and ground truth "
            "are unchanged; any difference from an earlier report is a change "
            "in measurement, not in model quality."
        ),
        "checkpoint": str(ck_path),
        "architecture": arch,
        "trained_epoch": ck.get("epoch"),
        "selection_metric": ck.get("best_metric"),
        "selection_value": ck.get("best_value"),
        "threshold": threshold,
        "tile_size": tile,
        "seed": a.seed,
        "inference_metadata": inf,
        "training_val_metrics": ck.get("metrics"),
        "splits": {},
    }

    # ---- 1 & 2: validation and held-out test -------------------------
    for split in ("val", "test"):
        tiles = load_tiles(data / split, a.limit)
        if not tiles:
            print(f"\n[{split}] no tiles found under {data/split} - skipped")
            report["splits"][split] = {"available": False}
            continue
        print(f"\n[{split}] {len(tiles)} tiles")
        rows = [score(predict(model, t["image"], device, threshold, tile),
                      t["mask"], None) for t in tiles]
        agg = aggregate(rows)
        report["splits"][split] = {"available": True, "n_tiles": len(tiles), **agg}
        print(f"       IoU {agg.get('iou')}  Dice {agg.get('dice')}  "
              f"P {agg.get('precision')}  R {agg.get('recall')}  "
              f"clDice {agg.get('cldice')}")

    v, t_ = report["splits"].get("val", {}), report["splits"].get("test", {})
    if v.get("available") and t_.get("available"):
        gap = round(v.get("iou", 0) - t_.get("iou", 0), 5)
        report["val_test_iou_gap"] = gap
        print(f"\n  val - test IoU gap: {gap:+.5f}  "
              f"({'possible overfit' if gap > 0.05 else 'consistent'})")

    # ---- 3: occlusion evaluation (paired) ----------------------------
    if not a.skip_occlusion:
        tiles = load_tiles(data / "test", a.limit) or load_tiles(data / "val", a.limit)
        if tiles:
            print(f"\n[occlusion] {len(tiles)} tiles x {len(CONDITIONS)} conditions")
            occ_res: dict[str, list[dict]] = {k: [] for k in CONDITIONS}
            for i, t in enumerate(tiles):
                for cond, cfg in CONDITIONS.items():
                    if cfg is None:
                        img, alpha = t["image"], None
                    else:
                        c = OcclusionConfig(**{**vars(cfg), "_rng": None,
                                               "rng_seed": a.seed + i})
                        # The source image and mask are never mutated in place.
                        img, alpha = apply_occlusions(t["image"], t["mask"], c)
                    pred = predict(model, img, device, threshold, tile)
                    occ_res[cond].append(score(pred, t["mask"], alpha))
                    if a.debug_viz and i < a.debug_viz:
                        save_debug(Path(a.out), t["id"], cond, t["image"], img,
                                   t["mask"], alpha, pred)
            occ_sum = {c: aggregate(rows) for c, rows in occ_res.items()}
            base_iou = occ_sum["normal"].get("iou", 0)
            base_dice = occ_sum["normal"].get("dice", 0)
            for c in CONDITIONS:
                if c != "normal":
                    occ_sum[c]["delta_iou_vs_normal"] = round(
                        occ_sum[c].get("iou", 0) - base_iou, 5)
                    occ_sum[c]["delta_dice_vs_normal"] = round(
                        occ_sum[c].get("dice", 0) - base_dice, 5)
            report["occlusion"] = {"conditions": CONDITION_NOTE, "summary": occ_sum,
                                   "n_tiles": len(tiles), "paired": True}
            for c in CONDITIONS:
                s = occ_sum[c]
                print(f"       {c:<8} IoU {s.get('iou')}  Dice {s.get('dice')}  "
                      f"P {s.get('precision')}  R {s.get('recall')}  "
                      f"OccRecall {s.get('occlusion_recall')}")

    # ---- 4: classical baseline on identical tiles --------------------
    tiles = load_tiles(data / "test", a.limit) or load_tiles(data / "val", a.limit)
    if tiles:
        print(f"\n[classical baseline] {len(tiles)} identical tiles")
        rows = [score(classical_extract(t["image"], pixel_size_m=0.5)["mask"],
                      t["mask"], None) for t in tiles]
        cls = aggregate(rows)
        report["classical_baseline"] = {"n_tiles": len(tiles), **cls}
        trained = report["splits"].get("test", report["splits"].get("val", {}))
        if trained.get("available"):
            report["model_vs_classical"] = {
                k: round(trained.get(k, 0) - cls.get(k, 0), 5)
                for k in ("iou", "dice", "precision", "recall", "cldice")
            }
            print(f"       classical IoU {cls.get('iou')}  Dice {cls.get('dice')}  "
                  f"clDice {cls.get('cldice')}")
            print(f"       trained - classical IoU "
                  f"{report['model_vs_classical']['iou']:+.5f}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation_report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "evaluation_report.md").write_text(render(report))
    print(f"\nWrote {out}/evaluation_report.json and .md")
    print("\n" + render(report))


def render(r: dict) -> str:
    def cell(d: dict, k: str) -> str:
        v = d.get(k)
        return f"{v:.4f}" if isinstance(v, (int, float)) else "\u2014"

    L = ["# Evaluation report", "",
         f"- checkpoint: `{r['checkpoint']}`",
         f"- architecture: `{r['architecture']}`  |  epoch {r.get('trained_epoch')}"
         f"  |  selected on **{r.get('selection_metric')}** = {r.get('selection_value')}",
         f"- threshold {r['threshold']}  |  tile {r['tile_size']}  |  seed {r['seed']}",
         ""]

    cols = ["iou", "dice", "precision", "recall", "cldice"]
    L += ["## 1-2. Validation and held-out test", "",
          "| split | n | " + " | ".join(cols) + " |",
          "|" + "|".join(["---"] * (len(cols) + 2)) + "|"]
    for sp in ("val", "test"):
        d = r["splits"].get(sp, {})
        if not d.get("available"):
            L.append(f"| {sp} | — | " + " | ".join(["—"] * len(cols)) + " |")
            continue
        L.append(f"| {sp} | {d['n_tiles']} | " + " | ".join(cell(d, c) for c in cols) + " |")
    if "val_test_iou_gap" in r:
        L += ["", f"val − test IoU gap: **{r['val_test_iou_gap']:+.5f}**"]

    if "occlusion" in r:
        oc = ["iou", "dice", "precision", "recall", "occlusion_recall",
              "occlusion_precision", "occlusion_iou", "occluded_road_fraction",
              "occluder_on_road_ratio"]
        L += ["", "## 3. Occlusion evaluation (paired, same tiles)", "",
              f"{r['occlusion']['n_tiles']} tiles. Occluders applied to the image "
              "only; the ground-truth mask is never modified.", "",
              "| condition | " + " | ".join(oc) + " |",
              "|" + "|".join(["---"] * (len(oc) + 1)) + "|"]
        for c in ("normal", "canopy", "partial", "clutter"):
            d = r["occlusion"]["summary"].get(c, {})
            L.append(f"| {c} | " + " | ".join(cell(d, k) for k in oc) + " |")
        L += ["", "Deltas against the normal control:", ""]
        for c in ("canopy", "partial", "clutter"):
            d = r["occlusion"]["summary"].get(c, {})
            L.append(f"- **{c}**: IoU {d.get('delta_iou_vs_normal', 0):+.4f}, "
                     f"Dice {d.get('delta_dice_vs_normal', 0):+.4f}")

    if "classical_baseline" in r:
        cb = r["classical_baseline"]
        tr = r["splits"].get("test", r["splits"].get("val", {}))
        L += ["", "## 4. Trained U-Net vs classical baseline", "",
              "Identical tiles, identical scoring.", "",
              "| method | " + " | ".join(cols) + " |",
              "|" + "|".join(["---"] * (len(cols) + 1)) + "|",
              "| trained U-Net | " + " | ".join(cell(tr, c) for c in cols) + " |",
              "| classical (Frangi) | " + " | ".join(cell(cb, c) for c in cols) + " |"]
        if "model_vs_classical" in r:
            d = r["model_vs_classical"]
            L += ["", "Difference (trained − classical):", "",
                  "- " + ",  ".join(f"{k} {d[k]:+.4f}" for k in cols)]
    return "\n".join(L)


if __name__ == "__main__":
    main()
