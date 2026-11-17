"""
Tuned and ensembled evaluation of one or more trained checkpoints.

    python scripts/ensemble_eval.py \
        --checkpoints unet.pt deeplab.pt segformer.pt \
        --data data/processed --out reports/ensemble

What it reports, for every model on its own and for the ensemble (mean of the
models' road probabilities):

  1. A decision threshold chosen on the VALIDATION split only, by mean per-tile
     IoU over a sweep of thresholds.
  2. That threshold applied once to the held-out TEST split, with full metrics.
  3. For transparency, the same test tiles scored at the default threshold 0.5.

The test split never influences any choice, so the tuned test figures remain an
honest held-out estimate. The primary protocol in evaluate_checkpoint.py
(threshold 0.5, no TTA) is not replaced; this is a second, labelled protocol.

Speed: inference runs in float16 on CUDA, skips the occlusion head, and loads
tiles one at a time, so it is several times faster than evaluate_checkpoint.py
and holds only one tile in memory.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import (normalize, percentile_stretch,  # noqa: E402
                                    stitch_tiles, tile_image)
from src.eval.metrics import aggregate, cl_dice, dice, iou, precision_recall  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
DEFAULT_THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(10)]  # 0.30 .. 0.75


# ----------------------------------------------------------------------- data
def list_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    img_dir, mask_dir = split_dir / "images", split_dir / "masks"
    if not img_dir.is_dir():
        return []
    pairs = []
    for ip in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        mp = next((mask_dir / (ip.stem + e) for e in IMG_EXT
                   if (mask_dir / (ip.stem + e)).exists()), None)
        if mp is not None:
            pairs.append((ip, mp))
    return pairs


def spread(pairs: list, limit: int) -> list:
    """Evenly spaced subset, so a limited sweep still covers the whole split."""
    if not limit or limit >= len(pairs):
        return pairs
    idx = np.linspace(0, len(pairs) - 1, limit).round().astype(int)
    return [pairs[i] for i in sorted(set(idx))]


def load_pair(ip: Path, mp: Path):
    import cv2

    im = cv2.imread(str(ip), cv2.IMREAD_COLOR)
    mk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if im is None or mk is None:
        return None, None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, mk > 127


# --------------------------------------------------------------------- models
def load_model(path: Path, device: str):
    import torch

    from src.models.baselines import build_model

    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    name = a.get("model", "unet_r34")
    m = build_model(name, a.get("in_channels", 3), encoder_weights=None,
                    hf_config=ck.get("hf_config"))
    m.load_state_dict(ck["model"], strict=True)
    m.eval().to(device)
    info = {"path": str(path), "architecture": name, "epoch": ck.get("epoch"),
            "selection_metric": ck.get("best_metric"),
            "selection_value": ck.get("best_value"),
            "tile_size": a.get("tile_size", 512),
            # The training config travels inside every checkpoint written by
            # scripts/kaggle_train.py; older checkpoints simply lack these keys.
            "no_occlusion": ck.get("config", {}).get("occlusion_augmentation") is False,
            "fine_tuned": bool(ck.get("config", {}).get("init_from"))}
    return m, info


def predict_prob(model, image: np.ndarray, device: str, tile: int, tta: bool,
                 fp16: bool, batch_size: int = 8) -> np.ndarray:
    """Same preprocessing and blending as src.models.inference.predict_scene,
    without the occlusion-head pass, optionally in float16."""
    import torch

    img = percentile_stretch(image)
    tiles = list(tile_image(img, tile, tile // 4))
    out = []
    use_amp = fp16 and device.startswith("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        for i in range(0, len(tiles), batch_size):
            chunk = tiles[i:i + batch_size]
            x = torch.from_numpy(np.stack(
                [normalize(t.array).transpose(2, 0, 1) for t in chunk])).float().to(device)
            acc = torch.sigmoid(model(x)["road_logits"].float())
            if tta:
                for dims in [(-1,), (-2,), (-1, -2)]:
                    p = torch.sigmoid(model(torch.flip(x, dims=dims))["road_logits"].float())
                    acc = acc + torch.flip(p, dims=dims)
                acc = acc / 4.0
            for t, p in zip(chunk, acc.cpu().numpy()[:, 0]):
                out.append((t, p))
    return stitch_tiles(out, image.shape[:2], tile)


def counts(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """TP, FP, FN pixel counts for one tile."""
    p, g = pred.astype(bool), gt.astype(bool)
    return np.array([np.sum(p & g), np.sum(p & ~g), np.sum(~p & g)], dtype=np.int64)


def pooled(c: np.ndarray) -> dict:
    """Dataset-level metrics from pixel counts summed over every tile.

    Many road-extraction papers report this form; the per-tile mean used as the
    primary figure here weights every tile equally, so tiles with very little
    road pull it down. Both are reported, each under its own name.
    """
    tp, fp, fn = (float(x) for x in c)
    return {"iou": tp / max(tp + fp + fn, 1.0), "dice": 2 * tp / max(2 * tp + fp + fn, 1.0),
            "precision": tp / max(tp + fp, 1.0), "recall": tp / max(tp + fn, 1.0)}


def full_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p_, r_ = precision_recall(pred, gt)
    return {"iou": iou(pred, gt), "dice": dice(pred, gt), "precision": p_, "recall": r_,
            "cldice": cl_dice(pred, gt)["cldice"],
            "pixel_accuracy": float((pred.astype(bool) == gt.astype(bool)).mean())}


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-limit", type=int, default=250,
                    help="validation tiles used for the threshold sweep (0 = all)")
    ap.add_argument("--test-limit", type=int, default=0, help="0 = all test tiles")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import torch

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    tta, fp16 = not a.no_tta, not a.no_fp16
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    models, infos = [], []
    for c in a.checkpoints:
        m, info = load_model(Path(c), device)
        models.append(m)
        infos.append(info)
    names = [i["architecture"] + ("_noocc" if i["no_occlusion"] else "")
             + ("_ft" if i["fine_tuned"] else "") for i in infos]
    if len(set(names)) != len(names):  # still ambiguous: number them
        names = [f"{n}#{k}" for k, n in enumerate(names)]
    methods = list(names)
    # Every combination of two or more models is a candidate ensemble. All are
    # swept on validation; the test split then scores the individual models,
    # the all-model ensemble and the validation-selected combination only.
    subsets = []
    if len(models) > 1 and not a.no_ensemble:
        for k in range(2, len(names) + 1):
            subsets += [list(c) for c in itertools.combinations(names, k)]
    sub_name = {tuple(s): "ensemble[" + "+".join(s) + "]" for s in subsets}
    sweep_methods = methods + list(sub_name.values())

    print("=" * 66)
    print("  Tuned / ensembled evaluation")
    print("=" * 66)
    for n, i in zip(names, infos):
        print(f"  {n:<22} epoch {i['epoch']}  {i['selection_metric']}={i['selection_value']}")
    print(f"  device {device}   TTA {'on' if tta else 'off'}   fp16 {'on' if fp16 and device.startswith('cuda') else 'off'}")

    def probs_for(image: np.ndarray, wanted: list[str]) -> dict[str, np.ndarray]:
        p = {n: predict_prob(m, image, device, i["tile_size"], tta, fp16)
             for n, m, i in zip(names, models, infos)}
        for s, sn in sub_name.items():
            if sn in wanted:
                p[sn] = np.mean([p[n] for n in s], axis=0)
        return {k: v for k, v in p.items() if k in wanted}

    # ---- 1. threshold sweep on validation ----------------------------------
    val = spread(list_pairs(data / "val"), a.val_limit)
    if not val:
        sys.exit(f"no validation tiles under {data / 'val'}")
    print(f"\n[1/2] threshold sweep on {len(val)} validation tiles")
    sums = {m: np.zeros(len(a.thresholds)) for m in sweep_methods}
    t0 = time.time()
    for k, (ip, mp) in enumerate(val):
        image, gt = load_pair(ip, mp)
        if image is None:
            continue
        for m, prob in probs_for(image, sweep_methods).items():
            for j, t in enumerate(a.thresholds):
                sums[m][j] += iou(prob > t, gt)
        if (k + 1) % 50 == 0:
            print(f"       {k + 1}/{len(val)} tiles  ({time.time() - t0:.0f}s)")
    chosen, sweep, best_val = {}, {}, {}
    for m in sweep_methods:
        mean = sums[m] / len(val)
        sweep[m] = {f"{t:.2f}": round(float(v), 5) for t, v in zip(a.thresholds, mean)}
        chosen[m] = float(a.thresholds[int(np.argmax(mean))])
        best_val[m] = float(mean.max())
        print(f"       {m:<40} best threshold {chosen[m]:.2f}   "
              f"val IoU {mean.max():.5f}  (at 0.50: {sweep[m].get('0.50', float('nan'))})")
    selected = None
    if subsets:
        full = sub_name[tuple(names)]
        selected = max(sub_name.values(), key=lambda s: best_val[s])
        methods.append(full)
        if selected != full:
            methods.append(selected)
        print(f"\n       ensemble selected on validation: {selected}")

    # ---- 2. held-out test ---------------------------------------------------
    test = spread(list_pairs(data / "test"), a.test_limit)
    if not test:
        sys.exit(f"no test tiles under {data / 'test'}")
    print(f"\n[2/2] {len(test)} test tiles at the validation-selected thresholds")
    rows = {m: [] for m in methods}
    rows05 = {m: [] for m in methods}
    pix = {m: np.zeros(3, dtype=np.int64) for m in methods}
    t0 = time.time()
    for k, (ip, mp) in enumerate(test):
        image, gt = load_pair(ip, mp)
        if image is None:
            continue
        for m, prob in probs_for(image, methods).items():
            pred = prob > chosen[m]
            rows[m].append(full_metrics(pred, gt))
            pix[m] += counts(pred, gt)
            p05 = prob > 0.5
            rows05[m].append({"iou": iou(p05, gt), "dice": dice(p05, gt)})
        if (k + 1) % 50 == 0:
            print(f"       {k + 1}/{len(test)} tiles  ({time.time() - t0:.0f}s)")

    report = {
        "protocol": "threshold chosen on validation (mean per-tile IoU), applied once to test",
        "tta": tta, "fp16": fp16 and device.startswith("cuda"),
        "n_val_tiles_for_threshold": len(val), "n_test_tiles": len(test),
        "models": dict(zip(names, infos)),
        "ensemble_members": names if subsets else [],
        "ensemble_selected_on_validation": selected,
        "val_iou_all_candidates": {m: round(best_val[m], 5) for m in sweep_methods},
        "results": {},
    }
    print("\n  " + "method".ljust(40) + "thr    IoU      Dice     Prec     Rec      clDice   PixAcc")
    for m in methods:
        agg, a05 = aggregate(rows[m]), aggregate(rows05[m])
        report["results"][m] = {"threshold": chosen[m], "val_iou_by_threshold": sweep[m],
                                "test": agg, "test_pooled": pooled(pix[m]),
                                "test_at_0.50": a05}
        print(f"  {m:<40} {chosen[m]:.2f}   " + "  ".join(
            f"{agg.get(k, float('nan')):.4f}" for k in
            ("iou", "dice", "precision", "recall", "cldice", "pixel_accuracy")))
    print("\n  pooled over all test pixels (the form many papers report):")
    for m in methods:
        pl = report["results"][m]["test_pooled"]
        print(f"  {m:<40} IoU {pl['iou']:.4f}  Dice {pl['dice']:.4f}  "
              f"P {pl['precision']:.4f}  R {pl['recall']:.4f}")

    (out / "ensemble_report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "ensemble_report.md").write_text(render(report))
    print(f"\nWrote {out}/ensemble_report.json and .md\n")
    print(render(report))


def render(r: dict) -> str:
    cols = ["iou", "dice", "precision", "recall", "cldice", "pixel_accuracy"]

    def c(d, k):
        v = d.get(k)
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    L = ["# Tuned and ensembled evaluation", "",
         f"- Threshold chosen on {r['n_val_tiles_for_threshold']} validation tiles "
         "(mean per-tile IoU), then applied once to the held-out test split.",
         f"- Test tiles: {r['n_test_tiles']}  |  TTA: {'4 flips' if r['tta'] else 'off'}"
         f"  |  float16 inference: {'yes' if r['fp16'] else 'no'}",
         ]
    if r["ensemble_members"]:
        L.append("- Ensembles average the models' road probabilities. Every combination "
                 "of two or more models was swept on validation; "
                 f"`{r['ensemble_selected_on_validation']}` scored best there and is the "
                 "only combination, besides the all-model ensemble, scored on test.")
    L += ["", "## Held-out test, validation-selected threshold (per-tile mean)", "",
          "| method | threshold | " + " | ".join(cols) + " |",
          "|" + "|".join(["---"] * (len(cols) + 2)) + "|"]
    for m, d in r["results"].items():
        L.append(f"| {m} | {d['threshold']:.2f} | " + " | ".join(c(d["test"], k) for k in cols) + " |")
    L += ["", "## Held-out test, pooled over all pixels", "",
          "TP, FP and FN summed over every test tile, then one IoU. Many road-extraction "
          "papers report this form. The table above is the per-tile mean.", "",
          "| method | iou | dice | precision | recall |", "|---|---|---|---|---|"]
    for m, d in r["results"].items():
        pl = d["test_pooled"]
        L.append(f"| {m} | " + " | ".join(c(pl, k) for k in ("iou", "dice", "precision", "recall")) + " |")
    L += ["", "## Same test tiles at the default threshold 0.50 (for comparison)", "",
          "| method | iou | dice |", "|---|---|---|"]
    for m, d in r["results"].items():
        L.append(f"| {m} | {c(d['test_at_0.50'], 'iou')} | {c(d['test_at_0.50'], 'dice')} |")
    L += ["", "## Validation IoU of every candidate (best threshold)", "",
          "| candidate | val IoU |", "|---|---|"]
    for m, v in r.get("val_iou_all_candidates", {}).items():
        L.append(f"| {m} | {v:.4f} |")
    L += ["", "## Validation IoU by threshold", ""]
    for m, d in r["results"].items():
        L.append(f"- **{m}**: " + ", ".join(f"{k}: {v:.4f}" for k, v in d["val_iou_by_threshold"].items()))
    return "\n".join(L)


if __name__ == "__main__":
    main()
