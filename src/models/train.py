"""
Training entry point - works for every architecture via the model factory.

    python -m src.models.train --model unet_r34        --data data/processed
    python -m src.models.train --model deeplabv3plus_r50 --data data/processed
    python -m src.models.train --model segformer_b3    --data data/processed

    # the key ablation: identical setup, occluders off
    python -m src.models.train --model segformer_b3 --data data/processed --no-occlusion

Validation reports IoU, clDice AND occlusion-split recall. Watch two things:

  * IoU high but clDice low  -> dotted-line roads: pixel-accurate, useless for
    routing. This is the failure the whole project exists to fix.
  * recall_visible high but recall_occluded low -> the model is seeing roads,
    not inferring them. If occlusion augmentation is on and this gap does not
    close, the project's central claim is unsupported and you want to know now.

Checkpoints are selected on clDice, not IoU, for exactly that reason.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import RoadDataset
from ..eval.metrics import evaluate_mask
from .baselines import MODEL_ZOO, build_model
from .losses import RoadLoss


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None) -> dict:
    """Full metric suite on the validation split, computed on numpy for exact
    parity with the benchmark runner - no train/eval metric drift."""
    model.eval()
    per_tile = []
    for i, (x, y, occ) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        prob = torch.sigmoid(model(x.to(device))["road_logits"]).cpu().numpy()
        pred = prob > 0.5
        for b in range(pred.shape[0]):
            per_tile.append(evaluate_mask(
                pred[b, 0], y[b, 0].numpy() > 0.5, occ[b, 0].numpy(), buffer_px=4
            ))
    if not per_tile:
        return {}
    keys = ["iou", "dice", "cldice", "relaxed_iou", "precision", "recall",
            "recall_visible", "recall_occluded", "occlusion_gap"]
    out = {}
    for k in keys:
        vals = np.array([t[k] for t in per_tile if t.get(k) is not None], float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[k] = round(float(vals.mean()), 5)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="segformer_b3", choices=MODEL_ZOO)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 6e-5 for transformers, 3e-4 for CNNs")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--no-occlusion", action="store_true",
                    help="ablation: train without synthetic occluders")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    if args.lr is None:
        args.lr = 6e-5 if args.model.startswith("segformer") else 3e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device found - training will be extremely slow. "
              "Check `python -c \"import torch; print(torch.cuda.is_available())\"` "
              "and reinstall torch from the correct CUDA index URL.")

    tag = f"{args.model}{'_noocc' if args.no_occlusion else ''}"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)

    train_ds = RoadDataset(Path(args.data) / "train", "train", args.tile_size,
                           synth_occlusion=not args.no_occlusion)
    val_ds = RoadDataset(Path(args.data) / "val", "val", args.tile_size)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=2)

    model = build_model(args.model, args.in_channels).to(device)
    criterion = RoadLoss()

    # No weight decay on norms/biases - standard for both CNN and transformer
    # fine-tuning; applying it to 1-D params measurably hurts.
    decay = [p for _, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for _, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(train_dl), 1), pct_start=0.1)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    best, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for x, y, occ in train_dl:
            x, y, occ = x.to(device), y.to(device), occ.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=device == "cuda"):
                parts = criterion(model(x), y, occ, epoch=epoch)
            scaler.scale(parts["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(parts["total"])

        m = evaluate(model, val_dl, device)
        m.update(epoch=epoch, model=args.model, train_loss=round(running / max(len(train_dl), 1), 5),
                 lr=round(sched.get_last_lr()[0], 8), secs=round(time.time() - t0))
        history.append(m)
        print(json.dumps(m))

        if m.get("cldice", -1) > best:
            best = m["cldice"]
            torch.save({"model": model.state_dict(), "args": vars(args), "metrics": m},
                       out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest clDice: {best:.4f}  ->  {out / 'best.pt'}")


if __name__ == "__main__":
    main()
