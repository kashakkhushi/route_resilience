"""
Kaggle training workflow - U-Net (ResNet34) on DeepGlobe road extraction.

Runs a full pre-flight verification before a single gradient step, because the
expensive failure mode is not a crash: it is a run that trains happily on
mispaired or non-binary data and produces a checkpoint nobody can trust.

    python scripts/kaggle_train.py --data /kaggle/working/processed \\
                                   --out /kaggle/working/checkpoints \\
                                   --epochs 40 --batch-size 8

Pre-flight (all fatal, all before training):
    1  dependencies importable, versions recorded
    2  dataset directory structure
    3  image and mask counts
    4  image/mask pairing and dimension agreement
    5  mask values are binary
    6  train/val/test separation with no leakage
    7  seed echoed and applied to python / numpy / torch / cudnn

Checkpoint contract: everything needed for offline inference travels inside the
.pt file. ImageNet weights may be used to INITIALISE the encoder on Kaggle, but
the saved state_dict contains those weights already, so deployment never needs
network access. The script verifies this by reloading the checkpoint with
`encoder_weights=None` before it exits.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ----------------------------------------------------------------- utilities
class Preflight:
    """Collects checks so every problem is reported, not just the first."""

    def __init__(self) -> None:
        self.ok: list[str] = []
        self.bad: list[tuple[str, str]] = []

    def check(self, name: str, cond: bool, detail: str = "") -> bool:
        (self.ok if cond else self.bad).append(name if cond else (name, detail))
        print(f"  {'OK  ' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
        return cond

    def finish(self) -> None:
        print(f"\n  pre-flight: {len(self.ok)} passed, {len(self.bad)} failed")
        if self.bad:
            print("\n  Training aborted. Fix these before spending GPU time:")
            for name, detail in self.bad:
                print(f"    - {name}: {detail}")
            raise SystemExit(2)


def set_seed(seed: int) -> dict:
    """Seed every source of randomness and report what was set."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # deterministic cudnn costs ~10% throughput and buys reproducible runs;
    # for a capstone that trade is worth making.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {"seed": seed, "cudnn_deterministic": True, "cudnn_benchmark": False}


def git_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def pairs_in(split_dir: Path) -> list[tuple[Path, Path]]:
    img_dir, mask_dir = split_dir / "images", split_dir / "masks"
    if not img_dir.is_dir():
        return []
    out = []
    for ip in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        mp = next((mask_dir / (ip.stem + e) for e in IMG_EXT
                   if (mask_dir / (ip.stem + e)).exists()), None)
        if mp is not None:
            out.append((ip, mp))
    return out


# ----------------------------------------------------------------- pre-flight
def preflight(data: Path, seed: int, sample: int, tile: int,
              allow_cpu: bool = False) -> dict:
    import cv2
    import torch

    pf = Preflight()
    report: dict = {}

    print("\n[1/7] Dependencies")
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "opencv": cv2.__version__}
    try:
        import segmentation_models_pytorch as smp

        versions["segmentation_models_pytorch"] = smp.__version__
        pf.check("segmentation-models-pytorch importable", True, smp.__version__)
    except ImportError as e:
        pf.check("segmentation-models-pytorch importable", False,
                 f"{e}. Run: pip install segmentation-models-pytorch")
    try:
        import albumentations as A

        versions["albumentations"] = A.__version__
    except ImportError:
        versions["albumentations"] = None
    pf.check("torch available", True, torch.__version__)
    cuda = torch.cuda.is_available()
    if cuda:
        pf.check("CUDA device present", True, torch.cuda.get_device_name(0))
    elif allow_cpu:
        # Explicit opt-in only. Never silently downgraded: a full DeepGlobe run
        # on CPU takes days, so a silent fallback would waste far more time than
        # an abort.
        print("  WARN  no CUDA device - proceeding on CPU because --allow-cpu "
              "was passed (smoke tests only)")
    else:
        pf.check("CUDA device present", False,
                 "training on CPU is impractically slow; pass --allow-cpu to "
                 "override for a smoke test")
    versions["cuda"] = torch.version.cuda if cuda else None
    versions["gpu"] = torch.cuda.get_device_name(0) if cuda else None
    report["versions"] = versions

    print("\n[2/7] Dataset structure")
    pf.check("data root exists", data.is_dir(), str(data))
    splits = {}
    for sp in ("train", "val", "test"):
        d = data / sp
        has = (d / "images").is_dir() and (d / "masks").is_dir()
        pf.check(f"{sp}/images and {sp}/masks exist", has or sp == "test",
                 "" if has else f"missing under {d}")
        splits[sp] = pairs_in(d) if has else []

    print("\n[3/7] Counts")
    counts = {}
    for sp, prs in splits.items():
        d = data / sp
        n_img = len([p for p in (d / "images").iterdir()
                     if p.suffix.lower() in IMG_EXT]) if (d / "images").is_dir() else 0
        n_msk = len([p for p in (d / "masks").iterdir()
                     if p.suffix.lower() in IMG_EXT]) if (d / "masks").is_dir() else 0
        counts[sp] = {"images": n_img, "masks": n_msk, "pairs": len(prs)}
        print(f"        {sp:<6} images={n_img:<6} masks={n_msk:<6} pairs={len(prs)}")
    pf.check("train split is non-empty", counts["train"]["pairs"] > 0,
             f"{counts['train']['pairs']} pairs")
    pf.check("val split is non-empty", counts["val"]["pairs"] > 0,
             f"{counts['val']['pairs']} pairs")
    report["counts"] = counts

    print("\n[4/7] Pairing")
    unpaired, dim_mismatch = [], []
    for sp, prs in splits.items():
        d = data / sp
        if not (d / "images").is_dir():
            continue
        imgs = {p.stem for p in (d / "images").iterdir() if p.suffix.lower() in IMG_EXT}
        paired = {ip.stem for ip, _ in prs}
        unpaired += [f"{sp}/{s}" for s in sorted(imgs - paired)]
        for ip, mp in prs[:sample]:
            im = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
            mk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if im is None or mk is None or im.shape[:2] != mk.shape[:2]:
                dim_mismatch.append(f"{sp}/{ip.stem}")
    pf.check("every image has a mask", not unpaired,
             f"{len(unpaired)} unpaired: {unpaired[:4]}")
    pf.check("image and mask dimensions agree", not dim_mismatch,
             f"{len(dim_mismatch)} mismatched: {dim_mismatch[:4]}")

    print("\n[5/7] Mask values")
    non_binary, road_fracs = [], []
    for sp, prs in splits.items():
        for _, mp in prs[:sample]:
            mk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mk is None:
                continue
            u = np.unique(mk)
            # DeepGlobe ships 0/255; anything with many intermediate values is
            # a soft or indexed mask and would silently corrupt the loss.
            if len(u) > 2 and np.isin(u, [0, 255]).mean() < 0.9:
                non_binary.append(f"{sp}/{mp.stem}:{len(u)} levels")
            road_fracs.append(float((mk > 127).mean()))
    pf.check("masks are binary", not non_binary,
             f"{len(non_binary)} non-binary: {non_binary[:3]}")
    mean_frac = float(np.mean(road_fracs)) if road_fracs else 0.0
    pf.check("road pixel fraction is plausible", 0.001 < mean_frac < 0.5,
             f"{mean_frac:.4f} (DeepGlobe is typically 0.02-0.08)")
    report["road_pixel_fraction"] = round(mean_frac, 5)
    # Class imbalance drives pos_weight; report it so the choice is inspectable.
    report["implied_pos_weight"] = round((1 - mean_frac) / max(mean_frac, 1e-6), 2)
    print(f"        mean road fraction {mean_frac:.4f} "
          f"-> background:road ratio {report['implied_pos_weight']:.1f}:1")

    print("\n[6/7] Split separation")
    stems = {sp: {ip.stem for ip, _ in prs} for sp, prs in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = stems[a] & stems[b]
        pf.check(f"{a}/{b} disjoint", not shared,
                 f"{len(shared)} shared: {sorted(shared)[:4]}")
    report["split_stems"] = {k: len(v) for k, v in stems.items()}

    print("\n[7/7] Seed")
    seed_cfg = set_seed(seed)
    print(f"        seed={seed}  cudnn.deterministic=True  cudnn.benchmark=False")
    pf.check("seed applied to python/numpy/torch/cuda", True, f"seed={seed}")
    report["seeding"] = seed_cfg

    pf.finish()
    return report


# ----------------------------------------------------------------- training
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="checkpoints/unet_r34")
    ap.add_argument("--arch", default="unet_r34")
    ap.add_argument("--encoder-weights", default="imagenet",
                    help="'imagenet' to initialise from pretrained weights, "
                         "'none' to train from scratch. Either way the saved "
                         "checkpoint is self-contained.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--monitor", default="cldice",
                    choices=["cldice", "iou", "dice"],
                    help="validation metric for best-checkpoint selection")
    ap.add_argument("--no-occlusion", action="store_true",
                    help="ablation: train without synthetic occluders")
    ap.add_argument("--preflight-sample", type=int, default=60)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit training without CUDA (smoke tests only)")
    ap.add_argument("--limit-train", type=int, default=0,
                    help="cap training tiles (smoke runs only)")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("  Route Resilience - U-Net(ResNet34) / DeepGlobe training")
    print("=" * 66)
    pre = preflight(data, args.seed, args.preflight_sample, args.tile_size,
                    allow_cpu=args.allow_cpu)

    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import RoadDataset
    from src.eval.metrics import evaluate_mask
    from src.models.baselines import build_model
    from src.models.losses import RoadLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_w = None if args.encoder_weights.lower() in ("none", "null", "") else args.encoder_weights

    # ---- config saved before training, so a crashed run is still reproducible
    config = {
        "architecture": args.arch,
        "encoder": "resnet34",
        "encoder_weights_init": enc_w,
        "in_channels": args.in_channels,
        "image_size": args.tile_size,
        "normalization": {
            "scheme": "percentile_stretch -> ImageNet mean/std",
            "mean": [0.485, 0.456, 0.406][: args.in_channels],
            "std": [0.229, 0.224, 0.225][: args.in_channels],
        },
        "threshold": args.threshold,
        "dataset": {"name": "DeepGlobe Road Extraction", "root": str(data),
                    "citation": "Demir et al., CVPR Workshops 2018",
                    **pre["counts"]},
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "monitor": args.monitor,
        "occlusion_augmentation": not args.no_occlusion,
        "occlusion_applies_to": "image only - ground-truth mask is never modified",
        "loss": "OcclusionWeightedBCE + SoftDice + SoftCLDice (+ aux occlusion BCE)",
        "device": device,
        "versions": pre["versions"],
        "git_commit": git_version(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\nConfiguration saved -> {out/'config.json'}")

    # ---- data
    train_ds = RoadDataset(data / "train", "train", args.tile_size,
                           synth_occlusion=not args.no_occlusion)
    val_ds = RoadDataset(data / "val", "val", args.tile_size)
    if args.limit_train:
        train_ds.images = train_ds.images[: args.limit_train]
    print(f"\nTrain tiles {len(train_ds)}   Val tiles {len(val_ds)}")
    print(f"Occlusion augmentation: {'ON (image only)' if not args.no_occlusion else 'OFF (ablation)'}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(device == "cuda"),
                          drop_last=len(train_ds) > args.batch_size, generator=g)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=max(args.num_workers // 2, 0))

    # ---- model
    try:
        model = build_model(args.arch, args.in_channels, encoder_weights=enc_w)
    except TypeError:
        model = build_model(args.arch, args.in_channels)
    model = model.to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Model {args.arch}  {n_par/1e6:.2f}M params  encoder init={enc_w}")

    criterion = RoadLoss()
    decay = [p for _, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for _, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(train_dl), 1), pct_start=0.1)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    history, best, best_epoch = [], -1.0, -1
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for x, y, occ in train_dl:
            x, y, occ = x.to(device), y.to(device), occ.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                parts = criterion(model(x), y, occ, epoch=epoch)
            scaler.scale(parts["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(parts["total"])

        # ---- validation: no occlusion, no augmentation, deterministic crops
        model.eval()
        per_tile = []
        with torch.no_grad():
            for x, y, occ in val_dl:
                prob = torch.sigmoid(model(x.to(device))["road_logits"]).cpu().numpy()
                pred = prob > args.threshold
                for b in range(pred.shape[0]):
                    per_tile.append(evaluate_mask(
                        pred[b, 0], y[b, 0].numpy() > 0.5, occ[b, 0].numpy(), buffer_px=4))

        def avg(k: str) -> float | None:
            v = np.array([t[k] for t in per_tile if t.get(k) is not None], float)
            v = v[np.isfinite(v)]
            return round(float(v.mean()), 5) if len(v) else None

        rec = {
            "epoch": epoch,
            "train_loss": round(running / max(len(train_dl), 1), 5),
            "lr": round(sched.get_last_lr()[0], 8),
            "val_iou": avg("iou"), "val_dice": avg("dice"),
            "val_precision": avg("precision"), "val_recall": avg("recall"),
            "val_cldice": avg("cldice"), "val_relaxed_iou": avg("relaxed_iou"),
            "secs": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(f"  epoch {epoch:>3}  loss {rec['train_loss']:.4f}  "
              f"IoU {rec['val_iou']}  Dice {rec['val_dice']}  "
              f"P {rec['val_precision']}  R {rec['val_recall']}  "
              f"clDice {rec['val_cldice']}  ({rec['secs']}s)")

        score = rec.get(f"val_{args.monitor}") or -1.0
        if score > best:
            best, best_epoch = score, epoch
            save_checkpoint(out / "best.pt", model, args, config, rec, epoch,
                            best_metric=args.monitor, best_value=score)

    duration = round(time.time() - t_start, 1)
    save_checkpoint(out / "final.pt", model, args, config, history[-1],
                    len(history) - 1, best_metric=args.monitor, best_value=best)
    (out / "history.json").write_text(json.dumps(history, indent=2))

    meta = {**config, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": duration,
            "best_epoch": best_epoch, "best_metric": args.monitor, "best_value": best,
            "epochs_completed": len(history),
            "final_val": history[-1] if history else None}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nTraining finished in {duration}s")
    print(f"  best {args.monitor} = {best} at epoch {best_epoch}")
    print(f"  {out/'best.pt'}  {out/'final.pt'}")
    print(f"  {out/'history.json'}  {out/'metadata.json'}  {out/'config.json'}")

    verify_offline(out / "best.pt")


def save_checkpoint(path: Path, model, args, config: dict, metrics: dict,
                    epoch: int, best_metric: str, best_value: float) -> None:
    """Self-contained checkpoint.

    The state_dict already holds the encoder weights, whether they were
    initialised from ImageNet or from scratch, so deployment rebuilds the model
    with encoder_weights=None and needs no network access.
    """
    import torch

    torch.save({
        "model": model.state_dict(),
        # `args` is what train.py and model_registry read to rebuild the model.
        "args": {"model": args.arch, "in_channels": args.in_channels,
                 "tile_size": args.tile_size, "threshold": args.threshold,
                 "seed": args.seed},
        "metrics": metrics,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_value": best_value,
        "config": config,
        "inference": {
            "architecture": args.arch,
            "encoder": "resnet34",
            "encoder_weights_required_at_load": None,
            "in_channels": args.in_channels,
            "image_size": args.tile_size,
            "normalization": config["normalization"],
            "threshold": args.threshold,
            "dataset": config["dataset"]["name"],
            "seed": args.seed,
            "git_commit": config["git_commit"],
        },
    }, path)


def verify_offline(ckpt: Path) -> None:
    """Prove the checkpoint deploys without network access before you leave Kaggle."""
    import socket

    import torch

    print("\nVerifying the checkpoint is deployment-ready (network disabled)...")
    real = socket.socket.connect

    def deny(self, addr):
        raise OSError(f"network blocked by verification (attempted {addr})")

    socket.socket.connect = deny
    try:
        from src.models.baselines import build_model

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        m = build_model(ck["args"]["model"], ck["args"]["in_channels"],
                        encoder_weights=None)
        res = m.load_state_dict(ck["model"], strict=True)
        missing = getattr(res, "missing_keys", [])
        unexpected = getattr(res, "unexpected_keys", [])
        assert not missing and not unexpected, (missing, unexpected)
        m.eval()
        with torch.no_grad():
            y = m(torch.zeros(1, ck["args"]["in_channels"], 64, 64))["road_logits"]
        print(f"  OK    loads offline, forward pass shape {tuple(y.shape)}")
        print(f"  OK    inference metadata present: {sorted(ck['inference'])}")
    except Exception as e:
        print(f"  FAIL  {type(e).__name__}: {e}")
        raise
    finally:
        socket.socket.connect = real


if __name__ == "__main__":
    main()
