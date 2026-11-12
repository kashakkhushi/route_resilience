"""
Prepare the DeepGlobe Road Extraction dataset for training.

    python scripts/prepare_deepglobe.py --src data/external/deepglobe --out data/processed

DeepGlobe ships as 1024x1024 RGB tiles with paired masks:

    <id>_sat.jpg    RGB imagery at ~0.5 m ground sample distance
    <id>_mask.png   binary road mask (white = road)

Only the `train/` split has masks; `valid/` and `test/` are unlabelled challenge
splits, so we ignore them and cut our own validation and test sets from `train/`.

Two decisions worth knowing about:

1. **Split by image, deterministically.** Tiles are geographically distinct, so a
   random per-tile split is legitimate - but it must be reproducible, or every
   run silently changes the test set and no two experiments are comparable.
   A fixed seed and a sorted file list give the same split every time.

2. **Optional retiling.** 1024 px tiles at 0.5 m cover ~500 m. Training at 512
   with random crops is usually better use of GPU memory, and the dataset class
   already crops, so retiling is off by default. Turn it on if you want fixed
   tiles instead of random crops.

Cite the dataset as Demir et al., DeepGlobe 2018 (CVPR Workshops) - not the
Kaggle mirror, which is a re-upload rather than the source.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMG_SUFFIX = "_sat.jpg"
MASK_SUFFIX = "_mask.png"


def find_pairs(src: Path) -> list[tuple[Path, Path]]:
    """Locate (image, mask) pairs, tolerating a nested train/ directory."""
    roots = [src, src / "train"]
    for root in roots:
        if not root.is_dir():
            continue
        imgs = sorted(root.glob(f"*{IMG_SUFFIX}"))
        pairs = []
        for im in imgs:
            mk = im.with_name(im.name.replace(IMG_SUFFIX, MASK_SUFFIX))
            if mk.exists():
                pairs.append((im, mk))
        if pairs:
            return pairs
    return []


def retile(img_path: Path, mask_path: Path, out_img: Path, out_mask: Path,
           tile: int, stride: int, min_road: float) -> int:
    """Cut a large tile into fixed sub-tiles, skipping near-empty ones.

    Empty tiles dominate DeepGlobe and teach the model to predict background,
    so sub-tiles below `min_road` road fraction are dropped.
    """
    import cv2

    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    mk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if im is None or mk is None:
        return 0
    h, w = mk.shape
    n = 0
    for r in range(0, max(h - tile + 1, 1), stride):
        for c in range(0, max(w - tile + 1, 1), stride):
            sub_m = mk[r:r + tile, c:c + tile]
            if sub_m.shape != (tile, tile):
                continue
            if (sub_m > 127).mean() < min_road:
                continue
            stem = f"{img_path.stem.replace('_sat', '')}_{r}_{c}"
            cv2.imwrite(str(out_img / f"{stem}.png"), im[r:r + tile, c:c + tile])
            cv2.imwrite(str(out_mask / f"{stem}.png"), sub_m)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/external/deepglobe")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--test-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N pairs (quick smoke run)")
    ap.add_argument("--retile", action="store_true",
                    help="cut fixed sub-tiles instead of copying whole tiles")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--min-road", type=float, default=0.01)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    pairs = find_pairs(src)
    if not pairs:
        raise SystemExit(
            f"No DeepGlobe pairs found under {src}.\n"
            f"Expected files named <id>{IMG_SUFFIX} and <id>{MASK_SUFFIX}, "
            f"either directly in {src} or in {src / 'train'}."
        )
    if args.limit:
        pairs = pairs[: args.limit]

    # Deterministic split: sorted input + fixed seed = reproducible test set.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pairs))
    n_val = int(len(pairs) * args.val_frac)
    n_test = int(len(pairs) * args.test_frac)
    splits = {
        "val": idx[:n_val],
        "test": idx[n_val:n_val + n_test],
        "train": idx[n_val + n_test:],
    }

    manifest = {"source": str(src), "seed": args.seed, "retiled": args.retile,
                "counts": {}, "citation": "Demir et al., DeepGlobe 2018 (CVPR Workshops)"}

    for split, ids in splits.items():
        di, dm = out / split / "images", out / split / "masks"
        di.mkdir(parents=True, exist_ok=True)
        dm.mkdir(parents=True, exist_ok=True)
        written = 0
        for j in ids:
            im, mk = pairs[int(j)]
            if args.retile:
                written += retile(im, mk, di, dm, args.tile, args.stride, args.min_road)
            else:
                stem = im.stem.replace("_sat", "")
                shutil.copy2(im, di / f"{stem}{im.suffix}")
                shutil.copy2(mk, dm / f"{stem}.png")
                written += 1
        manifest["counts"][split] = written
        print(f"  {split:<6} {written} tiles")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out}/manifest.json")
    print("\nNext:")
    print("  python -m src.models.train --model unet_r34 --data data/processed --epochs 40")
    print("  python -m src.models.train --model segformer_b3 --data data/processed --epochs 40")
    print("  python -m src.models.train --model segformer_b3 --data data/processed --no-occlusion")


if __name__ == "__main__":
    main()
