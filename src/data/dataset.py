"""
Dataset for road segmentation.

Directory layout expected (DeepGlobe / Massachusetts Roads / SpaceNet all fit):

    data_root/
        images/  xxx.png | xxx.tif
        masks/   xxx.png | xxx.tif      (binary, 0 = background, >0 = road)

Occlusions are applied on the fly. Crucially the mask is NOT occluded - see
occlusion.py for why. The generated alpha map is returned as a third tensor and
supervises the auxiliary head.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .occlusion import OcclusionConfig, apply_occlusions
from .preprocessing import normalize, percentile_stretch

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class RoadDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        tile_size: int = 512,
        synth_occlusion: bool = True,
        occlusion_cfg: OcclusionConfig | None = None,
        geometric_aug: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.tile_size = tile_size
        self.synth_occlusion = synth_occlusion and split == "train"
        self.geometric_aug = geometric_aug and split == "train"
        self.occ_cfg = occlusion_cfg or OcclusionConfig()

        img_dir = self.root / "images"
        self.images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        if not self.images:
            raise FileNotFoundError(f"no images found under {img_dir}")
        self.mask_dir = self.root / "masks"

    def __len__(self) -> int:
        return len(self.images)

    def _load_pair(self, idx: int):
        ip = self.images[idx]
        img = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"could not read {ip}")
        if img.ndim == 2:
            img = img[..., None]
        elif img.shape[-1] >= 3:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)

        candidates = [self.mask_dir / (ip.stem + e) for e in IMG_EXT]
        mp = next((p for p in candidates if p.exists()), None)
        if mp is None:
            raise FileNotFoundError(f"no mask for {ip.name}")
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        return img, (mask > 127).astype(np.float32)

    def _random_crop(self, img, mask, rng):
        h, w = mask.shape
        ts = self.tile_size
        if h < ts or w < ts:
            ph, pw = max(ts - h, 0), max(ts - w, 0)
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
            h, w = mask.shape
        # Bias crops toward road-containing regions; fully-empty tiles teach little.
        for _ in range(8):
            y, x = rng.integers(0, h - ts + 1), rng.integers(0, w - ts + 1)
            m = mask[y : y + ts, x : x + ts]
            if m.mean() > 0.01:
                return img[y : y + ts, x : x + ts], m
        return img[y : y + ts, x : x + ts], mask[y : y + ts, x : x + ts]

    def __getitem__(self, idx: int):
        rng = np.random.default_rng()
        img, mask = self._load_pair(idx)
        img = percentile_stretch(img)

        if self.split == "train":
            img, mask = self._random_crop(img, mask, rng)
        else:
            img = img[: self.tile_size, : self.tile_size]
            mask = mask[: self.tile_size, : self.tile_size]

        if self.geometric_aug:
            k = int(rng.integers(0, 4))
            img, mask = np.rot90(img, k, (0, 1)), np.rot90(mask, k, (0, 1))
            if rng.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)

        occ = np.zeros_like(mask)
        if self.synth_occlusion:
            img, occ = apply_occlusions(img, mask, self.occ_cfg)

        img = normalize(img)
        return (
            torch.from_numpy(img.transpose(2, 0, 1).copy()).float(),
            torch.from_numpy(mask[None].copy()).float(),
            torch.from_numpy(occ[None].copy()).float(),
        )
