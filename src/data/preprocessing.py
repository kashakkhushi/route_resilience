"""
Layer 2 - Data Preprocessing.

Implements the preprocessing pipeline from the architecture diagram:
    Image Tiling -> Radiometric Correction -> Normalization -> Augmentation -> Noise Removal

Works on plain numpy arrays so it can be unit-tested without rasterio/GDAL installed.
Georeferenced I/O helpers are isolated in `io_raster.py` and are optional.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Sensor profiles. Ground sample distance drives the tile size we need so that
# every tile covers a comparable ground footprint across sensors.
# ---------------------------------------------------------------------------
SENSOR_PROFILES = {
    "sentinel2": {"gsd_m": 10.0, "bands": ["B2", "B3", "B4", "B8"], "bit_depth": 12},
    "liss4": {"gsd_m": 5.8, "bands": ["G", "R", "NIR"], "bit_depth": 10},
    "cartosat3": {"gsd_m": 0.25, "bands": ["B", "G", "R", "NIR"], "bit_depth": 11},
}


@dataclass
class Tile:
    """A single image tile plus the offset needed to stitch it back."""

    array: np.ndarray  # (H, W, C)
    row: int
    col: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.array.shape[:2]


# ---------------------------------------------------------------------------
# 1. Image tiling
# ---------------------------------------------------------------------------
def tile_image(image: np.ndarray, tile_size: int = 512, overlap: int = 64) -> Iterator[Tile]:
    """Yield overlapping tiles. Overlap avoids roads being cut at tile borders,
    which is the single biggest source of false fragmentation at inference time."""
    if image.ndim == 2:
        image = image[..., None]
    h, w = image.shape[:2]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    for r in range(0, max(h - overlap, 1), stride):
        for c in range(0, max(w - overlap, 1), stride):
            r0, c0 = min(r, max(h - tile_size, 0)), min(c, max(w - tile_size, 0))
            patch = image[r0 : r0 + tile_size, c0 : c0 + tile_size]
            if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                patch = np.pad(
                    patch,
                    ((0, tile_size - patch.shape[0]), (0, tile_size - patch.shape[1]), (0, 0)),
                    mode="reflect",
                )
            yield Tile(patch, r0, c0)


def stitch_tiles(tiles, shape: Tuple[int, int], tile_size: int = 512) -> np.ndarray:
    """Average-blend overlapping tile predictions back into a full-size mask."""
    acc = np.zeros(shape, dtype=np.float32)
    wts = np.zeros(shape, dtype=np.float32)
    # Cosine window down-weights tile edges so seams do not appear as ridges.
    ramp = np.hanning(tile_size)[:, None] * np.hanning(tile_size)[None, :]
    ramp = np.clip(ramp, 1e-3, None).astype(np.float32)

    for tile, pred in tiles:
        h = min(tile_size, shape[0] - tile.row)
        w = min(tile_size, shape[1] - tile.col)
        acc[tile.row : tile.row + h, tile.col : tile.col + w] += (pred * ramp)[:h, :w]
        wts[tile.row : tile.row + h, tile.col : tile.col + w] += ramp[:h, :w]
    return acc / np.maximum(wts, 1e-6)


# ---------------------------------------------------------------------------
# 2. Radiometric correction
# ---------------------------------------------------------------------------
def dark_object_subtraction(image: np.ndarray, percentile: float = 1.0) -> np.ndarray:
    """Cheap atmospheric correction (DOS1). Subtracts the per-band haze floor,
    estimated as the darkest real pixel. Run before normalization."""
    img = image.astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]
    dark = np.percentile(img.reshape(-1, img.shape[-1]), percentile, axis=0)
    return np.clip(img - dark, 0, None)


def percentile_stretch(image: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Per-band contrast stretch to [0, 1]. Handles the different bit depths of
    Sentinel-2 (12-bit), LISS-IV (10-bit) and Cartosat-3 (11-bit) uniformly."""
    img = image.astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]
    out = np.empty_like(img)
    for b in range(img.shape[-1]):
        lo, hi = np.percentile(img[..., b], [low, high])
        out[..., b] = np.clip((img[..., b] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


# ---------------------------------------------------------------------------
# 3. Normalization
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(image: np.ndarray, mean=None, std=None) -> np.ndarray:
    """Standardize to the statistics the pretrained backbone expects."""
    img = image.astype(np.float32)
    c = img.shape[-1]
    if mean is None:
        mean = IMAGENET_MEAN[:c] if c <= 3 else np.full(c, 0.5, np.float32)
    if std is None:
        std = IMAGENET_STD[:c] if c <= 3 else np.full(c, 0.25, np.float32)
    return (img - mean) / std


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Vegetation index. Fed to the model as an extra channel, it gives an
    explicit cue for the tree canopy that causes most occlusions."""
    nir, red = nir.astype(np.float32), red.astype(np.float32)
    return (nir - red) / np.maximum(nir + red, 1e-6)


# ---------------------------------------------------------------------------
# 4. Noise removal
# ---------------------------------------------------------------------------
def denoise(image: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Speckle/sensor noise removal that preserves linear road edges.
    Bilateral is the default because Gaussian blurs away thin rural roads."""
    img = np.ascontiguousarray(image.astype(np.float32))
    if method == "bilateral":
        if img.shape[-1] > 3:
            return np.stack(
                [cv2.bilateralFilter(img[..., b], 5, 0.1, 5) for b in range(img.shape[-1])],
                axis=-1,
            )
        return cv2.bilateralFilter(img, 5, 0.1, 5)
    if method == "median":
        return cv2.medianBlur((img * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    raise ValueError(f"unknown denoise method: {method}")


def preprocess(image: np.ndarray, sensor: str = "sentinel2") -> np.ndarray:
    """Full Layer-2 pipeline for a single image (no augmentation - see occlusion.py)."""
    if sensor not in SENSOR_PROFILES:
        raise ValueError(f"unknown sensor {sensor}; expected one of {list(SENSOR_PROFILES)}")
    x = dark_object_subtraction(image)
    x = percentile_stretch(x)
    x = denoise(x)
    return x
