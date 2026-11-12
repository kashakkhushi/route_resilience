"""
Occlusion simulation - the core of the "occlusion-robust" claim.

Standard segmentation training teaches a model to find visible road pixels.
If you want a model that infers road *continuity* under a tree canopy, you must
show it images where the road is hidden but the label still says "road".
That is exactly what this module manufactures: the occluder is painted onto the
image only, never onto the mask.

Four occluder families, matching the failure modes named in the problem statement:
    tree canopy      - blobby, textured, green, hard-edged
    building shadow  - directional, low-luminance, sharp-edged polygon
    cloud / haze     - large, soft-edged, brightening + contrast-crushing
    vehicles         - small, high-contrast, on-road rectangles
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np


@dataclass
class OcclusionConfig:
    tree_prob: float = 0.7
    shadow_prob: float = 0.5
    cloud_prob: float = 0.25
    vehicle_prob: float = 0.6
    # Fraction of road pixels we are willing to hide in one sample.
    max_road_occlusion: float = 0.45
    sun_azimuth_deg: Tuple[float, float] = (0.0, 360.0)
    rng_seed: int | None = None
    _rng: np.random.Generator = field(default=None, repr=False)

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.rng_seed)
        return self._rng


def _perlin_like(shape, rng, scale: int = 8) -> np.ndarray:
    """Cheap smooth noise field (upsampled white noise) for organic blob shapes."""
    small = rng.random((max(shape[0] // scale, 2), max(shape[1] // scale, 2))).astype(np.float32)
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def tree_canopy(shape, rng, coverage: float = 0.15) -> np.ndarray:
    """Soft blobby alpha map imitating a tree line or canopy patch."""
    field_ = _perlin_like(shape, rng, scale=rng.integers(6, 16))
    thresh = np.quantile(field_, 1.0 - coverage)
    alpha = np.clip((field_ - thresh) * 6.0, 0, 1)
    return cv2.GaussianBlur(alpha, (0, 0), sigmaX=2.0)


def building_shadow(shape, rng, azimuth_deg: float, n: int = 3) -> np.ndarray:
    """Directional hard-edged quadrilaterals cast off building footprints."""
    alpha = np.zeros(shape, np.float32)
    h, w = shape
    theta = np.deg2rad(azimuth_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    for _ in range(n):
        x, y = rng.integers(0, w), rng.integers(0, h)
        bw, bh = rng.integers(15, 60), rng.integers(15, 60)
        length = rng.integers(20, 90)
        pts = np.array(
            [
                [x, y],
                [x + bw, y],
                [x + bw + dx * length, y + bh + dy * length],
                [x + dx * length, y + bh + dy * length],
            ],
            np.int32,
        )
        cv2.fillPoly(alpha, [pts], 1.0)
    return cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.0)


def cloud(shape, rng, coverage: float = 0.2) -> np.ndarray:
    """Large soft alpha region - the only occluder that brightens rather than darkens."""
    field_ = _perlin_like(shape, rng, scale=rng.integers(20, 40))
    thresh = np.quantile(field_, 1.0 - coverage)
    alpha = np.clip((field_ - thresh) * 3.0, 0, 1)
    return cv2.GaussianBlur(alpha, (0, 0), sigmaX=8.0)


# Plausible vehicle body colours, as (low, high) ranges sampled per vehicle.
# Deliberately NOT a single uniform random colour: the previous implementation
# painted every vehicle in one arbitrary colour and placed them only on road
# pixels, which drew a coloured dotted line along the carriageway. That acted as
# a road MARKER rather than an occluder and inflated recall inside occluded
# regions. See docs/OCCLUSION_HARNESS.md.
_VEHICLE_PALETTE = (
    (0.05, 0.18),   # black / very dark
    (0.20, 0.35),   # dark grey, navy, maroon
    (0.45, 0.60),   # mid grey, muted blue/green
    (0.72, 0.92),   # white / silver
)


def road_corridor(road_mask: np.ndarray, road_width_px: int = 7) -> np.ndarray:
    """Road plus a margin either side.

    Vehicles are sampled from this corridor rather than from road pixels alone,
    so placements span on-road, straddling the edge, and immediately adjacent
    (verge, parking, driveway). Sampling only road pixels makes the occluder
    perfectly correlated with the label, which is what corrupted the previous
    evaluation.
    """
    k = max(int(round(road_width_px * 1.6)) | 1, 3)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(road_mask.astype(np.uint8), kern, iterations=1).astype(bool)


def vehicles(shape, rng, road_mask: np.ndarray, n_channels: int = 3,
             n: int = 25, road_width_px: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Vehicle-like occluders with realistic placement and appearance.

    Returns (alpha, colour_layer). `alpha` is the occluder geometry and is the
    ONLY thing used to define "occluded region" downstream - it is never derived
    from a prediction.

    Placement: centres drawn uniformly from the road corridor, so roughly
    on-road, edge-straddling and adjacent cases appear in natural proportion
    rather than being forced on-road.

    Appearance: per-vehicle colour from a plausible palette, per-vehicle size and
    orientation, a soft cast shadow, and mild per-pixel texture. No single colour
    is reused across vehicles, so no road-aligned chromatic cue is created.

    Fully deterministic given `rng`.
    """
    alpha = np.zeros(shape, np.float32)
    # Per-channel 2-D buffers: cv2 cannot draw into a non-contiguous
    # colour[..., c] view, so channels are painted separately and stacked.
    chan = [np.zeros(shape, np.float32) for _ in range(n_channels)]
    if not road_mask.any():
        return alpha, np.stack(chan, axis=-1)

    corridor = road_corridor(road_mask, road_width_px)
    ys, xs = np.nonzero(corridor)
    if len(ys) == 0:
        return alpha, np.stack(chan, axis=-1)

    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    for i in idx:
        cy, cx = float(ys[i]), float(xs[i])
        # Size: cars through to small trucks at ~0.5 m GSD.
        length = float(rng.uniform(4.0, 13.0))
        width = float(rng.uniform(2.5, 6.0))
        angle = float(rng.uniform(0.0, 180.0))

        lo, hi = _VEHICLE_PALETTE[int(rng.integers(0, len(_VEHICLE_PALETTE)))]
        base = float(rng.uniform(lo, hi))
        # Slight per-channel tint so vehicles are not pure greys.
        tint = rng.normal(0.0, 0.04, size=n_channels).astype(np.float32)
        body = np.clip(base + tint, 0.0, 1.0)

        box = cv2.boxPoints(((cx, cy), (length, width), angle))

        # Cast shadow first, offset and darker, so the vehicle sits over it.
        sh = np.deg2rad(float(rng.uniform(0, 360)))
        off = float(rng.uniform(1.0, 2.5))
        shadow = np.int32(box + np.array([np.cos(sh) * off, np.sin(sh) * off]))
        cv2.fillConvexPoly(alpha, shadow, 0.55)
        for c in range(n_channels):
            cv2.fillConvexPoly(chan[c], shadow, float(body[c]) * 0.35)

        pts = np.int32(box)
        cv2.fillConvexPoly(alpha, pts, 1.0)
        for c in range(n_channels):
            cv2.fillConvexPoly(chan[c], pts, float(body[c]))

    # Mild texture so vehicles are not flat patches.
    colour = np.stack(chan, axis=-1)
    tex = rng.normal(0.0, 0.03, colour.shape).astype(np.float32)
    colour = np.clip(colour + tex * (alpha[..., None] > 0), 0.0, 1.0)
    return alpha, colour


def apply_occlusions(
    image: np.ndarray,
    road_mask: np.ndarray,
    cfg: OcclusionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (occluded_image, occlusion_alpha).

    The road mask is deliberately returned unchanged by the caller: the label
    still marks road under the occluder, which is what forces the network to
    learn contextual completion instead of pure appearance matching.
    """
    cfg = cfg or OcclusionConfig()
    rng = cfg.rng()
    img = image.astype(np.float32).copy()
    if img.ndim == 2:
        img = img[..., None]
    shape = img.shape[:2]
    total_alpha = np.zeros(shape, np.float32)

    if rng.random() < cfg.tree_prob:
        a = tree_canopy(shape, rng, coverage=float(rng.uniform(0.08, 0.28)))
        green = np.array([0.12, 0.30, 0.10], np.float32)[: img.shape[-1]]
        img = img * (1 - a[..., None] * 0.85) + a[..., None] * 0.85 * green
        total_alpha = np.maximum(total_alpha, a)

    if rng.random() < cfg.shadow_prob:
        az = float(rng.uniform(*cfg.sun_azimuth_deg))
        a = building_shadow(shape, rng, az, n=int(rng.integers(2, 6)))
        img = img * (1 - a[..., None] * 0.65)  # shadows darken, keep some detail
        total_alpha = np.maximum(total_alpha, a * 0.8)

    if rng.random() < cfg.cloud_prob:
        a = cloud(shape, rng, coverage=float(rng.uniform(0.1, 0.35)))
        img = img * (1 - a[..., None]) + a[..., None] * 0.92  # bright, low contrast
        total_alpha = np.maximum(total_alpha, a)

    if rng.random() < cfg.vehicle_prob:
        a, veh_colour = vehicles(shape, rng, road_mask, n_channels=img.shape[-1],
                                 n=int(rng.integers(10, 40)))
        img = img * (1 - a[..., None]) + a[..., None] * veh_colour
        total_alpha = np.maximum(total_alpha, a)

    # Safety valve: if we hid too much road, blend some of the clean image back.
    road = road_mask.astype(bool)
    if road.any():
        hidden = float((total_alpha[road] > 0.5).mean())
        if hidden > cfg.max_road_occlusion:
            keep = cfg.max_road_occlusion / hidden
            img = img * keep + image.astype(np.float32) * (1 - keep)
            total_alpha *= keep

    return np.clip(img, 0, 1), total_alpha


def occlusion_report(alpha: np.ndarray, road_mask: np.ndarray) -> dict:
    """Diagnostics used in the evaluation table: how much road was hidden, and
    therefore how much of the model's recall came from contextual inference."""
    road = road_mask.astype(bool)
    occ = alpha > 0.5
    return {
        "scene_occluded_frac": float(occ.mean()),
        "road_occluded_frac": float(occ[road].mean()) if road.any() else 0.0,
        "road_visible_px": int((road & ~occ).sum()),
        "road_hidden_px": int((road & occ).sum()),
    }
