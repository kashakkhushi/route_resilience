"""
Synthetic city generator.

Purpose: make the whole pipeline runnable and testable with no dataset download,
no GPU and no network access. It produces a road mask with realistic structure -
a distorted grid, a ring road, radial arterials and a river with a small number
of crossings - so that the criticality analysis has something meaningful to find.

The river crossings matter: they are deliberately few, which manufactures genuine
articulation points and bridge edges. A pure grid has almost no single points of
failure and makes the resilience module look trivially optimistic.
"""
from __future__ import annotations

import cv2
import numpy as np


def generate_city(
    size: int = 768,
    grid_spacing: int = 96,
    road_width: int = 4,
    n_radials: int = 6,
    river: bool = True,
    n_crossings: int = 2,
    jitter: float = 12.0,
    seed: int = 0,
) -> dict:
    """Return dict with 'mask' (bool), 'image' (float RGB), 'river' (bool)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), np.uint8)
    c = size // 2

    def wobbly_line(p0, p1, width, segments=8):
        """Draw a slightly irregular line - straight lines skeletonize too cleanly
        and hide the artefacts the pruning code exists to handle."""
        p0, p1 = np.array(p0, float), np.array(p1, float)
        ts = np.linspace(0, 1, segments + 1)[:, None]
        pts = p0[None] * (1 - ts) + p1[None] * ts
        pts[1:-1] += rng.normal(0, jitter, pts[1:-1].shape)
        pts = pts.round().astype(np.int32)
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(mask, tuple(a), tuple(b), 255, width)

    # --- distorted grid ---------------------------------------------------
    for x in range(grid_spacing, size, grid_spacing):
        wobbly_line((x, 0), (x, size - 1), road_width)
    for y in range(grid_spacing, size, grid_spacing):
        wobbly_line((0, y), (size - 1, y), road_width)

    # --- ring road --------------------------------------------------------
    ring_r = int(size * 0.36)
    ring_pts = []
    for a in np.linspace(0, 2 * np.pi, 40):
        rr = ring_r + rng.normal(0, jitter * 0.6)
        ring_pts.append((int(c + rr * np.cos(a)), int(c + rr * np.sin(a))))
    cv2.polylines(mask, [np.array(ring_pts, np.int32)], True, 255, road_width + 2)

    # --- radial arterials -------------------------------------------------
    for a in np.linspace(0, 2 * np.pi, n_radials, endpoint=False):
        end = (int(c + size * 0.55 * np.cos(a)), int(c + size * 0.55 * np.sin(a)))
        wobbly_line((c, c), end, road_width + 2, segments=6)

    mask = mask > 0

    # --- river: cuts the network, leaving only n_crossings bridges ---------
    river_mask = np.zeros((size, size), bool)
    if river:
        ys = np.arange(size)
        xs = (c + 0.32 * size * np.sin(ys / size * 2.6 * np.pi + 0.7)).astype(int)
        band = int(size * 0.035)
        for y, x in zip(ys, xs):
            river_mask[y, max(x - band, 0) : min(x + band, size)] = True

        keep = np.zeros((size, size), bool)
        for cy in np.linspace(size * 0.2, size * 0.8, n_crossings):
            cy = int(cy)
            keep[max(cy - road_width * 2, 0) : cy + road_width * 2, :] = True

        mask = mask & ~(river_mask & ~keep)

    # --- synthesize a plausible RGB scene ---------------------------------
    image = _render_scene(mask, river_mask, rng)
    return {"mask": mask, "image": image, "river": river_mask}


def _render_scene(mask: np.ndarray, river: np.ndarray, rng) -> np.ndarray:
    """Grey asphalt on mixed terrain, with texture noise. Not photorealistic -
    it exists so the occlusion module and the visualizations have something
    to draw on, not to train a deployable model."""
    h, w = mask.shape
    small = rng.random((h // 24, w // 24)).astype(np.float32)
    terrain = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 0.32 + 0.18 * terrain      # soil / rooftops
    img[..., 1] = 0.38 + 0.22 * terrain      # vegetation
    img[..., 2] = 0.26 + 0.14 * terrain
    img[river] = np.array([0.12, 0.20, 0.35], np.float32)
    img[mask] = np.array([0.46, 0.45, 0.44], np.float32)

    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def fragment_mask(mask: np.ndarray, n_gaps: int = 40, max_gap: int = 22,
                  seed: int = 0) -> np.ndarray:
    """Punch holes in a clean mask to emulate an occlusion-damaged segmentation.

    This is how we test the healing stage without training a model: we know the
    ground-truth topology, so we can measure exactly how much of it healing
    recovers."""
    rng = np.random.default_rng(seed)
    out = mask.copy()
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return out
    idx = rng.choice(len(ys), size=min(n_gaps, len(ys)), replace=False)
    for i in idx:
        r = int(rng.integers(6, max_gap))
        cv2.circle(out.view(np.uint8), (int(xs[i]), int(ys[i])), r, 0, -1)
    return out.astype(bool)
