"""
Tiled inference over full satellite scenes.

A Sentinel-2 tile is 10980x10980 px and will not fit in GPU memory, so we run
overlapping windows and blend. The Hann-window blending in preprocessing.stitch_tiles
matters more than it looks: hard tile boundaries produce a grid of seams, and the
skeletonizer faithfully turns every seam into a spurious road.
"""
from __future__ import annotations

import numpy as np
import torch

import torch.nn as nn

from ..data.preprocessing import normalize, percentile_stretch, stitch_tiles, tile_image


@torch.no_grad()
def predict_logits(model: "nn.Module", x: torch.Tensor, tta: bool = True) -> torch.Tensor:
    """Test-time augmentation over the dihedral group.

    Lives here rather than in segformer.py because it is architecture-agnostic:
    it only needs a model returning {"road_logits": ...}. Keeping it in the
    SegFormer module forced every inference run to import `transformers`, so a
    U-Net checkpoint could not be served without the whole HuggingFace stack
    installed.

    Roads are direction-agnostic, so averaging over 4 flips is a free ~1-2 IoU
    point gain and noticeably reduces fragmentation on diagonal roads.
    """
    model.eval()
    acc = torch.sigmoid(model(x)["road_logits"])
    if not tta:
        return acc
    n = 1
    for dims in [(-1,), (-2,), (-1, -2)]:
        p = torch.sigmoid(model(torch.flip(x, dims=dims))["road_logits"])
        acc = acc + torch.flip(p, dims=dims)
        n += 1
    return acc / n


@torch.no_grad()
def predict_scene(
    model,
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 128,
    batch_size: int = 8,
    device: str = "cuda",
    tta: bool = True,
    threshold: float = 0.5,
) -> dict:
    """Return {'prob', 'mask', 'occlusion'} at full scene resolution."""
    model.eval().to(device)
    img = percentile_stretch(image)
    tiles = list(tile_image(img, tile_size, overlap))

    road_out, occ_out = [], []
    for i in range(0, len(tiles), batch_size):
        chunk = tiles[i : i + batch_size]
        batch = np.stack([normalize(t.array).transpose(2, 0, 1) for t in chunk])
        x = torch.from_numpy(batch).float().to(device)

        prob = predict_logits(model, x, tta=tta).cpu().numpy()[:, 0]
        for t, p in zip(chunk, prob):
            road_out.append((t, p))

        if getattr(model, "aux_occlusion_head", False):
            o = torch.sigmoid(model(x)["occlusion_logits"]).cpu().numpy()[:, 0]
            for t, p in zip(chunk, o):
                occ_out.append((t, p))

    shape = image.shape[:2]
    prob = stitch_tiles(road_out, shape, tile_size)
    result = {"prob": prob, "mask": prob > threshold}
    if occ_out:
        result["occlusion"] = stitch_tiles(occ_out, shape, tile_size)
    return result


def evidence_map_from_prediction(pred: dict, w_occlusion: float = 0.6) -> np.ndarray:
    """Build the evidence raster that healing.py uses to price bridges.

    A gap is cheap to bridge if EITHER the model still had some road probability
    there, OR it confidently detected an occluder. The second term is the point
    of the auxiliary head: "I see a tree, not an empty field" is positive evidence
    that a road continues underneath."""
    prob = pred["prob"]
    if "occlusion" in pred:
        return np.clip(prob + w_occlusion * pred["occlusion"], 0, 1)
    return prob
