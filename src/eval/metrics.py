"""
Segmentation metrics - implements the evaluation table from the problem statement.

    IoU / Dice          standard pixel accuracy
    Occlusion-Recall    recall computed ONLY on road pixels that were occluded.
                        This is the headline number: it isolates the project's
                        central claim (inferring roads under canopy/shadow) from
                        ordinary segmentation performance.
    Relaxed IoU         buffered comparison with a 3-5 px tolerance, so a road
                        predicted 2 px off-centre is not punished as a total miss.
    clDice              centreline (topology) agreement - see models/losses.py.

All functions take numpy bool/float arrays and are framework-free, so they can
be used to score any model (U-Net baseline, DeepLabV3+, SegFormer) identically.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

EPS = 1e-9


def _b(x) -> np.ndarray:
    return np.asarray(x).astype(bool)


# ---------------------------------------------------------------- basic
def confusion(pred, target) -> dict:
    p, t = _b(pred), _b(target)
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def iou(pred, target) -> float:
    c = confusion(pred, target)
    return c["tp"] / (c["tp"] + c["fp"] + c["fn"] + EPS)


def dice(pred, target) -> float:
    c = confusion(pred, target)
    return 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"] + EPS)


def precision_recall(pred, target) -> tuple[float, float]:
    c = confusion(pred, target)
    return (c["tp"] / (c["tp"] + c["fp"] + EPS),
            c["tp"] / (c["tp"] + c["fn"] + EPS))


# ---------------------------------------------------------------- occlusion
def occlusion_recall(pred, target, occlusion_mask, threshold: float = 0.5) -> dict:
    """Recall split by whether the ground-truth road pixel was occluded.

    The `visible` figure is what an ordinary segmentation paper reports.
    The `occluded` figure is what this project is actually about, and the
    `gap` between them is the honest measure of how much the model is truly
    inferring versus merely observing.

    A model trained without synthetic occluders typically shows a large gap
    (e.g. 0.88 visible vs 0.34 occluded); the whole point of the occlusion
    augmentation is to close it.
    """
    p, t = _b(pred), _b(target)
    occ = np.asarray(occlusion_mask) > threshold

    road_vis = t & ~occ
    road_occ = t & occ

    r_vis = (p & road_vis).sum() / (road_vis.sum() + EPS) if road_vis.any() else float("nan")
    r_occ = (p & road_occ).sum() / (road_occ.sum() + EPS) if road_occ.any() else float("nan")

    return {
        "recall_visible": float(r_vis),
        "recall_occluded": float(r_occ),
        "occlusion_gap": float(r_vis - r_occ) if road_occ.any() else float("nan"),
        "road_px_visible": int(road_vis.sum()),
        "road_px_occluded": int(road_occ.sum()),
        "occluded_fraction": float(road_occ.sum() / (t.sum() + EPS)),
    }


# ---------------------------------------------------------------- relaxed
def buffered_masks(pred, target, buffer_px: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Dilate each mask by `buffer_px` using an exact Euclidean distance
    transform (not a square structuring element, which would over-dilate
    diagonally by a factor of sqrt(2))."""
    p, t = _b(pred), _b(target)
    # distance_transform_edt measures distance to the nearest zero, so invert.
    d_pred = ndimage.distance_transform_edt(~p)
    d_true = ndimage.distance_transform_edt(~t)
    return d_pred <= buffer_px, d_true <= buffer_px


def relaxed_metrics(pred, target, buffer_px: int = 4) -> dict:
    """Length-complete / relaxed IoU with a tolerance buffer.

    Correctness (precision): predicted pixels within `buffer_px` of true road.
    Completeness (recall):   true road pixels within `buffer_px` of a prediction.
    This is the standard road-extraction relaxed scoring (Wiedemann et al.) and
    it is what prevents sub-pixel georeferencing shifts from destroying the score.
    """
    p, t = _b(pred), _b(target)
    p_buf, t_buf = buffered_masks(pred, target, buffer_px)

    correctness = (p & t_buf).sum() / (p.sum() + EPS)
    completeness = (t & p_buf).sum() / (t.sum() + EPS)
    quality = (correctness * completeness) / (
        correctness + completeness - correctness * completeness + EPS
    )
    f1 = 2 * correctness * completeness / (correctness + completeness + EPS)

    return {
        "buffer_px": buffer_px,
        "relaxed_precision": float(correctness),   # "correctness"
        "relaxed_recall": float(completeness),     # "completeness"
        "relaxed_iou": float(quality),             # "quality"
        "relaxed_f1": float(f1),
    }


# ---------------------------------------------------------------- topology
def cl_dice(pred, target) -> dict:
    """Centreline Dice on binary masks (numpy counterpart of the training loss).

    A mask with a 3-px break scores nearly full IoU but loses clDice sharply,
    which is exactly the failure mode that breaks the downstream graph.
    """
    p, t = _b(pred), _b(target)
    if not p.any() or not t.any():
        return {"cldice": 0.0, "topology_precision": 0.0, "topology_recall": 0.0}

    sp, st = skeletonize(p), skeletonize(t)
    t_prec = (sp & t).sum() / (sp.sum() + EPS)   # predicted centreline on true road
    t_rec = (st & p).sum() / (st.sum() + EPS)    # true centreline covered
    return {
        "cldice": float(2 * t_prec * t_rec / (t_prec + t_rec + EPS)),
        "topology_precision": float(t_prec),
        "topology_recall": float(t_rec),
    }


def connectivity_metrics(pred, target, min_object_px: int = 32) -> dict:
    """Fragmentation of the prediction versus ground truth.

    `component_ratio` > 1 means the prediction is more fragmented than reality;
    this is the number that should fall towards 1.0 after MST healing.
    """
    p, t = _b(pred), _b(target)
    lab_p, n_p = ndimage.label(p, structure=np.ones((3, 3)))
    lab_t, n_t = ndimage.label(t, structure=np.ones((3, 3)))

    # Ignore speckle so the ratio reflects real fragmentation, not noise.
    if n_p:
        n_p = int((np.bincount(lab_p.ravel())[1:] >= min_object_px).sum())
    if n_t:
        n_t = int((np.bincount(lab_t.ravel())[1:] >= min_object_px).sum())

    largest_p = (np.bincount(lab_p.ravel())[1:].max() if n_p else 0)
    return {
        "components_pred": n_p,
        "components_true": n_t,
        "component_ratio": float(n_p / max(n_t, 1)),
        "largest_component_fraction": float(largest_p / (p.sum() + EPS)),
    }


# ---------------------------------------------------------------- aggregate
def evaluate_mask(pred, target, occlusion_mask=None, buffer_px: int = 4) -> dict:
    """One call producing every pixel-level number in the evaluation table."""
    prec, rec = precision_recall(pred, target)
    out = {
        "iou": iou(pred, target),
        "dice": dice(pred, target),
        "precision": prec,
        "recall": rec,
        **relaxed_metrics(pred, target, buffer_px),
        **cl_dice(pred, target),
        **connectivity_metrics(pred, target),
    }
    if occlusion_mask is not None:
        out.update(occlusion_recall(pred, target, occlusion_mask))
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}


def aggregate(results: list[dict]) -> dict:
    """Mean +- std across tiles. Report std: a model with IoU 0.62 +- 0.03 is
    deployable, one with 0.62 +- 0.25 is not, and the mean hides the difference."""
    if not results:
        return {}
    keys = [k for k in results[0] if isinstance(results[0][k], (int, float))]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in results if r.get(k) is not None], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[k] = round(float(vals.mean()), 5)
            out[f"{k}_std"] = round(float(vals.std()), 5)
    out["n_tiles"] = len(results)
    return out
