"""
Model registry and the provenance contract.

This module exists to make one guarantee enforceable in code: the system never
presents a non-learned result as a deep-learning prediction.

Every road mask leaving this module carries a `ModelProvenance` describing what
actually produced it. The API returns it, and the UI is required to display it.

Three modes, in order of preference:

    TRAINED_MODEL   a real .pt checkpoint was found and loaded (torch present)
    CLASSICAL_BASELINE   no checkpoint; a genuine multi-scale ridge-detection
                         extractor ran on the real pixels. This is NOT a neural
                         network and is never labelled as one. It is also not a
                         mock: it performs real computation on the uploaded image
                         and its output varies with the input, as it must.
    DEMO_DATA            no image at all; a deterministic synthetic city. Only
                         used when the user explicitly loads the demo.

The classical baseline is a Frangi-style vesselness filter. Roads are elongated
bright ridges at a known width range, which is exactly the structure the Hessian
eigenvalue ratio responds to. It is a defensible baseline from the pre-deep-learning
road-extraction literature, not a placeholder.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


class ExtractionMode(str, Enum):
    TRAINED_MODEL = "TRAINED_MODEL"
    CLASSICAL_BASELINE = "CLASSICAL_BASELINE"
    DEMO_DATA = "DEMO_DATA"


@dataclass
class ModelProvenance:
    """Attached to every mask. The UI must render `label` verbatim."""

    mode: ExtractionMode
    label: str
    description: str
    is_learned: bool
    checkpoint: str | None = None
    architecture: str | None = None
    metrics_available: bool = False

    def dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d


PROVENANCE = {
    ExtractionMode.CLASSICAL_BASELINE: ModelProvenance(
        mode=ExtractionMode.CLASSICAL_BASELINE,
        label="CLASSICAL BASELINE - NOT A TRAINED MODEL",
        description=(
            "No trained checkpoint is present, so roads were extracted by a "
            "multi-scale Hessian ridge filter (Frangi vesselness) followed by "
            "morphological cleanup. This performs real computation on the uploaded "
            "pixels but is not a deep-learning prediction, and its accuracy is "
            "substantially below a trained segmentation model."
        ),
        is_learned=False,
        architecture="Frangi vesselness + morphological post-processing",
        metrics_available=False,
    ),
    ExtractionMode.DEMO_DATA: ModelProvenance(
        mode=ExtractionMode.DEMO_DATA,
        label="DEMO DATA - SYNTHETIC CITY",
        description=(
            "A deterministic synthetic road network with known ground-truth "
            "topology. No satellite imagery was processed. Used to demonstrate the "
            "graph, criticality and simulation stages offline; segmentation metrics "
            "shown against it measure the healing algorithm, not a trained model."
        ),
        is_learned=False,
        architecture="procedural generator",
        metrics_available=True,  # ground truth IS known here
    ),
    # Template for the trained path. neural_extract() overrides `architecture`
    # and `label` with the values read from the checkpoint; this entry exists so
    # /api/health can report provenance without having to load the model.
    ExtractionMode.TRAINED_MODEL: ModelProvenance(
        mode=ExtractionMode.TRAINED_MODEL,
        label="TRAINED MODEL",
        description=(
            "Roads were extracted by a trained deep-learning checkpoint running "
            "real inference on the input pixels. Validation metrics stored in the "
            "checkpoint were measured at training time on the validation split "
            "and are not test-set performance."
        ),
        is_learned=True,
        architecture=None,
        metrics_available=False,
    ),
}


def find_checkpoint(search_dirs=("checkpoints",)) -> Path | None:
    """Locate a trained checkpoint, if one exists."""
    env = os.getenv("ROUTE_RESILIENCE_CHECKPOINT")
    if env and Path(env).is_file():
        return Path(env)
    for d in search_dirs:
        p = Path(d)
        if p.is_dir():
            found = sorted(p.rglob("*.pt"))
            if found:
                return found[0]
    return None


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Classical baseline - real computation, honestly labelled
# ---------------------------------------------------------------------------
def _frangi_vesselness(gray: np.ndarray, scales, beta: float = 0.5,
                       c: float = 0.12) -> np.ndarray:
    """Multi-scale ridge filter.

    At each scale we build the Hessian, take its eigenvalues, and score pixels
    that look like a bright elongated ridge: one small eigenvalue along the road
    and one large one across it. `scales` should bracket the expected road
    half-width in pixels, which is why the caller derives them from the GSD.
    """
    best = np.zeros_like(gray, dtype=np.float32)

    for s in scales:
        sm = cv2.GaussianBlur(gray, (0, 0), sigmaX=s)
        # Hessian components, normalized by s^2 so scales are comparable.
        gxx = ndimage.sobel(ndimage.sobel(sm, axis=1), axis=1) * (s ** 2)
        gyy = ndimage.sobel(ndimage.sobel(sm, axis=0), axis=0) * (s ** 2)
        gxy = ndimage.sobel(ndimage.sobel(sm, axis=1), axis=0) * (s ** 2)

        tmp = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4 * gxy ** 2, 0))
        l1 = 0.5 * (gxx + gyy + tmp)
        l2 = 0.5 * (gxx + gyy - tmp)

        # order so |lam1| <= |lam2|
        swap = np.abs(l1) > np.abs(l2)
        lam1 = np.where(swap, l2, l1)
        lam2 = np.where(swap, l1, l2)

        rb = np.abs(lam1) / np.maximum(np.abs(lam2), 1e-9)   # blobness
        sness = np.sqrt(lam1 ** 2 + lam2 ** 2)               # structureness

        v = np.exp(-(rb ** 2) / (2 * beta ** 2)) * (1 - np.exp(-(sness ** 2) / (2 * c ** 2)))
        # Roads are brighter than surroundings -> ridge, so lam2 must be negative.
        v[lam2 > 0] = 0
        best = np.maximum(best, v.astype(np.float32))

    return best


def classical_extract(image: np.ndarray, pixel_size_m: float = 1.0,
                      road_width_m: float = 8.0,
                      threshold: float | None = None,
                      road_fraction_prior: float = 0.08) -> dict:
    """Extract a road probability map without any trained model.

    Returns the same contract as the neural path: {'prob', 'mask', 'evidence',
    'provenance'} - so the downstream graph pipeline is identical either way.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img[..., :3] if img.shape[-1] >= 3 else img[..., 0]
        gray = img.mean(axis=-1) if img.ndim == 3 else img
    else:
        gray = img
    if gray.max() > 1.5:
        gray = gray / 255.0

    # Local contrast normalization: satellite scenes have large illumination
    # gradients that otherwise dominate the Hessian response.
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(gray.shape) / 40.0)
    norm = np.clip(gray - bg + 0.5, 0, 1)

    half_w_px = max(road_width_m / (2.0 * max(pixel_size_m, 1e-6)), 1.0)
    scales = [half_w_px * f for f in (0.6, 1.0, 1.6, 2.4)]
    vess = _frangi_vesselness(norm, scales)

    if vess.max() > 0:
        vess = vess / vess.max()

    # Hysteresis thresholding rather than a single global cut.
    #
    # A single threshold on a ridge response either floods the scene (low) or
    # shatters roads into dashes (high). Hysteresis takes confident ridge cores
    # as seeds and grows them only through connected medium-response pixels, so
    # a road dimmed by shadow stays attached to its bright neighbour instead of
    # becoming a separate fragment. Otsu was tried first and over-predicted
    # heavily, because the response histogram is dominated by near-zero
    # background and the between-class split lands far too low.
    #
    # Levels are set by percentile against a plausible urban road fraction
    # rather than absolute response values, which are scene-dependent.
    hi_frac = float(np.clip(road_fraction_prior, 0.02, 0.35))
    t_hi = float(np.quantile(vess, 1.0 - hi_frac))
    t_lo = float(np.quantile(vess, 1.0 - min(hi_frac * 2.5, 0.6)))
    if threshold is not None:            # caller override wins
        t_hi = threshold
        t_lo = threshold * 0.6

    seeds = vess >= t_hi
    grow = vess >= t_lo
    lab, n = ndimage.label(grow, structure=np.ones((3, 3)))
    if n:
        keep_ids = np.unique(lab[seeds])
        keep_ids = keep_ids[keep_ids != 0]
        mask = np.isin(lab, keep_ids)
    else:
        mask = seeds
    threshold = t_hi

    # Morphological cleanup: close small gaps, drop speckle.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=1) > 0
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n:
        sizes = np.bincount(lab.ravel())
        keep = np.where(sizes >= max(int(0.00002 * mask.size), 30))[0]
        keep = keep[keep != 0]
        mask = np.isin(lab, keep)

    return {
        "prob": vess,
        "mask": mask,
        # No occlusion head exists in this mode, so evidence is the response
        # itself. Healing degrades gracefully: it prices bridges on distance
        # and angle alone.
        "evidence": vess,
        "threshold": threshold,
        "provenance": PROVENANCE[ExtractionMode.CLASSICAL_BASELINE],
    }


def neural_extract(image: np.ndarray, checkpoint: Path, device: str = "cpu",
                   threshold: float = 0.5) -> dict:
    """Trained-model path. Only reachable when torch and a checkpoint exist."""
    import torch  # noqa: F401

    from src.models.baselines import build_model
    from src.models.inference import evidence_map_from_prediction, predict_scene

    import torch as _t

    ckpt = _t.load(checkpoint, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    name = args.get("model", "segformer_b3")

    # encoder_weights=None is REQUIRED here, not an optimisation. The default
    # ("imagenet") downloads ~85 MB of pretrained encoder weights that are then
    # immediately overwritten by the checkpoint's state_dict. On an offline or
    # firewalled machine that download fails and inference dies with an HTTP
    # error, despite every weight it needs already being on disk.
    try:
        model = build_model(name, args.get("in_channels", 3), encoder_weights=None)
    except TypeError:
        # SegFormer variants take no encoder_weights argument.
        model = build_model(name, args.get("in_channels", 3))

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match architecture '{name}': "
            f"{len(missing)} missing and {len(unexpected)} unexpected tensors. "
            "The checkpoint was probably trained with a different --model."
        )
    model = model.to(device).eval()

    pred = predict_scene(model, image, device=device, threshold=threshold)
    pred["evidence"] = evidence_map_from_prediction(pred)
    pred["provenance"] = ModelProvenance(
        mode=ExtractionMode.TRAINED_MODEL,
        label=f"TRAINED MODEL - {name}",
        description=(
            f"Segmentation produced by {name} loaded from {checkpoint.name}. "
            "Reported validation metrics come from the checkpoint's own training run."
        ),
        is_learned=True,
        checkpoint=str(checkpoint),
        architecture=name,
        metrics_available=bool(ckpt.get("metrics")),
    )
    pred["checkpoint_metrics"] = ckpt.get("metrics")
    return pred


def simulated_demo_extract(gt_mask: np.ndarray, occlusion: np.ndarray | None = None,
                           gaps: int = 70, seed: int = 1) -> dict:
    """Deterministic stand-in for an occlusion-damaged segmentation.

    This does NOT pretend to be inference. It takes the synthetic city's known
    ground truth and punches occlusion-shaped holes in it, reproducing the
    failure mode the project exists to fix - fragmentation - with the true
    topology still known. That is what makes the demo scientifically useful:
    healing quality can be MEASURED against the real graph rather than admired.

    It is only ever reachable from a demo session, and is always reported as
    DEMO_DATA so no viewer can mistake it for a model output.
    """
    from src.data.synthetic import fragment_mask

    mask = fragment_mask(gt_mask, n_gaps=gaps, max_gap=22, seed=seed + 1)
    prob = mask.astype(np.float32)
    # Evidence tells healing where road is likely HIDDEN rather than absent, so
    # bridging under a detected occluder is cheap. With no occlusion map we fall
    # back to the mask itself.
    evidence = (np.clip(0.25 + 0.75 * occlusion, 0, 1).astype(np.float32)
                if occlusion is not None else prob)
    return {
        "prob": prob,
        "mask": mask,
        "evidence": evidence,
        "threshold": 0.5,
        "provenance": PROVENANCE[ExtractionMode.DEMO_DATA],
    }


def extract_roads(image: np.ndarray, pixel_size_m: float = 1.0,
                  threshold: float | None = None, prefer_neural: bool = True,
                  road_width_m: float = 8.0, road_fraction_prior: float = 0.08) -> dict:
    """Single entry point. Uses the trained model when genuinely available,
    otherwise the classical baseline - and says which, every time."""
    if prefer_neural and torch_available():
        ckpt = find_checkpoint()
        if ckpt is not None:
            return neural_extract(image, ckpt, threshold=threshold or 0.5)
    return classical_extract(image, pixel_size_m, road_width_m=road_width_m,
                             threshold=threshold, road_fraction_prior=road_fraction_prior)


def capability_report() -> dict:
    """What this deployment can actually do right now. Surfaced by /api/health
    so the UI never advertises a capability the server lacks."""
    ckpt = find_checkpoint()
    has_torch = torch_available()

    def has(m):
        try:
            __import__(m)
            return True
        except ImportError:
            return False

    active = (ExtractionMode.TRAINED_MODEL
              if (has_torch and ckpt) else ExtractionMode.CLASSICAL_BASELINE)

    # Provenance must be populated for BOTH modes. Returning None for the
    # trained case left the UI badge empty exactly when a real model was
    # active - the one case where the label matters most.
    prov = PROVENANCE[active].dict()
    ckpt_meta: dict = {}
    if active is ExtractionMode.TRAINED_MODEL:
        prov["checkpoint"] = str(ckpt)
        try:
            import torch as _t

            meta = _t.load(ckpt, map_location="cpu", weights_only=False)
            prov["architecture"] = meta.get("args", {}).get("model")
            ckpt_meta = {
                "architecture": meta.get("args", {}).get("model"),
                "epoch": meta.get("epoch"),
                "selection_metric": meta.get("best_metric"),
                "selection_value": meta.get("best_value"),
                # Validation metrics recorded at training time. Labelled as such
                # so they are never read as test-set performance.
                "validation_metrics_at_training_time": meta.get("metrics"),
                "inference": meta.get("inference"),
            }
            prov["label"] = (f"TRAINED MODEL - {ckpt_meta['architecture']}"
                             if ckpt_meta.get("architecture") else prov["label"])
        except Exception as e:  # unreadable checkpoint must not crash health
            ckpt_meta = {"error": f"checkpoint present but unreadable: {e}"}

    return {
        "extraction_mode": active.value,
        "provenance": prov,
        "checkpoint_metadata": ckpt_meta or None,
        "torch_available": has_torch,
        "checkpoint_found": str(ckpt) if ckpt else None,
        "geotiff_support": has("rasterio"),
        "osm_ground_truth": has("osmnx") and has("geopandas"),
        "notes": (
            "No trained checkpoint present: road extraction will use the classical "
            "ridge-filter baseline and will be labelled as such throughout the UI."
            if active is ExtractionMode.CLASSICAL_BASELINE else
            "Trained checkpoint loaded."
        ),
    }
