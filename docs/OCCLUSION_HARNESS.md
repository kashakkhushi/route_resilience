# Occlusion Harness — correction record

## Status

| version | occluder | status |
|---|---|---|
| **v1** | vehicles sampled from road pixels only, one flat random colour per image | **superseded — design flaw** |
| **v2-corrected-occluder** | vehicles sampled from the road corridor, per-vehicle appearance | current |

Reports produced under v1 carry no `harness_version` field. Keep them; rename
the folder `reports_v1_superseded/` and cite this document.

## The flaw

`vehicles()` sampled placement centres directly from `np.nonzero(road_mask)` and
painted every vehicle in the image a single `rng.random(3)` colour.

Consequences:

1. **The occluder was perfectly correlated with the label.** Every occluder
   pixel sat on road (`occluder_on_road_ratio` ≈ 1.0). A uniformly-coloured
   dotted line running along the carriageway *marks* the road rather than
   hiding it.
2. **Only the easy road pixels were occluded.** Vehicles 2–5 px wide on a ~7 px
   road cover road *centres*. "Visible" road therefore contained the hard
   boundary pixels, so `occlusion_gap` was biased negative by construction.
3. **Tree and shadow blobs suppressed background false positives**, raising
   precision, while the vehicle cue raised recall. Both moved the same way, so
   IoU *rose* under occlusion — which is impossible for a genuine occluder.

Observed symptom: `partial` +0.032 IoU and `clutter` +0.019 IoU relative to the
unoccluded control.

## The correction

- Placement centres are drawn from a **road corridor** (road dilated by ≈1.6×
  road width), so on-road, edge-straddling and adjacent placements occur in
  natural proportion. Measured `occluder_on_road_ratio` falls from ≈1.0 to
  roughly 0.07–0.27.
- **Per-vehicle appearance**: colour from a plausible palette (black, dark grey,
  mid-tone, white/silver) with a per-channel tint, independent size
  (4–13 × 2.5–6 px), free orientation, a soft offset cast shadow, and mild
  texture. No colour is shared between vehicles, so no road-aligned chromatic
  cue exists.
- Determinism is unchanged: everything draws from the seeded `rng`, so a given
  `rng_seed` reproduces the scene exactly.
- The ground-truth mask is still never touched; only the image is modified.

## New diagnostics

The occluded region is defined from the synthetic occluder geometry (`alpha`),
never from the prediction — defining it from the prediction would be circular.

| field | meaning |
|---|---|
| `occlusion_recall` | recall restricted to occluded road pixels |
| `occlusion_precision` | precision restricted to the occluded region |
| `occlusion_iou` | IoU restricted to the occluded region |
| `occluded_road_fraction` | share of ground-truth road that is occluded |
| `occluded_background_fraction` | share of background that is occluded |
| `occluder_on_road_ratio` | label correlation of the occluder; ≈1.0 indicates the v1 flaw |

**`occlusion_precision` is the metric that matters.** High `occlusion_recall`
with low `occlusion_precision` means the model *floods* ambiguous regions with
road predictions — good recall, useless graph. Genuine contextual inference
scores acceptably on both.

## What this correction does and does not mean

It changes **measurement only**. Model weights, architecture, threshold, dataset
splits, ground truth and training procedure are untouched. Any difference
between a v1 and a v2 report reflects a corrected instrument, not an improved
model, and must not be reported as a gain.

## Known limitation: train/eval occluder mismatch

`src/data/occlusion.py` is shared by training augmentation and evaluation. A
checkpoint trained before this correction was trained against the v1 occluder
and is evaluated against v2. This is the scientifically correct choice — the
evaluation occluder should be realistic regardless of what augmentation produced
the model — but it must be disclosed. Retraining with v2 would remove the
mismatch and is the cleaner long-term position.

## Debug output

```bash
python scripts/evaluate_checkpoint.py --checkpoint <ckpt> --data <data> \
    --out outputs/reports --debug-viz 3
```

Writes, per condition, for the first N tiles: original, occluded, GT mask,
occlusion mask, and an overlay (green = GT road, red = occluder,
blue = prediction). Inspect these before trusting any occlusion number.
