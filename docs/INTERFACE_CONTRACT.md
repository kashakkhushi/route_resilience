# Interface Contract

The brief's 30-hour plan depends on two sub-teams working **in parallel**:

- **Team A (Vision)** — Phase I: preprocessing, occlusion augmentation, baselines,
  SegFormer, training, inference.
- **Team B (Graph)** — Phase II–IV: skeletonization, healing, criticality,
  simulation, resilience, dashboard.

Team B cannot wait for a trained model, and Team A cannot wait for a graph stack.
This document is the seam between them. **Agree on it before writing code, then
neither team blocks the other.**

---

## The single hand-off point

Team A produces exactly one artefact. Team B consumes exactly that artefact.

```python
{
    "prob":       np.ndarray,  # (H, W) float32 in [0, 1]  - road probability
    "mask":       np.ndarray,  # (H, W) bool               - prob > threshold
    "occlusion":  np.ndarray,  # (H, W) float32 in [0, 1]  - optional, aux head
    "evidence":   np.ndarray,  # (H, W) float32 in [0, 1]  - optional, see below
}
```

Nothing else crosses the boundary. No model objects, no tensors, no CUDA.

### `evidence` — why it exists

Healing needs to know whether a gap is *road hidden under a tree* or *genuinely
empty ground*. `evidence = clip(prob + 0.6 * occlusion, 0, 1)` encodes exactly
that: "I see a tree, not a field" is positive evidence a road continues
underneath. It is the one place Team A's auxiliary head feeds Team B's algorithm,
and it is the most defensible novelty in the project.

If Team A's model has no occlusion head, pass `prob` as `evidence`. Healing
degrades gracefully — it just prices bridges on distance and angle alone.

---

## How Team B works before any model exists

`src/data/synthetic.py` generates a city with **known ground-truth topology**,
then damages it the way an occluded segmentation would:

```python
from src.data.synthetic import generate_city, fragment_mask

city = generate_city(size=768, seed=1)
frag = fragment_mask(city["mask"], n_gaps=70, max_gap=22)
# frag is a drop-in stand-in for prediction["mask"]
```

Because ground truth is known, Team B can *measure* healing quality on hour one
rather than eyeballing it:

```bash
python scripts/run_benchmark.py --scenes 5
```

This is not a mock that gets thrown away. It stays in the repo as the regression
test for the healing algorithm.

---

## How Team A works before any graph stack exists

Metrics are framework-free and depend only on numpy:

```python
from src.eval.metrics import evaluate_mask
evaluate_mask(pred_mask, gt_mask, occlusion_mask, buffer_px=4)
# -> iou, dice, cldice, relaxed_iou, recall_visible, recall_occluded, occlusion_gap
```

`recall_occluded` is available from the first training run, so Team A can tell
immediately whether occlusion augmentation is working — without waiting for APLS
or any graph code.

---

## Model interface (so architectures are swappable)

Every model returns the same dict, whatever the architecture:

```python
forward(x: Tensor[B, C, H, W]) -> {
    "road_logits":      Tensor[B, 1, H, W],   # raw logits, NOT sigmoid
    "occlusion_logits": Tensor[B, 1, H, W],   # optional
}
```

Register in `src/models/baselines.py::build_model` and `train.py`,
`inference.py` and the whole metrics suite work unchanged. This is what makes the
U-Net vs DeepLabV3+ vs SegFormer comparison table cheap to produce.

---

## Graph contract

```python
G.nodes[n]["pos"]     # (row, col) float pixel coordinates - ALWAYS this order
G.nodes[n]["kind"]    # "junction" | "endpoint"
G[u][v]["length_px"]  # float, polyline length in pixels
G[u][v]["length_m"]   # float, metres - what all shortest paths are weighted by
G[u][v]["kind"]       # "observed" | "healed"   <-- never drop this
G[u][v]["pts"]        # [(row, col), ...] pixel chain for drawing
```

Rules:

- Node IDs are plain Python `int`. Numpy integer types break GraphML and JSON
  export — there is a test for this.
- `kind="healed"` marks road the system *inferred*. The dashboard draws it in
  red so a human can audit it. Dropping the label makes hallucinated roads
  indistinguishable from observed ones.
- `pos` is `(row, col)` everywhere upstream of `graph/georef.py`. Matplotlib and
  GeoJSON want `(x, y)` — convert at the boundary, never internally.

---

## Suggested 30-hour split

| Hours | Team A (Vision) | Team B (Graph) |
|---|---|---|
| 0–2 | env + CUDA, download DeepGlobe | env (core only), run demo, read healing code |
| 2–6 | OSM ground-truth pipeline, tiling | tune healing on synthetic benchmark |
| 6–12 | U-Net baseline training | criticality + gatekeepers, ablation |
| 12–20 | SegFormer + clDice, occlusion ablation | disaster scenarios, resilience index |
| 20–26 | inference on real scene → hand off `prob`/`evidence` | dashboard, Leaflet map |
| 26–30 | **joint**: real-scene run, benchmark table, report figures | |

The hand-off at hour 20 is the only hard dependency in the plan. Everything
before it runs in parallel.

---

## Definition of done

- [ ] `make test` passes
- [ ] `make demo` runs end to end and writes figures
- [ ] Benchmark table has ≥2 models with `recall_occluded` reported
- [ ] Occlusion ablation run (same architecture ± synthetic occluders)
- [ ] One real GeoTIFF processed end to end, GeoJSON opens in QGIS
- [ ] APLS reported against OSM
- [ ] Dashboard runs and the disaster simulator responds to clicks
