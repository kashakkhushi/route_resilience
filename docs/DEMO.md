# Faculty Demonstration Script

Target: a reviewer understands the system in **two minutes**, and can verify
nothing is faked in **five**.

---

## Setup (before they arrive)

```bash
uvicorn backend.app.main:app --port 8000     # terminal 1
cd frontend && npm run dev                   # terminal 2
```

Open `http://localhost:5173`. Leave it on the landing page.

No internet required. No GPU required. No dataset required.

---

## The two-minute run

Click **Launch Urban Digital Twin** → Overview → **Run** on *Guided demonstration*.

Seven steps execute against the real backend. Narrate as they run:

| Step | What to say |
|---|---|
| 1. Load study area | "A synthetic city with a river crossed at only two points. Few crossings create genuine single points of failure — a plain grid would make the resilience analysis look trivially good." |
| 2. Inspect occlusion | "Canopy, shadow, cloud and vehicles are painted on the **image only**. The label is untouched. That asymmetry is what forces a model to infer road continuity instead of matching appearance." |
| 3. Extract roads | "The mask fragments wherever an occluder sat on a road. That fragmentation is the problem this project exists to fix." |
| 4. Heal topology | "Union–find plus a minimum spanning tree. It adds exactly (components − 1) bridges — provably the minimum. A second pass repairs breaks inside loops that a spanning tree can never reach." |
| 5. Criticality | "Betweenness finds busy corridors. Articulation points find gatekeepers, where losing one intersection disconnects a district. Busy and critical are different things." |
| 6. Targeted failure | "Five intersections removed, re-ranking after each — the network reorganises, so batch removal understates the damage." |
| 7. Random comparison | "Same number of failures, chosen at random. Compare the two numbers." |

**The closing line:** targeted failures sever a far larger share of trips than
random ones. That gap is the quantified argument for hardening specific
intersections rather than upgrading the network uniformly.

---

## If they ask "is any of this faked?"

Show them, in this order:

1. **The provenance banner.** Every mask says what produced it. On the demo it
   reads `DEMO DATA`. Switch Extraction mode to *Classical baseline* and rerun —
   the banner changes to `CLASSICAL BASELINE — NOT A TRAINED MODEL` and IoU drops
   from ~0.80 to ~0.16. Two honest numbers for two different methods.

2. **The Evaluation tab with an uploaded image.** No ground truth exists, so every
   metric renders as an em dash with instructions for generating OSM reference
   masks. It does not invent numbers.

3. **The test suites.**
   ```bash
   python tests/run_tests_nopytest.py               # scientific core
   python backend/tests/test_api_e2e.py             # API + sanity
   python backend/tests/test_contract.py            # frontend/backend contract
   python backend/tests/test_resilience_metric.py   # metric is bounded + monotone
   python backend/tests/test_deployment.py          # offline / no-transformers serving
   python scripts/audit_ml_pipeline.py --stage data
   python scripts/audit_ml_pipeline.py --stage model
   python scripts/audit_ml_pipeline.py --stage backend
   ```

4. **`/docs`.** Every endpoint, schema and validation rule, generated from the code.

---

## Likely questions

**"Did you train a model?"**
Be direct about the current state. The inference pipeline, losses, baselines and
training script are complete; whether a checkpoint exists is shown live on
`/api/health` and in the top bar. With no checkpoint the system runs the classical
baseline and says so everywhere — it never implies a network produced the output.

**"Why clDice and not IoU?"**
A road with a three-pixel break keeps nearly all its IoU but becomes two graph
components. Demonstrable: a thinned road loses more pixels (IoU 0.50) yet stays
connected (clDice 1.00), while a broken road loses fewer pixels (IoU 0.68) and
breaks (clDice 0.80). The metrics rank them in opposite orders. There is a unit
test pinning exactly this.

**"How is the Resilience Index defined?"**
The primary metric is **NER — Network Efficiency Resilience**, shown on screen
with its formula:

```
NER = E(G') / E(G),   E(H) = mean over a fixed OD sample of 1/d_H(s,t),  1/inf := 0
```

It is the fraction of baseline routing efficiency retained. `1.00` means nothing
was lost; `0.34` means 34% retained and 66% lost.

**"How do you know it can't go above 1?"**
It is a theorem about the construction, not a clamp. Removing infrastructure can
never create a shorter path, so every term `1/d` can only fall, so
`E(G') <= E(G)`. Three regression tests pin this: an empty scenario scores
exactly `1.0`, the score is non-increasing under nested removals
(`1.000 → 0.949 → 0.911 → 0.841 → 0.727 → 0.622 → 0.486 → 0.360 → 0.283`), and a
catastrophic failure that severs 92% of trips cannot raise it.

**"Why does the brief's own formula read 0.00 then?"**
It is shown under *Secondary readings*, labelled degenerate. `ASPL_baseline /
ASPL_perturbed` divides by a mean path length that becomes infinite the instant
one OD pair disconnects, so it reads 0.00 for nearly every realistic scenario and
cannot rank them. We report it for compliance and rank with NER.

**"There's a figure above 1 in the secondary panel — did the network improve?"**
No, and that is why it is not the index. The connected-only ratio averages over
*surviving* pairs, and the survivors are the short trips, so it rises as the
network collapses — 5.33 in the shattered-chain test where 92% of trips became
impossible. It carries `diagnostic_is_resilience_score: false` in the API and
lives in a collapsed panel in the UI. It is a diagnostic of which trips survived,
never a resilience score.

**"How would this work on real satellite data?"**
Upload a georeferenced GeoTIFF under Imagery. Ground truth comes from
`scripts/prepare_groundtruth.py`, which rasterises OpenStreetMap for the scene
footprint — zero manual annotation. Sensor-specific handling for Sentinel-2,
LISS-IV and Cartosat-3 is in `src/data/geo_io.py`.

**"What are the limitations?"**
Healing infers bridges and can hallucinate; betweenness assumes shortest-path
routing with uniform demand; travel time uses one assumed speed; flood scenarios
are geometric, not hydrological. All seven are listed in the README, and the UI
warns when more than 20% of network length was inferred.

---

## What NOT to claim

- Not "the model sees through clouds" → it infers likely continuity from
  contextual spatial evidence.
- Not "predicts floods" → it simulates the consequences of road closures.
- Not "100% accurate" → report the measured numbers with their provenance.
