# Backend — Route Resilience AI

FastAPI service wrapping the tested scientific core in `src/`. The core is not
duplicated or rewritten here; `backend/` is a transport layer over it.

## Run

```bash
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000    # from the repo ROOT
```

OpenAPI docs: <http://localhost:8000/docs>

## Validate

```bash
python backend/tests/test_api_e2e.py     # 41 checks, full pipeline
python tests/run_tests_nopytest.py       # 48 checks, scientific core
```

## The provenance contract

Every response derived from a road mask carries `provenance`:

| mode | meaning |
|---|---|
| `TRAINED_MODEL` | a real `.pt` was found and loaded |
| `CLASSICAL_BASELINE` | no checkpoint — multi-scale Hessian ridge filter ran on the real pixels. **Not** a neural prediction, and never labelled as one |
| `DEMO_DATA` | deterministic synthetic city; no imagery was processed |

`GET /api/health` reports which mode is active, whether torch/rasterio/osmnx are
present, and whether a checkpoint was found. Clients must render `provenance.label`.

Segmentation metrics are returned **only when ground truth genuinely exists**.
Otherwise `metrics.available` is `false` with a `reason` — never a fabricated number.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | capability + active extraction mode |
| POST | `/api/imagery/demo` | load a deterministic synthetic city (known ground truth) |
| POST | `/api/imagery/upload` | GeoTIFF (needs rasterio) or PNG/JPEG |
| POST | `/api/extract-roads` | segmentation → probability + mask + metrics |
| POST | `/api/build-graph` | skeletonize → prune → union-find/MST heal → GeoJSON |
| POST | `/api/criticality` | betweenness, gatekeepers, per-node explanations |
| POST | `/api/simulate-failure` | node/edge/flood/targeted/random ablation |
| POST | `/api/route` | shortest path before vs after disruption |

## Design notes

**In-process session store, no Redis/Celery.** For a single-machine demonstration
a job queue adds two failure modes and buys nothing. `SessionStore` is behind an
interface so it can be swapped later.

**Centrality is cached per session** and keyed on graph size. Exact Brandes is
O(V·E); above 4,000 nodes the server samples pivots and sets `exact: false` in the
response rather than silently redefining the metric.

**Travel time uses a disclosed uniform speed assumption.** Road class is not
recoverable from segmentation alone, so distance is measured and time is derived;
the assumption is stated in every route response.
