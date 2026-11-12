# Architecture

```
┌─ frontend/ ──────────────────────────────────────────────┐
│  React 18 + TS + Vite + Tailwind                         │
│  usePipeline (single reducer)  →  canvas renderer        │
└───────────────────────────── /api (proxied, no keys) ────┘
                                     │
┌─ backend/ ──────────────────────────────────────────────┐
│  FastAPI  routes → services → session store              │
│  Pydantic schemas enforce provenance on every response   │
└──────────────────────────── imports ────────────────────┘
                                     │
┌─ src/ ──────────────────────────────────────────────────┐
│  data/       preprocessing, occlusion, OSM GT, geo I/O   │
│  models/     baselines, SegFormer, losses, train, infer  │
│  graph/      skeletonize, healing, georef                │
│  analytics/  criticality, simulation, resilience         │
│  eval/       metrics, topology (APLS), benchmark         │
└──────────────────────────────────────────────────────────┘
```

## Layer rules

Layers are strictly ordered; a lower layer never imports from a higher one.

**`src/graph/` and `src/analytics/` are pure numpy/networkx.** Verified — the only
module-level torch imports in the repo are in `src/models/` and
`src/data/dataset.py`. Geospatial imports are function-level. This is what keeps
the scientific core testable in seconds without CUDA or GDAL.

**The backend contains no algorithms.** Routes choose parameters and shape
responses; every computation is delegated to `src/`, which is unit-tested
independently of the web layer.

## Session state

An in-process LRU dict with TTL eviction, not a database. A demo runs one backend
process and a handful of sessions, so that is the right amount of machinery.
Swapping in Redis touches one module.

Each session records `provenance` alongside its artefacts, so any response can
state whether the user is looking at a model prediction, a classical baseline, an
OSM reference, or demo data.

## Why no job queue

Exact betweenness is O(V·E). The obvious reaction is Celery + Redis. We did not,
because a faculty demonstration values reliability over distributed architecture,
and a queue adds two failure modes to buy latency we do not need at demo scale.

Instead the graph size decides the algorithm: exact Brandes below
`EXACT_BETWEENNESS_MAX_NODES`, k-pivot estimation above it — and the response
carries `exact: false` with the pivot count so the UI can label the number as an
estimate. An estimate is never silently substituted for an exact value.

## Coordinate conventions

- Everything upstream of `graph/georef.py` works in `(row, col)` pixel space.
- Georeferencing happens **once**, explicitly. CRS logic is not scattered.
- GeoJSON uses `(x, y)`; conversion happens at the boundary only.
- Lengths are computed as true polyline length in a projected CRS, never as
  pixel count × GSD, which is wrong away from the reference meridian and biases
  every centrality number weighted by length.

## Healing thresholds are in metres

A fixed pixel threshold is meaningless across a 40× GSD range: 60 px is 600 m on
Sentinel-2 and 15 m on Cartosat-3. The API takes `max_gap_m` and converts using
the scene GSD.

## Frontend state

One reducer owns the whole chain because the stages are genuinely dependent:
new imagery invalidates the mask, the graph, the criticality and any scenario.
Letting each view hold its own copy is how a dashboard ends up showing a
criticality map computed from a mask that no longer exists.

## Security

- No secrets in the bundle; the dev server and nginx both proxy `/api`.
- Upload validation: extension allow-list, size cap, safe temp handling.
- Container runs as a non-root user; uploads are untrusted input.
- CORS is an explicit allow-list from the environment.
- `.env` is gitignored; `.env.example` documents every variable.
