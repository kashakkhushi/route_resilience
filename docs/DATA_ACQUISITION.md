# Data Acquisition

Everything here is optional for development — `make demo` and the full test suite
run on synthetic data. Follow this only when moving to real imagery.

---

## Priority order

Do **not** start with satellite imagery. Start with DeepGlobe: it is labelled,
tiled, RGB, and downloads in minutes. Get a baseline training end to end on it,
*then* introduce the harder real-sensor data. Teams that begin with raw Sentinel-2
typically lose their first eight hours to CRS bugs and band scaling before
training a single epoch.

| Order | Dataset | Why |
|---|---|---|
| 1 | DeepGlobe Road Extraction | labelled, RGB, 50 cm, ready to train |
| 2 | SpaceNet Roads | georeferenced + APLS baseline scores to compare against |
| 3 | Sentinel-2 | free, global, 10 m — the realistic operational sensor |
| 4 | LISS-IV / Cartosat-3 | ISRO indigenous, 5.8 m / 0.25 m |

---

## 1. DeepGlobe (start here)

Kaggle: `balraj98/deepglobe-road-extraction-dataset`

```
data/external/deepglobe/
  train/  *_sat.jpg  *_mask.png
```

1024×1024 RGB tiles at ~50 cm. Masks are already binary. Convert to the project
layout (`images/`, `masks/`) with `scripts/prepare_dataset.py`.

## 2. SpaceNet Roads

AWS open data, requester-pays:

```bash
aws s3 ls s3://spacenet-dataset/spacenet/SN3_roads/ --request-payer requester
```

Labels are GeoJSON centrelines, not masks — buffer them the same way
`osm_groundtruth.rasterize_roads` does. Worth the extra step because SpaceNet
publishes APLS leaderboard scores, giving you an external reference point.

## 3. Sentinel-2 (Copernicus Data Space)

Register at `dataspace.copernicus.eu`, then use the OData/OpenSearch API or
`sentinelsat`. **Request L2A, not L1C** — L2A is atmospherically corrected and
skipping that costs you a preprocessing stage.

Bands to fetch: B2, B3, B4 (10 m RGB) and B8 (10 m NIR).

Three things that bite:

- **DN range is 0–10000, not 0–1.** Divide by 10000 (`geo_io.to_reflectance`).
  Skipping this fails silently — the model just never converges.
- Bands are separate JP2 files. Stack them into one GeoTIFF before tiling.
- At 10 m GSD, a 7 m residential road is under one pixel. Only `tertiary` and
  above are learnable; `osm_groundtruth` drops the rest and tells you what it
  dropped. Do not fight this — it is physics, not a bug.

## 4. Resourcesat LISS-IV / Cartosat-3 (ISRO Bhoonidhi)

`bhoonidhi.nrsc.gov.in` — free registration, Indian users. Cartosat-3 is
restricted and per the brief is provided to participants during the event.

- **LISS-IV has no blue band** (G, R, NIR). `geo_io.select_rgb` synthesizes a
  plausible blue so ImageNet-pretrained stems still work, but say so in your
  results table — it is a genuine domain shift, not a cosmetic detail.
- **Cartosat-3 at 0.25 m** changes every spatial constant. A 512 px tile covers
  128 m, which is too little context for a transformer to reason across an
  occlusion. Use `configs/data/cartosat3.yaml` (1024 px tiles) and remember
  healing gaps are set in metres, not pixels.

---

## Ground truth from OSM (zero manual labelling)

```bash
pip install -r requirements-geo.txt
python scripts/prepare_groundtruth.py --raster data/raw/sentinel2 --sensor sentinel2
```

For each GeoTIFF this queries OSM for the matching bbox, buffers centrelines by
road class, and burns a pixel-aligned mask.

**Verify before training.** The script prints a road-pixel fraction per scene;
an urban tile should land around 2–15%. Below 0.5% almost always means a CRS
mismatch, and the warning fires. Open one mask over its image in QGIS and look
at it — five minutes here saves hours of debugging a "model problem" that is
really a label problem.

Known limitation: OSM is current, your image has an acquisition date. New roads
appear as false negatives, demolished ones as false positives. Record the
acquisition date, and if precision drops sharply in one tile, suspect the data
before the model.

---

## Suggested Indian study areas

| City | Why |
|---|---|
| Bengaluru | named in the brief; dense canopy on arterials — ideal occlusion case |
| Chennai | recurrent flooding; real scenarios for the simulation module |
| Guwahati | Brahmaputra crossings create genuine gatekeeper nodes |
| Dehradun | forested hill terrain — tests generalisation beyond flat urban |

Pick **two contrasting** areas, not five similar ones. The "Generalisation"
row in the rubric asks for success across dense urban, forested suburban and
rural — two well-chosen cities demonstrate that; five similar ones do not.

---

## Layout after preparation

```
data/
  raw/sentinel2/bengaluru_2024.tif
  interim/masks/bengaluru_2024_roadmask.tif
  processed/
    train/images/*.png   masks/*.png
    val/images/*.png     masks/*.png
    test/images/*.png    masks/*.png
  osm/osm_bengaluru.gpkg          # cached; makes runs reproducible
```

`data/` is gitignored except `.gitkeep` and manifests. Never commit imagery —
one Sentinel-2 tile is ~1 GB and will permanently bloat the repo history.
