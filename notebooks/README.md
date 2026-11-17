# Executed Kaggle notebooks

These are the notebooks exactly as they ran on Kaggle (GPU: 2 × Tesla T4), saved
**with their outputs**, so every number in the main README can be traced to a log.
Each one copies the project code from an attached Kaggle dataset and calls the
scripts in `scripts/`; nothing is computed inside the notebook that is not in the repo.

| Notebook | What it runs | Key result (held-out test) |
|---|---|---|
| `01_segformer_b3_training.ipynb` | SegFormer-B3 training on DeepGlobe (`kaggle_train.py`), then `evaluate_checkpoint.py` | IoU 0.6441, Dice 0.7732 (498 tiles) |
| `02_occlusion_test_final_segformer.ipynb` | Final fine-tuned SegFormer: validation, test and occlusion harness | IoU 0.6688, Dice 0.7917; canopy 0.6578, partial 0.7029, clutter 0.6952 |
| `03_apls_spacenet_las_vegas.ipynb` | `apls_spacenet.py`: 60 SpaceNet 3 tiles, zero-shot, APLS vs SpaceNet labels and OSM | APLS 0.42 (raw) / 0.40 (healed), relaxed IoU 0.71 (54 tiles) |
| `04_liss4_hyderabad.ipynb` | `prepare_liss4.py` on a ResourceSat-2A LISS-IV scene (Bhoonidhi, 14 Nov 2026) + OSM labels; zero-shot, fine-tune, evaluate | IoU 0.138 → 0.254 (54 tiles) |
| `05_sentinel2_hyderabad.ipynb` | `fetch_sentinel2.py` + `prepare_liss4.py --sensor sentinel2`; zero-shot, fine-tune, evaluate | IoU 0.225 → 0.460 (72 tiles) |
| `06_export_demo_tiles.ipynb` | Rebuilds the same splits and exports a few **test** tiles + masks (LISS-IV, Sentinel-2, optionally DeepGlobe) for the live dashboard demo. The tiles are already stretched and upsampled to 2.5 m/px, exactly what the fine-tuned models saw | (not run yet: no outputs) |

The U-Net, DeepLabV3+ and first fine-tuning runs were made from the same template
(`kaggle_train.py` with different `--arch` / `--init-from` arguments); their reports
are summarised in the main README.

To re-run one: create a Kaggle notebook, attach the project code as a dataset (and
DeepGlobe / the LISS-IV scene where the notebook says so), enable GPU and Internet,
import the `.ipynb` and choose *Save Version → Save & Run All*.
