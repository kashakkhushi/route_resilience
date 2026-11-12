# Kaggle Training Guide

U-Net (ResNet34) on DeepGlobe, trained on Kaggle GPU, deployed on a local laptop.

The governing constraint: **ImageNet weights may initialise the encoder on
Kaggle, but the resulting checkpoint must run offline.** The saved `state_dict`
already contains those weights, so deployment rebuilds the model with
`encoder_weights=None` and never touches the network. `kaggle_train.py` proves
this before it exits by reloading the checkpoint with sockets disabled.

---

## 1. Notebook setup

Create a new Kaggle Notebook.

- **Accelerator**: GPU T4 x2 (or P100)
- **Internet**: **ON** during training (needed for pip and ImageNet weights).
  It is *not* needed afterwards.
- **Add Data** → search `deepglobe-road-extraction-dataset` → add
  `balraj98/deepglobe-road-extraction-dataset`.

It mounts read-only at `/kaggle/input/deepglobe-road-extraction-dataset/`.

## 2. Cell 1 — dependencies

```python
!pip -q install segmentation-models-pytorch
import torch, segmentation_models_pytorch as smp
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("smp", smp.__version__)
```

**That is the only package you need to install.** `segmentation-models-pytorch`
is not in the default Kaggle image and `unet_r34` cannot be built without it.
Everything else the training path imports — torch, cv2, numpy, scipy,
scikit-image, networkx — is preinstalled.

Do **not** install `albumentations`: the augmentation in this repo is
hand-written in `src/data/dataset.py` (rotations/flips) and
`src/data/occlusion.py` (occluders). `albumentations` appears only in a version
probe and is never used.

## 3. Cell 2 — project code

Either clone your repo, or upload it as a Kaggle Dataset and copy it in:

```python
!git clone https://github.com/<you>/route-resilience-ai.git /kaggle/working/rr
%cd /kaggle/working/rr
```

## 4. Cell 3 — prepare the dataset

```python
!python scripts/prepare_deepglobe.py \
    --src /kaggle/input/deepglobe-road-extraction-dataset \
    --out /kaggle/working/processed \
    --seed 42
```

Only DeepGlobe's `train/` split carries masks; `valid/` and `test/` are
unlabelled challenge splits. The script therefore cuts its own val and test sets
from `train/`, deterministically — the same seed always yields the same test set,
so two runs are comparable.

Expect roughly: train 4,978 / val 747 / test 498 tiles from 6,226 pairs.

## 5. Cell 4 — train

```python
!python scripts/kaggle_train.py \
    --data /kaggle/working/processed \
    --out /kaggle/working/checkpoints/unet_r34 \
    --arch unet_r34 \
    --encoder-weights imagenet \
    --epochs 40 \
    --batch-size 8 \
    --lr 3e-4 \
    --tile-size 512 \
    --seed 42 \
    --monitor cldice \
    --num-workers 2
```

Seven pre-flight checks run before the first gradient step and **abort on
failure**, because the expensive mistake is not a crash — it is a run that
trains happily on mispaired or non-binary data.

### The ablation (run this too)

Identical settings, occlusion augmentation off. This is the cleanest experiment
in the project: one variable, compared on `occlusion_recall`.

```python
!python scripts/kaggle_train.py \
    --data /kaggle/working/processed \
    --out /kaggle/working/checkpoints/unet_r34_noocc \
    --epochs 40 --batch-size 8 --seed 42 --no-occlusion
```

## 6. Cell 5 — evaluate

```python
!python scripts/evaluate_checkpoint.py \
    --checkpoint /kaggle/working/checkpoints/unet_r34/best.pt \
    --data /kaggle/working/processed \
    --out /kaggle/working/reports
```

Produces validation, held-out test, paired occlusion evaluation, and the
classical-baseline comparison in one report. It **refuses to run without a
checkpoint** rather than scoring the baseline and labelling it model
performance.

## 7. Cell 6 — package for download

```python
import shutil
shutil.make_archive("/kaggle/working/unet_r34_package", "zip",
                    "/kaggle/working/checkpoints/unet_r34")
```

Download from the notebook's **Output** tab.

---

## 8. Local deployment (Galaxy Book 3)

```bash
mkdir -p checkpoints/unet_r34
unzip ~/Downloads/unet_r34_package.zip -d checkpoints/unet_r34/

python -c "import torch; ck=torch.load('checkpoints/unet_r34/best.pt', map_location='cpu', weights_only=False); print(ck['inference'])"

uvicorn backend.app.main:app --reload --port 8000     # terminal 1
cd frontend && npm run dev                             # terminal 2
```

`/api/health` should report:

```json
{ "extraction_mode": "TRAINED_MODEL",
  "checkpoint_found": "checkpoints/unet_r34/best.pt",
  "provenance": { "label": "TRAINED MODEL - unet_r34", "is_learned": true } }
```

The top bar shows **TRAINED MODEL - unet_r34** in accent blue, not the amber
`CLASSICAL BASELINE` or `DEMO DATA` badge.

CPU inference on a 512 px tile takes a few seconds; that is expected and fine
for interactive use.

---

## Checkpoint contents

```python
ck["model"]       # state_dict, encoder weights included
ck["args"]        # model, in_channels, tile_size, threshold, seed
ck["epoch"]       # selected epoch
ck["best_metric"] # "cldice"
ck["best_value"]
ck["metrics"]     # validation metrics at that epoch
ck["config"]      # full training configuration
ck["inference"]   # architecture, encoder, in_channels, image_size,
                  # normalization, threshold, dataset, seed, git_commit
```

Alongside it: `config.json`, `history.json`, `metadata.json`.

---

## Reading the results honestly

**Watch the val−test gap.** Above ~0.05 IoU suggests the selection split leaked
into model choice. The evaluation script prints it.

**Watch precision against recall.** Recall near 1.0 with low precision means the
model is over-predicting — painting most of the tile as road. It will look fine
on recall and produce a useless graph. Raise `--threshold` or check
`implied_pos_weight` in the pre-flight output.

**clDice is the selection metric, not IoU.** A mask with high IoU and low clDice
is a dotted line: pixel-accurate and unusable for routing.

**Occlusion gap is the project's actual claim.** `recall_visible −
occlusion_recall` measures how much the model infers versus merely observes.
With occlusion augmentation on it should be materially smaller than without —
that comparison is the ablation above.

If the model performs poorly, report it and diagnose. Do not raise the threshold
until the number looks better; that changes the question rather than the answer.

---

## Runtime expectations

| Setting | Effect |
|---|---|
| batch 8, 512 px, T4 | roughly 3–6 min/epoch on ~5k tiles |
| 40 epochs | expect several hours; Kaggle caps sessions at 12 h |
| out of memory | drop `--batch-size` to 4, then `--tile-size` to 384 |

Save intermediate checkpoints by keeping the notebook's output directory under
`/kaggle/working`, which persists into the Output tab.
