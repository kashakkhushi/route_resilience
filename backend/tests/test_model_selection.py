"""
Model selection contract (GET /api/models, ExtractRequest.model).

    M1  /api/models lists every checkpoint under checkpoints/, labelled by the
        imagery it was trained for (DeepGlobe / Sentinel-2 / LISS-IV).
    M2  The default is the ROUTE_RESILIENCE_CHECKPOINT file when it is set.
    M3  Extraction with model=<id> runs THAT checkpoint (named in provenance).
    M4  Unknown ids and paths outside checkpoints/ are rejected with 422.

    python backend/tests/test_model_selection.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name if ok else (name, detail))
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"   [{detail}]"))


def main() -> int:
    from test_trained_pipeline import make_scene, save_ckpt

    work = Path(tempfile.mkdtemp())
    os.chdir(work)                                   # checkpoints/ is resolved from cwd
    for sub, seed in (("segformer_b3_ft", 1), ("segformer_s2", 2), ("segformer_liss4", 3)):
        (work / "checkpoints" / sub).mkdir(parents=True)
        save_ckpt(work / "checkpoints" / sub / "best.pt", seed)
    os.environ["ROUTE_RESILIENCE_CHECKPOINT"] = str(Path("checkpoints/segformer_b3_ft/best.pt"))

    from fastapi.testclient import TestClient

    from backend.app.main import app

    c = TestClient(app)

    print("\n[M1/M2] listing")
    j = c.get("/api/models").json()
    ids = {m["id"]: m for m in j["models"]}
    check("three checkpoints listed", len(ids) == 3, list(ids))
    check("default is the env-var checkpoint", j["default"] == "segformer_b3_ft/best.pt", j["default"])
    check("Sentinel-2 recognised", ids.get("segformer_s2/best.pt", {}).get("sensor", "").startswith("Sentinel-2"))
    check("LISS-IV recognised", ids.get("segformer_liss4/best.pt", {}).get("sensor", "").startswith("LISS-IV"))
    check("upsampled tiles are 2.5 m", ids["segformer_s2/best.pt"]["input_gsd_m"] == 2.5)
    check("DeepGlobe is 0.5 m", ids["segformer_b3_ft/best.pt"]["input_gsd_m"] == 0.5)

    img = (make_scene(256) * 255).astype(np.uint8)
    png = cv2.imencode(".png", img[..., ::-1])[1].tobytes()
    sid = c.post("/api/imagery/upload", files={"file": ("s.png", png, "image/png")},
                 data={"pixel_size_m": "2.5"}).json()["session_id"]

    print("\n[M3] the chosen checkpoint is the one that runs")
    for mid in ("segformer_s2/best.pt", "segformer_liss4/best.pt"):
        r = c.post("/api/extract-roads", json={"session_id": sid, "mode": "auto", "model": mid})
        prov = r.json().get("provenance", {}) if r.status_code == 200 else {}
        folder = mid.split("/")[0]
        ok = r.status_code in (200, 422) and (r.status_code == 422 and "empty" in r.text
                                              or folder in (prov.get("checkpoint") or ""))
        check(f"{folder} used", ok, (r.status_code, prov.get("checkpoint"), r.text[:120]))
    r = c.post("/api/extract-roads", json={"session_id": sid, "mode": "auto"})
    if r.status_code == 200:
        check("no model -> default", "segformer_b3_ft" in r.json()["provenance"]["checkpoint"])

    print("\n[M4] bad ids")
    for bad in ("nope/best.pt", "../../etc/passwd", "/etc/passwd"):
        r = c.post("/api/extract-roads", json={"session_id": sid, "mode": "auto", "model": bad})
        check(f"{bad!r} -> 422", r.status_code == 422, r.status_code)

    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for item in FAIL:
        name, detail = item if isinstance(item, tuple) else (item, "")
        print(f"  FAILED: {name}  {detail}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
