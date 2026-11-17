"""
Ground-truth upload contract.

POST /api/imagery/{session_id}/ground-truth attaches a reference road mask to an
uploaded scene so the dashboard can report real IoU / Dice for it. The rules:

    G1  Before ground truth exists, metrics are reported as unavailable.
    G2  A mask with the wrong size is rejected (never resized onto the image).
    G3  An empty mask is rejected.
    G4  A correct mask returns metrics at once if roads were already extracted,
        and labels where the ground truth came from.
    G5  The next extraction uses the attached mask too.
    G6  Demo scenes refuse an upload: they already carry exact ground truth.

    python backend/tests/test_ground_truth_upload.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name if ok else (name, detail))
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"   [{detail}]"))


def png(a: np.ndarray) -> bytes:
    return cv2.imencode(".png", a)[1].tobytes()


def main() -> int:
    from fastapi.testclient import TestClient

    import backend.app.services.model_registry as mr
    mr.find_checkpoint = lambda *a, **k: None      # classical path: fast, no torch model
    from backend.app.main import app
    from src.data.synthetic import generate_city

    c = TestClient(app)
    city = generate_city(size=384, seed=7)
    img = (city["image"][..., ::-1] * 255).astype(np.uint8)
    mask = (city["mask"] * 255).astype(np.uint8)

    sid = c.post("/api/imagery/upload",
                 files={"file": ("t_sat.png", png(img), "image/png")}).json()["session_id"]

    print("\n[G1] no ground truth yet")
    m = c.post("/api/extract-roads", json={"session_id": sid, "mode": "classical"}).json()["metrics"]
    check("metrics unavailable before upload", m["available"] is False, m)

    print("\n[G2/G3] bad masks are rejected")
    r = c.post(f"/api/imagery/{sid}/ground-truth",
               files={"file": ("m.png", png(mask[:300]), "image/png")})
    check("wrong size -> 422", r.status_code == 422, r.status_code)
    r = c.post(f"/api/imagery/{sid}/ground-truth",
               files={"file": ("m.png", png(np.zeros_like(mask)), "image/png")})
    check("empty mask -> 422", r.status_code == 422, r.status_code)

    print("\n[G4] correct mask gives metrics immediately")
    r = c.post(f"/api/imagery/{sid}/ground-truth",
               files={"file": ("t_mask.png", png(mask), "image/png")})
    j = r.json()
    check("status 200", r.status_code == 200, r.status_code)
    check("ground_truth layer returned", "ground_truth" in j.get("layers", {}))
    check("metrics returned", bool(j.get("metrics")) and j["metrics"]["available"], j.get("metrics"))
    check("IoU in [0,1]", 0.0 <= j["metrics"]["iou"] <= 1.0, j["metrics"].get("iou"))
    check("source names the file", "t_mask.png" in j["source"], j["source"])

    print("\n[G5] later extractions use it")
    m2 = c.post("/api/extract-roads", json={"session_id": sid, "mode": "classical"}).json()["metrics"]
    check("metrics available after re-extract", m2["available"] is True)
    check("same IoU as returned at upload", abs(m2["iou"] - j["metrics"]["iou"]) < 1e-9,
          (m2["iou"], j["metrics"]["iou"]))

    print("\n[G6] demo scenes refuse an upload")
    demo = c.post("/api/imagery/demo", json={"city": "grid_river", "size": 384, "seed": 1,
                                             "occlusion": True}).json()["session_id"]
    r = c.post(f"/api/imagery/{demo}/ground-truth",
               files={"file": ("m.png", png(mask), "image/png")})
    check("demo -> 422", r.status_code == 422, r.status_code)

    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for item in FAIL:
        name, detail = item if isinstance(item, tuple) else (item, "")
        print(f"  FAILED: {name}  {detail}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
