"""
Deployment regression tests.

These pin two bugs found during the production-readiness audit. Both were
invisible in a fully-provisioned dev environment and only bite on the target
deployment: a laptop running a Kaggle-trained U-Net checkpoint.

    python backend/tests/test_deployment.py

1. A U-Net checkpoint must be servable without `transformers` installed.
   `inference.py` used to import `predict_logits` from `segformer.py`, which
   imports HuggingFace at module level, so serving a 24 M-parameter U-Net
   dragged in the entire transformers stack or crashed with ImportError.

2. Loading a checkpoint must not require network access. The model was rebuilt
   with `encoder_weights="imagenet"`, downloading ~85 MB of pretrained encoder
   weights that were then immediately overwritten by the checkpoint. Offline or
   behind a firewall, inference died with an HTTP error despite every weight it
   needed already being on disk.
"""
from __future__ import annotations

import builtins
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

PASS: list[str] = []
FAIL: list[tuple[str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    (PASS if cond else FAIL).append(name if cond else (name, detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    return cond


class block_imports:
    """Context manager that makes chosen top-level packages un-importable.

    Simulates the deployment machine rather than trusting that the dev machine
    resembles it.
    """

    def __init__(self, *names: str):
        self.names = names
        self._real = builtins.__import__
        self._purged: dict[str, object] = {}

    def __enter__(self):
        for mod in list(sys.modules):
            if any(mod == n or mod.startswith(n + ".") for n in self.names):
                self._purged[mod] = sys.modules.pop(mod)

        def guard(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split(".")[0]
            if root in self.names:
                raise ModuleNotFoundError(f"No module named {root!r} (blocked by test)")
            return self._real(name, globals, locals, fromlist, level)

        builtins.__import__ = guard
        return self

    def __exit__(self, *exc):
        builtins.__import__ = self._real
        sys.modules.update(self._purged)
        return False


class block_network:
    """Make any outbound HTTP attempt fail loudly instead of silently working."""

    def __enter__(self):
        import socket

        self._real = socket.socket.connect

        def deny(self_sock, addr):
            raise OSError(f"network blocked by test (attempted {addr})")

        socket.socket.connect = deny
        return self

    def __exit__(self, *exc):
        import socket

        socket.socket.connect = self._real
        return False


def make_checkpoint(tmp: Path, arch: str = "unet_r34") -> Path:
    import torch

    from src.models.baselines import build_model

    m = build_model(arch, 3, encoder_weights=None).eval()
    ck = tmp / "best.pt"
    torch.save(
        {
            "model": m.state_dict(),
            "args": {"model": arch, "in_channels": 3},
            "metrics": {"cldice": 0.61, "iou": 0.55},
        },
        ck,
    )
    return ck


def main() -> int:
    tmp = Path(tempfile.mkdtemp())

    print("\n[1] Source-level decoupling")
    inf_src = (ROOT / "src/models/inference.py").read_text()
    seg_src = (ROOT / "src/models/segformer.py").read_text()
    check(
        "inference.py does not import from segformer.py",
        "from .segformer import" not in inf_src,
        "predict_logits now lives in inference.py",
    )
    check(
        "predict_logits defined in inference.py",
        "def predict_logits" in inf_src,
    )
    check(
        "segformer.py no longer owns predict_logits",
        "def predict_logits" not in seg_src,
    )
    check(
        "baselines.py imports segformer lazily",
        'if n.startswith("segformer")' in (ROOT / "src/models/baselines.py").read_text(),
        "only imported when a segformer variant is requested",
    )

    print("\n[2] U-Net inference with transformers uninstalled")
    ck = make_checkpoint(tmp)
    with block_imports("transformers"):
        try:
            import torch  # noqa: F401

            from src.models.baselines import build_model
            from src.models.inference import predict_scene

            m = build_model("unet_r34", 3, encoder_weights=None).eval()
            img = np.random.default_rng(0).random((192, 192, 3)).astype(np.float32)
            out = predict_scene(m, img, tile_size=128, overlap=32,
                                device="cpu", tta=False)
            ok = out["mask"].shape == (192, 192) and out["prob"].min() >= 0
            check("predict_scene runs without transformers", ok,
                  f"mask {out['mask'].shape}, prob in [{out['prob'].min():.2f}, "
                  f"{out['prob'].max():.2f}]")
        except ModuleNotFoundError as e:
            check("predict_scene runs without transformers", False, str(e))

    with block_imports("transformers"):
        try:
            from src.models.baselines import build_model

            build_model("segformer_b0", 3)
            check("segformer still fails loudly without transformers", False,
                  "expected ModuleNotFoundError")
        except ModuleNotFoundError:
            check("segformer still fails loudly without transformers", True,
                  "lazy import raises only when actually requested")

    print("\n[3] Offline checkpoint loading")
    src = (ROOT / "backend/app/services/model_registry.py").read_text()
    check("neural_extract builds with encoder_weights=None",
          "encoder_weights=None" in src)
    check("architecture mismatch is detected, not silently ignored",
          "strict=False" in src and "does not match architecture" in src)

    with block_network():
        try:
            import torch

            from src.models.baselines import build_model

            ckpt = torch.load(ck, map_location="cpu", weights_only=False)
            m = build_model(ckpt["args"]["model"], ckpt["args"]["in_channels"],
                            encoder_weights=None)
            res = m.load_state_dict(ckpt["model"], strict=True)
            clean = (not getattr(res, "missing_keys", [])
                     and not getattr(res, "unexpected_keys", []))
            check("checkpoint loads with no network access", clean,
                  "no encoder download attempted")
        except OSError as e:
            check("checkpoint loads with no network access", False, str(e)[:110])

    # This check deliberately triggers the old, broken path. Silence its retry
    # logging: the failure is the expected result, not an error to report.
    import logging as _lg
    for _n in ("huggingface_hub", "timm", "urllib3", "filelock"):
        _lg.getLogger(_n).setLevel(_lg.CRITICAL)
    _lg.disable(_lg.WARNING)
    with block_network():
        try:
            from src.models.baselines import build_model

            build_model("unet_r34", 3, encoder_weights="imagenet")
            check("the old code path would have needed the network", False,
                  "download unexpectedly succeeded; test cannot prove the fix")
        except Exception:
            check("the old code path would have needed the network", True,
                  "confirms encoder_weights='imagenet' requires a download")
    _lg.disable(_lg.NOTSET)

    print("\n[4] End-to-end trained-model extraction, offline")
    with block_network():
        try:
            from backend.app.services.model_registry import neural_extract

            img = np.random.default_rng(1).random((160, 160, 3)).astype(np.float32)
            out = neural_extract(img, ck, device="cpu", threshold=0.5)
            prov = out["provenance"]
            mode = getattr(prov, "mode", None) or prov.get("mode")
            mode = getattr(mode, "value", mode)
            check("neural_extract completes offline", "mask" in out and "prob" in out,
                  f"road fraction {float(out['mask'].mean()):.4f}")
            check("result is labelled TRAINED_MODEL", str(mode) == "TRAINED_MODEL",
                  str(mode))
            learned = getattr(prov, "is_learned", None)
            if learned is None and isinstance(prov, dict):
                learned = prov.get("is_learned")
            check("result is flagged as learned", bool(learned))
        except Exception as e:
            check("neural_extract completes offline", False,
                  f"{type(e).__name__}: {str(e)[:110]}")

    import shutil

    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for item in FAIL:
        name, detail = item if isinstance(item, tuple) else (item, "")
        print(f"  FAILED: {name}  {detail}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
