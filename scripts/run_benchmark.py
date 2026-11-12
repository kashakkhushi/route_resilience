"""
Benchmark the graph/healing stack, and optionally trained checkpoints.

    # no torch needed - validates healing against known ground truth
    python scripts/run_benchmark.py --out outputs/reports

    # with trained models
    python scripts/run_benchmark.py --checkpoints checkpoints/*.pt --data data/processed/test

The first form is what the analytics sub-team runs on hour one to tune healing
parameters, long before any model has finished training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.benchmark import format_table, save_benchmark, synthetic_benchmark  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/reports")
    ap.add_argument("--scenes", type=int, default=5)
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--data", default=None, help="test split, required with --checkpoints")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== graph / healing benchmark (synthetic ground truth) ===")
    res = synthetic_benchmark(args.scenes)
    print(json.dumps(res["mean"], indent=2))
    gain = res["loop_pass_apls_gain"]
    print(f"\nloop-restoration APLS gain: {gain:+.4f}")
    (out / "graph_benchmark.json").write_text(json.dumps(res, indent=2))

    if args.checkpoints:
        if not args.data:
            raise SystemExit("--data is required when scoring checkpoints")
        from src.data.dataset import RoadDataset
        from src.eval.benchmark import evaluate_checkpoint

        ds = RoadDataset(args.data, split="test")
        tiles = []
        for i in range(len(ds)):
            img, mask, occ = ds[i]
            tiles.append({
                "image": img.numpy().transpose(1, 2, 0),
                "mask": mask.numpy()[0],
                "occlusion": occ.numpy()[0],
            })

        rows = [evaluate_checkpoint(c, tiles, args.device) for c in args.checkpoints]
        save_benchmark(rows, out, "model_benchmark")
        print("\n=== model comparison ===")
        print(format_table(rows))

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
