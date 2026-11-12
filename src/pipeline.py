"""
End-to-end orchestration: satellite scene -> decision support.

Heavy dependencies (torch, rasterio) are imported inside methods, never at module
level. That is deliberate: it lets the graph/analytics half of the system run and
be tested on a machine with neither CUDA nor GDAL installed.

    from src.pipeline import RouteResiliencePipeline, PipelineConfig

    p = RouteResiliencePipeline(checkpoint="checkpoints/best.pt")
    r = p.run_from_geotiff("data/raw/sentinel2/bengaluru.tif", sensor="sentinel2")
    p.save(r, "outputs/bengaluru")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from .analytics.criticality import (
    criticality_scores,
    network_statistics,
    rank_critical_edges,
    rank_critical_nodes,
)
from .analytics.resilience import full_resilience_report
from .analytics.simulation import Scenario, run_scenario_suite
from .graph.healing import HealingConfig, heal_graph, largest_component
from .graph.skeletonize import build_raw_graph

# Healing gaps are expressed in METRES and converted to pixels per sensor.
# A fixed pixel threshold is meaningless across a 40x GSD range: 60 px is 600 m
# on Sentinel-2 and 15 m on Cartosat-3.
DEFAULT_MAX_GAP_M = 400.0
DEFAULT_MIN_SPUR_M = 60.0


@dataclass
class PipelineConfig:
    sensor: str = "sentinel2"
    pixel_size_m: float = 10.0
    threshold: float = 0.5
    max_gap_m: float = DEFAULT_MAX_GAP_M
    min_spur_m: float = DEFAULT_MIN_SPUR_M
    healing: HealingConfig = field(default_factory=HealingConfig)
    top_k: int = 20
    betweenness_pivots: int | None = None   # None = exact Brandes
    od_pairs: int = 300
    ablation_nodes: int = 12

    def scaled_healing(self) -> HealingConfig:
        """Convert metre thresholds to pixels for this scene's GSD."""
        cfg = HealingConfig(**vars(self.healing))
        cfg.max_gap_px = self.max_gap_m / max(self.pixel_size_m, 1e-6)
        cfg.max_cost = cfg.max_gap_px * 2.5
        cfg.loop_max_cost = cfg.max_gap_px * 1.4
        return cfg

    @property
    def min_spur_px(self) -> float:
        return self.min_spur_m / max(self.pixel_size_m, 1e-6)


class RouteResiliencePipeline:
    def __init__(self, checkpoint: str | None = None,
                 config: PipelineConfig | None = None, device: str = "cuda"):
        self.cfg = config or PipelineConfig()
        self.device = device
        self.checkpoint = checkpoint
        self._model = None
        self._geo = None  # (transform, crs) when the source was a GeoTIFF

    # ---------------- stage 1-3: image -> probability ----------------
    def _load_model(self):
        if self._model is None:
            import torch

            from .models.baselines import build_model

            ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            name = ckpt.get("args", {}).get("model", "segformer_b3")
            model = build_model(name, in_channels=ckpt.get("args", {}).get("in_channels", 3))
            model.load_state_dict(ckpt["model"])
            self._model = model.to(self.device).eval()
        return self._model

    def segment(self, image: np.ndarray) -> dict:
        from .models.inference import evidence_map_from_prediction, predict_scene

        if self.checkpoint is None:
            raise ValueError("no checkpoint supplied; use run_from_mask() instead")
        pred = predict_scene(self._load_model(), image, device=self.device,
                             threshold=self.cfg.threshold)
        pred["evidence"] = evidence_map_from_prediction(pred)
        return pred

    # ---------------- stage 4-5: mask -> healed graph ----------------
    def build_graph(self, mask: np.ndarray, evidence: np.ndarray | None = None):
        raw = build_raw_graph(mask, self.cfg.pixel_size_m, self.cfg.min_spur_px)
        healed, report = heal_graph(raw, self.cfg.scaled_healing(), evidence)
        return healed, raw, report

    # ---------------- stage 6-8: analysis ----------------
    def analyze(self, G: nx.Graph, scenarios: list[Scenario] | None = None) -> dict:
        Gc = largest_component(G)
        k = self.cfg.betweenness_pivots
        crit = criticality_scores(Gc, k=k)
        out = {
            "statistics": network_statistics(G),
            "criticality": {
                "scores": {int(n): round(s, 5) for n, s in crit["scores"].items()},
                "gatekeepers": sorted(int(n) for n in crit["components"]["gatekeeper"]),
                "top_nodes": rank_critical_nodes(Gc, self.cfg.top_k, k=k),
                "top_edges": rank_critical_edges(Gc, self.cfg.top_k, k=k),
                "meta": crit["meta"],
            },
        }
        if scenarios:
            out["scenarios"] = run_scenario_suite(Gc, scenarios, self.cfg.od_pairs)
            out["resilience"] = full_resilience_report(
                Gc, scenarios, self.cfg.od_pairs, self.cfg.ablation_nodes
            )
        return out

    # ---------------- entry points ----------------
    def run_from_mask(self, mask: np.ndarray, evidence=None, scenarios=None) -> dict:
        G, raw, healing = self.build_graph(mask, evidence)
        result = self.analyze(G, scenarios)
        result["healing"] = healing
        result["_graph"], result["_raw_graph"] = G, raw
        return result

    def run_from_image(self, image: np.ndarray, scenarios=None) -> dict:
        pred = self.segment(image)
        r = self.run_from_mask(pred["mask"], pred.get("evidence"), scenarios)
        r["_prediction"] = pred
        return r

    def run_from_geotiff(self, path: str | Path, sensor: str | None = None,
                         scenarios=None) -> dict:
        """Full georeferenced run: GeoTIFF in, GeoJSON-ready graph out."""
        from .data.geo_io import SENSOR_BANDS, read_raster, select_rgb, to_reflectance
        from .graph.georef import georeference_graph

        sensor = sensor or self.cfg.sensor
        raster = read_raster(path)
        self.cfg.pixel_size_m = raster["gsd_m"]
        self._geo = (raster["transform"], raster["crs"])

        img = select_rgb(to_reflectance(raster["array"], sensor), sensor)
        result = self.run_from_image(img, scenarios)
        result["_graph"] = georeference_graph(result["_graph"], *self._geo)
        result["source"] = {"raster": str(path), "sensor": sensor,
                            "gsd_m": raster["gsd_m"], "crs": str(raster["crs"])}
        return result

    # ---------------- persistence ----------------
    def save(self, result: dict, outdir: str | Path) -> Path:
        """Write GraphML + GeoJSON + JSON report."""
        from .graph.georef import graph_to_geojson, save_graphml

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        G = result.get("_graph")

        if G is not None:
            save_graphml(G, outdir / "network.graphml")
            if G.graph.get("georeferenced"):
                graph_to_geojson(
                    G, outdir / "network.geojson",
                    scores=result.get("criticality", {}).get("scores"),
                    gatekeepers=result.get("criticality", {}).get("gatekeepers"),
                )

        payload = {k: v for k, v in result.items() if not k.startswith("_")}
        (outdir / "analysis.json").write_text(json.dumps(payload, indent=2, default=str))
        return outdir
