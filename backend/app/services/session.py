"""
Session store and pipeline orchestration.

A session holds one study area as it moves through the pipeline. State is kept
in-process: for a single-machine faculty demonstration that is the right call,
because Redis and Celery would add two failure modes and buy nothing. The store
is behind an interface so it can be swapped for Redis later without touching
route handlers.

Expensive derived values (centrality, baseline shortest paths) are cached on the
session, because exact betweenness is O(V*E) and recomputing it on every slider
move would make the UI unusable.
"""
from __future__ import annotations

import base64
import io
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from src.analytics.criticality import (
    criticality_scores,
    network_statistics,
    rank_critical_edges,
)
from src.analytics.simulation import _path_lengths, sample_od_pairs
from src.eval.topology import connectivity_ratio
from src.graph.healing import HealingConfig, heal_graph, largest_component
from src.graph.skeletonize import build_raw_graph

from .model_registry import ExtractionMode, ModelProvenance, PROVENANCE

MAX_SESSIONS = 24
SESSION_TTL_S = 60 * 60 * 3


@dataclass
class Session:
    id: str
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)

    # imagery
    image: np.ndarray | None = None
    pixel_size_m: float = 1.0
    transform: Any = None
    crs: Any = None
    source: str = "unknown"
    seed: int = 1                       # reproduces the demo scene deterministically
    meta: dict = field(default_factory=dict)

    # ground truth (only when genuinely known)
    gt_mask: np.ndarray | None = None
    occlusion: np.ndarray | None = None
    gt_graph: nx.Graph | None = None

    # extraction
    prob: np.ndarray | None = None
    mask: np.ndarray | None = None
    evidence: np.ndarray | None = None
    provenance: ModelProvenance | None = None
    threshold: float | None = None

    # graph
    raw_graph: nx.Graph | None = None
    graph: nx.Graph | None = None
    healing: dict | None = None

    # caches
    _crit: dict | None = None
    _crit_key: tuple | None = None
    _od: list | None = None
    _baseline: dict | None = None

    def touch(self):
        self.touched = time.time()

    # -- cached criticality ------------------------------------------------
    def criticality(self, pivots: int | None = None) -> dict:
        key = (self.graph.number_of_nodes() if self.graph else 0,
               self.graph.number_of_edges() if self.graph else 0, pivots)
        if self._crit is None or self._crit_key != key:
            self._crit = criticality_scores(self.graph, k=pivots)
            self._crit_key = key
        return self._crit

    # -- cached OD baseline ------------------------------------------------
    def baseline(self, n_pairs: int = 300) -> tuple[list, dict]:
        if self._od is None or len(self._od) != n_pairs:
            self._od = sample_od_pairs(self.graph, n_pairs, seed=42)
            self._baseline = _path_lengths(self.graph, self._od, "length_m")
        return self._od, self._baseline

    def invalidate(self):
        self._crit = self._crit_key = self._od = self._baseline = None


class SessionStore:
    def __init__(self):
        self._d: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        with self._lock:
            self._evict()
            s = Session(id=uuid.uuid4().hex[:16])
            self._d[s.id] = s
            return s

    def get(self, sid: str) -> Session | None:
        with self._lock:
            s = self._d.get(sid)
            if s:
                s.touch()
            return s

    def _evict(self):
        now = time.time()
        stale = [k for k, v in self._d.items() if now - v.touched > SESSION_TTL_S]
        for k in stale:
            self._d.pop(k, None)
        while len(self._d) >= MAX_SESSIONS:
            oldest = min(self._d.values(), key=lambda s: s.touched)
            self._d.pop(oldest.id, None)

    def stats(self) -> dict:
        return {"sessions": len(self._d), "max": MAX_SESSIONS}


STORE = SessionStore()


# ---------------------------------------------------------------- rendering
def to_png_data_uri(arr: np.ndarray, cmap: str | None = None,
                    max_dim: int = 1400) -> str:
    """Encode an array as a PNG data URI for the client.

    Downscaled to `max_dim` because a 10k x 10k raster would be a 300 MB payload
    that the browser cannot render anyway. Analysis always uses full resolution;
    only the display copy is reduced.
    """
    import cv2

    a = np.asarray(arr)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    elif a.dtype != np.uint8:
        a = a.astype(np.float32)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = np.zeros_like(a) if hi - lo < 1e-9 else (a - lo) / (hi - lo)
        a = (a * 255).astype(np.uint8)

    h, w = a.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        a = cv2.resize(a, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    if cmap == "magma" and a.ndim == 2:
        a = cv2.applyColorMap(a, cv2.COLORMAP_MAGMA)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    elif cmap == "roads" and a.ndim == 2:
        rgb = np.zeros((*a.shape, 4), np.uint8)
        rgb[..., 0] = 255
        rgb[..., 1] = 210
        rgb[..., 2] = 60
        rgb[..., 3] = a
        a = rgb

    if a.ndim == 3 and a.shape[-1] == 4:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA))
    elif a.ndim == 3:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    else:
        ok, buf = cv2.imencode(".png", a)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------- graph -> geojson
def graph_to_display_geojson(G: nx.Graph, scores: dict | None = None,
                             gatekeepers=None, simplify_px: float = 1.5) -> dict:
    """Pixel-space or geographic GeoJSON for the map.

    Geometry is simplified for display only; all metrics are computed on the
    full-resolution graph.
    """
    gates = set(gatekeepers or [])
    feats = []
    geo = G.graph.get("georeferenced", False)

    def xy(n):
        d = G.nodes[n]
        if geo and "lon" in d:
            return [d["lon"], d["lat"]]
        r, c = d["pos"]
        return [float(c), float(-r)]   # flip row so north is up in pixel space

    for u, v, d in G.edges(data=True):
        pts = d.get("pts")
        if pts is not None and len(pts) > 2 and not geo:
            a = np.asarray(pts, float)
            step = max(int(simplify_px), 1)
            a = a[::step] if len(a) > 2 * step else a
            coords = [[float(c), float(-r)] for r, c in a]
        else:
            coords = [xy(u), xy(v)]
        sc = 0.5 * (scores.get(u, 0.0) + scores.get(v, 0.0)) if scores else None
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "u": int(u), "v": int(v),
                "length_m": round(float(d.get("length_m", 0.0)), 2),
                "kind": d.get("kind", "observed"),
                "healed": d.get("kind") == "healed",
                "criticality": round(sc, 4) if sc is not None else None,
            },
        })

    for n, d in G.nodes(data=True):
        deg = G.degree(n)
        if deg == 2 and n not in gates:
            continue          # interior nodes add payload without information
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": xy(n)},
            "properties": {
                "node": int(n), "degree": deg,
                "criticality": round(scores.get(n, 0.0), 4) if scores else None,
                "gatekeeper": n in gates,
            },
        })

    return {"type": "FeatureCollection",
            "crs_mode": "geographic" if geo else "pixel",
            "features": feats}


# ---------------------------------------------------------------- pipeline
def build_and_heal(session: Session, cfg: HealingConfig,
                   min_spur_m: float = 60.0) -> dict:
    """Mask -> pruned graph -> healed graph, with the connectivity report."""
    px = session.pixel_size_m
    min_spur_px = min_spur_m / max(px, 1e-6)

    raw = build_raw_graph(session.mask, px, min_spur_px)
    if raw.number_of_nodes() == 0:
        raise ValueError("no road structure found in mask")

    healed, report = heal_graph(raw, cfg, session.evidence)
    G = largest_component(healed)

    session.raw_graph = raw
    session.graph = G
    session.healing = report
    session.invalidate()

    return {
        "healing": report,
        "connectivity_ratio": connectivity_ratio(raw, healed),
        "stats": network_statistics(G),
    }


def explain_node(G: nx.Graph, node: int, crit: dict, od_pairs, baseline) -> dict:
    """Generate a justification from computed statistics only.

    Every clause below is backed by a number we actually calculated; nothing is
    asserted that we cannot show.
    """
    from src.analytics.simulation import Scenario, evaluate_scenario

    bc = crit["components"]["betweenness"].get(node, 0.0)
    gates = crit["components"]["gatekeeper"]
    deg = G.degree(node)
    load = sum(d.get("length_m", 0.0) for _, _, d in G.edges(node, data=True))

    res = evaluate_scenario(G, Scenario(f"probe_{node}", removed_nodes=[node]),
                            od_pairs, baseline)
    res.pop("_graph", None)
    od_pct = res["disconnection_rate"] * 100
    eff_loss = (1.0 - res["efficiency_retained"]) * 100
    detour = res["mean_detour_ratio"]

    parts = []
    if node in gates:
        parts.append(
            "removing this intersection splits the network into disconnected "
            "components, so it is an articulation point (gatekeeper) rather than "
            "merely a busy junction"
        )
    pct_rank = 100.0 * sum(1 for v in crit["components"]["betweenness"].values()
                           if v <= bc) / max(len(crit["components"]["betweenness"]), 1)
    parts.append(f"it carries a betweenness of {bc:.4f}, higher than "
                 f"{pct_rank:.0f}% of intersections in this network")
    if od_pct > 0.5:
        parts.append(f"disabling it makes {od_pct:.0f}% of sampled origin-destination "
                     "trips impossible")
    if detour and detour > 1.01:
        parts.append(f"trips that survive become {detour:.2f}x longer on average")
    if eff_loss > 0.5:
        parts.append(f"network efficiency falls by {eff_loss:.0f}%")

    return {
        "node": int(node),
        "criticality": round(float(crit["scores"].get(node, 0.0)), 4),
        "betweenness": round(float(bc), 5),
        "degree": int(deg),
        "is_gatekeeper": node in gates,
        "incident_length_m": round(float(load), 1),
        "od_pairs_affected_pct": round(od_pct, 2),
        "efficiency_loss_pct": round(eff_loss, 2),
        "explanation": "Node {}: ".format(node) + "; ".join(parts) + ".",
    }
