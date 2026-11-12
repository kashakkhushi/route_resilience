"""API routes. Each stage of the pipeline is a real computation over session
state; nothing returns a canned result."""
from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.analytics.criticality import network_statistics, rank_critical_edges
from src.analytics.resilience import resilience_index_aspl
from src.analytics.simulation import (
    Scenario,
    alternate_route,
    evaluate_scenario,
    flood_scenario,
    random_failure,
    targeted_attack,
)
from src.data.synthetic import fragment_mask, generate_city
from src.eval.metrics import evaluate_mask
from src.graph.healing import HealingConfig

from ...schemas.models import (
    CriticalityRequest,
    CriticalityResponse,
    ExtractRequest,
    ExtractResponse,
    GraphResponse,
    HealRequest,
    HealthResponse,
    ImageryResponse,
    LoadDemoRequest,
    RouteRequest,
    RouteResponse,
    SimulateRequest,
    SimulateResponse,
)
from ...services.model_registry import (
    simulated_demo_extract,
    ExtractionMode,
    PROVENANCE,
    capability_report,
    extract_roads,
)
from ...services.session import (
    STORE,
    build_and_heal,
    explain_node,
    graph_to_display_geojson,
    to_png_data_uri,
)

router = APIRouter()
VERSION = "0.3.0"

DEMO_PRESETS = {
    "grid_river": dict(grid_spacing=96, n_radials=6, river=True, n_crossings=2),
    "dense_urban": dict(grid_spacing=64, n_radials=8, river=True, n_crossings=3),
    "sparse_rural": dict(grid_spacing=150, n_radials=3, river=True, n_crossings=1),
}


def _need(sid: str, stage: str = "session"):
    s = STORE.get(sid)
    if s is None:
        raise HTTPException(404, detail=f"Unknown session '{sid}'. Load imagery first.")
    return s


# ------------------------------------------------------------------ health
@router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    cap = capability_report()
    return HealthResponse(version=VERSION, **cap)


# ------------------------------------------------------------------ imagery
@router.post("/imagery/demo", response_model=ImageryResponse, tags=["imagery"])
def load_demo(req: LoadDemoRequest):
    """Deterministic synthetic city with KNOWN ground-truth topology.

    This is the only path where segmentation metrics are meaningful without an
    external label set, because we generated the truth ourselves.
    """
    s = STORE.create()
    preset = DEMO_PRESETS[req.city.value]
    city = generate_city(size=req.size, seed=req.seed, **preset)

    s.image = city["image"]
    s.gt_mask = city["mask"]
    s.pixel_size_m = 2.0
    s.source = f"DEMO:{req.city.value}"
    s.seed = req.seed
    s.meta = {"demo": True, "preset": req.city.value}

    layers = {"original": to_png_data_uri(city["image"]),
              "ground_truth": to_png_data_uri(city["mask"], cmap="roads")}

    if req.occlusion:
        from src.data.occlusion import OcclusionConfig, apply_occlusions

        occluded, alpha = apply_occlusions(city["image"], city["mask"],
                                           OcclusionConfig(rng_seed=req.seed))
        s.image = occluded
        s.occlusion = alpha
        layers["original"] = to_png_data_uri(occluded)
        layers["clean"] = to_png_data_uri(city["image"])
        layers["occlusion"] = to_png_data_uri(alpha, cmap="magma")

    return ImageryResponse(
        session_id=s.id,
        meta={
            "width": s.image.shape[1], "height": s.image.shape[0],
            "bands": s.image.shape[2] if s.image.ndim == 3 else 1,
            "crs": None, "pixel_size_m": s.pixel_size_m, "bounds": None,
            "dtype": str(s.image.dtype), "source": s.source,
            "is_georeferenced": False,
        },
        layers=layers,
        provenance=PROVENANCE[ExtractionMode.DEMO_DATA].dict(),
    )


@router.post("/imagery/upload", response_model=ImageryResponse, tags=["imagery"])
async def upload(file: UploadFile = File(...)):
    """Accept a GeoTIFF or ordinary raster.

    GeoTIFF requires rasterio; when it is absent we say so rather than failing
    obscurely. Plain PNG/JPEG always works but carries no CRS, so all downstream
    lengths are in pixels and every response flags `is_georeferenced: false`.
    """
    import cv2

    raw = await file.read()
    if len(raw) > 120 * 1024 * 1024:
        raise HTTPException(413, detail="File exceeds the 120 MB upload limit.")
    name = (file.filename or "upload").lower()

    s = STORE.create()
    if name.endswith((".tif", ".tiff")):
        try:
            import rasterio
            from rasterio.io import MemoryFile

            from src.data.geo_io import select_rgb, to_reflectance
        except ImportError:
            raise HTTPException(
                501,
                detail=("GeoTIFF support requires rasterio, which is not installed "
                        "in this deployment. Install requirements-geo.txt, or upload "
                        "a PNG/JPEG for pixel-space analysis."),
            )
        try:
            with MemoryFile(raw) as mem, mem.open() as src:
                arr = np.transpose(src.read(), (1, 2, 0)).astype(np.float32)
                s.transform, s.crs = src.transform, src.crs
                s.pixel_size_m = abs(src.transform.a)
                bounds = list(src.bounds)
                sensor = ("sentinel2" if 8 < s.pixel_size_m < 12
                          else "liss4" if 4 < s.pixel_size_m < 7 else "cartosat3")
                img = select_rgb(to_reflectance(arr, sensor), sensor)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(422, detail=f"Could not read GeoTIFF: {e}")
        crs = str(s.crs) if s.crs else None
        georef = s.crs is not None
    else:
        a = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if a is None:
            raise HTTPException(422, detail="Unsupported or corrupted image file.")
        img = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        bounds, crs, georef = None, None, False
        s.pixel_size_m = 1.0

    if img.shape[0] < 64 or img.shape[1] < 64:
        raise HTTPException(422, detail="Image too small: minimum 64x64 pixels.")

    s.image = img
    s.source = file.filename or "upload"
    return ImageryResponse(
        session_id=s.id,
        meta={
            "width": img.shape[1], "height": img.shape[0],
            "bands": img.shape[2] if img.ndim == 3 else 1,
            "crs": crs, "pixel_size_m": s.pixel_size_m, "bounds": bounds,
            "dtype": str(img.dtype), "source": s.source,
            "is_georeferenced": georef,
        },
        layers={"original": to_png_data_uri(img)},
        provenance=None,
    )


# ------------------------------------------------------------------ extraction
@router.post("/extract-roads", response_model=ExtractResponse, tags=["extraction"])
def extract(req: ExtractRequest):
    t0 = time.time()
    s = _need(req.session_id)
    if s.image is None:
        raise HTTPException(409, detail="No imagery loaded for this session.")

    # Demo scenes default to the deterministic simulated segmentation, because
    # running the classical baseline on a synthetic render produces a graph too
    # sparse to demonstrate criticality or stress testing. The caller can still
    # force the real extractor with mode="classical" to compare the two.
    is_demo = s.source.startswith("DEMO:")
    use_demo = (req.mode or ("demo" if is_demo else "auto")) == "demo"
    if use_demo and not is_demo:
        raise HTTPException(
            422,
            detail="mode='demo' is only valid for a demo session; it simulates a "
                   "fragmented segmentation from known ground truth.",
        )
    if use_demo and s.gt_mask is not None:
        out = simulated_demo_extract(s.gt_mask, s.occlusion, seed=s.seed)
    else:
        # An explicit "classical" request must be honoured even when a trained
        # checkpoint exists, otherwise the side-by-side comparison in the demo
        # silently runs the model twice and labels both TRAINED_MODEL.
        out = extract_roads(s.image, s.pixel_size_m, req.threshold,
                            prefer_neural=(req.mode != "classical"),
                            road_width_m=req.road_width_m)
    s.prob, s.mask = out["prob"], out["mask"]
    s.evidence = out.get("evidence", out["prob"])
    s.provenance = out["provenance"]
    s.threshold = out.get("threshold", req.threshold or 0.5)
    s.invalidate()

    if not s.mask.any():
        raise HTTPException(
            422,
            detail=("Extraction produced an empty road mask. Try lowering the "
                    "threshold or check that the imagery contains visible roads."),
        )

    # Metrics ONLY when ground truth genuinely exists.
    if s.gt_mask is not None:
        m = evaluate_mask(s.mask, s.gt_mask, s.occlusion, buffer_px=4)
        metrics = {"available": True, **{k: m.get(k) for k in (
            "iou", "dice", "precision", "recall", "cldice", "relaxed_iou",
            "recall_visible", "recall_occluded", "occlusion_gap")}}
    else:
        metrics = {
            "available": False,
            "reason": ("No ground truth for this scene. Load a demo city, or run "
                       "scripts/prepare_groundtruth.py to generate OpenStreetMap "
                       "reference masks for georeferenced imagery."),
        }

    return ExtractResponse(
        session_id=s.id,
        provenance=s.provenance.dict(),
        threshold_used=float(s.threshold),
        road_pixel_fraction=float(s.mask.mean()),
        layers={
            "probability": to_png_data_uri(s.prob, cmap="magma"),
            "mask": to_png_data_uri(s.mask, cmap="roads"),
        },
        metrics=metrics,
        elapsed_ms=int((time.time() - t0) * 1000),
    )


# ------------------------------------------------------------------ graph
@router.post("/build-graph", response_model=GraphResponse, tags=["graph"])
def build_graph(req: HealRequest):
    """Skeletonize, prune, then heal with union-find + MST."""
    t0 = time.time()
    s = _need(req.session_id)
    if s.mask is None:
        raise HTTPException(409, detail="Run /extract-roads before building the graph.")

    px = max(s.pixel_size_m, 1e-6)
    cfg = HealingConfig(
        max_gap_px=req.max_gap_m / px,
        w_angle=req.w_angle,
        w_evidence=req.w_evidence,
        max_cost=(req.max_gap_m / px) * 2.5,
        loop_max_cost=(req.max_gap_m / px) * 1.4,
        restore_loops=req.restore_loops,
    )
    try:
        res = build_and_heal(s, cfg, req.min_spur_m)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))

    if s.graph.number_of_nodes() < 3:
        raise HTTPException(
            422, detail="Extracted network is too small to analyse (fewer than 3 nodes).")

    return GraphResponse(
        session_id=s.id,
        stats=res["stats"],
        healing=res["healing"],
        connectivity_ratio=res["connectivity_ratio"],
        geojson=graph_to_display_geojson(s.graph),
        provenance=s.provenance.dict(),
        elapsed_ms=int((time.time() - t0) * 1000),
    )


# ------------------------------------------------------------------ criticality
@router.post("/criticality", response_model=CriticalityResponse, tags=["analysis"])
def criticality(req: CriticalityRequest):
    t0 = time.time()
    s = _need(req.session_id)
    if s.graph is None:
        raise HTTPException(409, detail="Build the graph before running criticality.")

    n = s.graph.number_of_nodes()
    pivots = req.pivots
    # Guard the O(V*E) cost: above ~4000 nodes exact Brandes stalls the request,
    # so we sample pivots and SAY SO in the response rather than silently
    # changing the definition of the metric.
    if pivots is None and n > 4000:
        pivots = 512
    crit = s.criticality(pivots)

    od, base = s.baseline(300)
    ranked = sorted(crit["scores"], key=crit["scores"].get, reverse=True)
    k = max(int(n * req.top_percent / 100.0), 1)
    top = ranked[:min(k, 40)]

    return CriticalityResponse(
        session_id=s.id,
        exact=pivots is None,
        pivots=pivots,
        weight=crit["meta"]["weight"],
        top_nodes=[explain_node(s.graph, nd, crit, od, base) for nd in top[:12]],
        top_edges=rank_critical_edges(s.graph, 12, k=pivots),
        gatekeepers=[int(x) for x in sorted(crit["components"]["gatekeeper"])],
        bridge_edges=[list(map(int, e)) for e in
                      __import__("networkx").bridges(s.graph)],
        scores={str(int(kk)): round(float(v), 4) for kk, v in crit["scores"].items()},
        elapsed_ms=int((time.time() - t0) * 1000),
    )


# ------------------------------------------------------------------ simulation
@router.post("/simulate-failure", response_model=SimulateResponse, tags=["analysis"])
def simulate(req: SimulateRequest):
    t0 = time.time()
    s = _need(req.session_id)
    if s.graph is None:
        raise HTTPException(409, detail="Build the graph before simulating failures.")
    G = s.graph

    kind = req.kind.value
    if kind == "node_failure":
        bad = [n for n in req.nodes if n not in G]
        if bad:
            raise HTTPException(422, detail=f"Nodes not in network: {bad[:5]}")
        if not req.nodes:
            raise HTTPException(422, detail="Select at least one intersection.")
        sc = Scenario("node_failure", removed_nodes=list(req.nodes),
                      description=f"{len(req.nodes)} intersection(s) disabled")
    elif kind == "edge_closure":
        edges = [tuple(e) for e in req.edges]
        bad = [e for e in edges if not G.has_edge(*e)]
        if bad:
            raise HTTPException(422, detail=f"Segments not in network: {bad[:5]}")
        if not edges:
            raise HTTPException(422, detail="Select at least one road segment.")
        sc = Scenario("edge_closure", removed_edges=edges,
                      description=f"{len(edges)} segment(s) closed")
    elif kind == "flood_radial":
        if not req.center or req.radius_px is None:
            raise HTTPException(422, detail="Flood scenario needs center and radius_px.")
        sc = flood_scenario(G, tuple(req.center), req.radius_px)
    elif kind == "targeted_attack":
        sc = targeted_attack(G, req.count, k=256 if G.number_of_nodes() > 2000 else None)
    else:
        sc = random_failure(G, req.count, seed=req.seed)

    od, base = s.baseline(req.od_pairs)
    res = evaluate_scenario(G, sc, od, base)
    H = res.pop("_graph")
    ri = resilience_index_aspl(G, sc, od, base)

    base_stats = network_statistics(G)
    dis_stats = network_statistics(H) if H.number_of_nodes() else {"nodes": 0, "edges": 0}

    removed = set(sc.removed_nodes)
    feats = []
    for n in removed:
        if n in G:
            d = G.nodes[n]
            r, c = d["pos"]
            feats.append({"type": "Feature",
                          "geometry": {"type": "Point",
                                       "coordinates": [float(c), float(-r)]},
                          "properties": {"node": int(n), "status": "disabled"}})
    for u, v in sc.removed_edges:
        if G.has_edge(u, v):
            pu, pv = G.nodes[u]["pos"], G.nodes[v]["pos"]
            feats.append({"type": "Feature",
                          "geometry": {"type": "LineString",
                                       "coordinates": [[float(pu[1]), float(-pu[0])],
                                                       [float(pv[1]), float(-pv[0])]]},
                          "properties": {"u": int(u), "v": int(v), "status": "closed"}})

    # Normalized resilience: 0 = severely disrupted, 1 = highly resilient.
    # Built from efficiency retention, which degrades smoothly and stays defined
    # when the network fragments - unlike the ASPL ratio, which becomes 0.
    norm = float(np.clip(ri["efficiency_ratio"], 0, 1))

    return SimulateResponse(
        session_id=s.id,
        scenario=sc.name,
        description=sc.description,
        removed_nodes=len(sc.removed_nodes),
        removed_edges=len(sc.removed_edges),
        # The ids, not just the counts: the map has to mark WHICH intersections
        # failed. Returning only a count leaves the UI unable to render the
        # scenario it just ran.
        removed_node_ids=[int(n) for n in sc.removed_nodes],
        removed_edge_ids=[[int(u), int(v)] for u, v in sc.removed_edges],
        baseline={"nodes": base_stats["nodes"], "edges": base_stats["edges"],
                  "components": base_stats["components"],
                  "aspl_m": ri["aspl_baseline_m"],
                  "efficiency": res["efficiency_before"],
                  "total_length_km": base_stats["total_length_km"]},
        disrupted={"nodes": dis_stats.get("nodes", 0), "edges": dis_stats.get("edges", 0),
                   "components": res["components_after"],
                   "aspl_m": ri["aspl_perturbed_m"],
                   "efficiency": res["efficiency_after"],
                   "giant_fraction": res["giant_fraction_after"]},
        delta={"disconnection_rate": res["disconnection_rate"],
               "mean_detour_ratio": res["mean_detour_ratio"],
               "p95_detour_ratio": res["p95_detour_ratio"],
               "efficiency_retained": res["efficiency_retained"],
               "aspl_increase_pct": ri["aspl_increase_pct"],
               "od_pairs": res["od_pairs"],
               "od_disconnected": res["od_disconnected"]},
        resilience={k: ri[k] for k in (
            "resilience_index", "resilience_metric", "resilience_formula",
            "resilience_interpretation", "normalized_resilience", "efficiency_ratio",
            "grade",
            "resilience_index_literal_aspl", "literal_aspl_is_degenerate",
            "literal_aspl_note",
            "diagnostic_aspl_ratio_connected_only", "diagnostic_is_resilience_score",
            "diagnostic_survivorship_biased", "diagnostic_note",
            "disconnection_rate", "aspl_baseline_m", "aspl_perturbed_m",
            "aspl_increase_pct", "network_fully_connected_after", "note")},
        affected_geojson={"type": "FeatureCollection", "features": feats},
        elapsed_ms=int((time.time() - t0) * 1000),
    )


# ------------------------------------------------------------------ routing
@router.post("/route", response_model=RouteResponse, tags=["analysis"])
def route(req: RouteRequest):
    s = _need(req.session_id)
    if s.graph is None:
        raise HTTPException(409, detail="Build the graph before routing.")
    G = s.graph
    for n in (req.source, req.target):
        if n not in G:
            raise HTTPException(422, detail=f"Node {n} is not in the network.")
    if req.source == req.target:
        raise HTTPException(422, detail="Origin and destination must differ.")

    sc = Scenario("route_probe",
                  removed_nodes=[n for n in req.disabled_nodes if n in G],
                  removed_edges=[tuple(e) for e in req.disabled_edges
                                 if G.has_edge(*e)])
    rr = alternate_route(G, sc, req.source, req.target)
    mps = req.speed_kmh * 1000 / 60.0        # metres per minute

    def leg(path, dist):
        if path is None or dist is None or not np.isfinite(dist):
            return {"reachable": False, "path": None, "distance_m": None,
                    "travel_time_min": None, "geojson": None}
        coords = []
        for a, b in zip(path[:-1], path[1:]):
            d = G[a][b]
            pts = d.get("pts")
            if pts is not None and len(pts) > 1:
                coords += [[float(c), float(-r)] for r, c in pts]
            else:
                pa, pb = G.nodes[a]["pos"], G.nodes[b]["pos"]
                coords += [[float(pa[1]), float(-pa[0])], [float(pb[1]), float(-pb[0])]]
        return {
            "reachable": True, "path": [int(x) for x in path],
            "distance_m": round(float(dist), 1),
            "travel_time_min": round(float(dist) / mps, 2),
            "geojson": {"type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": coords},
                        "properties": {}},
        }

    o = leg(rr["original_path"], rr["original_length_m"])
    d = leg(rr["alternate_path"], rr["alternate_length_m"])
    inc = (None if not (o["reachable"] and d["reachable"])
           else round(100 * (d["distance_m"] - o["distance_m"]) / max(o["distance_m"], 1e-6), 2))

    return RouteResponse(
        session_id=s.id, original=o, disrupted=d,
        distance_increase_pct=inc, time_increase_pct=inc,   # uniform speed => equal
        detour_ratio=rr["detour_ratio"],
        speed_assumption_kmh=req.speed_kmh,
        note=("Travel time assumes a uniform free-flow speed of "
              f"{req.speed_kmh} km/h across all road classes, because road class and "
              "speed limits are not available from segmentation alone. Distance "
              "figures are measured; time figures are derived from that assumption."),
    )
