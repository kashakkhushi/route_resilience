"""
Frontend/backend contract validation (project spec section 34).

Asserts that every field the React app reads actually exists in the backend
response. This is the class of bug that does not show up in backend tests - the
API is perfectly valid, the UI simply renders `undefined` - so it gets its own
check.

    python backend/tests/test_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import logging  # noqa: E402

logging.disable(logging.INFO)

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402

c = TestClient(app)
PASS: list[str] = []
FAIL: list[tuple[str, str]] = []


def need(obj: dict, path: str, label: str) -> None:
    """Assert a dotted path exists and is not None-by-absence."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            if not cur:
                FAIL.append((label, f"{path}: list is empty"))
                print(f"  FAIL  {label}: {path} (empty list)")
                return
            cur = cur[0]
        if not isinstance(cur, dict) or part not in cur:
            FAIL.append((label, f"{path}: missing '{part}'"))
            print(f"  FAIL  {label}: {path} (missing '{part}')")
            return
        cur = cur[part]
    PASS.append(label)
    print(f"  PASS  {label}: {path}")


def main() -> int:
    print("\n[contract] imagery -> ImageryResponse")
    r = c.post("/api/imagery/demo",
               json={"city": "grid_river", "size": 512, "seed": 1, "occlusion": True})
    assert r.status_code == 200, r.text
    img = r.json()
    sid = img["session_id"]
    for p in ("session_id", "layers", "meta.width", "meta.height", "meta.bands",
              "meta.dtype", "meta.pixel_size_m", "meta.is_georeferenced",
              "meta.source"):
        need(img, p, "imagery")

    print("\n[contract] extract -> ExtractResponse")
    ex = c.post("/api/extract-roads", json={"session_id": sid}).json()
    for p in ("provenance.mode", "provenance.label", "provenance.description",
              "provenance.is_learned", "layers.mask", "layers.probability",
              "metrics.available", "road_pixel_fraction", "threshold_used",
              "elapsed_ms"):
        need(ex, p, "extract")

    print("\n[contract] build-graph -> GraphResponse")
    g = c.post("/api/build-graph",
               json={"session_id": sid, "max_gap_m": 140, "restore_loops": True}).json()
    for p in ("stats.nodes", "stats.edges", "stats.total_length_km",
              "stats.gatekeepers", "stats.bridge_edges", "stats.alpha_index",
              "stats.intersections", "stats.dead_ends", "stats.mean_degree",
              "healing.components_before", "healing.components_after",
              "healing.bridges_added", "healing.loops_restored",
              "healing.healed_length_fraction",
              "connectivity_ratio.lcc_node_increase_pct",
              "connectivity_ratio.lcc_length_increase_pct",
              "connectivity_ratio.lcc_coverage_after",
              "geojson.features"):
        need(g, p, "graph")

    # The canvas needs both geometry types and the properties it draws from.
    feats = g["geojson"]["features"]
    lines = [f for f in feats if f["geometry"]["type"] == "LineString"]
    points = [f for f in feats if f["geometry"]["type"] == "Point"]
    for label, cond, msg in [
        ("graph geojson has edges", bool(lines), "no LineString features"),
        ("graph geojson has nodes", bool(points), "no Point features"),
        ("edge props complete",
         bool(lines) and all(k in lines[0]["properties"] for k in ("u", "v", "healed")),
         "edge missing u/v/healed"),
        ("node props complete",
         bool(points) and "node" in points[0]["properties"],
         "node missing 'node'"),
    ]:
        (PASS if cond else FAIL).append(label if cond else (label, msg))
        print(f"  {'PASS' if cond else 'FAIL'}  {label}" + ("" if cond else f": {msg}"))

    print("\n[contract] criticality -> CriticalityResponse")
    k = c.post("/api/criticality", json={"session_id": sid, "top_percent": 10}).json()
    for p in ("exact", "pivots", "weight", "gatekeepers", "bridge_edges", "scores",
              "top_nodes.node", "top_nodes.criticality", "top_nodes.betweenness",
              "top_nodes.degree", "top_nodes.is_gatekeeper",
              "top_nodes.incident_length_m", "top_nodes.explanation"):
        need(k, p, "criticality")

    # scores keys must be strings: the canvas indexes scores[String(id)].
    ok = all(isinstance(x, str) for x in list(k["scores"])[:5])
    (PASS if ok else FAIL).append("scores keyed by string" if ok
                                  else ("scores keyed by string", "keys are not strings"))
    print(f"  {'PASS' if ok else 'FAIL'}  scores keyed by string")

    print("\n[contract] simulate -> SimulateResponse")
    s = c.post("/api/simulate-failure",
               json={"session_id": sid, "kind": "targeted_attack", "count": 4}).json()
    for p in ("scenario", "description", "removed_nodes", "removed_edges",
              "baseline", "disrupted", "delta.mean_detour_ratio",
              "delta.p95_detour_ratio", "resilience.resilience_index",
              "resilience.normalized_resilience", "resilience.efficiency_ratio",
              "resilience.resilience_metric", "resilience.resilience_formula",
              "resilience.resilience_interpretation",
              "resilience.resilience_index_literal_aspl",
              "resilience.literal_aspl_is_degenerate", "resilience.literal_aspl_note",
              "resilience.diagnostic_aspl_ratio_connected_only",
              "resilience.diagnostic_is_resilience_score",
              "resilience.diagnostic_survivorship_biased", "resilience.diagnostic_note",
              "resilience.disconnection_rate", "resilience.aspl_baseline_m",
              "resilience.grade", "resilience.note",
              "resilience.network_fully_connected_after"):
        need(s, p, "simulate")

    # The stress-test view paints removed nodes on the map; it needs the ids.
    has_ids = ("removed_nodes_list" in s.get("baseline", {})
               or "removed_nodes_list" in s
               or "removed_node_ids" in s)
    (PASS if has_ids else FAIL).append(
        "removed node ids exposed for map rendering" if has_ids
        else ("removed node ids exposed for map rendering",
              "no removed_nodes_list / removed_node_ids in response; the UI "
              "cannot mark which intersections failed"))
    print(f"  {'PASS' if has_ids else 'FAIL'}  removed node ids exposed for map rendering")

    print("\n[contract] route -> RouteResponse")
    nodes = sorted(int(f["properties"]["node"]) for f in points)
    rt = c.post("/api/route",
                json={"session_id": sid, "source": nodes[0], "target": nodes[-1]}).json()
    for p in ("original.reachable", "original.path", "original.distance_m",
              "original.travel_time_min", "disrupted.reachable",
              "speed_assumption_kmh", "note"):
        need(rt, p, "route")

    print("\n" + "=" * 58)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for item in FAIL:
        label, msg = item if isinstance(item, tuple) else (item, "")
        print(f"  - {label}: {msg}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
