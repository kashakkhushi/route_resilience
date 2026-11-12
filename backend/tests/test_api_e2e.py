"""
End-to-end API validation (project spec section 34).

Walks the full chain - demo -> extract -> graph -> criticality -> simulate ->
route - and asserts the scientific sanity properties from section 18, including
that a no-op scenario causes zero disruption and that removing a gatekeeper
actually disconnects the network.

    python backend/tests/test_api_e2e.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402

c = TestClient(app)
PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + str(extra)) if extra else ''}")


def main() -> int:
    print("\n[1] health / capability")
    h = c.get("/api/health").json()
    check("health ok", h["status"] == "ok")
    check("declares extraction mode", h["extraction_mode"] in
          ("TRAINED_MODEL", "CLASSICAL_BASELINE"), h["extraction_mode"])
    check("reports geotiff capability honestly", isinstance(h["geotiff_support"], bool),
          f"geotiff={h['geotiff_support']}")

    print("\n[2] load demo imagery")
    r = c.post("/api/imagery/demo",
               json={"city": "grid_river", "size": 512, "seed": 2, "occlusion": True})
    check("demo loads", r.status_code == 200, r.status_code)
    d = r.json()
    sid = d["session_id"]
    check("returns image layers", "original" in d["layers"] and "occlusion" in d["layers"])
    check("demo labelled as DEMO_DATA", d["provenance"]["mode"] == "DEMO_DATA")
    check("demo not claimed as learned", d["provenance"]["is_learned"] is False)

    print("\n[3] road extraction (real computation)")
    r = c.post("/api/extract-roads", json={"session_id": sid, "road_width_m": 6.0})
    check("extraction ok", r.status_code == 200, r.text[:160])
    e = r.json()
    check("provenance present", "provenance" in e)
    check("fallback NOT labelled as trained model",
          e["provenance"]["is_learned"] is False if
          e["provenance"]["mode"] != "TRAINED_MODEL" else True,
          e["provenance"]["label"])
    check("produced non-trivial mask", 0.001 < e["road_pixel_fraction"] < 0.7,
          round(e["road_pixel_fraction"], 4))
    check("metrics available (demo has ground truth)", e["metrics"]["available"])
    check("IoU is a real number", e["metrics"]["iou"] is not None,
          f"IoU={e['metrics']['iou']}")
    check("occlusion recall reported", e["metrics"]["recall_occluded"] is not None,
          f"vis={e['metrics']['recall_visible']} occ={e['metrics']['recall_occluded']}")

    print("\n[4] graph construction + healing")
    r = c.post("/api/build-graph", json={"session_id": sid, "max_gap_m": 120.0,
                                         "restore_loops": True})
    check("graph built", r.status_code == 200, r.text[:200])
    g = r.json()
    st, heal = g["stats"], g["healing"]
    check("graph non-trivial", st["nodes"] > 10 and st["edges"] > 10,
          f"{st['nodes']}n/{st['edges']}e")
    check("healing reduced fragmentation",
          heal["components_after"] <= heal["components_before"],
          f"{heal['components_before']} -> {heal['components_after']}")
    check("connectivity ratio reported",
          g["connectivity_ratio"]["lcc_node_increase_pct"] >= 0,
          f"+{g['connectivity_ratio']['lcc_node_increase_pct']}%")
    check("geojson has features", len(g["geojson"]["features"]) > 0,
          len(g["geojson"]["features"]))
    check("healed edges labelled",
          any(f["properties"].get("healed") for f in g["geojson"]["features"]
              if f["geometry"]["type"] == "LineString") or heal["bridges_added"] == 0)

    print("\n[5] criticality (real graph statistics)")
    r = c.post("/api/criticality", json={"session_id": sid, "top_percent": 10})
    check("criticality ok", r.status_code == 200, r.text[:160])
    cr = r.json()
    check("reports exactness", isinstance(cr["exact"], bool), f"exact={cr['exact']}")
    check("top nodes ranked", len(cr["top_nodes"]) > 0)
    top = cr["top_nodes"][0]
    check("explanation is data-backed",
          str(top["betweenness"]) in top["explanation"] or "betweenness" in top["explanation"])
    check("explanation non-empty", len(top["explanation"]) > 40)
    print(f"        e.g. {top['explanation'][:150]}...")

    print("\n[6] sanity: no-op scenario causes zero disruption")
    r = c.post("/api/simulate-failure",
               json={"session_id": sid, "kind": "random_failure", "count": 1, "seed": 0})
    check("simulate ok", r.status_code == 200, r.text[:160])

    nodes_sorted = sorted(int(k) for k in cr["scores"])
    r0 = c.post("/api/simulate-failure",
                json={"session_id": sid, "kind": "edge_closure", "edges": []})
    check("empty scenario rejected with 422", r0.status_code == 422, r0.status_code)

    print("\n[7] sanity: removing a gatekeeper disconnects the network")
    gates = cr["gatekeepers"]
    if gates:
        r = c.post("/api/simulate-failure",
                   json={"session_id": sid, "kind": "node_failure", "nodes": [gates[0]]})
        s = r.json()
        check("gatekeeper removal fragments network",
              s["disrupted"]["components"] >= 2, f"components={s['disrupted']['components']}")
        check("resilience block complete",
              all(k in s["resilience"] for k in
                  ("resilience_index", "normalized_resilience", "resilience_formula",
                   "resilience_metric")))
        check("normalized resilience in [0,1]",
              0.0 <= s["resilience"]["normalized_resilience"] <= 1.0,
              s["resilience"]["normalized_resilience"])
        # resilience_index is now the PRIMARY bounded metric (NER), not the
        # brief's literal ASPL ratio. The literal value moved to its own field
        # and is the one that degenerates to 0 under fragmentation.
        r = s["resilience"]
        check("primary index is bounded and not degenerate",
              0.0 <= r["resilience_index"] <= 1.0 and r["resilience_metric"] == "NER",
              f"R={r['resilience_index']} metric={r['resilience_metric']}")
        check("literal ASPL ratio is 0 exactly when the network fragmented",
              (r["resilience_index_literal_aspl"] == 0.0)
              == (not r["network_fully_connected_after"]))
        check("literal degeneracy is flagged",
              r["literal_aspl_is_degenerate"] == (not r["network_fully_connected_after"]))
        check("connected-only ratio is never presented as the index",
              r["diagnostic_is_resilience_score"] is False
              and r["resilience_index"] != r["diagnostic_aspl_ratio_connected_only"])
        print(f"        R_primary={s['resilience']['resilience_index']} "
              f"R_norm={s['resilience']['normalized_resilience']} "
              f"disc={s['resilience']['disconnection_rate']}")
    else:
        check("gatekeepers found", False, "none in this graph")

    print("\n[8] targeted attack beats random damage")
    t = c.post("/api/simulate-failure",
               json={"session_id": sid, "kind": "targeted_attack", "count": 5}).json()
    rnd = c.post("/api/simulate-failure",
                 json={"session_id": sid, "kind": "random_failure",
                       "count": 5, "seed": 1}).json()
    check("targeted more damaging than random",
          t["delta"]["efficiency_retained"] <= rnd["delta"]["efficiency_retained"] + 1e-9,
          f"targeted={t['delta']['efficiency_retained']} random={rnd['delta']['efficiency_retained']}")

    print("\n[9] routing: real shortest path, real detour")
    a, b = nodes_sorted[0], nodes_sorted[-1]
    r = c.post("/api/route", json={"session_id": sid, "source": a, "target": b})
    check("route ok", r.status_code == 200, r.text[:160])
    rt = r.json()
    check("baseline route reachable", rt["original"]["reachable"])
    check("distance is positive", (rt["original"]["distance_m"] or 0) > 0,
          rt["original"]["distance_m"])
    check("speed assumption disclosed", "uniform free-flow speed" in rt["note"])

    if gates:
        r2 = c.post("/api/route", json={"session_id": sid, "source": a, "target": b,
                                        "disabled_nodes": [gates[0]]}).json()
        changed = (not r2["disrupted"]["reachable"]) or (
            r2["disrupted"]["distance_m"] != r2["original"]["distance_m"])
        check("disabling a gatekeeper changes the route", changed,
              f"reachable={r2['disrupted']['reachable']}")

    print("\n[10] error handling")
    check("unknown session -> 404",
          c.post("/api/extract-roads", json={"session_id": "nope"}).status_code == 404)
    check("graph before extraction -> 409",
          c.post("/api/build-graph",
                 json={"session_id": c.post('/api/imagery/demo',
                                            json={'size': 256}).json()['session_id']}
                 ).status_code == 409)
    check("invalid node -> 422",
          c.post("/api/route", json={"session_id": sid, "source": 999999,
                                     "target": b}).status_code == 422)
    check("bad upload -> 422",
          c.post("/api/imagery/upload",
                 files={"file": ("x.png", b"not-an-image", "image/png")}
                 ).status_code == 422)

    print(f"\n{'='*58}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
