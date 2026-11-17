"""
Trip "what if roads fail" contract (POST /api/route/what-if).

    T1  The original route is returned with distance and path.
    T2  Every intermediate junction on the route is checked.
    T3  Junctions are ranked: trip-cutting ones first, and among those the one
        that splits off more of the network first (a bridge outranks the corner
        of a dead-end street).
    T4  The worst junction is never the origin or destination.
    T5  For a non-cutting worst junction the alternate route avoids it.
    T6  Bad requests are rejected (same node; unknown node).

    python backend/tests/test_trip_what_if.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name if ok else (name, detail))
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + ("" if ok else f"   [{detail}]"))


def main() -> int:
    from fastapi.testclient import TestClient

    from backend.app.main import app

    c = TestClient(app)
    sid = c.post("/api/imagery/demo", json={"city": "grid_river", "size": 768, "seed": 1,
                                            "occlusion": True}).json()["session_id"]
    c.post("/api/extract-roads", json={"session_id": sid, "mode": "demo"})
    g = c.post("/api/build-graph", json={"session_id": sid, "max_gap_m": 140,
                                         "restore_loops": True}).json()
    pts = sorted(((f["properties"]["node"], f["geometry"]["coordinates"])
                  for f in g["geojson"]["features"] if f["geometry"]["type"] == "Point"),
                 key=lambda t: t[1][0])
    a, b = pts[3][0], pts[-4][0]

    print("\n[T1-T4] trip across the river")
    r = c.post("/api/route/what-if", json={"session_id": sid, "source": a, "target": b})
    j = r.json()
    check("status 200", r.status_code == 200, r.status_code)
    check("original route reachable", j["original"]["reachable"] is True)
    path = j["original"]["path"]
    check("path starts at A and ends at B", path[0] == a and path[-1] == b)
    check("all intermediate junctions checked", j["junctions_checked"] == len(path) - 2,
          (j["junctions_checked"], len(path)))
    crit = j["critical"]
    cuts = [x for x in crit if x["cuts_trip"]]
    check("cutting junctions ranked first",
          all(x["cuts_trip"] for x in crit[:len(cuts)]))
    sizes = [x["isolated_junctions"] for x in cuts]
    check("cuts ranked by size of the split", sizes == sorted(sizes, reverse=True), sizes)
    check("worst is not A or B", j["worst"]["node"] not in (a, b))

    print("\n[T5] alternate route avoids the closed junction")
    # search trips until one has a non-cutting worst junction (a loop around it)
    import random
    rng = random.Random(0)
    nodes = [n for n, _ in pts]
    ok_found = False
    for _ in range(80):
        s_, t_ = rng.sample(nodes, 2)
        jj = c.post("/api/route/what-if", json={"session_id": sid, "source": s_,
                                                "target": t_}).json()
        w = jj.get("worst")
        if w and not w["cuts_trip"]:
            alt = jj["alternate"]
            check("alternate reachable", alt["reachable"] is True)
            check("alternate avoids the closed junction", w["node"] not in alt["path"])
            check("alternate is not shorter", alt["distance_m"] >= jj["original"]["distance_m"])
            ok_found = True
            break
    check("found a trip with an alternative", ok_found)

    print("\n[T6] bad requests")
    check("same node -> 422", c.post("/api/route/what-if", json={
        "session_id": sid, "source": a, "target": a}).status_code == 422)
    check("unknown node -> 422", c.post("/api/route/what-if", json={
        "session_id": sid, "source": a, "target": 10 ** 9}).status_code == 422)

    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for item in FAIL:
        name, detail = item if isinstance(item, tuple) else (item, "")
        print(f"  FAILED: {name}  {detail}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
