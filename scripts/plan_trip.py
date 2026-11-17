"""
Plan a trip from the command line - the same code the Trip Planner page calls.
Use it to check the set-up and to pre-download an area before a demo (the
result is cached in data/osm and data/plan_cache, so the page then opens it
instantly, even offline).

    python scripts/plan_trip.py --a 17.4401,78.3489 --b 17.4504,78.3810

Run from the project root, with internet, after
    pip install -r requirements-geo.txt
and with the Sentinel-2 checkpoint at checkpoints/segformer_s2/best.pt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="lat,lon of A")
    ap.add_argument("--b", required=True, help="lat,lon of B")
    ap.add_argument("--no-satellite", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    args = ap.parse_args()
    a = tuple(float(v) for v in args.a.split(","))
    b = tuple(float(v) for v in args.b.split(","))

    from backend.app.services import trip_planner as T
    from src.routing import sources as S

    print("Sentinel-2 checkpoint:", S.s2_checkpoint() or "NOT FOUND")
    try:
        r = T.plan(a, b, satellite=not args.no_satellite, refresh=args.refresh)
    except T.PlanError as e:
        raise SystemExit(f"FAILED ({e.status}): {e.detail}")

    sat = r["satellite"]
    print(f"\nroads in area      : {r['road_count']}")
    if sat["available"]:
        sc = sat["scene"]
        print(f"satellite image    : {sc['id']}  date {sc['date']}  "
              f"cloud over area {sc['box_unusable_pct']}%  (cached: {sat['from_cache']})")
        print(f"road check         : {sat['counts']}")
        h = sat.get("healing") or {}
        print(f"healing            : {h.get('gaps_joined', 0)} gaps joined "
              f"(up to {h.get('max_gap_m')} m){'  ERROR ' + h['error'] if h.get('error') else ''}")
    else:
        print(f"satellite          : {sat['reason']}")
    rt = r["route"]
    if rt["reachable"]:
        print(f"route              : {rt['length_m'] / 1000:.2f} km, {rt['time_min']} min "
              f"(seen by satellite {rt['verified_length_m'] / 1000:.2f} km, "
              f"joined by healing {rt.get('bridged_length_m', 0) / 1000:.2f} km, "
              f"not seen {rt['unverified_length_m'] / 1000:.2f} km)")
    else:
        print("route              : A and B are not connected")
    for i, c in enumerate(r["critical_roads"], 1):
        eff = "cuts the trip" if c["cuts_trip"] else f"+{c['extra_min']} min"
        print(f"  #{i} {c['name'] or 'unnamed ' + c['class']:<40} {eff}")
    for w in r["warnings"]:
        print("warning:", w)
    print("timings (s)        :", r["timings"])


if __name__ == "__main__":
    main()
