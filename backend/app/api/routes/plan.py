"""Coordinate trip planner: A/B as lat/lon on real OpenStreetMap roads, checked
against the latest clear Sentinel-2 image by the fine-tuned S2 model."""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...services import trip_planner as T

router = APIRouter(tags=["trip planner"])
log = logging.getLogger("route_resilience.plan")


class LatLon(BaseModel):
    lat: float = Field(..., ge=-85, le=85)
    lon: float = Field(..., ge=-180, le=180)


class PlanRequest(BaseModel):
    a: LatLon
    b: LatLon
    satellite: bool = Field(True, description="Check every road against the latest clear Sentinel-2 image.")
    refresh: bool = Field(False, description="Ignore the cached image/model result for this area.")
    optimise: Literal["time", "distance"] = "time"


class Flood(BaseModel):
    lat: float = Field(..., ge=-85, le=85)
    lon: float = Field(..., ge=-180, le=180)
    radius_m: float = Field(..., gt=0, le=3000)


class RerouteRequest(BaseModel):
    closed: list[int] = Field(default_factory=list, max_length=500)
    flood: Flood | None = None
    avoid_unverified: bool = False


@router.post("/plan")
def plan(req: PlanRequest):
    try:
        return T.plan((req.a.lat, req.a.lon), (req.b.lat, req.b.lon),
                      satellite=req.satellite, refresh=req.refresh, optimise=req.optimise)
    except T.PlanError as e:
        raise HTTPException(e.status, detail=e.detail)
    except Exception as e:                  # say what broke, not just "500"
        log.exception("plan failed")
        raise HTTPException(500, detail=f"Planning failed: {type(e).__name__}: {e}")


@router.post("/plan/{plan_id}/reroute")
def reroute(plan_id: str, req: RerouteRequest):
    try:
        return T.reroute(plan_id, closed=req.closed,
                         flood=req.flood.model_dump() if req.flood else None,
                         avoid_unverified=req.avoid_unverified)
    except T.PlanError as e:
        raise HTTPException(e.status, detail=e.detail)
