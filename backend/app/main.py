"""
Route Resilience AI - API server.

    uvicorn backend.app.main:app --reload --port 8000

OpenAPI docs at /docs. Run from the repository root so `src` is importable.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.api.routes.pipeline import VERSION, router  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("route-resilience")

app = FastAPI(
    title="Route Resilience AI",
    version=VERSION,
    description=(
        "Occlusion-robust road extraction and graph-theoretic criticality analysis.\n\n"
        "**Provenance contract:** every response derived from a road mask includes a "
        "`provenance` object stating what produced it - a trained checkpoint, the "
        "classical ridge-filter baseline, or demo data. Clients must display it. "
        "Segmentation metrics are returned only when ground truth genuinely exists; "
        "otherwise `metrics.available` is false with a reason."
    ),
)

# CORS is restricted to the dev frontend origins by default; override with
# ALLOWED_ORIGINS for deployment. Wildcards are never used with credentials.
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.time()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        raise
    ms = int((time.time() - t0) * 1000)
    response.headers["X-Process-Time-ms"] = str(ms)
    if ms > 2000:
        log.warning("slow request %s %s took %dms", request.method, request.url.path, ms)
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    """Never leak a stack trace to the client."""
    log.exception("unhandled: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "The server could not complete this operation.",
            "hint": "Check the server log for the full traceback.",
        },
    )


app.include_router(router, prefix="/api")


@app.get("/", tags=["system"])
def index():
    return {"name": "Route Resilience AI", "version": VERSION, "docs": "/docs"}
