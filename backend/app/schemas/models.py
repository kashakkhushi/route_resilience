"""Pydantic schemas. Every response that derives from a road mask carries
`provenance`, so the client can never render a result without knowing what
produced it."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ------------------------------------------------------------------ common
class Provenance(BaseModel):
    mode: str
    label: str
    description: str
    is_learned: bool
    checkpoint: str | None = None
    architecture: str | None = None
    metrics_available: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    extraction_mode: str
    torch_available: bool
    checkpoint_found: str | None
    geotiff_support: bool
    osm_ground_truth: bool
    checkpoint_metadata: dict | None = Field(
        None, description="Architecture, epoch and training-time validation "
                          "metrics when a trained checkpoint is active.")
    notes: str
    provenance: Provenance | None = None


class SessionRef(BaseModel):
    session_id: str = Field(..., description="Opaque handle for this study area")


# ------------------------------------------------------------------ imagery
class DemoCity(str, Enum):
    grid_river = "grid_river"
    dense_urban = "dense_urban"
    sparse_rural = "sparse_rural"


class LoadDemoRequest(BaseModel):
    city: DemoCity = DemoCity.grid_river
    size: int = Field(768, ge=256, le=1536)
    seed: int = Field(1, ge=0, le=9999)
    occlusion: bool = True


class RasterMeta(BaseModel):
    width: int
    height: int
    bands: int
    crs: str | None = None
    pixel_size_m: float
    bounds: list[float] | None = None
    dtype: str
    source: str
    is_georeferenced: bool


class ImageryResponse(SessionRef):
    meta: RasterMeta
    layers: dict[str, str] = Field(
        ..., description="layer name -> PNG data URI for display"
    )
    provenance: Provenance | None = None


# ------------------------------------------------------------------ extraction
class ExtractRequest(SessionRef):
    mode: Literal["auto", "demo", "classical"] | None = Field(
        None,
        description="auto = trained model if present, else classical baseline. "
                    "demo = deterministic simulated occluded segmentation "
                    "(demo sessions only). The mode is always reported back in "
                    "`provenance`, never inferred by the client.",
    )
    threshold: float | None = Field(None, ge=0.01, le=0.99)
    road_width_m: float = Field(8.0, gt=0, le=60)

    @field_validator("threshold")
    @classmethod
    def _t(cls, v):
        return v


class SegmentationMetrics(BaseModel):
    """Null when ground truth is unavailable - never fabricated."""

    available: bool
    reason: str | None = None
    iou: float | None = None
    dice: float | None = None
    precision: float | None = None
    recall: float | None = None
    cldice: float | None = None
    relaxed_iou: float | None = None
    recall_visible: float | None = None
    recall_occluded: float | None = None
    occlusion_gap: float | None = None


class ExtractResponse(SessionRef):
    provenance: Provenance
    threshold_used: float
    road_pixel_fraction: float
    layers: dict[str, str]
    metrics: SegmentationMetrics
    elapsed_ms: int


# ------------------------------------------------------------------ graph
class HealRequest(SessionRef):
    max_gap_m: float = Field(400.0, gt=0, le=5000)
    w_angle: float = Field(2.0, ge=0, le=10)
    w_evidence: float = Field(1.5, ge=0, le=10)
    restore_loops: bool = True
    min_spur_m: float = Field(60.0, ge=0, le=1000)


class HealingReport(BaseModel):
    components_before: int
    components_after: int
    components_merged: int
    bridges_added: int
    loops_restored: int
    candidates_considered: int
    candidates_rejected_as_cycle: int
    healed_length_fraction: float
    mean_bridge_cost: float


class ConnectivityRatio(BaseModel):
    lcc_nodes_before: int
    lcc_nodes_after: int
    lcc_node_increase_pct: float
    lcc_length_increase_pct: float
    lcc_coverage_after: float


class NetworkStats(BaseModel):
    nodes: int
    edges: int
    components: int
    giant_component_fraction: float
    total_length_km: float
    healed_length_km: float
    mean_degree: float
    dead_ends: int
    intersections: int
    gatekeepers: int
    bridge_edges: int
    alpha_index: float
    avg_shortest_path_m: float | None = None


class GraphResponse(SessionRef):
    stats: NetworkStats
    healing: HealingReport
    connectivity_ratio: ConnectivityRatio
    geojson: dict[str, Any]
    provenance: Provenance
    elapsed_ms: int


# ------------------------------------------------------------------ criticality
class CriticalityRequest(SessionRef):
    top_percent: float = Field(10.0, gt=0, le=100)
    pivots: int | None = Field(None, ge=16, le=4096,
                               description="null = exact Brandes")


class NodeExplanation(BaseModel):
    node: int
    criticality: float
    betweenness: float
    degree: int
    is_gatekeeper: bool
    incident_length_m: float
    od_pairs_affected_pct: float | None = None
    efficiency_loss_pct: float | None = None
    explanation: str


class CriticalityResponse(SessionRef):
    exact: bool
    pivots: int | None
    weight: str
    top_nodes: list[NodeExplanation]
    top_edges: list[dict[str, Any]]
    gatekeepers: list[int]
    bridge_edges: list[list[int]]
    scores: dict[str, float]
    elapsed_ms: int


# ------------------------------------------------------------------ simulation
class ScenarioKind(str, Enum):
    node_failure = "node_failure"
    edge_closure = "edge_closure"
    flood_radial = "flood_radial"
    targeted_attack = "targeted_attack"
    random_failure = "random_failure"


class SimulateRequest(SessionRef):
    kind: ScenarioKind
    nodes: list[int] = Field(default_factory=list)
    edges: list[list[int]] = Field(default_factory=list)
    center: list[float] | None = None
    radius_px: float | None = Field(None, gt=0)
    count: int = Field(5, ge=1, le=50)
    seed: int = 0
    od_pairs: int = Field(300, ge=20, le=2000)


class ResilienceBlock(BaseModel):
    """Resilience readings, with exactly one designated as THE index.

    `resilience_index` is Network Efficiency Resilience (NER): bounded in [0,1]
    by construction, 1 for an undamaged network, and monotonically
    non-increasing as more infrastructure is removed.

    The other two ratios are reported for completeness and are explicitly not
    resilience scores - one is degenerate under fragmentation, the other is
    survivorship-biased and can exceed 1.
    """

    # --- primary -----------------------------------------------------
    resilience_index: float = Field(
        ..., ge=0.0, le=1.0,
        description="PRIMARY. Network Efficiency Resilience, bounded [0,1].")
    resilience_metric: str = Field("NER", description="Identifier of the primary metric.")
    resilience_formula: str
    resilience_interpretation: str
    normalized_resilience: float = Field(..., ge=0.0, le=1.0,
                                         description="Alias of resilience_index.")
    efficiency_ratio: float = Field(..., ge=0.0, le=1.0,
                                    description="Alias of resilience_index.")
    grade: str

    # --- compliance: the brief's literal definition -------------------
    resilience_index_literal_aspl: float = Field(
        ..., description="ASPL_baseline / ASPL_perturbed, per the project brief. "
                         "Reads 0 whenever any OD pair disconnects.")
    literal_aspl_is_degenerate: bool = False
    literal_aspl_note: str

    # --- diagnostic: NOT a resilience score ---------------------------
    diagnostic_aspl_ratio_connected_only: float = Field(
        ..., description="Diagnostic only. Unbounded and survivorship-biased.")
    diagnostic_is_resilience_score: bool = Field(
        False, description="Always false. Present so clients cannot misread it.")
    diagnostic_survivorship_biased: bool = False
    diagnostic_note: str

    # --- supporting detail --------------------------------------------
    disconnection_rate: float
    aspl_baseline_m: float
    aspl_perturbed_m: float | None
    aspl_increase_pct: float | None
    network_fully_connected_after: bool
    note: str


class SimulateResponse(SessionRef):
    scenario: str
    description: str
    removed_nodes: int
    removed_edges: int
    removed_node_ids: list[int] = Field(
        default_factory=list,
        description="Ids of the intersections removed, so the client can render "
                    "exactly which ones failed.")
    removed_edge_ids: list[list[int]] = Field(default_factory=list)
    baseline: dict[str, Any]
    disrupted: dict[str, Any]
    delta: dict[str, Any]
    resilience: ResilienceBlock
    affected_geojson: dict[str, Any]
    elapsed_ms: int


# ------------------------------------------------------------------ routing
class RouteRequest(SessionRef):
    source: int
    target: int
    disabled_nodes: list[int] = Field(default_factory=list)
    disabled_edges: list[list[int]] = Field(default_factory=list)
    speed_kmh: float = Field(30.0, gt=0, le=150,
                             description="assumed uniform free-flow speed")


class RouteLeg(BaseModel):
    reachable: bool
    path: list[int] | None
    distance_m: float | None
    travel_time_min: float | None
    geojson: dict[str, Any] | None


class RouteResponse(SessionRef):
    original: RouteLeg
    disrupted: RouteLeg
    distance_increase_pct: float | None
    time_increase_pct: float | None
    detour_ratio: float | None
    speed_assumption_kmh: float
    note: str


# ------------------------------------------------------------------ errors
class ErrorResponse(BaseModel):
    error: str
    detail: str
    hint: str | None = None
