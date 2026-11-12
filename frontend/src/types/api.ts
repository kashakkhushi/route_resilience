/**
 * Types mirroring the backend Pydantic schemas.
 *
 * `Provenance` is the most important type in this file. Every view that renders
 * a road mask, a graph or a metric must display it, so the UI can never show a
 * result without stating what produced it.
 */

export type ExtractionMode =
  | "TRAINED_MODEL"
  | "CLASSICAL_BASELINE"
  | "DEMO_DATA"
  | "OSM_REFERENCE";

export interface Provenance {
  mode: ExtractionMode | string;
  label: string;
  description: string;
  /** True ONLY for a trained deep-learning checkpoint. */
  is_learned: boolean;
  checkpoint?: string | null;
  architecture?: string | null;
  metrics_available?: boolean;
}

export interface Health {
  status: "ok";
  version: string;
  extraction_mode: string;
  torch_available: boolean;
  checkpoint_found: string | null;
  geotiff_support: boolean;
  osm_ground_truth: boolean;
  notes: string;
  provenance?: Provenance | null;
}

export interface RasterMeta {
  width: number;
  height: number;
  bands: number;
  crs: string | null;
  pixel_size_m: number;
  bounds: number[] | null;
  dtype: string;
  source: string;
  is_georeferenced: boolean;
}

export interface ImageryResponse {
  session_id: string;
  meta: RasterMeta;
  /** layer name -> PNG data URI */
  layers: Record<string, string>;
  provenance?: Provenance | null;
}

/** Null fields mean "no ground truth" - they are never filled with guesses. */
export interface SegmentationMetrics {
  available: boolean;
  reason?: string | null;
  iou?: number | null;
  dice?: number | null;
  precision?: number | null;
  recall?: number | null;
  cldice?: number | null;
  relaxed_iou?: number | null;
  recall_visible?: number | null;
  recall_occluded?: number | null;
  occlusion_gap?: number | null;
}

export interface ExtractResponse {
  session_id: string;
  provenance: Provenance;
  threshold_used: number;
  road_pixel_fraction: number;
  layers: Record<string, string>;
  metrics: SegmentationMetrics;
  elapsed_ms: number;
}

export interface HealingReport {
  components_before: number;
  components_after: number;
  components_merged: number;
  bridges_added: number;
  /** Gaps inside cycles that a pure MST pass can never repair. */
  loops_restored: number;
  candidates_considered: number;
  candidates_rejected_as_cycle: number;
  healed_length_fraction: number;
  mean_bridge_cost: number;
}

export interface ConnectivityRatio {
  lcc_nodes_before: number;
  lcc_nodes_after: number;
  lcc_node_increase_pct: number;
  lcc_length_increase_pct: number;
  lcc_coverage_after: number;
}

export interface NetworkStats {
  nodes: number;
  edges: number;
  components: number;
  giant_component_fraction: number;
  total_length_km: number;
  healed_length_km: number;
  mean_degree: number;
  dead_ends: number;
  intersections: number;
  gatekeepers: number;
  bridge_edges: number;
  /** Meshedness: 0 = tree (no alternative routes), higher = more redundancy. */
  alpha_index: number;
  avg_shortest_path_m?: number | null;
}

export interface GraphResponse {
  session_id: string;
  stats: NetworkStats;
  healing: HealingReport;
  connectivity_ratio: ConnectivityRatio;
  geojson: GeoJSONCollection;
  provenance: Provenance;
  elapsed_ms: number;
}

export interface GeoJSONCollection {
  type: "FeatureCollection";
  crs_mode?: "pixel" | "wgs84";
  features: GeoFeature[];
}

export interface GeoFeature {
  type: "Feature";
  geometry:
    | { type: "LineString"; coordinates: [number, number][] }
    | { type: "Point"; coordinates: [number, number] };
  properties: Record<string, any>;
}

export interface NodeExplanation {
  node: number;
  criticality: number;
  betweenness: number;
  degree: number;
  is_gatekeeper: boolean;
  incident_length_m: number;
  od_pairs_affected_pct?: number | null;
  efficiency_loss_pct?: number | null;
  /** Generated from computed graph statistics only - never templated prose. */
  explanation: string;
}

export interface CriticalityResponse {
  session_id: string;
  /** False means betweenness was estimated from pivots, and the UI must say so. */
  exact: boolean;
  pivots: number | null;
  weight: string;
  top_nodes: NodeExplanation[];
  top_edges: Record<string, any>[];
  gatekeepers: number[];
  bridge_edges: number[][];
  scores: Record<string, number>;
  elapsed_ms: number;
}

export interface ResilienceBlock {
  /** PRIMARY index: Network Efficiency Resilience, bounded [0,1].
   *  1 = no routing capability lost, 0 = every sampled trip severed. */
  resilience_index: number;
  resilience_metric: string;
  resilience_formula: string;
  resilience_interpretation: string;
  /** Aliases of resilience_index, kept for older clients. */
  normalized_resilience: number;
  efficiency_ratio: number;
  grade: string;

  /** The brief's literal ASPL ratio. Reads 0 under any fragmentation. */
  resilience_index_literal_aspl: number;
  literal_aspl_is_degenerate: boolean;
  literal_aspl_note: string;

  /** Diagnostic only — unbounded and survivorship-biased. Never the index. */
  diagnostic_aspl_ratio_connected_only: number;
  diagnostic_is_resilience_score: boolean;
  diagnostic_survivorship_biased: boolean;
  diagnostic_note: string;

  disconnection_rate: number;
  aspl_baseline_m: number;
  aspl_perturbed_m: number | null;
  aspl_increase_pct: number | null;
  network_fully_connected_after: boolean;
  note: string;
}

export type ScenarioKind =
  | "node_failure"
  | "edge_closure"
  | "flood_radial"
  | "targeted_attack"
  | "random_failure";

export interface SimulateResponse {
  session_id: string;
  scenario: string;
  description: string;
  removed_nodes: number;
  removed_edges: number;
  /** Ids of what failed, so the map can mark them. */
  removed_node_ids: number[];
  removed_edge_ids: number[][];
  baseline: Record<string, any>;
  disrupted: Record<string, any>;
  delta: Record<string, any>;
  resilience: ResilienceBlock;
  affected_geojson: GeoJSONCollection;
  elapsed_ms: number;
}

export interface RouteLeg {
  reachable: boolean;
  path: number[] | null;
  distance_m: number | null;
  travel_time_min: number | null;
  geojson: GeoJSONCollection | null;
}

export interface RouteResponse {
  session_id: string;
  original: RouteLeg;
  disrupted: RouteLeg;
  distance_increase_pct: number | null;
  time_increase_pct: number | null;
  detour_ratio: number | null;
  speed_assumption_kmh: number;
  note: string;
}

export type Stage =
  | "idle"
  | "imagery"
  | "extracted"
  | "graph"
  | "criticality"
  | "simulated";
