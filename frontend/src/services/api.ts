/**
 * API client.
 *
 * Every function here hits the real backend. There is deliberately no mock
 * layer and no cached sample payload: if the backend is down the UI shows an
 * error, it does not fall back to canned data that would look like a result.
 */
import type {
  CriticalityResponse,
  ExtractResponse,
  GraphResponse,
  Health,
  ImageryResponse,
  RouteResponse,
  ScenarioKind,
  SimulateResponse,
} from "@/types/api";

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly hint?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: init?.body instanceof FormData
        ? undefined
        : { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError(
      "Cannot reach the analysis server.",
      0,
      "Start the backend with `uvicorn backend.app.main:app --reload` on port 8000.",
    );
  }

  if (!res.ok) {
    let detail = res.statusText;
    let hint: string | undefined;
    try {
      const body = await res.json();
      detail = body.detail ?? body.error ?? detail;
      hint = body.hint;
    } catch {
      /* non-JSON error body; keep the status text */
    }
    throw new ApiError(String(detail), res.status, hint);
  }
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body: unknown) =>
  req<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => req<Health>("/health"),

  loadDemo: (p: { city: string; size: number; seed: number; occlusion: boolean }) =>
    post<ImageryResponse>("/imagery/demo", p),

  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<ImageryResponse>("/imagery/upload", { method: "POST", body: fd });
  },

  extract: (p: {
    session_id: string;
    mode?: "auto" | "demo" | "classical" | null;
    threshold?: number | null;
    road_width_m?: number;
  }) => post<ExtractResponse>("/extract-roads", p),

  buildGraph: (p: {
    session_id: string;
    max_gap_m: number;
    min_spur_m?: number;
    w_angle?: number;
    w_evidence?: number;
    restore_loops: boolean;
  }) => post<GraphResponse>("/build-graph", p),

  criticality: (p: { session_id: string; top_percent: number; pivots?: number | null }) =>
    post<CriticalityResponse>("/criticality", p),

  simulate: (p: {
    session_id: string;
    kind: ScenarioKind;
    nodes?: number[];
    edges?: number[][];
    center?: number[] | null;
    radius_px?: number | null;
    count?: number;
    seed?: number;
    od_pairs?: number;
  }) => post<SimulateResponse>("/simulate-failure", p),

  route: (p: {
    session_id: string;
    source: number;
    target: number;
    disabled_nodes?: number[];
    disabled_edges?: number[][];
    speed_kmh?: number;
  }) => post<RouteResponse>("/route", p),
};
