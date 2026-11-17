/**
 * Pipeline state.
 *
 * One reducer owns the whole chain, because the stages are genuinely dependent:
 * changing the imagery must invalidate the mask, the graph, the criticality and
 * any scenario derived from them. Letting each view hold its own copy is how a
 * dashboard ends up showing a criticality map computed from a mask that no
 * longer exists.
 */
import { useCallback, useReducer, useRef } from "react";
import { ApiError, api } from "@/services/api";
import type {
  CriticalityResponse, ExtractResponse, GraphResponse, Health,
  ImageryResponse, RouteResponse, ScenarioKind, SimulateResponse, GroundTruthResponse } from "@/types/api";

export interface PipelineState {
  health: Health | null;
  imagery: ImageryResponse | null;
  extract: ExtractResponse | null;
  graph: GraphResponse | null;
  crit: CriticalityResponse | null;
  sim: SimulateResponse | null;
  route: RouteResponse | null;
  busy: string | null;
  error: { title: string; detail?: string; hint?: string } | null;
}

const init: PipelineState = {
  health: null, imagery: null, extract: null, graph: null,
  crit: null, sim: null, route: null, busy: null, error: null,
};

type Action =
  | { t: "busy"; v: string | null }
  | { t: "error"; v: PipelineState["error"] }
  | { t: "health"; v: Health }
  | { t: "imagery"; v: ImageryResponse }
  | { t: "gt"; v: GroundTruthResponse }
  | { t: "extract"; v: ExtractResponse }
  | { t: "graph"; v: GraphResponse }
  | { t: "crit"; v: CriticalityResponse }
  | { t: "sim"; v: SimulateResponse | null }
  | { t: "route"; v: RouteResponse | null };

function reducer(s: PipelineState, a: Action): PipelineState {
  switch (a.t) {
    case "busy":  return { ...s, busy: a.v, error: a.v ? null : s.error };
    case "error": return { ...s, error: a.v, busy: null };
    case "health": return { ...s, health: a.v };
    // Each stage clears everything downstream of it.
    case "imagery": return { ...init, health: s.health, imagery: a.v };
    // Attaching ground truth keeps everything: it only adds a layer and, if
    // roads were already extracted, fills in the metrics computed against it.
    case "gt": return {
      ...s,
      imagery: s.imagery && { ...s.imagery, layers: { ...s.imagery.layers, ...a.v.layers } },
      extract: s.extract && a.v.metrics ? { ...s.extract, metrics: a.v.metrics } : s.extract,
    };
    case "extract": return { ...s, extract: a.v, graph: null, crit: null, sim: null, route: null };
    case "graph":   return { ...s, graph: a.v, crit: null, sim: null, route: null };
    case "crit":    return { ...s, crit: a.v };
    case "sim":     return { ...s, sim: a.v };
    case "route":   return { ...s, route: a.v };
  }
}

export function usePipeline() {
  const [state, dispatch] = useReducer(reducer, init);

  const run = useCallback(async <T,>(label: string, fn: () => Promise<T>,
                                     ok: (v: T) => Action) => {
    dispatch({ t: "busy", v: label });
    try {
      const v = await fn();
      dispatch(ok(v));
      return v;
    } catch (e) {
      const err = e as ApiError;
      dispatch({ t: "error", v: {
        title: label + " failed", detail: err.message, hint: err.hint } });
      return null;
    } finally {
      dispatch({ t: "busy", v: null });
    }
  }, []);

  // The live session id. A ref, not the render-time value: multi-step flows
  // (guided demo, one-click analysis) call several actions inside one async
  // function, and a closure over `state` would still hold the PREVIOUS session
  // right after a new image is loaded.
  const sidRef = useRef<string | undefined>(undefined);
  if (state.imagery?.session_id) sidRef.current = state.imagery.session_id;
  const withSid = <T,>(fn: (sid: string) => Promise<T> | null): Promise<T> | null =>
    sidRef.current ? fn(sidRef.current) : null;
  const newSession = <T extends { session_id: string }>(v: T | null) => {
    if (v) sidRef.current = v.session_id;
    return v;
  };

  return {
    state,
    checkHealth: () => run("Health check", api.health, (v) => ({ t: "health", v })),

    loadDemo: async (city: string, size: number, seed: number) =>
      newSession(await run("Loading demo city",
                           () => api.loadDemo({ city, size, seed, occlusion: true }),
                           (v) => ({ t: "imagery", v }))),

    upload: async (f: File, pixel_size_m?: number) =>
      newSession(await run("Uploading imagery", () => api.upload(f, pixel_size_m),
                           (v) => ({ t: "imagery", v }))),

    uploadGroundTruth: (f: File) =>
      withSid((sid) => run("Attaching ground truth", () => api.uploadGroundTruth(sid, f),
                           (v) => ({ t: "gt", v }))),

    extract: (mode?: "auto" | "demo" | "classical", model?: string | null) =>
      withSid((sid) => run("Road extraction",
                           () => api.extract({ session_id: sid, mode: mode ?? null,
                                               model: model ?? null }),
                           (v) => ({ t: "extract", v }))),

    buildGraph: (max_gap_m: number, restore_loops: boolean) =>
      withSid((sid) => run("Graph construction",
                           () => api.buildGraph({ session_id: sid, max_gap_m, restore_loops }),
                           (v) => ({ t: "graph", v }))),

    criticality: (top_percent: number) =>
      withSid((sid) => run("Criticality analysis",
                           () => api.criticality({ session_id: sid, top_percent }),
                           (v) => ({ t: "crit", v }))),

    simulate: (p: { kind: ScenarioKind; nodes?: number[]; edges?: number[][];
                    count?: number; center?: number[]; radius_px?: number; seed?: number }) =>
      withSid((sid) => run("Failure simulation",
                           () => api.simulate({ session_id: sid, ...p }),
                           (v) => ({ t: "sim", v }))),

    route: (source: number, target: number, disabled_nodes: number[] = []) =>
      withSid((sid) => run("Route computation",
                           () => api.route({ session_id: sid, source, target, disabled_nodes }),
                           (v) => ({ t: "route", v }))),

    /** Trip analysis: kept out of shared state; the planner holds the result. */
    tripWhatIf: (source: number, target: number) =>
      withSid((sid) => run("Trip analysis",
                           () => api.tripWhatIf({ session_id: sid, source, target, top: 5 }),
                           () => ({ t: "busy", v: null }))),

    clearSim: () => { dispatch({ t: "sim", v: null }); dispatch({ t: "route", v: null }); },
  };
}
