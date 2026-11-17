/**
 * Route planner: one image in, a resilience-aware trip out.
 *
 * 1. Analyse: upload one image (optionally its ground-truth mask). The whole
 *    pipeline runs by itself - extraction, graph + healing, criticality.
 * 2. Click A, then B, on the extracted network: shortest route.
 * 3. "What if roads fail?": every junction on the route is closed in turn; the
 *    one that hurts this trip most is shown with the alternate route, or the
 *    trip is reported as cut off.
 *
 * Every number comes from the backend for the image that was just analysed.
 */
import { useEffect, useRef, useState } from "react";
import {
  CheckCircle2, Circle, Loader2, MapPin, Play, RotateCcw, ShieldAlert, Upload,
} from "lucide-react";
import clsx from "clsx";
import NetworkCanvas from "@/components/NetworkCanvas";
import ModelPicker from "@/components/ModelPicker";
import { Chip, Empty, Metric, Note, Panel } from "@/components/ui";
import type { TripWhatIfResponse } from "@/types/api";

type StepId = "upload" | "gt" | "extract" | "graph" | "crit";
const STEP_LABEL: Record<StepId, string> = {
  upload: "Load image",
  gt: "Attach ground truth",
  extract: "Extract roads",
  graph: "Build + heal road graph",
  crit: "Find critical junctions",
};

type StepState = Partial<Record<StepId, "run" | "ok" | "fail" | "skip">>;

// The analysis keeps running when the user switches tabs, but this component
// is unmounted and mounted again. Its progress therefore lives here, outside
// the component, so a remounted panel still shows it and still blocks a second
// "Analyse" - two overlapping runs would otherwise mix up their sessions
// ("Run /extract-roads before building the graph").
const RUN = {
  id: 0,
  busy: false,
  steps: {} as StepState,
  subs: new Set<() => void>(),
};
const emit = () => RUN.subs.forEach((f) => f());

export default function RoutePlannerSection({ p }: any) {
  const { state } = p;
  const imgRef = useRef<HTMLInputElement>(null);
  const gtRef = useRef<HTMLInputElement>(null);
  const [img, setImg] = useState<File | null>(null);
  const [gt, setGt] = useState<File | null>(null);
  const [gsd, setGsd] = useState(0.5);
  const [model, setModel] = useState<string | null>(null);
  const [steps, setStepsLocal] = useState<StepState>(RUN.steps);
  const [running, setRunningLocal] = useState(RUN.busy);
  useEffect(() => {
    const f = () => { setStepsLocal(RUN.steps); setRunningLocal(RUN.busy); };
    RUN.subs.add(f);
    f();
    return () => { RUN.subs.delete(f); };
  }, []);
  const setSteps = (u: StepState | ((s: StepState) => StepState)) => {
    RUN.steps = typeof u === "function" ? u(RUN.steps) : u;
    emit();
  };
  const setRunning = (v: boolean) => { RUN.busy = v; emit(); };

  const [a, setA] = useState<number | null>(null);
  const [b, setB] = useState<number | null>(null);
  const [trip, setTrip] = useState<TripWhatIfResponse | null>(null);
  const [showFail, setShowFail] = useState(false);

  // A new network invalidates any trip drawn on the old one.
  useEffect(() => { setA(null); setB(null); setTrip(null); setShowFail(false); },
            [state.graph]);

  const mark = (id: StepId, v: "run" | "ok" | "fail" | "skip") =>
    setSteps((s) => ({ ...s, [id]: v }));

  async function analyse(kind: "upload" | "demo") {
    if (RUN.busy) return;
    const me = ++RUN.id;
    setRunning(true);
    setSteps({});
    const order: StepId[] = ["upload", "gt", "extract", "graph", "crit"];
    const ok = async (id: StepId, fn: () => Promise<any>) => {
      if (me !== RUN.id) return false;          // a newer run took over
      mark(id, "run");
      const r = await fn();
      if (me !== RUN.id) return false;
      mark(id, r ? "ok" : "fail");
      return !!r;
    };
    try {
      if (kind === "upload") {
        if (!img) return;
        if (!await ok("upload", () => p.upload(img, gsd))) return;
        if (gt) { if (!await ok("gt", () => p.uploadGroundTruth(gt))) return; }
        else mark("gt", "skip");
        if (!await ok("extract", () => p.extract("auto", model))) return;
        // Healing reach scales with resolution: ~60 px, kept within 15-140 m.
        const gap = Math.min(140, Math.max(15, Math.round(60 * gsd)));
        if (!await ok("graph", () => p.buildGraph(gap, true))) return;
      } else {
        if (!await ok("upload", () => p.loadDemo("grid_river", 768, 1))) return;
        mark("gt", "skip");
        if (!await ok("extract", () => p.extract("demo"))) return;
        if (!await ok("graph", () => p.buildGraph(140, true))) return;
      }
      await ok("crit", () => p.criticality(10));
    } finally {
      // steps never reached stay empty
      if (me === RUN.id) {
        setSteps((s) => Object.fromEntries(order.filter((k) => s[k]).map((k) => [k, s[k]])));
        setRunning(false);
      }
    }
  }

  async function pick(n: number) {
    if (running) return;
    if (a === null || (a !== null && b !== null)) {
      setA(n); setB(null); setTrip(null); setShowFail(false);
      return;
    }
    if (n === a) return;
    setB(n);
    setShowFail(false);
    const r = await p.tripWhatIf(a, n);
    setTrip(r ?? null);
  }

  const reset = () => { setA(null); setB(null); setTrip(null); setShowFail(false); };
  const o = trip?.original;
  const w = trip?.worst;
  const alt = trip?.alternate;
  const markers = [
    ...(a !== null ? [{ node: a, label: "A", color: "#34D399" }] : []),
    ...(b !== null ? [{ node: b, label: "B", color: "#F472B6" }] : []),
  ];

  return (
    <div className="flex-1 grid grid-cols-12 gap-3 min-h-0">
      <Panel className="col-span-12 lg:col-span-8" title="Road network"
        actions={state.graph && (
          <div className="flex items-center gap-2">
            <Chip tone="accent">{state.graph.stats.nodes} junctions</Chip>
            <span className="text-[11px] text-ink-500">
              {a === null ? "Click a junction to set A"
                : b === null ? "Now click a junction to set B"
                : "Click again to start a new trip"}
            </span>
          </div>
        )}>
        {state.graph ? (
          <NetworkCanvas
            className="w-full h-full min-h-[460px]"
            geojson={state.graph.geojson}
            scores={state.crit?.scores}
            gatekeepers={state.crit?.gatekeepers}
            onSelectNode={pick}
            markers={markers}
            routeOriginal={o?.reachable ? o.path : null}
            routeAlternate={showFail && alt?.reachable ? alt.path : null}
            disabledNodes={showFail && w ? [w.node] : []}
            backdrop={state.imagery?.layers?.original ?? null}
            backdropExtent={state.imagery ? { width: state.imagery.meta.width,
                                              height: state.imagery.meta.height } : null}
            backdropOpacity={0.5}
          />
        ) : (
          <Empty icon={MapPin} title="No road network yet"
                 hint="Upload one satellite image on the right and press Analyse. Everything else runs automatically." />
        )}
      </Panel>

      <div className="col-span-12 lg:col-span-4 flex flex-col gap-3 overflow-auto">
        <Panel title="1 · Analyse an image" dense className="shrink-0">
          <div className="space-y-2.5">
            <input ref={imgRef} type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" className="hidden"
                   onChange={(e) => setImg(e.target.files?.[0] ?? null)} />
            <input ref={gtRef} type="file" accept=".png,.tif,.tiff,.jpg,.jpeg" className="hidden"
                   onChange={(e) => setGt(e.target.files?.[0] ?? null)} />
            <button className="btn-ghost w-full justify-between" disabled={running}
                    onClick={() => imgRef.current?.click()}>
              <span className="flex items-center gap-2"><Upload size={14} /> Satellite image</span>
              <span className="text-[11px] text-ink-500 truncate max-w-[55%]">{img?.name ?? "choose…"}</span>
            </button>
            <button className="btn-ghost w-full justify-between" disabled={running}
                    onClick={() => gtRef.current?.click()}>
              <span className="flex items-center gap-2"><Upload size={14} /> Ground-truth mask</span>
              <span className="text-[11px] text-ink-500 truncate max-w-[55%]">{gt?.name ?? "optional"}</span>
            </button>
            <ModelPicker value={model} disabled={running}
                         onChange={(id, m) => { setModel(id); if (m) setGsd(m.input_gsd_m); }} />
            <label className="flex items-center justify-between text-[11px] text-ink-300">
              Pixel size (m) for plain images
              <input type="number" min={0.1} max={30} step={0.1} value={gsd}
                     onChange={(e) => setGsd(+e.target.value || 0.5)}
                     className="input w-20 text-right" />
            </label>
            <p className="text-[10px] text-ink-500 leading-relaxed">
              DeepGlobe is 0.5 m; the LISS-IV and Sentinel-2 demo tiles are 2.5 m. A GeoTIFF's own pixel size is used automatically.
            </p>
            <button className="btn-primary w-full justify-center" disabled={!img || running}
                    onClick={() => analyse("upload")}>
              <Play size={14} /> Analyse
            </button>
            <button className="btn-ghost w-full justify-center text-[12px]" disabled={running}
                    onClick={() => analyse("demo")}>
              Or use the demo city (river with two bridges)
            </button>

            {Object.keys(steps).length > 0 && (
              <ol className="space-y-1 pt-1">
                {(Object.keys(STEP_LABEL) as StepId[]).filter((k) => steps[k]).map((k) => (
                  <li key={k} className="flex items-center gap-2 text-[12px]">
                    {steps[k] === "run" ? <Loader2 size={13} className="animate-spin text-accent" />
                     : steps[k] === "ok" ? <CheckCircle2 size={13} className="text-risk-low" />
                     : steps[k] === "fail" ? <ShieldAlert size={13} className="text-risk-critical" />
                     : <Circle size={13} className="text-ink-500" />}
                    <span className={clsx(steps[k] === "skip" ? "text-ink-500" : "text-ink-100")}>
                      {STEP_LABEL[k]}{steps[k] === "skip" ? " (skipped)" : ""}
                    </span>
                  </li>
                ))}
              </ol>
            )}
            {steps.extract === "run" && (
              <p className="text-[10px] text-ink-500">
                On a laptop CPU the trained model takes 20–60 s per image.
              </p>
            )}
          </div>
        </Panel>

        <Panel title="2 · Trip A → B" dense className="shrink-0"
          actions={(a !== null || b !== null) && (
            <button className="btn-ghost text-xs" onClick={reset}><RotateCcw size={12} /> Reset</button>
          )}>
          {!state.graph ? (
            <p className="text-[12px] text-ink-500">Analyse an image first.</p>
          ) : !o ? (
            <p className="text-[12px] text-ink-300">
              {a === null ? "Click a junction on the map to set A." : "Click a second junction to set B."}
            </p>
          ) : (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <Metric size="sm" label="Distance"
                        value={o.distance_m != null ? (o.distance_m / 1000).toFixed(2) : null} unit="km" />
                <Metric size="sm" label="Time" value={o.travel_time_min} unit="min"
                        hint="Assumes 30 km/h on every road" />
              </div>
              <p className="text-[11px] text-ink-500">
                Blue: shortest route on the roads extracted from this image.
              </p>
            </div>
          )}
        </Panel>

        {o && (
          <Panel title="3 · What if roads fail?" dense className="shrink-0">
            <div className="space-y-3">
              <button className={clsx("w-full justify-center", showFail ? "btn-ghost" : "btn-primary")}
                      onClick={() => setShowFail((v) => !v)} disabled={!w}>
                <ShieldAlert size={14} /> {showFail ? "Show normal route" : "Close the most critical junction"}
              </button>

              {!w ? (
                <Note>The route has no intermediate junction to close.</Note>
              ) : showFail && (
                w.cuts_trip ? (
                  <Note tone="warn">
                    <strong>No alternate route: A and B are cut off.</strong> The junction marked ✕ is a
                    single point of failure for this trip
                    {w.isolated_junctions != null ? ` — closing it splits ${w.isolated_junctions} junctions off from the rest of the route's network` : ""}.
                  </Note>
                ) : (
                  <div className="grid grid-cols-2 gap-3">
                    <Metric size="sm" label="Alternate distance"
                            value={alt?.distance_m != null ? (alt.distance_m / 1000).toFixed(2) : null} unit="km" />
                    <Metric size="sm" label="Longer by" tone="warn"
                            value={w.distance_increase_pct != null ? `+${w.distance_increase_pct}` : null} unit="%" />
                  </div>
                )
              )}

              {trip && trip.critical.length > 0 && (
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-ink-500 mb-1.5">
                    Weakest junctions on this trip ({trip.junctions_checked} checked)
                  </div>
                  <ul className="space-y-1">
                    {trip.critical.map((j, i) => (
                      <li key={j.node} className="flex justify-between text-[12px]">
                        <span className="text-ink-300">#{i + 1} junction {j.node}</span>
                        {j.cuts_trip
                          ? <span className="text-risk-critical font-medium">
                              cuts the trip{j.isolated_junctions != null
                                ? ` · splits off ${j.isolated_junctions} junctions` : ""}
                            </span>
                          : <span className="text-ink-100 font-mono">+{j.distance_increase_pct}%</span>}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {trip && <p className="text-[10px] text-ink-500 leading-relaxed">{trip.note}</p>}
            </div>
          </Panel>
        )}
      </div>
    </div>
  );
}
