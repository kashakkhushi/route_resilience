/**
 * Route Resilience AI - application shell.
 *
 * Landing gives the 30-second story; the command centre is the working surface.
 * Every number rendered here comes from a backend response. Nothing on screen is
 * a placeholder, and any value the backend could not compute renders as an em
 * dash with the backend's own reason attached.
 */
import { useEffect, useState } from "react";
import {
  Activity, BarChart3, ChevronRight, Crosshair,
  Image as ImageIcon, Layers, Network, Play, Satellite, ShieldAlert, Waypoints, Route, Map as MapIcon,
  BookOpen, CheckCircle2, Circle, Loader2, Plus, GitBranch, Ruler, AlertOctagon,
} from "lucide-react";
import clsx from "clsx";
import { usePipeline } from "@/hooks/usePipeline";
import NetworkCanvas from "@/components/NetworkCanvas";
import HeroScene, { type HeroAnchors } from "@/components/HeroScene";
import {
  Chip, Empty, ErrorNote, Metric, Note, Panel, fmt,
} from "@/components/ui";
import ImagerySection from "@/features/ImagerySection";
import ExtractionSection from "@/features/ExtractionSection";
import CriticalitySection from "@/features/CriticalitySection";
import StressTestSection from "@/features/StressTestSection";
import EvaluationSection from "@/features/EvaluationSection";
import { DemoRunner } from "@/features/DemoRunner";
import RoutePlannerSection from "@/features/RoutePlannerSection";
import TripPlannerSection from "@/features/TripPlannerSection";

type Tab = "overview" | "trip" | "planner" | "imagery" | "extraction" | "network"
         | "criticality" | "stress" | "evaluation";

const TABS: { id: Tab; label: string; icon: typeof Layers }[] = [
  { id: "overview",    label: "Overview",        icon: Layers },
  { id: "trip",        label: "Trip Planner (Map)", icon: MapIcon },
  { id: "planner",     label: "Image Route Planner", icon: Route },
  { id: "imagery",     label: "Imagery",         icon: ImageIcon },
  { id: "extraction",  label: "Road Extraction", icon: Satellite },
  { id: "network",     label: "Network",         icon: Network },
  { id: "criticality", label: "Criticality",     icon: Crosshair },
  { id: "stress",      label: "Stress Test",     icon: ShieldAlert },
  { id: "evaluation",  label: "Evaluation",      icon: BarChart3 },
];

export default function App() {
  const p = usePipeline();
  const { state } = p;
  const [launched, setLaunched] = useState(false);
  const [tab, setTab] = useState<Tab>("overview");
  const [selected, setSelected] = useState<number | null>(null);

  useEffect(() => { p.checkHealth(); }, []);
  // Remember the last result of each scenario kind, so the overview can show
  // random damage next to targeted failure (both are real backend results).
  useEffect(() => { for (const k of Object.keys(SIM_HISTORY)) delete SIM_HISTORY[k]; }, [state.graph]);
  useEffect(() => {
    if (state.sim) { const k = String(state.sim.scenario); delete SIM_HISTORY[k]; SIM_HISTORY[k] = state.sim.resilience; }
  }, [state.sim]);

  if (!launched) {
    return <Landing health={state.health}
                    onLaunch={(t) => { if (t) setTab(t); setLaunched(true); }} />;
  }

  const stages = stageStatus(state);
  const nav = (ids: Tab[]) => TABS.filter((t) => ids.includes(t.id)).map(({ id, label, icon: Icon }) => {
    const st = stages.byTab[id];
    const on = tab === id;
    return (
      <button key={id} onClick={() => setTab(id)}
        className={clsx(
          "group relative w-full flex items-center gap-3 pl-3 pr-3 py-2.5 rounded-xl text-[14.5px] font-medium transition-all",
          on ? "bg-gradient-to-r from-accent/20 via-accent/8 to-transparent text-white"
             : "text-ink-300 hover:bg-base-700/70 hover:text-white",
        )}>
        {on && <span className="absolute left-0 top-2 bottom-2 w-1 rounded-r-full bg-accent shadow-[0_0_10px_rgba(34,211,238,0.9)]" />}
        <span className={clsx("grid h-8 w-8 shrink-0 place-items-center rounded-lg border transition-colors",
          on ? "border-accent/40 bg-accent/15 text-accent shadow-[0_0_12px_rgba(34,211,238,0.3)]"
             : "border-line-soft bg-base-700/60 text-ink-300 group-hover:text-accent group-hover:border-accent/30")}>
          <Icon size={16} />
        </span>
        <span className="flex-1 text-left">{label}</span>
        {st === "done" && <CheckCircle2 size={16} className="text-risk-low" />}
        {st === "busy" && <Loader2 size={16} className="text-risk-med animate-spin" />}
        {st === "todo" && <Circle size={14} className="text-base-600" />}
      </button>
    );
  });

  return (
    <div className="h-screen flex flex-col">
      <TopBar state={state} onNew={() => setTab("imagery")} />

      <div className="flex-1 flex min-h-0">
        {/* ---------------------------------------------------- sidebar */}
        <nav className="w-[17rem] shrink-0 border-r border-line-soft bg-gradient-to-b from-base-800/80 to-base-900/80 backdrop-blur flex flex-col">
          <div className="px-3 pt-5 space-y-1">
            <div className="label px-3 pb-1.5">Workspace</div>
            {nav(["overview", "trip", "planner"])}
          </div>
          <div className="px-3 pt-6 space-y-1">
            <div className="label px-3 pb-1.5">Analysis pipeline</div>
            {nav(["imagery", "extraction", "network", "criticality", "stress", "evaluation"])}
          </div>

          <div className="mt-auto p-3">
            <div className="rounded-xl border border-line-soft bg-base-700/50 p-3.5">
              <div className="flex items-center justify-between text-[12px] text-ink-300">
                <span className="font-medium">Pipeline progress</span>
                <span>{stages.done} / {stages.total} stages</span>
              </div>
              <div className="mt-1 text-[30px] leading-tight font-bold text-white metric">
                {Math.round((stages.done / stages.total) * 100)}<span className="text-sm text-ink-500">%</span>
              </div>
              <div className="text-[12.5px] text-ink-300 mt-0.5 truncate">
                {state.busy ? <span className="text-risk-med">{state.busy}...</span>
                  : stages.done === stages.total ? "All stages complete"
                  : stages.done === 0 ? "Load a study area to begin" : `Next: ${stages.next}`}
              </div>
              <div className="mt-3 grid grid-cols-5 gap-1.5">
                {stages.list.map(([l, s]) => (
                  <div key={l} title={l} className={clsx("h-1.5 rounded-full",
                    s === "done" ? "bg-accent shadow-[0_0_6px_rgba(34,211,238,0.6)]"
                    : s === "busy" ? "bg-risk-med animate-pulse" : "bg-base-600")} />
                ))}
              </div>
            </div>
          </div>
        </nav>

        {/* ------------------------------------------------------ centre */}
        <main className="flex-1 min-w-0 flex flex-col gap-3 p-4 overflow-auto">
          {state.error && (
            <ErrorNote title={state.error.title} detail={state.error.detail}
                       hint={state.error.hint} />
          )}

          {tab === "overview" && <OverviewSection p={{...p, onTab: setTab}}
                                                  selected={selected} onSelect={setSelected} />}
          {tab === "imagery" && <ImagerySection p={p} />}
          {tab === "extraction" && <ExtractionSection p={p} />}
          {tab === "network" && <NetworkSection p={p} selected={selected}
                                                onSelect={setSelected} />}
          {tab === "criticality" && <CriticalitySection p={p} selected={selected}
                                                        onSelect={setSelected} />}
          {tab === "stress" && <StressTestSection p={p} selected={selected}
                                                  onSelect={setSelected} />}
          {tab === "evaluation" && <EvaluationSection p={p} />}
          {/* The two planners stay mounted when you switch tabs, so a planned trip,
              the picked A/B and any closures are still there when you come back. */}
          <div className={tab === "planner" ? "contents" : "hidden"}><RoutePlannerSection p={p} /></div>
          <div className={tab === "trip" ? "contents" : "hidden"}><TripPlannerSection /></div>
        </main>
      </div>
    </div>
  );
}

/** Last resilience result per scenario kind (cleared when the graph changes). */
const SIM_HISTORY: Record<string, any> = {};

type StageState = "done" | "busy" | "todo";
function stageStatus(state: any) {
  const list: [string, boolean, Tab][] = [
    ["Imagery", !!state.imagery, "imagery"],
    ["Extraction", !!state.extract, "extraction"],
    ["Graph + healing", !!state.graph, "network"],
    ["Criticality", !!state.crit, "criticality"],
    ["Simulation", !!state.sim, "stress"],
  ];
  const firstTodo = list.findIndex(([, d]) => !d);
  const withState: [string, StageState][] = list.map(([l, d], i) =>
    [l, d ? "done" : state.busy && i === firstTodo ? "busy" : "todo"]);
  const byTab: Partial<Record<Tab, StageState>> = {};
  list.forEach(([, , t], i) => { byTab[t] = withState[i][1]; });
  const ev = state.extract?.metrics?.available;
  byTab.evaluation = ev ? "done" : "todo";
  return {
    list: withState, byTab,
    done: list.filter(([, d]) => d).length, total: list.length,
    next: firstTodo >= 0 ? list[firstTodo][0] : "",
  };
}

/* ------------------------------------------------------------- landing */
// Held-out test results of the deployed model (DeepGlobe test split, 498
// tiles, threshold 0.5, no test-time augmentation). Source: the model's
// evaluation_report.md; see README "Results". Update here if the model changes.
const FINAL_RESULTS = {
  model: "SegFormer-B3 (fine-tuned)",
  iou: "0.669", dice: "0.792", cldice: "0.868",
  occludedRecall: "0.55\u21920.89",
};

function Landing({ health, onLaunch }: { health: any; onLaunch: (t?: Tab) => void }) {
  const arch = health?.checkpoint_metadata?.architecture;
  const [anchors, setAnchors] = useState<HeroAnchors>({ map: null });
  const hi = "text-cyan-200 font-semibold font-mono";
  const glance: JSX.Element[] = [
    <>Finds the roads in satellite images, including roads hidden under{" "}
      <b className="text-white font-semibold">trees, shadows and vehicles</b>.</>,
    <>Repairs broken roads into a <b className="text-white font-semibold">routable network</b>, finds the
      junctions a city cannot afford to lose, and simulates floods, closures and failures.</>,
    <>Final model <b className="text-white font-semibold">{FINAL_RESULTS.model}</b>: test IoU{" "}
      <span className={hi}>{FINAL_RESULTS.iou}</span> &middot; Dice{" "}
      <span className={hi}>{FINAL_RESULTS.dice}</span> &middot; clDice{" "}
      <span className={hi}>{FINAL_RESULTS.cldice}</span> on 498 held-out DeepGlobe tiles.</>,
    <>Occlusion training: hidden-road recall under synthetic clutter{" "}
      <span className={hi}>0.55</span> &rarr; <span className={hi}>0.89</span>.</>,
    <>Tested on Indian satellites: <b className="text-white font-semibold">Sentinel-2</b> and{" "}
      <b className="text-white font-semibold">ResourceSat-2A LISS-IV</b> over Hyderabad.</>,
  ];
  const m = anchors.map;
  return (
    <div className="relative min-h-screen overflow-hidden text-slate-300
                    bg-[radial-gradient(ellipse_at_25%_55%,#0B2846_0%,#06182D_35%,#030B17_70%,#02060E_100%)]">
      {/* nebula haze on the right */}
      <div className="absolute inset-0 pointer-events-none
                      bg-[radial-gradient(ellipse_at_80%_20%,rgba(59,130,246,0.10),transparent_45%),radial-gradient(ellipse_at_95%_80%,rgba(34,211,238,0.07),transparent_40%)]" />
      <HeroScene className="absolute inset-0 w-full h-full" onLayout={setAnchors} />

      {m && (<>
        <div className="absolute z-10 pointer-events-none rounded-lg border border-cyan-300/30 bg-[#06182D]/80
                        backdrop-blur px-3 py-2 shadow-[0_0_20px_rgba(34,211,238,0.15)]"
             style={{ left: m.x - 10, top: m.y + m.h * 0.04 }}>
          <div className="text-[10px] font-mono tracking-[0.18em] text-cyan-300/80">FOCUS REGION</div>
          <div className="text-[13px] text-white">India &middot; road network</div>
        </div>
        <div className="absolute z-10 pointer-events-none rounded-lg border border-cyan-300/30 bg-[#06182D]/80
                        backdrop-blur px-3 py-2 shadow-[0_0_20px_rgba(34,211,238,0.15)]"
             style={{ left: m.x - 10, top: m.y + m.h * 0.84 }}>
          <div className="text-[10px] font-mono tracking-[0.18em] text-cyan-300/80">ORBIT</div>
          <div className="text-[13px] text-white">Sentinel-2 &middot; LISS-IV</div>
        </div>
      </>)}

      <div className="relative z-10 min-h-screen flex flex-col justify-center px-6 sm:px-12 lg:px-14 py-10
                      pointer-events-none [&>*]:pointer-events-auto">
        <div className="max-w-[47rem] self-start">
          <div className="flex items-center gap-4">
            <div className="grid h-16 w-16 shrink-0 place-items-center rounded-2xl border border-cyan-300/40
                            bg-cyan-300/10 backdrop-blur shadow-[0_0_30px_rgba(34,211,238,0.4)]">
              <Waypoints size={34} className="text-cyan-300" />
            </div>
            <h1 className="whitespace-nowrap text-5xl sm:text-[4.9rem] font-bold leading-[1.02] tracking-tight text-white
                           [text-shadow:0_2px_4px_rgba(0,0,0,0.8),0_0_24px_rgba(2,8,20,0.9)]">
              Route Resilience{" "}
              <span className="bg-gradient-to-r from-cyan-300 to-teal-300 bg-clip-text text-transparent [text-shadow:none] drop-shadow-[0_0_16px_rgba(34,211,238,0.45)]">AI</span>
            </h1>
          </div>
          <p className="mt-3 text-[1.38rem] leading-relaxed text-white [text-shadow:0_1px_3px_rgba(0,0,0,0.95),0_0_14px_rgba(2,8,20,0.95)]">
            Seeing roads through trees and shadows, and finding the junctions a city
            cannot afford to lose.
          </p>

          <div className="mt-6 rounded-2xl border border-white/10 bg-[#0B1E33]/50 backdrop-blur-[3px]
                          shadow-[0_20px_50px_rgba(0,0,0,0.35)] px-7 py-5">
            <div className="flex items-center gap-3">
              <BookOpen size={26} className="text-cyan-200" />
              <h2 className="text-[22px] font-semibold text-white">Project Glance:</h2>
            </div>
            <ul className="mt-3.5 space-y-2 pl-1">
              {glance.map((body, i) => (
                <li key={i} className="flex gap-3 text-[17.5px] leading-relaxed text-slate-100/90">
                  <span className="mt-[0.6em] h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-300" />
                  <span>{body}</span>
                </li>
              ))}
            </ul>
            <div className="mt-5 flex flex-wrap items-center gap-3 text-[16px]">
              <span className="font-semibold text-white">Model status:</span>
              {health ? (
                <span className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-mono
                                  ${health.checkpoint_found
                                    ? "border-cyan-300/40 bg-cyan-300/5 text-cyan-100"
                                    : "border-amber-300/60 text-amber-200"}`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${health.checkpoint_found ? "bg-emerald-400" : "bg-amber-400"}`} />
                  {health.checkpoint_found
                    ? `Trained checkpoint${arch ? ` \u00b7 ${arch}` : ""}`
                    : "No checkpoint"}
                </span>
              ) : <span className="text-xs text-slate-400">connecting to backend...</span>}
            </div>
            {health && !health.checkpoint_found && (
              <div className="mt-3"><Note tone="warn">{health.notes}</Note></div>
            )}
            <p className="mt-3 text-[14.5px] leading-relaxed text-slate-400">
              Scores are on the DeepGlobe test split (0.5&nbsp;m imagery). Results on LISS-IV and
              Sentinel-2 are in View Results.
            </p>

            <div className="mt-5 flex flex-wrap gap-3">
              <button onClick={() => onLaunch("overview")}
                      className="inline-flex items-center gap-2 rounded-lg bg-gradient-to-r from-cyan-300 to-teal-300
                                 px-6 py-3.5 text-[15px] font-bold uppercase tracking-[0.1em] whitespace-nowrap text-slate-900
                                 shadow-[0_6px_20px_rgba(34,211,238,0.45)] hover:brightness-110 transition">
                Launch Dashboard <ChevronRight size={16} />
              </button>
              {([["trip", "Plan a Trip"], ["evaluation", "View Results"]] as [Tab, string][]).map(([k, label]) => (
                <button key={k} onClick={() => onLaunch(k)}
                        className="inline-flex items-center gap-2 rounded-lg border border-white/15 bg-white/[0.06]
                                   px-6 py-3.5 text-[15px] font-bold uppercase tracking-[0.1em] whitespace-nowrap text-white
                                   shadow-[0_6px_20px_rgba(0,0,0,0.35)] hover:border-cyan-300/60 hover:bg-cyan-300/10 transition">
                  {label} <ChevronRight size={16} />
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="absolute bottom-5 left-6 sm:left-12 lg:left-14 hidden lg:block text-xs text-slate-500">
          &#10022; Drag the globe to rotate it
        </div>
        <a className="absolute bottom-4 right-5 hidden lg:block text-[10px] text-slate-600 hover:text-slate-400"
           href="https://github.com/datameet/maps" target="_blank" rel="noreferrer">
          India boundary: DataMeet, CC BY 4.0
        </a>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------- topbar */
function TopBar({ state, onNew }: { state: any; onNew: () => void }) {
  const h = state.health;
  const pill = (ok: boolean, label: string) => (
    <span className={clsx("inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-[12px]",
      ok ? "border-risk-low/25 bg-risk-low/5 text-ink-300" : "border-risk-med/30 bg-risk-med/5 text-risk-med")}>
      <span className={clsx("h-1.5 w-1.5 rounded-full", ok ? "bg-risk-low" : "bg-risk-med")} />{label}
    </span>
  );
  return (
    <header className="h-16 shrink-0 border-b border-line-soft bg-base-800/70 backdrop-blur
                       flex items-center px-4 gap-4">
      <div className="flex items-center gap-2.5">
        <div className="grid h-8 w-8 place-items-center rounded-lg border border-accent/30 bg-accent/10 shadow-glow">
          <Waypoints size={17} className="text-accent" />
        </div>
        <span className="font-bold text-[17px] text-white">Route Resilience <span className="text-accent">AI</span></span>
        {h && <span className="rounded-md border border-line px-1.5 py-0.5 text-[10px] text-ink-500 font-mono">v{h.version}</span>}
      </div>

      {state.extract?.provenance && (
        <Chip tone={state.extract.provenance.is_learned ? "accent" : "warn"}>
          {state.extract.provenance.label}
        </Chip>
      )}

      <div className="ml-auto flex items-center gap-2">
        {state.busy && (
          <span className="mr-2 flex items-center gap-1.5 text-[11px] text-accent">
            <Activity size={12} className="animate-pulse" /> {state.busy}
          </span>
        )}
        {h && <>
          {pill(h.torch_available, h.torch_available ? "PyTorch" : "No PyTorch")}
          {pill(!!h.checkpoint_found, h.checkpoint_found ? "Checkpoint" : "No checkpoint")}
          {pill(h.geotiff_support, "GeoTIFF")}
        </>}
        <button onClick={onNew}
          className="ml-2 inline-flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-cyan-300 to-sky-400
                     px-3.5 py-1.5 text-[13px] font-semibold text-slate-900 shadow-glow hover:brightness-110">
          <Plus size={15} /> New analysis
        </button>
      </div>
    </header>
  );
}

/* ------------------------------------------------------------ overview */
const KPI_TONE = {
  cyan:   { text: "text-cyan-300",   ring: "from-cyan-400/60",   icon: "bg-cyan-400/10 text-cyan-300",     bar: "bg-cyan-400" },
  blue:   { text: "text-sky-300",    ring: "from-sky-400/60",    icon: "bg-sky-400/10 text-sky-300",       bar: "bg-sky-400" },
  violet: { text: "text-violet-300", ring: "from-violet-400/60", icon: "bg-violet-400/10 text-violet-300", bar: "bg-violet-400" },
  red:    { text: "text-rose-300",   ring: "from-rose-400/70",   icon: "bg-rose-400/10 text-rose-300",     bar: "bg-rose-400" },
  amber:  { text: "text-amber-300",  ring: "from-amber-400/70",  icon: "bg-amber-400/10 text-amber-300",   bar: "bg-amber-400" },
  green:  { text: "text-emerald-300", ring: "from-emerald-400/70", icon: "bg-emerald-400/10 text-emerald-300", bar: "bg-emerald-400" },
};
const BADGE = {
  good: "bg-emerald-400/10 text-emerald-300 border-emerald-400/25",
  warn: "bg-amber-400/10 text-amber-300 border-amber-400/25",
  bad:  "bg-rose-400/10 text-rose-300 border-rose-400/25",
  info: "bg-cyan-400/10 text-cyan-200 border-cyan-400/20",
};

function Kpi({ icon: Icon, label, value, unit, badge, tone, meter, hint }: {
  icon: any; label: string; value: string | number | null; unit?: string;
  badge?: [string, keyof typeof BADGE] | null; tone: keyof typeof KPI_TONE;
  meter?: number | null; hint?: string;
}) {
  const t = KPI_TONE[tone];
  return (
    <div className="relative panel p-4 overflow-hidden" title={hint}>
      <div className={clsx("absolute inset-x-0 top-0 h-px bg-gradient-to-r to-transparent", t.ring)} />
      <div className="flex items-center gap-2 text-[13px] font-medium text-ink-300">
        <span className={clsx("grid h-7 w-7 place-items-center rounded-md", t.icon)}><Icon size={13} /></span>
        {label}
      </div>
      <div className="mt-2 flex items-baseline gap-1.5">
        <span className={clsx("text-[28px] leading-none font-bold metric", value == null ? "text-ink-500" : "text-white")}>
          {value ?? "—"}
        </span>
        {value != null && unit && <span className="text-sm text-ink-500">{unit}</span>}
      </div>
      <div className="mt-3 flex items-center gap-2 min-h-[22px]">
        {badge && <span className={clsx("rounded-md border px-2 py-0.5 text-[11.5px] font-medium whitespace-nowrap", BADGE[badge[1]])}>{badge[0]}</span>}
        {meter != null && (
          <div className="ml-auto h-1.5 w-16 rounded-full bg-base-600 overflow-hidden">
            <div className={clsx("h-full rounded-full", t.bar)} style={{ width: `${Math.max(3, Math.min(100, meter * 100))}%` }} />
          </div>
        )}
      </div>
    </div>
  );
}

function Gauge100({ value }: { value: number | null }) {
  const r = 52, c = 2 * Math.PI * r, v = value == null ? 0 : Math.max(0, Math.min(1, value));
  const col = value == null ? "#1F3A4C" : v > 0.7 ? "#34D399" : v > 0.5 ? "#FBBF24" : "#FB7185";
  return (
    <svg viewBox="0 0 130 130" className="h-32 w-32 shrink-0 -rotate-90">
      <circle cx="65" cy="65" r={r} fill="none" stroke="#15293A" strokeWidth="11" />
      <circle cx="65" cy="65" r={r} fill="none" stroke={col} strokeWidth="11" strokeLinecap="round"
              strokeDasharray={`${c * v} ${c}`} style={{ filter: `drop-shadow(0 0 6px ${col})`, transition: "stroke-dasharray .8s" }} />
      <g transform="rotate(90 65 65)">
        <text x="65" y="66" textAnchor="middle" className="fill-white" style={{ font: "700 30px Inter" }}>
          {value == null ? "—" : Math.round(v * 100)}
        </text>
        <text x="65" y="86" textAnchor="middle" style={{ font: "500 10px Inter", fill: "#9DB4C4" }}>Resilience</text>
      </g>
    </svg>
  );
}

function OverviewSection({ p, selected, onSelect }: any) {
  const { state } = p;
  const st = state.graph?.stats;
  const res = state.sim?.resilience;
  const [view, setView] = useState<"graph" | "overlay">("graph");
  const im = state.imagery?.meta;
  const areaKm2 = im ? (im.width * im.height * im.pixel_size_m ** 2) / 1e6 : null;
  const alphaLabel = (a: number): [string, "warn" | "good" | "bad"] =>
    a < 0.1 ? ["Low", "bad"] : a < 0.3 ? ["Moderate", "warn"] : ["High", "good"];
  const lastOf = (prefix: string) => {
    const k = Object.keys(SIM_HISTORY).filter((x) => x.startsWith(prefix)).pop();
    return k ? SIM_HISTORY[k] : null;
  };
  const targeted = lastOf("targeted"), random = lastOf("random");
  const gradeTone = (g?: string) => !g ? "info" : /good|excellent|high/i.test(g) ? "good" : /moderate|fair/i.test(g) ? "warn" : "bad";

  return (
    <div className="flex-1 flex flex-col gap-3 min-h-0">
      {/* KPI row */}
      <div className="grid grid-cols-2 lg:grid-cols-3 2xl:grid-cols-6 gap-3">
        <Kpi icon={Ruler} label="Network length" tone="cyan" unit="km"
             value={st ? st.total_length_km.toFixed(1) : null}
             badge={st && st.healed_length_km > 0 ? [`+${st.healed_length_km.toFixed(2)} km healed`, "good"] : null} />
        <Kpi icon={Crosshair} label="Intersections" tone="blue"
             value={st ? st.intersections.toLocaleString() : null}
             badge={st ? [`${st.dead_ends} dead ends`, "info"] : null} />
        <Kpi icon={GitBranch} label="Segments" tone="violet"
             value={st ? st.edges.toLocaleString() : null}
             badge={st ? [`${st.components} component${st.components === 1 ? "" : "s"}`, st.components === 1 ? "good" : "warn"] : null} />
        <Kpi icon={AlertOctagon} label="Gatekeepers" tone="red" unit={st ? "critical" : undefined}
             value={st ? st.gatekeepers : null}
             hint="Articulation points: removing one disconnects part of the network"
             badge={st ? [`${((st.gatekeepers / Math.max(1, st.nodes)) * 100).toFixed(1)}% of nodes`, "bad"] : null}
             meter={st ? st.gatekeepers / Math.max(1, st.nodes) : null} />
        <Kpi icon={Network} label="Redundancy α" tone="amber"
             value={st ? st.alpha_index.toFixed(2) : null}
             hint="Meshedness: 0 = tree (no alternative routes), higher = more redundancy"
             badge={st ? alphaLabel(st.alpha_index) : null} meter={st ? st.alpha_index : null} />
        <Kpi icon={ShieldAlert} label="Resilience" tone="green" unit={res ? "/ 100" : undefined}
             value={res ? Math.round(res.normalized_resilience * 100) : null}
             hint="Network Efficiency Resilience after the last simulated failure (0-100)"
             badge={res ? [res.grade, gradeTone(res.grade) as any] : st ? ["run a stress test", "info"] : null}
             meter={res ? res.normalized_resilience : null} />
      </div>

      <div className="flex-1 grid grid-cols-12 gap-3 min-h-0">
        {/* network */}
        <section className="panel col-span-12 xl:col-span-8 relative flex flex-col min-h-[460px] overflow-hidden">
          <div className="absolute left-4 top-4 z-10 flex items-center gap-3 rounded-xl border border-line-soft
                          bg-base-800/85 backdrop-blur px-3.5 py-2.5">
            <Network size={16} className="text-accent" />
            <div>
              <div className="text-[13px] font-semibold text-white">Road network</div>
              <div className="text-[11px] text-ink-500">
                {im ? `${im.width}×${im.height} px · ${im.pixel_size_m} m/px · ${areaKm2!.toFixed(2)} km²`
                    : "no study area loaded"}
              </div>
            </div>
            {state.graph && (
              <span className="ml-1 inline-flex items-center gap-1.5 rounded-full border border-emerald-400/30
                               bg-emerald-400/10 px-2 py-0.5 text-[10.5px] text-emerald-300">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" /> Live graph
              </span>
            )}
          </div>
          {state.graph && state.imagery && (
            <div className="absolute right-4 top-4 z-10 flex rounded-lg border border-line-soft bg-base-800/85 p-0.5 text-[12px]">
              {(["graph", "overlay"] as const).map((v) => (
                <button key={v} onClick={() => setView(v)}
                  className={clsx("rounded-md px-3 py-1 capitalize",
                    view === v ? "bg-accent/15 text-accent" : "text-ink-300 hover:text-white")}>{v}</button>
              ))}
            </div>
          )}

          {state.graph ? (
            <NetworkCanvas
              className="w-full h-full min-h-[460px]"
              geojson={state.graph.geojson}
              scores={state.crit?.scores}
              gatekeepers={state.crit?.gatekeepers}
              selected={selected}
              onSelectNode={onSelect}
              routeOriginal={state.route?.original?.path}
              routeAlternate={state.route?.disrupted?.path}
              backdrop={view === "overlay" ? state.imagery?.layers?.original ?? null : null}
              backdropExtent={im ? { width: im.width, height: im.height } : null}
              backdropOpacity={0.55}
            />
          ) : (
            <div className="flex-1 grid place-items-center">
              <Empty icon={Network} title="No network yet"
                     hint="Run the guided demonstration on the right, or load a study area under Imagery." />
            </div>
          )}

          {state.graph && (
            <div className="absolute left-4 bottom-4 z-10 rounded-xl border border-line-soft bg-base-800/85
                            backdrop-blur px-3.5 py-2.5 text-[11px] space-y-1.5">
              <div className="font-semibold text-ink-100 text-[12px]">Legend</div>
              {[["bg-cyan-400", "Extracted road", false], ["bg-cyan-50", "Healed gap (inferred)", true],
                ["bg-rose-400", "Gatekeeper junction", false], ["bg-amber-400", "High-betweenness junction", false]]
                .map(([c, l, dash]) => (
                  <div key={l as string} className="flex items-center gap-2 text-ink-300">
                    {String(l).includes("junction")
                      ? <span className={clsx("h-2.5 w-2.5 rounded-full", c as string)} />
                      : <span className={clsx("h-0.5 w-5", c as string, dash && "opacity-60 [mask-image:repeating-linear-gradient(90deg,#000_0_4px,transparent_4px_7px)]")} />}
                    {l}
                  </div>
                ))}
            </div>
          )}
        </section>

        {/* right column */}
        <div className="col-span-12 xl:col-span-4 flex flex-col gap-3 min-h-0">
          <section className="panel p-4 flex items-center gap-4">
            <Gauge100 value={res ? res.normalized_resilience : null} />
            <div className="flex-1 min-w-0">
              <div className="text-[15px] font-semibold text-white">
                {res ? (res.normalized_resilience > 0.7 ? "Holds up well" : res.normalized_resilience > 0.5 ? "Under strain" : "Fragile")
                     : "Not stress-tested yet"}
              </div>
              <p className="mt-1 text-[11.5px] leading-relaxed text-ink-300">
                {res ? `${Math.round(res.disconnection_rate * 100)}% of sampled trips cut off by the last scenario.`
                     : "Run the guided demonstration or a stress test to measure resilience."}
              </p>
              {(targeted || random) && (
                <div className="mt-2.5 space-y-2 text-[11px]">
                  {([["Random damage", random, "bg-emerald-400"], ["Targeted failure", targeted, "bg-rose-400"]] as const)
                    .filter(([, r]) => r).map(([l, r, c]) => (
                    <div key={l}>
                      <div className="flex justify-between text-ink-300">
                        <span>{l}</span>
                        <span className="metric text-white">{Math.round(r.normalized_resilience * 100)} / 100</span>
                      </div>
                      <div className="mt-1 h-1.5 rounded-full bg-base-600 overflow-hidden">
                        <div className={clsx("h-full rounded-full", c)} style={{ width: `${r.normalized_resilience * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>

          <DemoRunner p={p} onTab={p.onTab} />

          <section className="panel p-4">
            <div className="flex items-center justify-between">
              <div className="text-[13px] font-semibold text-white flex items-center gap-2">
                <ShieldAlert size={14} className="text-accent" /> Provenance
              </div>
              {state.extract?.provenance && (
                <span className={clsx("text-[10.5px]", state.extract.provenance.is_learned ? "text-emerald-300" : "text-amber-300")}>
                  {state.extract.provenance.is_learned ? "trained model" : "not a trained model"}
                </span>
              )}
            </div>
            {state.extract?.provenance ? (
              <div className="mt-2.5 flex flex-wrap gap-1.5 text-[10.5px]">
                {[["model", state.extract.provenance.architecture ?? state.extract.provenance.mode],
                  ["source", im?.source], ["res", im ? `${im.pixel_size_m} m/px` : null],
                  ["mode", state.extract.provenance.mode]].filter(([, v]) => v).map(([k, v]) => (
                  <span key={k as string} className="rounded-md border border-line bg-base-700/60 px-2 py-1 text-ink-300">
                    <span className="text-ink-500">{k}</span> {String(v)}
                  </span>
                ))}
              </div>
            ) : <p className="mt-2 text-[11.5px] text-ink-500">Appears as soon as a road mask exists.</p>}
          </section>
        </div>
      </div>
    </div>
  );
}

export function Row({ k, v }: { k: string; v: any }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-ink-500">{k}</span>
      <span className="text-ink-100 font-mono text-right truncate">{String(v)}</span>
    </div>
  );
}

/* ------------------------------------------------------------- network */
function NetworkSection({ p, selected, onSelect }: any) {
  const { state } = p;
  const [gap, setGap] = useState(140);
  const [loops, setLoops] = useState(true);

  return (
    <div className="flex-1 grid grid-cols-12 gap-3 min-h-0">
      <Panel className="col-span-12 lg:col-span-8" title="Routable graph">
        {state.graph ? (
          <NetworkCanvas className="w-full h-full min-h-[440px]"
            geojson={state.graph.geojson} scores={state.crit?.scores}
            gatekeepers={state.crit?.gatekeepers} selected={selected}
            onSelectNode={onSelect} />
        ) : <Empty icon={Network} title="Graph not built"
                   hint="Run road extraction first, then build the graph." />}
      </Panel>

      <div className="col-span-12 lg:col-span-4 flex flex-col gap-3 min-h-0 overflow-auto">
        <Panel title="Topology healing" dense>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span className="text-ink-300">Max bridge gap</span>
                <span className="metric text-ink-100">{gap} m</span>
              </div>
              <input type="range" min={20} max={800} step={10} value={gap}
                     onChange={(e) => setGap(+e.target.value)}
                     className="w-full accent-accent" />
            </div>
            <label className="flex items-center gap-2 text-[11px] text-ink-300">
              <input type="checkbox" checked={loops} className="accent-accent"
                     onChange={(e) => setLoops(e.target.checked)} />
              Restore loops (second healing pass)
            </label>
            <Note>
              Without the second pass, a gap inside a cycle is never repaired:
              both sides are already connected the long way round, so union&ndash;find
              rejects the bridge. Connectivity looks perfect while redundancy
              collapses and gatekeepers are over-reported.
            </Note>
            <button className="btn-primary w-full justify-center"
                    disabled={!state.extract || !!state.busy}
                    onClick={() => p.buildGraph(gap, loops)}>
              <Play size={14} /> Build graph
            </button>
          </div>
        </Panel>

        {state.graph && (
          <>
            <Panel title="Healing result" dense>
              <div className="grid grid-cols-2 gap-3">
                <Metric size="sm" label="Components" tone="good"
                        value={`${state.graph.healing.components_before} \u2192 ${state.graph.healing.components_after}`} />
                <Metric size="sm" label="Bridges added" value={state.graph.healing.bridges_added} />
                <Metric size="sm" label="Loops restored" value={state.graph.healing.loops_restored} />
                <Metric size="sm" label="Inferred length"
                        value={fmt.pct(state.graph.healing.healed_length_fraction)}
                        tone={state.graph.healing.healed_length_fraction > 0.2 ? "warn" : "default"} />
              </div>
              {state.graph.healing.healed_length_fraction > 0.2 && (
                <div className="mt-3">
                  <Note tone="warn">
                    Over 20% of network length was inferred rather than observed.
                    Treat downstream criticality as low-confidence and consider a
                    smaller maximum bridge gap.
                  </Note>
                </div>
              )}
            </Panel>

            <Panel title="Network statistics" dense>
              <div className="space-y-1.5 text-[11px]">
                <Row k="Nodes" v={state.graph.stats.nodes} />
                <Row k="Edges" v={state.graph.stats.edges} />
                <Row k="Total length" v={`${state.graph.stats.total_length_km} km`} />
                <Row k="Mean degree" v={state.graph.stats.mean_degree} />
                <Row k="Dead ends" v={state.graph.stats.dead_ends} />
                <Row k="Gatekeepers" v={state.graph.stats.gatekeepers} />
                <Row k="Bridge edges" v={state.graph.stats.bridge_edges} />
                <Row k="Redundancy α" v={state.graph.stats.alpha_index} />
                <Row k="LCC coverage"
                     v={fmt.pct(state.graph.connectivity_ratio.lcc_coverage_after) ?? "\u2014"} />
              </div>
            </Panel>
          </>
        )}
      </div>
    </div>
  );
}
