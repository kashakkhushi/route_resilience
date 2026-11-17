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
  Activity, AlertTriangle, BarChart3, BookOpen, ChevronRight, Crosshair, Github,
  Image as ImageIcon, Layers, Network, Play, Satellite, ShieldAlert, Waypoints, Route, Map as MapIcon,
} from "lucide-react";
import clsx from "clsx";
import { usePipeline } from "@/hooks/usePipeline";
import NetworkCanvas from "@/components/NetworkCanvas";
import Globe from "@/components/Globe";
import {
  Chip, Empty, ErrorNote, Metric, Note, Panel, ProvenanceBanner, StatusDot, fmt,
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

  if (!launched) {
    return <Landing health={state.health}
                    onLaunch={(t) => { if (t) setTab(t); setLaunched(true); }} />;
  }

  return (
    <div className="h-screen flex flex-col bg-base-900">
      <TopBar state={state} />

      <div className="flex-1 flex min-h-0">
        {/* ---------------------------------------------------- sidebar */}
        <nav className="w-52 shrink-0 border-r border-line-soft bg-base-800/50 flex flex-col">
          <div className="p-2 space-y-0.5">
            {TABS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={clsx(
                  "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-md text-sm transition-colors",
                  tab === id
                    ? "bg-accent/10 text-accent border border-accent/25"
                    : "text-ink-300 hover:bg-base-700 border border-transparent",
                )}
              >
                <Icon size={15} />
                {label}
              </button>
            ))}
          </div>

          <div className="mt-auto p-3 border-t border-line-soft space-y-2">
            <div className="label">Pipeline</div>
            <StageList state={state} />
          </div>
        </nav>

        {/* ------------------------------------------------------ centre */}
        <main className="flex-1 min-w-0 flex flex-col gap-3 p-3">
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
          {tab === "planner" && <RoutePlannerSection p={p} />}
          {tab === "trip" && <TripPlannerSection />}
        </main>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------- landing */
// Held-out test results of the deployed model (DeepGlobe test split, 498
// tiles, threshold 0.5, no test-time augmentation). Source: the model's
// evaluation_report.md; see README "Results". Update here if the model changes.
const FINAL_RESULTS = {
  model: "SegFormer-B3 (fine-tuned)",
  iou: "0.669", dice: "0.792", cldice: "0.868",
  occludedRecall: "0.55 \u2192 0.89",
};

function Landing({ health, onLaunch }: { health: any; onLaunch: (t?: Tab) => void }) {
  const arch = health?.checkpoint_metadata?.architecture;
  return (
    <div className="relative min-h-screen overflow-hidden
                    bg-[radial-gradient(ellipse_at_70%_50%,#0B2540_0%,#050B16_55%,#02060D_100%)]">
      <Globe className="absolute inset-0 w-full h-full" />

      <div className="relative z-10 min-h-screen flex flex-col px-6 sm:px-12 lg:px-16 py-10">
        {/* brand */}
        <div className="flex items-center gap-3">
          <Waypoints size={40} className="text-cyan-300 drop-shadow-[0_0_12px_rgba(34,211,238,0.7)]" />
          <div className="leading-none">
            <div className="text-3xl sm:text-4xl font-semibold tracking-tight text-white">
              Route Resilience
            </div>
            <div className="text-right text-lg sm:text-xl font-medium text-cyan-300 mt-1">
              AI
            </div>
          </div>
        </div>

        {/* glance card */}
        <div className="mt-10 lg:mt-14 max-w-2xl rounded-2xl border border-cyan-300/15
                        bg-[#0A1A2E]/55 backdrop-blur-md shadow-[0_20px_60px_rgba(0,0,0,0.55)]
                        p-6 sm:p-8">
          <div className="flex items-center gap-3 mb-5">
            <BookOpen size={26} className="text-ink-300" />
            <h2 className="text-xl font-semibold text-white">Project Glance:</h2>
          </div>

          <ul className="space-y-3 text-[15px] sm:text-base leading-relaxed text-ink-300
                         list-disc pl-6 marker:text-cyan-300/70">
            <li>
              Extracts road networks from satellite imagery, including roads hidden
              under <span className="text-white font-medium">trees, shadows and vehicles</span>.
            </li>
            <li>
              Repairs broken roads into a routable graph, finds the
              <span className="text-white font-medium"> critical intersections</span>, and
              simulates floods, closures and failures.
            </li>
            <li>
              Final model: <span className="text-white font-semibold">{FINAL_RESULTS.model}</span>.
              Test IoU <span className="text-cyan-300 font-semibold">{FINAL_RESULTS.iou}</span>,
              Dice <span className="text-cyan-300 font-semibold">{FINAL_RESULTS.dice}</span>,
              clDice <span className="text-cyan-300 font-semibold">{FINAL_RESULTS.cldice}</span>
              {" "}on 498 held-out DeepGlobe tiles.
            </li>
            <li>
              Occlusion training: recall on hidden roads under clutter{" "}
              <span className="text-cyan-300 font-semibold">{FINAL_RESULTS.occludedRecall}</span>.
            </li>
            <li className="flex flex-wrap items-center gap-2">
              Model status:
              {health ? (
                <Chip tone={health.checkpoint_found ? "accent" : "warn"}>
                  {health.checkpoint_found
                    ? `Trained checkpoint${arch ? ` \u00b7 ${arch}` : ""}`
                    : "No checkpoint"}
                </Chip>
              ) : <span className="skeleton h-5 w-28 inline-block" />}
            </li>
          </ul>

          <p className="mt-5 text-sm text-ink-500 leading-relaxed">
            Note: results above are measured on the DeepGlobe test split (0.5&nbsp;m
            imagery). Accuracy on other sensors has not been measured yet.
          </p>

          {health && !health.checkpoint_found && (
            <div className="mt-4"><Note tone="warn">{health.notes}</Note></div>
          )}

          <div className="mt-7 flex flex-wrap gap-4">
            <button onClick={() => onLaunch("overview")} className="glow-btn">
              Launch Dashboard <ChevronRight size={16} />
            </button>
            <button onClick={() => onLaunch("trip")} className="glow-btn">
              Plan a Trip
            </button>
            <button onClick={() => onLaunch("evaluation")} className="glow-btn">
              View Results
            </button>
          </div>
        </div>

        {/* footer */}
        <div className="mt-auto pt-10 flex flex-wrap gap-x-6 gap-y-2 text-sm text-ink-300">
          <a className="footer-link" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">
            <BookOpen size={15} /> API Docs
          </a>
          <a className="footer-link" href="https://github.com/kashakkhushi/route_resilience"
             target="_blank" rel="noreferrer">
            <Github size={15} /> GitHub
          </a>
          <span className="footer-link"><Satellite size={15} /> DeepGlobe Road Extraction</span>
          <span className="footer-link"><ShieldAlert size={15} /> VIT-AP Capstone 2026&ndash;27</span>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------- topbar */
function TopBar({ state }: { state: any }) {
  const h = state.health;
  return (
    <header className="h-12 shrink-0 border-b border-line-soft bg-base-800/70
                       flex items-center px-4 gap-4">
      <div className="flex items-center gap-2">
        <Waypoints size={16} className="text-accent" />
        <span className="font-semibold text-sm">Route Resilience AI</span>
        {h && <span className="text-[10px] text-ink-500 font-mono">v{h.version}</span>}
      </div>

      {state.extract?.provenance && (
        <Chip tone={state.extract.provenance.is_learned ? "accent" : "warn"}>
          {state.extract.provenance.label}
        </Chip>
      )}

      <div className="ml-auto flex items-center gap-4">
        {h && <>
          <StatusDot ok={h.torch_available} label={h.torch_available ? "PyTorch" : "No PyTorch"} />
          <StatusDot ok={!!h.checkpoint_found} label={h.checkpoint_found ? "Checkpoint" : "No checkpoint"} />
          <StatusDot ok={h.geotiff_support} label="GeoTIFF" />
        </>}
        {state.busy && (
          <span className="flex items-center gap-1.5 text-[11px] text-accent">
            <Activity size={12} className="animate-pulse" /> {state.busy}
          </span>
        )}
      </div>
    </header>
  );
}

function StageList({ state }: { state: any }) {
  const stages = [
    ["Imagery", !!state.imagery],
    ["Extraction", !!state.extract],
    ["Graph + healing", !!state.graph],
    ["Criticality", !!state.crit],
    ["Simulation", !!state.sim],
  ] as [string, boolean][];
  return (
    <div className="space-y-1">
      {stages.map(([l, done]) => (
        <div key={l} className="flex items-center gap-2 text-[11px]">
          <span className={clsx("w-1.5 h-1.5 rounded-full",
            done ? "bg-risk-low" : "bg-base-600")} />
          <span className={done ? "text-ink-300" : "text-ink-500"}>{l}</span>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------ overview */
function OverviewSection({ p, selected, onSelect }: any) {
  const { state } = p;
  const st = state.graph?.stats;
  const res = state.sim?.resilience;

  return (
    <div className="flex-1 grid grid-cols-12 gap-3 min-h-0">
      <div className="col-span-12 grid grid-cols-2 lg:grid-cols-6 gap-3">
        <div className="panel p-3"><Metric label="Network length" value={st ? st.total_length_km : null} unit="km" /></div>
        <div className="panel p-3"><Metric label="Intersections" value={st?.intersections ?? null} /></div>
        <div className="panel p-3"><Metric label="Segments" value={st?.edges ?? null} /></div>
        <div className="panel p-3"><Metric label="Gatekeepers" value={st?.gatekeepers ?? null}
          tone={st && st.gatekeepers > 0 ? "warn" : "default"}
          hint="Articulation points: removing one disconnects part of the network" /></div>
        <div className="panel p-3"><Metric label="Redundancy α" value={st ? st.alpha_index : null}
          hint="0 = tree (no alternative routes); higher = more redundancy" /></div>
        <div className="panel p-3"><Metric label="Resilience"
          value={res ? res.normalized_resilience.toFixed(2) : null}
          tone={res ? (res.normalized_resilience > 0.7 ? "good"
                     : res.normalized_resilience > 0.5 ? "warn" : "bad") : "default"} /></div>
      </div>

      <Panel className="col-span-12 lg:col-span-8" title="Network"
             actions={state.graph && <Chip tone="accent">{state.graph.stats.nodes} nodes</Chip>}>
        {state.graph ? (
          <NetworkCanvas
            className="w-full h-full min-h-[420px]"
            geojson={state.graph.geojson}
            scores={state.crit?.scores}
            gatekeepers={state.crit?.gatekeepers}
            selected={selected}
            onSelectNode={onSelect}
            routeOriginal={state.route?.original?.path}
            routeAlternate={state.route?.disrupted?.path}
          />
        ) : (
          <Empty icon={Network} title="No network yet"
                 hint="Load a study area under Imagery, run extraction, then build the graph." />
        )}
      </Panel>

      <div className="col-span-12 lg:col-span-4 flex flex-col gap-3 min-h-0">
        <Panel title="Study area" dense>
          {state.imagery ? (
            <div className="space-y-2 text-[11px]">
              <Row k="Source" v={state.imagery.meta.source} />
              <Row k="Size" v={`${state.imagery.meta.width} \u00d7 ${state.imagery.meta.height}`} />
              <Row k="GSD" v={`${state.imagery.meta.pixel_size_m} m/px`} />
              <Row k="CRS" v={state.imagery.meta.crs ?? "not georeferenced"} />
            </div>
          ) : <Empty icon={ImageIcon} title="No study area loaded" />}
        </Panel>

        <DemoRunner p={p} onTab={p.onTab} />

        <Panel title="Provenance" dense className="min-h-0">
          {state.extract?.provenance
            ? <ProvenanceBanner p={state.extract.provenance} />
            : <Empty icon={AlertTriangle} title="Nothing extracted yet"
                     hint="Provenance appears as soon as a road mask exists." />}
        </Panel>
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
