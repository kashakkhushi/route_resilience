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
  Activity, AlertTriangle, BarChart3, ChevronRight, Crosshair,
  Image as ImageIcon, Layers, Network, Play, Satellite, ShieldAlert, Waypoints,
} from "lucide-react";
import clsx from "clsx";
import { usePipeline } from "@/hooks/usePipeline";
import NetworkCanvas from "@/components/NetworkCanvas";
import {
  Chip, Empty, ErrorNote, Metric, Note, Panel, ProvenanceBanner, StatusDot, fmt,
} from "@/components/ui";
import ImagerySection from "@/features/ImagerySection";
import ExtractionSection from "@/features/ExtractionSection";
import CriticalitySection from "@/features/CriticalitySection";
import StressTestSection from "@/features/StressTestSection";
import EvaluationSection from "@/features/EvaluationSection";
import { DemoRunner } from "@/features/DemoRunner";

type Tab = "overview" | "imagery" | "extraction" | "network"
         | "criticality" | "stress" | "evaluation";

const TABS: { id: Tab; label: string; icon: typeof Layers }[] = [
  { id: "overview",    label: "Overview",        icon: Layers },
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
    return <Landing health={state.health} onLaunch={() => setLaunched(true)} />;
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
        </main>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------- landing */
function Landing({ health, onLaunch }: { health: any; onLaunch: () => void }) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6
                    bg-[radial-gradient(ellipse_at_top,#0F1A26_0%,#080B10_60%)]">
      <div className="max-w-3xl w-full space-y-8">
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Waypoints size={18} className="text-accent" />
            <span className="label">Urban Infrastructure Intelligence</span>
          </div>
          <h1 className="text-4xl font-bold tracking-tight">
            Route Resilience <span className="text-accent">AI</span>
          </h1>
          <p className="text-ink-300 text-lg">
            AI-Powered Routable Road Network Intelligence
          </p>
          <p className="text-ink-500 max-w-2xl leading-relaxed">
            Extract roads from satellite imagery, reconstruct missing topology,
            identify systemic bottlenecks, and simulate the network-wide impact
            of road failures.
          </p>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {[
            ["Processed area", "\u2014"], ["Roads extracted", "\u2014"],
            ["Network nodes", "\u2014"], ["Critical links", "\u2014"],
            ["Network resilience", "\u2014"],
          ].map(([l, v]) => (
            <div key={l} className="panel p-3">
              <Metric label={l} value={v} size="sm" />
            </div>
          ))}
          <div className="panel p-3">
            <div className="label">Model status</div>
            <div className="mt-1.5">
              {health ? (
                <Chip tone={health.checkpoint_found ? "accent" : "warn"}>
                  {health.checkpoint_found ? "Trained checkpoint" : "No checkpoint"}
                </Chip>
              ) : <div className="skeleton h-5 w-24" />}
            </div>
          </div>
        </div>

        {health && !health.checkpoint_found && (
          <Note tone="warn">
            {health.notes}
          </Note>
        )}

        <button onClick={onLaunch} className="btn-primary text-base px-5 py-2.5">
          Launch Urban Digital Twin <ChevronRight size={16} />
        </button>

        <p className="text-[11px] text-ink-500 leading-relaxed max-w-2xl">
          Statistics populate from real computation once a study area is loaded.
          They are shown as em dashes until then rather than pre-filled with
          example values.
        </p>
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
