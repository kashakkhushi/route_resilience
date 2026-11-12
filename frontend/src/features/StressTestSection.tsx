/**
 * Stress test: disable infrastructure, recompute, compare baseline vs disrupted.
 * Includes the Resilience Index with its formula and the survivorship-bias
 * warning, per the brief's requirement that metric orientation be explained.
 */
import { useState } from "react";
import { RotateCcw, RouteIcon, ShieldAlert, Zap } from "lucide-react";
import NetworkCanvas from "@/components/NetworkCanvas";
import { Chip, Empty, Metric, Note, Panel, fmt } from "@/components/ui";

const KINDS = [
  { id: "node_failure",    label: "Intersection failure", hint: "Disable selected intersections" },
  { id: "edge_closure",    label: "Road closure",         hint: "Close selected segments" },
  { id: "flood_radial",    label: "Flood scenario",       hint: "Closure of all roads within a radius" },
  { id: "targeted_attack", label: "Targeted attack",      hint: "Sequentially remove highest-betweenness nodes" },
  { id: "random_failure",  label: "Random failure",       hint: "Uniformly random failures (baseline)" },
] as const;

export default function StressTestSection({ p, selected, onSelect }: any) {
  const { state } = p;
  const [kind, setKind] = useState<typeof KINDS[number]["id"]>("targeted_attack");
  const [count, setCount] = useState(5);
  const [confirm, setConfirm] = useState(false);
  const [src, setSrc] = useState<number | "">("");
  const [dst, setDst] = useState<number | "">("");

  const res = state.sim?.resilience;
  const removed: number[] = state.sim?.removed_node_ids ?? [];
  const removedEdges: number[][] = state.sim?.removed_edge_ids ?? [];

  const run = () => {
    const body: any = { kind, count };
    if (kind === "node_failure") body.nodes = selected != null ? [selected] : [];
    if (kind === "flood_radial") {
      const m = state.imagery?.meta;
      body.center = [m ? m.height / 2 : 320, m ? m.width / 2 : 320];
      body.radius_px = m ? Math.min(m.width, m.height) * 0.16 : 100;
    }
    p.simulate(body);
    setConfirm(false);
  };

  return (
    <div className="flex-1 grid grid-cols-12 gap-3 min-h-0">
      <Panel className="col-span-12 lg:col-span-8"
        title={state.sim ? `Disrupted network \u2014 ${state.sim.scenario}` : "Network"}
        actions={state.sim && <Chip tone="bad">{state.sim.removed_nodes} nodes removed</Chip>}>
        {state.graph ? (
          <NetworkCanvas className="w-full h-full min-h-[440px]"
            geojson={state.graph.geojson} scores={state.crit?.scores}
            gatekeepers={state.crit?.gatekeepers} disabledNodes={removed} disabledEdges={removedEdges}
            selected={selected} onSelectNode={onSelect}
            routeOriginal={state.route?.original?.path}
            routeAlternate={state.route?.disrupted?.path} />
        ) : <Empty icon={ShieldAlert} title="Build the graph first" />}
      </Panel>

      <div className="col-span-12 lg:col-span-4 flex flex-col gap-3 overflow-auto">
        <Panel title="Scenario" dense>
          <div className="space-y-3">
            <div className="space-y-1.5">
              {KINDS.map((k) => (
                <button key={k.id} onClick={() => setKind(k.id)}
                  className={`w-full text-left px-2.5 py-2 rounded-md border text-[12px] transition-colors ${
                    kind === k.id ? "border-accent/40 bg-accent/10" : "border-line hover:bg-base-700"}`}>
                  <div className="font-medium">{k.label}</div>
                  <div className="text-[10px] text-ink-500 mt-0.5">{k.hint}</div>
                </button>
              ))}
            </div>

            {kind === "node_failure" && (
              <Note tone={selected == null ? "warn" : "info"}>
                {selected == null
                  ? "Click an intersection on the map to select it."
                  : `Intersection #${selected} selected.`}
              </Note>
            )}
            {(kind === "targeted_attack" || kind === "random_failure") && (
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-ink-300">Intersections removed</span>
                  <span className="metric">{count}</span>
                </div>
                <input type="range" min={1} max={25} value={count}
                       onChange={(e) => setCount(+e.target.value)}
                       className="w-full accent-accent" />
              </div>
            )}

            {kind === "flood_radial" && (
              <Note>
                This simulates the <strong>closure</strong> of roads inside a radius.
                It does not predict flooding: no hydrological model or flood
                probability data is used.
              </Note>
            )}

            {!confirm ? (
              <button className="btn-danger w-full justify-center"
                      disabled={!state.graph || !!state.busy}
                      onClick={() => setConfirm(true)}>
                <Zap size={14} /> Run simulation
              </button>
            ) : (
              <div className="space-y-2">
                <Note tone="warn">
                  This recomputes routing across the whole network. Continue?
                </Note>
                <div className="flex gap-2">
                  <button className="btn-danger flex-1 justify-center" onClick={run}>Confirm</button>
                  <button className="btn-ghost flex-1 justify-center"
                          onClick={() => setConfirm(false)}>Cancel</button>
                </div>
              </div>
            )}

            {state.sim && (
              <button className="btn-ghost w-full justify-center" onClick={p.clearSim}>
                <RotateCcw size={14} /> Reset simulation
              </button>
            )}
          </div>
        </Panel>

        {res && (
          <>
            <Panel title="Resilience Index" dense>
              <div className="space-y-3">
                <div className="flex items-end gap-3">
                  <div className="metric text-3xl font-bold"
                       style={{ color: res.resilience_index > 0.7 ? "#22C55E"
                              : res.resilience_index > 0.5 ? "#EAB308" : "#EF4444" }}>
                    {res.resilience_index.toFixed(2)}
                  </div>
                  <div className="pb-1">
                    <div className="text-[11px] text-ink-300">{res.grade}</div>
                    <div className="text-[10px] text-ink-500">
                      {res.resilience_metric} &middot; bounded 0&ndash;1
                    </div>
                  </div>
                </div>

                <p className="text-[11px] text-ink-300">{res.resilience_interpretation}</p>

                <div className="rounded-md bg-base-900 border border-line-soft p-2.5">
                  <div className="label mb-1">Formula &mdash; {res.resilience_metric}</div>
                  <p className="text-[10px] text-ink-300 font-mono leading-relaxed">
                    {res.resilience_formula}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <Metric size="sm" label="Trips disconnected"
                          value={fmt.pct(res.disconnection_rate)} tone="bad" />
                  <Metric size="sm" label="ASPL change"
                          value={res.aspl_increase_pct != null
                                 ? `${res.aspl_increase_pct > 0 ? "+" : ""}${res.aspl_increase_pct}%` : null} />
                </div>

                {/* Secondary readings, visually demoted so neither can be
                    mistaken for the index above. */}
                <details className="rounded-md border border-line-soft bg-base-900/60">
                  <summary className="px-2.5 py-1.5 text-[11px] text-ink-500 cursor-pointer">
                    Secondary readings (not the Resilience Index)
                  </summary>
                  <div className="px-2.5 pb-2.5 pt-1 space-y-2">
                    <div>
                      <div className="label">Literal ASPL ratio (project brief)</div>
                      <div className="metric text-sm">
                        {res.resilience_index_literal_aspl.toFixed(2)}
                      </div>
                      <p className="text-[10px] text-ink-500 mt-1 leading-relaxed">
                        {res.literal_aspl_note}
                      </p>
                    </div>
                    <div>
                      <div className="label">
                        Connected-only ASPL ratio &mdash; diagnostic
                      </div>
                      <div className={`metric text-sm ${
                        res.diagnostic_survivorship_biased ? "text-risk-med" : ""}`}>
                        {res.diagnostic_aspl_ratio_connected_only.toFixed(2)}
                      </div>
                      <p className="text-[10px] text-ink-500 mt-1 leading-relaxed">
                        {res.diagnostic_note}
                      </p>
                    </div>
                  </div>
                </details>

                {!res.network_fully_connected_after && (
                  <Note tone="warn">{res.note}</Note>
                )}
              </div>
            </Panel>

            <Panel title="Baseline vs disrupted" dense>
              <div className="grid grid-cols-3 gap-2 text-[11px]">
                <div className="label">Metric</div>
                <div className="label text-right">Baseline</div>
                <div className="label text-right">Disrupted</div>
                {[
                  ["Components", state.sim.baseline.components, state.sim.disrupted.components],
                  ["ASPL (m)", res.aspl_baseline_m, res.aspl_perturbed_m ?? "\u221e"],
                  ["Giant fraction", fmt.pct(state.sim.baseline.giant_fraction ?? 1),
                                     fmt.pct(state.sim.disrupted.giant_fraction)],
                ].map(([k, a, b]) => (
                  <>
                    <div className="text-ink-500">{k}</div>
                    <div className="metric text-right">{String(a)}</div>
                    <div className="metric text-right text-risk-high">{String(b)}</div>
                  </>
                ))}
              </div>
              <div className="mt-3 grid grid-cols-2 gap-3">
                <Metric size="sm" label="Mean detour"
                        value={state.sim.delta.mean_detour_ratio
                               ? `${state.sim.delta.mean_detour_ratio}\u00d7` : null} />
                <Metric size="sm" label="p95 detour"
                        value={state.sim.delta.p95_detour_ratio
                               ? `${state.sim.delta.p95_detour_ratio}\u00d7` : null} />
              </div>
            </Panel>
          </>
        )}

        <Panel title="Route comparison" dense>
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <input className="input" placeholder="Origin node" value={src}
                     onChange={(e) => setSrc(e.target.value === "" ? "" : +e.target.value)} />
              <input className="input" placeholder="Destination" value={dst}
                     onChange={(e) => setDst(e.target.value === "" ? "" : +e.target.value)} />
            </div>
            <button className="btn-ghost w-full justify-center"
                    disabled={src === "" || dst === "" || !!state.busy}
                    onClick={() => p.route(Number(src), Number(dst), removed)}>
              <RouteIcon size={14} /> Compute route
            </button>
            {state.route && (
              <div className="space-y-2 pt-1">
                <div className="grid grid-cols-2 gap-3">
                  <Metric size="sm" label="Original"
                          value={fmt.km(state.route.original.distance_m)} unit="km" />
                  <Metric size="sm" label="Alternate"
                          value={state.route.disrupted.reachable
                                 ? fmt.km(state.route.disrupted.distance_m) : null}
                          unit="km" tone={state.route.disrupted.reachable ? "warn" : "bad"} />
                  <Metric size="sm" label="Detour"
                          value={state.route.detour_ratio ? `${state.route.detour_ratio}\u00d7` : null} />
                  <Metric size="sm" label="Time change"
                          value={state.route.time_increase_pct != null
                                 ? `+${state.route.time_increase_pct}%` : null} />
                </div>
                {!state.route.disrupted.reachable && (
                  <Note tone="warn">Destination unreachable under this scenario.</Note>
                )}
                <Note>{state.route.note}</Note>
              </div>
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}
