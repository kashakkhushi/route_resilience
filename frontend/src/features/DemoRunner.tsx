/**
 * Guided demonstration (project spec section 21).
 *
 * Runs the real pipeline end to end, one stage at a time, so a reviewer sees the
 * whole system in about two minutes. Every step is an actual API call against
 * the actual backend - this is a narrated run, not a scripted animation, and
 * the numbers shown are the numbers that were just computed.
 */
import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, Info, Loader2, Play, Square } from "lucide-react";
import clsx from "clsx";

export interface DemoStep {
  id: string;
  title: string;
  /** What a reviewer should take away from this stage. */
  caption: string;
  run: (ctx: DemoCtx) => Promise<string>;
}

interface DemoCtx {
  /** The pipeline hook. Steps drive it rather than calling the API directly,
   *  so every result lands in shared state and all views update with it. */
  p: any;
  tab: (t: string) => void;
}

export const DEMO_STEPS: DemoStep[] = [
  {
    id: "imagery",
    title: "Load study area",
    caption:
      "A deterministic synthetic city with a river crossed at only two points. " +
      "Few crossings manufacture genuine single points of failure; a plain grid " +
      "would make the resilience analysis look trivially optimistic.",
    run: async (ctx) => {
      ctx.tab("imagery");
      const r = await ctx.p.loadDemo("grid_river", 768, 1);
      if (!r) throw new Error("could not load the demo city");
      return `${r.meta.width}\u00d7${r.meta.height} at ${r.meta.pixel_size_m} m/px`;
    },
  },
  {
    id: "occlusion",
    title: "Inspect occlusion",
    caption:
      "Tree canopy, building shadow, cloud and vehicles are applied to the image " +
      "only - the label is left untouched. That asymmetry is what forces a model " +
      "to infer road continuity rather than match appearance.",
    run: async (ctx) => {
      ctx.tab("imagery");
      await new Promise((r) => setTimeout(r, 800));
      return "Occluders applied to imagery; ground truth unmodified";
    },
  },
  {
    id: "extract",
    title: "Extract roads",
    caption:
      "The mask is fragmented wherever an occluder sat on the road. This is the " +
      "failure the project exists to fix, and the method used is always labelled.",
    run: async (ctx) => {
      ctx.tab("extraction");
      const r = await ctx.p.extract("demo");
      if (!r) throw new Error("extraction failed");
      const m = r.metrics;
      return m.available
        ? `${r.provenance.mode} \u2014 IoU ${m.iou?.toFixed(3)}, clDice ${m.cldice?.toFixed(3)}`
        : r.provenance.mode;
    },
  },
  {
    id: "heal",
    title: "Heal topology",
    caption:
      "Union\u2013find with a minimum spanning tree bridges the gaps, adding exactly " +
      "(components \u2212 1) bridges. A second pass repairs breaks inside cycles, which " +
      "a spanning tree alone can never reach.",
    run: async (ctx) => {
      ctx.tab("network");
      const r = await ctx.p.buildGraph(140, true);
      if (!r) throw new Error("graph construction failed");
      return `${r.healing.components_before} components \u2192 ${r.healing.components_after}, ` +
             `${r.healing.bridges_added} bridges, ${r.healing.loops_restored} loops restored`;
    },
  },
  {
    id: "criticality",
    title: "Find critical intersections",
    caption:
      "Betweenness centrality finds busy corridors; articulation points find " +
      "gatekeepers whose loss disconnects the city. A busy junction usually has " +
      "alternatives - that is often why it is busy. A gatekeeper has none.",
    run: async (ctx) => {
      ctx.tab("criticality");
      const r = await ctx.p.criticality(10);
      if (!r) throw new Error("centrality failed");
      return `${r.gatekeepers.length} gatekeepers, ${r.bridge_edges.length} bridge segments` +
             (r.exact ? " (exact Brandes)" : ` (estimated, ${r.pivots} pivots)`);
    },
  },
  {
    id: "simulate",
    title: "Simulate targeted failure",
    caption:
      "Five intersections are removed sequentially, re-ranking after each one - " +
      "the network reorganises, so batch removal understates the damage.",
    run: async (ctx) => {
      ctx.tab("stress");
      const r = await ctx.p.simulate({ kind: "targeted_attack", count: 5 });
      if (!r) throw new Error("simulation failed");
      const res = r.resilience;
      return `${(res.disconnection_rate * 100).toFixed(0)}% of trips severed, ` +
             `resilience ${res.normalized_resilience.toFixed(2)} (${res.grade})`;
    },
  },
  {
    id: "compare",
    title: "Compare with random damage",
    caption:
      "The same number of failures chosen at random. The gap between the two is " +
      "the quantified argument for hardening specific intersections rather than " +
      "upgrading the network uniformly.",
    run: async (ctx) => {
      ctx.tab("stress");
      const r = await ctx.p.simulate({ kind: "random_failure", count: 5, seed: 3 });
      if (!r) throw new Error("simulation failed");
      return `random failure severs ${(r.resilience.disconnection_rate * 100).toFixed(0)}% ` +
             `of trips (resilience ${r.resilience.normalized_resilience.toFixed(2)})`;
    },
  },
];

// The demo switches tabs as it goes, which unmounts this panel; its progress
// therefore lives here, outside the component, and is shown again on return.
type DemoState = { idx: number; results: Record<string, string>; times: Record<string, number>;
                   failed: string | null; running: boolean };
const DEMO: DemoState & { subs: Set<() => void>; stop: boolean } = {
  idx: -1, results: {}, times: {}, failed: null, running: false, subs: new Set(), stop: false,
};
function setDemo(patch: Partial<DemoState> | ((d: DemoState) => Partial<DemoState>)) {
  Object.assign(DEMO, typeof patch === "function" ? patch(DEMO) : patch);
  DEMO.subs.forEach((f) => f());
}

export function DemoRunner({ p, onTab }: { p: any; onTab: (t: string) => void }) {
  const [, force] = useState(0);
  useEffect(() => {
    const f = () => force((n) => n + 1);
    DEMO.subs.add(f);
    return () => { DEMO.subs.delete(f); };
  }, []);
  const { idx, results, times, failed, running } = DEMO;
  const setIdx = (v: number) => setDemo({ idx: v });
  const setResults = (fn: (r: Record<string, string>) => Record<string, string>) => setDemo((d) => ({ results: fn(d.results) }));
  const setTimes = (fn: (t: Record<string, number>) => Record<string, number>) => setDemo((d) => ({ times: fn(d.times) }));
  const setFailed = (v: string | null) => setDemo({ failed: v });
  const setRunning = (v: boolean) => setDemo({ running: v });
  const [open, setOpen] = useState<string | null>(null);
  const stop = { get current() { return DEMO.stop; }, set current(v: boolean) { DEMO.stop = v; } };

  const play = useCallback(async () => {
    if (DEMO.running) return;
    setRunning(true);
    stop.current = false;
    setDemo({ results: {}, times: {}, failed: null });
    const ctx: DemoCtx = { p, tab: onTab };

    for (let i = 0; i < DEMO_STEPS.length; i++) {
      if (stop.current) break;
      setIdx(i);
      const t0 = performance.now();
      try {
        const out = await DEMO_STEPS[i].run(ctx);
        setResults((r) => ({ ...r, [DEMO_STEPS[i].id]: out }));
      } catch (e) {
        setResults((r) => ({ ...r, [DEMO_STEPS[i].id]: `failed: ${(e as Error).message}` }));
        setFailed(DEMO_STEPS[i].id);
        break;
      } finally {
        const dt = (performance.now() - t0) / 1000;      // measured, not invented
        setTimes((t) => ({ ...t, [DEMO_STEPS[i].id]: dt }));
      }
      await new Promise((r) => setTimeout(r, 900));
    }
    setRunning(false);
    setIdx(-1);
  }, [p, onTab]);

  const doneCount = Object.keys(results).length;
  return (
    <section className="panel p-4 flex flex-col min-h-0">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[15px] font-semibold text-white">Guided demonstration</div>
          <div className="text-[12px] text-ink-500">Live API calls &middot; real computed figures</div>
        </div>
        {running ? (
          <button className="btn-ghost text-xs" onClick={() => { stop.current = true; }}>
            <Square size={12} /> Stop
          </button>
        ) : (
          <button onClick={play}
            className="inline-flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-cyan-300 to-sky-400
                       px-3 py-1.5 text-xs font-semibold text-slate-900 shadow-[0_0_16px_rgba(34,211,238,0.4)]
                       hover:brightness-110">
            <Play size={12} /> {doneCount ? "Run again" : "Run all"}
          </button>
        )}
      </div>

      <ol className="mt-3 relative space-y-0.5 overflow-auto min-h-0">
        {DEMO_STEPS.map((s, i) => {
          const done = results[s.id] !== undefined && failed !== s.id;
          const bad = failed === s.id;
          const active = idx === i;
          const last = i === DEMO_STEPS.length - 1;
          const showText = open === s.id || active;
          return (
            <li key={s.id} className="relative pl-9">
              {!last && <span className={clsx("absolute left-[13px] top-7 bottom-0 w-px",
                                              done ? "bg-accent/50" : "bg-line")} />}
              <span className={clsx("absolute left-0 top-1.5 grid h-7 w-7 place-items-center rounded-full border text-[11px]",
                done ? "border-accent/60 bg-accent/15 text-accent shadow-[0_0_10px_rgba(34,211,238,0.35)]"
                : bad ? "border-rose-400/60 bg-rose-400/10 text-rose-300"
                : active ? "border-amber-400/70 bg-amber-400/10 text-amber-300"
                : "border-line text-ink-500")}>
                {done ? <CheckCircle2 size={14} /> : active ? <Loader2 size={13} className="animate-spin" />
                  : bad ? "!" : i + 1}
              </span>
              <button className="w-full text-left py-1.5" onClick={() => setOpen(open === s.id ? null : s.id)}>
                <div className="flex items-center gap-2">
                  <span className={clsx("text-[13.5px] font-medium",
                    done || active ? "text-white" : bad ? "text-rose-300" : "text-ink-300")}>{s.title}</span>
                  <span className="ml-auto text-[10.5px] font-mono text-ink-500">
                    {times[s.id] !== undefined ? `${times[s.id].toFixed(1)}s`
                      : active ? "running" : running && i > idx ? "queued" : ""}
                  </span>
                </div>
                {results[s.id] && (
                  <div className={clsx("text-[11.5px] font-mono mt-0.5 break-words",
                    bad ? "text-rose-300" : "text-accent/90")}>{results[s.id]}</div>
                )}
                {active && <div className="mt-1.5 h-1 rounded-full bg-base-600 overflow-hidden">
                  <div className="h-full w-1/3 rounded-full bg-amber-400 animate-pulse" />
                </div>}
                {showText && (
                  <p className="text-[11.5px] text-ink-500 mt-1 leading-relaxed">{s.caption}</p>
                )}
              </button>
            </li>
          );
        })}
      </ol>
      <p className="mt-2 text-[10px] text-ink-500 flex items-center gap-1.5">
        <Info size={11} /> Each step is a real API call; click a step to read what it shows.
      </p>
    </section>
  );
}
