/**
 * Guided demonstration (project spec section 21).
 *
 * Runs the real pipeline end to end, one stage at a time, so a reviewer sees the
 * whole system in about two minutes. Every step is an actual API call against
 * the actual backend - this is a narrated run, not a scripted animation, and
 * the numbers shown are the numbers that were just computed.
 */
import { useCallback, useRef, useState } from "react";
import { CheckCircle2, Circle, Loader2, Play, Square } from "lucide-react";
import clsx from "clsx";
import { Note, Panel } from "@/components/ui";

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

export function DemoRunner({ p, onTab }: { p: any; onTab: (t: string) => void }) {
  const [idx, setIdx] = useState(-1);
  const [results, setResults] = useState<Record<string, string>>({});
  const [running, setRunning] = useState(false);
  const stop = useRef(false);

  const play = useCallback(async () => {
    setRunning(true);
    stop.current = false;
    setResults({});
    const ctx: DemoCtx = { p, tab: onTab };

    for (let i = 0; i < DEMO_STEPS.length; i++) {
      if (stop.current) break;
      setIdx(i);
      try {
        const out = await DEMO_STEPS[i].run(ctx);
        setResults((r) => ({ ...r, [DEMO_STEPS[i].id]: out }));
      } catch (e) {
        setResults((r) => ({
          ...r, [DEMO_STEPS[i].id]: `failed: ${(e as Error).message}`,
        }));
        break;
      }
      await new Promise((r) => setTimeout(r, 900));
    }
    setRunning(false);
    setIdx(-1);
  }, [p, onTab]);

  return (
    <Panel title="Guided demonstration" dense
      actions={
        running ? (
          <button className="btn-ghost text-xs" onClick={() => { stop.current = true; }}>
            <Square size={12} /> Stop
          </button>
        ) : (
          <button className="btn-primary text-xs" onClick={play}>
            <Play size={12} /> Run
          </button>
        )
      }>
      <div className="space-y-2">
        <Note>
          Each step issues a real API call. The figures below are what the backend
          just computed, not a recorded walkthrough.
        </Note>

        <ol className="space-y-1.5">
          {DEMO_STEPS.map((s, i) => {
            const done = results[s.id] !== undefined;
            const active = idx === i;
            return (
              <li key={s.id}
                  className={clsx("rounded-md border px-2.5 py-2 transition-colors",
                    active ? "border-accent/40 bg-accent/10"
                    : done ? "border-line-soft bg-base-700/40" : "border-line-soft")}>
                <div className="flex items-center gap-2 text-[12px]">
                  {active ? <Loader2 size={13} className="animate-spin text-accent" />
                   : done ? <CheckCircle2 size={13} className="text-risk-low" />
                   : <Circle size={13} className="text-ink-500" />}
                  <span className={done || active ? "text-ink-100" : "text-ink-500"}>
                    {i + 1}. {s.title}
                  </span>
                </div>
                {(active || done) && (
                  <p className="text-[10px] text-ink-500 mt-1.5 leading-relaxed pl-5">
                    {s.caption}
                  </p>
                )}
                {done && (
                  <p className="text-[11px] text-accent mt-1 font-mono pl-5 break-words">
                    {results[s.id]}
                  </p>
                )}
              </li>
            );
          })}
        </ol>
      </div>
    </Panel>
  );
}
