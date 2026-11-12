/**
 * Road extraction. Shows before/after and the real metrics - or an explicit
 * "ground truth unavailable" state. Metrics are never invented.
 */
import { useState } from "react";
import { Play, Satellite } from "lucide-react";
import {
  Empty, Metric, Note, Panel, ProvenanceBanner, UnavailableMetric, fmt,
} from "@/components/ui";

export default function ExtractionSection({ p }: any) {
  const { state } = p;
  const [mode, setMode] = useState<"auto" | "demo" | "classical">("auto");
  const isDemo = state.imagery?.meta?.source?.startsWith("DEMO:");
  const m = state.extract?.metrics;

  return (
    <div className="flex-1 grid grid-cols-12 gap-3 min-h-0">
      <Panel className="col-span-12 lg:col-span-8" title="Before / after">
        {state.extract ? (
          <div className="grid grid-cols-2 gap-3 h-full">
            <Frame label="Input imagery" src={state.imagery.layers.original} />
            <Frame label="Road mask" src={state.extract.layers.mask} />
            <Frame label="Probability" src={state.extract.layers.probability} />
            <Frame label={isDemo ? "Occlusion map" : "Ground truth"}
                   src={state.imagery.layers.occlusion ?? state.imagery.layers.ground_truth} />
          </div>
        ) : (
          <Empty icon={Satellite} title="No extraction yet"
                 hint="Load imagery, then run extraction. The method used is always reported." />
        )}
      </Panel>

      <div className="col-span-12 lg:col-span-4 flex flex-col gap-3 overflow-auto">
        <Panel title="Run extraction" dense>
          <div className="space-y-3">
            <div className="space-y-1.5">
              {([
                ["auto", "Auto", "Trained model if a checkpoint exists, else classical baseline"],
                ["demo", "Simulated occluded segmentation", "Demo scenes only: fragments known ground truth"],
                ["classical", "Classical baseline", "Frangi ridge filter \u2014 not deep learning"],
              ] as const).map(([id, label, note]) => (
                <button key={id} onClick={() => setMode(id)}
                  disabled={id === "demo" && !isDemo}
                  className={`w-full text-left px-2.5 py-2 rounded-md border text-[12px] transition-colors
                    disabled:opacity-35 disabled:cursor-not-allowed ${
                    mode === id ? "border-accent/40 bg-accent/10" : "border-line hover:bg-base-700"}`}>
                  <div className="font-medium">{label}</div>
                  <div className="text-[10px] text-ink-500 mt-0.5">{note}</div>
                </button>
              ))}
            </div>
            <button className="btn-primary w-full justify-center"
                    disabled={!state.imagery || !!state.busy}
                    onClick={() => p.extract(mode)}>
              <Play size={14} /> Run extraction
            </button>
          </div>
        </Panel>

        {state.extract && (
          <>
            <Panel title="Provenance" dense>
              <ProvenanceBanner p={state.extract.provenance} />
            </Panel>

            <Panel title="Segmentation metrics" dense>
              {m?.available ? (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <Metric size="sm" label="IoU" value={fmt.n(m.iou, 3)} />
                    <Metric size="sm" label="Dice" value={fmt.n(m.dice, 3)} />
                    <Metric size="sm" label="Precision" value={fmt.n(m.precision, 3)} />
                    <Metric size="sm" label="Recall" value={fmt.n(m.recall, 3)} />
                    <Metric size="sm" label="clDice" value={fmt.n(m.cldice, 3)}
                            hint="Centreline agreement: punishes breaks that IoU ignores" />
                    <Metric size="sm" label="Relaxed IoU" value={fmt.n(m.relaxed_iou, 3)}
                            hint="4 px tolerance buffer" />
                  </div>
                  {m.recall_occluded !== null && m.recall_occluded !== undefined && (
                    <div className="mt-3 pt-3 border-t border-line-soft grid grid-cols-3 gap-3">
                      <Metric size="sm" label="Recall visible" value={fmt.n(m.recall_visible, 3)} />
                      <Metric size="sm" label="Recall occluded" value={fmt.n(m.recall_occluded, 3)}
                              tone="accent" hint="The number this project is actually about" />
                      <Metric size="sm" label="Gap" value={fmt.n(m.occlusion_gap, 3)}
                              tone={(m.occlusion_gap ?? 0) > 0.2 ? "warn" : "good"} />
                    </div>
                  )}
                </>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    {["IoU", "Dice", "Precision", "Recall"].map((k) => (
                      <UnavailableMetric key={k} label={k} reason={m?.reason} />
                    ))}
                  </div>
                  <div className="mt-3">
                    <Note tone="warn">
                      <strong>Ground truth unavailable.</strong> {m?.reason}
                    </Note>
                  </div>
                </>
              )}
            </Panel>

            <Panel title="Output" dense>
              <div className="grid grid-cols-2 gap-3">
                <Metric size="sm" label="Road pixels"
                        value={fmt.pct(state.extract.road_pixel_fraction)} />
                <Metric size="sm" label="Threshold" value={fmt.n(state.extract.threshold_used, 2)} />
                <Metric size="sm" label="Elapsed" value={state.extract.elapsed_ms} unit="ms" />
              </div>
            </Panel>
          </>
        )}
      </div>
    </div>
  );
}

function Frame({ label, src }: { label: string; src?: string }) {
  return (
    <div className="flex flex-col min-h-0">
      <div className="label mb-1">{label}</div>
      <div className="flex-1 bg-base-900 rounded-md overflow-hidden flex items-center justify-center">
        {src ? <img src={src} alt={label} className="max-w-full max-h-[230px] object-contain" />
             : <span className="text-[11px] text-ink-500">unavailable</span>}
      </div>
    </div>
  );
}
