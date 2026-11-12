/** Evaluation. Shows real numbers when ground truth exists, and an explicit
 *  "awaiting ground truth" state with instructions when it does not. */
import { BarChart3 } from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Empty, Metric, Note, Panel, ProvenanceBanner, fmt } from "@/components/ui";
import { Row } from "@/App";

export default function EvaluationSection({ p }: any) {
  const { state } = p;
  const m = state.extract?.metrics;
  const g = state.graph;

  if (!state.extract) {
    return <Panel title="Evaluation" className="flex-1">
      <Empty icon={BarChart3} title="Nothing to evaluate yet"
             hint="Run extraction and build the graph first." />
    </Panel>;
  }

  const bars = m?.available ? [
    { k: "IoU", v: m.iou ?? 0 },
    { k: "Dice", v: m.dice ?? 0 },
    { k: "clDice", v: m.cldice ?? 0 },
    { k: "Relaxed IoU", v: m.relaxed_iou ?? 0 },
    { k: "Precision", v: m.precision ?? 0 },
    { k: "Recall", v: m.recall ?? 0 },
  ] : [];

  return (
    <div className="flex-1 grid grid-cols-12 gap-3 min-h-0 overflow-auto">
      <Panel className="col-span-12 lg:col-span-7" title="Segmentation accuracy">
        {m?.available ? (
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bars} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1A2330" vertical={false} />
                <XAxis dataKey="k" tick={{ fill: "#647A93", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 1]} tick={{ fill: "#647A93", fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: "#0D1117", border: "1px solid #243040",
                                         borderRadius: 6, fontSize: 12 }}
                         formatter={(v: any) => Number(v).toFixed(4)} />
                <Bar dataKey="v" radius={[3, 3, 0, 0]}>
                  {bars.map((b, i) => (
                    <Cell key={i} fill={b.k === "clDice" ? "#38BDF8" : "#3A5A78"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="space-y-3">
            <Empty icon={BarChart3} title="Awaiting ground truth" hint={m?.reason ?? undefined} />
            <Note tone="warn">
              <strong>To run a real evaluation:</strong> generate OpenStreetMap
              reference masks with <code className="font-mono">scripts/prepare_groundtruth.py
              --raster data/raw/&lt;scene&gt;.tif</code>, or load a demo city where the
              ground truth is known by construction.
            </Note>
          </div>
        )}
      </Panel>

      <Panel className="col-span-12 lg:col-span-5" title="Provenance">
        <div className="space-y-3">
          <ProvenanceBanner p={state.extract.provenance} />
          <Note>
            Metrics describe this scene only. They are not a claim about
            generalisation, and validation numbers recorded at training time are
            reported separately from any test-set figure.
          </Note>
        </div>
      </Panel>

      {m?.available && m.recall_occluded != null && (
        <Panel className="col-span-12 lg:col-span-5" title="Occlusion recall">
          <div className="grid grid-cols-3 gap-3">
            <Metric label="Visible" value={fmt.n(m.recall_visible, 3)} size="sm" />
            <Metric label="Occluded" value={fmt.n(m.recall_occluded, 3)} size="sm" tone="accent" />
            <Metric label="Gap" value={fmt.n(m.occlusion_gap, 3)} size="sm"
                    tone={(m.occlusion_gap ?? 0) > 0.2 ? "warn" : "good"} />
          </div>
          <div className="mt-3">
            <Note>
              Recall split by whether the road pixel was occluded. The gap is the
              honest measure of how much the extractor <em>infers</em> versus merely
              <em> observes</em> &mdash; the central claim of this project.
            </Note>
          </div>
        </Panel>
      )}

      {g && (
        <Panel className="col-span-12 lg:col-span-7" title="Topology & network">
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-1.5 text-[11px]">
              <div className="label mb-1">Topology</div>
              <Row k="Components before healing" v={g.healing.components_before} />
              <Row k="Components after" v={g.healing.components_after} />
              <Row k="LCC node increase" v={`${g.connectivity_ratio.lcc_node_increase_pct}%`} />
              <Row k="LCC length increase" v={`${g.connectivity_ratio.lcc_length_increase_pct}%`} />
              <Row k="Inferred length" v={fmt.pct(g.healing.healed_length_fraction)} />
            </div>
            <div className="space-y-1.5 text-[11px]">
              <div className="label mb-1">Network</div>
              <Row k="Nodes" v={g.stats.nodes} />
              <Row k="Edges" v={g.stats.edges} />
              <Row k="Gatekeepers" v={g.stats.gatekeepers} />
              <Row k="Bridge edges" v={g.stats.bridge_edges} />
              <Row k="Redundancy α" v={g.stats.alpha_index} />
              <Row k="Mean shortest path"
                   v={g.stats.avg_shortest_path_m ? `${g.stats.avg_shortest_path_m} m` : "\u2014"} />
            </div>
          </div>
          <div className="mt-3">
            <Note>
              Connectivity Ratio is the growth of the largest connected component
              after MST healing &mdash; the direct measure of what topology repair bought.
            </Note>
          </div>
        </Panel>
      )}
    </div>
  );
}
