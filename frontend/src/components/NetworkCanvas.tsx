/**
 * Network canvas - the centrepiece view.
 *
 * Renders the routable graph returned by the backend. Canvas rather than SVG
 * because a real city graph runs to tens of thousands of segments and SVG nodes
 * would stall the browser; canvas redraws the whole scene in one pass.
 *
 * It draws only what the backend sent. Nothing is generated client-side, and
 * healed (inferred) segments are always drawn distinctly from observed ones so
 * an inferred road can never be mistaken for an observed one.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GeoFeature, GeoJSONCollection } from "@/types/api";
import { riskColor } from "./ui";

interface Props {
  geojson: GeoJSONCollection | null;
  scores?: Record<string, number>;
  gatekeepers?: number[];
  disabledNodes?: number[];
  disabledEdges?: number[][];
  routeOriginal?: number[] | null;
  routeAlternate?: number[] | null;
  selected?: number | null;
  onSelectNode?: (n: number) => void;
  /** Labelled pins (e.g. trip origin "A" and destination "B"). */
  markers?: { node: number; label: string; color: string }[];
  showHealed?: boolean;
  showCriticality?: boolean;
  backdrop?: string | null;
  /** Image size in pixels. When given, the backdrop is drawn at its true
   *  position (graph coordinates are [col, -row]); otherwise it is stretched
   *  to the network's bounding box. */
  backdropExtent?: { width: number; height: number } | null;
  /** Backdrop opacity, 0-1. */
  backdropOpacity?: number;
  className?: string;
}

interface Layout {
  nodes: Map<number, { x: number; y: number }>;
  edges: { u: number; v: number; pts: [number, number][]; healed: boolean; crit: number }[];
  bbox: [number, number, number, number];
}

export default function NetworkCanvas({
  geojson, scores = {}, gatekeepers = [], disabledNodes = [], disabledEdges = [],
  routeOriginal, routeAlternate, selected, onSelectNode, markers = [],
  showHealed = true, showCriticality = true, backdrop, backdropExtent,
  backdropOpacity = 0.34, className,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ k: 1, tx: 0, ty: 0 });
  const [hover, setHover] = useState<number | null>(null);
  const drag = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);
  const bgImg = useRef<HTMLImageElement | null>(null);

  /* ---------------------------------------------------------- layout */
  const layout = useMemo<Layout | null>(() => {
    if (!geojson?.features?.length) return null;
    const nodes = new Map<number, { x: number; y: number }>();
    const edges: Layout["edges"] = [];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    const track = (x: number, y: number) => {
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    };

    for (const f of geojson.features as GeoFeature[]) {
      if (f.geometry.type === "Point") {
        const [x, y] = f.geometry.coordinates;
        nodes.set(Number(f.properties.node), { x, y });
        track(x, y);
      } else {
        const pts = f.geometry.coordinates as [number, number][];
        pts.forEach(([x, y]) => track(x, y));
        edges.push({
          u: Number(f.properties.u),
          v: Number(f.properties.v),
          pts,
          healed: Boolean(f.properties.healed),
          crit: Number(f.properties.criticality ?? 0),
        });
      }
    }
    if (!isFinite(minX)) return null;
    return { nodes, edges, bbox: [minX, minY, maxX, maxY] };
  }, [geojson]);

  useEffect(() => {
    if (!backdrop) { bgImg.current = null; return; }
    const img = new Image();
    img.onload = () => { bgImg.current = img; draw(); };
    img.src = backdrop;
  }, [backdrop]);

  /* ------------------------------------------------- world -> screen */
  const transform = useCallback(() => {
    const cv = ref.current;
    if (!cv || !layout) return null;
    const [minX, minY, maxX, maxY] = layout.bbox;
    const w = cv.clientWidth, h = cv.clientHeight;
    const pad = 28;
    const s = Math.min((w - pad * 2) / Math.max(maxX - minX, 1e-6),
                       (h - pad * 2) / Math.max(maxY - minY, 1e-6));
    const k = s * view.k;
    const ox = (w - (maxX - minX) * k) / 2 - minX * k + view.tx;
    // Flip Y: GeoJSON y grows north, canvas y grows down.
    const oy = (h + (maxY - minY) * k) / 2 + minY * k + view.ty;
    return { k, ox, oy, toScreen: (x: number, y: number): [number, number] =>
      [x * k + ox, oy - y * k] };
  }, [layout, view]);

  /* --------------------------------------------------------- drawing */
  const draw = useCallback(() => {
    const cv = ref.current;
    const t = transform();
    if (!cv || !layout || !t) return;

    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== w * dpr || cv.height !== h * dpr) {
      cv.width = w * dpr; cv.height = h * dpr;
    }
    const g = cv.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    g.fillStyle = "#080B10";
    g.fillRect(0, 0, w, h);

    if (bgImg.current) {
      const [minX, minY, maxX, maxY] = backdropExtent
        ? [0, -backdropExtent.height, backdropExtent.width, 0]
        : layout.bbox;
      const [x0, y1] = t.toScreen(minX, maxY);
      const [x1, y0] = t.toScreen(maxX, minY);
      g.globalAlpha = backdropOpacity;
      g.drawImage(bgImg.current, x0, y1, x1 - x0, y0 - y1);
      g.globalAlpha = 1;
    }

    const deadNodes = new Set(disabledNodes);
    const deadEdges = new Set(disabledEdges.map((e) => `${Math.min(e[0], e[1])}-${Math.max(e[0], e[1])}`));

    // Edges
    for (const e of layout.edges) {
      const key = `${Math.min(e.u, e.v)}-${Math.max(e.u, e.v)}`;
      const dead = deadEdges.has(key) || deadNodes.has(e.u) || deadNodes.has(e.v);
      g.beginPath();
      e.pts.forEach(([x, y], i) => {
        const [sx, sy] = t.toScreen(x, y);
        i === 0 ? g.moveTo(sx, sy) : g.lineTo(sx, sy);
      });

      if (dead) {
        g.strokeStyle = "#EF4444"; g.lineWidth = 2.4; g.setLineDash([]);
      } else if (e.healed && showHealed) {
        g.strokeStyle = "#38BDF8"; g.lineWidth = 1.9; g.setLineDash([5, 4]);
      } else if (showCriticality && e.crit > 0) {
        g.strokeStyle = riskColor(e.crit);
        g.lineWidth = 0.9 + 2.6 * e.crit;
        g.setLineDash([]);
      } else {
        g.strokeStyle = "#3A4A5E"; g.lineWidth = 1.2; g.setLineDash([]);
      }
      g.globalAlpha = dead ? 0.85 : 1;
      g.stroke();
      g.globalAlpha = 1;
    }
    g.setLineDash([]);

    // Road shape per junction pair (shortest one if two roads join the same pair)
    const plen = (q: [number, number][]) =>
      q.reduce((acc, p, i) => i ? acc + Math.hypot(p[0] - q[i - 1][0], p[1] - q[i - 1][1]) : 0, 0);
    const edgeShape = new Map<string, [number, number][]>();
    for (const e of layout.edges) {
      const key = `${Math.min(e.u, e.v)}-${Math.max(e.u, e.v)}`;
      const prev = edgeShape.get(key);
      if (!prev || plen(e.pts) < plen(prev)) edgeShape.set(key, e.pts);
    }

    // Routes drawn above the network
    const drawRoute = (path: number[] | null | undefined, color: string, width: number) => {
      if (!path?.length) return;
      // Follow each road's real shape (its pixel trace), not a straight line
      // between its two junctions.
      const d2 = (p: [number, number], q: { x: number; y: number }) =>
        (p[0] - q.x) ** 2 + (p[1] - q.y) ** 2;
      const pts: [number, number][] = [];
      for (let i = 0; i < path.length - 1; i++) {
        const a = layout.nodes.get(path[i]), b = layout.nodes.get(path[i + 1]);
        if (!a || !b) continue;
        const shape = edgeShape.get(`${Math.min(path[i], path[i + 1])}-${Math.max(path[i], path[i + 1])}`);
        let seg: [number, number][] = shape && shape.length > 1 ? shape : [[a.x, a.y], [b.x, b.y]];
        if (d2(seg[0], a) > d2(seg[seg.length - 1], a)) seg = [...seg].reverse();
        pts.push([a.x, a.y], ...seg, [b.x, b.y]);
      }
      if (path.length === 1) {
        const a = layout.nodes.get(path[0]);
        if (a) pts.push([a.x, a.y]);
      }
      g.beginPath();
      pts.forEach(([x, y], i) => {
        const [sx, sy] = t.toScreen(x, y);
        i === 0 ? g.moveTo(sx, sy) : g.lineTo(sx, sy);
      });
      g.strokeStyle = color; g.lineWidth = width;
      g.lineCap = "round"; g.lineJoin = "round";
      g.stroke();
    };
    drawRoute(routeOriginal, "#38BDF8", 3.4);
    drawRoute(routeAlternate, "#F97316", 3.4);

    // Nodes: only meaningful ones, so the view stays readable
    const gates = new Set(gatekeepers);
    for (const [id, p] of layout.nodes) {
      const sc = scores[String(id)] ?? 0;
      const isGate = gates.has(id);
      const isSel = selected === id;
      const isHover = hover === id;
      const isDead = deadNodes.has(id);
      if (!isGate && !isSel && !isHover && !isDead && sc < 0.25) continue;

      const [x, y] = t.toScreen(p.x, p.y);
      const r = isSel ? 7 : isHover ? 6 : 2.6 + 4 * sc;

      g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2);
      g.fillStyle = isDead ? "#EF4444" : showCriticality ? riskColor(sc) : "#9FB0C4";
      g.fill();

      if (isGate) {
        g.beginPath(); g.arc(x, y, r + 3.5, 0, Math.PI * 2);
        g.strokeStyle = "#38BDF8"; g.lineWidth = 1.3; g.stroke();
      }
      if (isSel) {
        g.beginPath(); g.arc(x, y, r + 7, 0, Math.PI * 2);
        g.strokeStyle = "#E8EDF4"; g.lineWidth = 1.6; g.stroke();
      }
      if (isDead) {
        const q = r + 7;
        g.lineCap = "round";
        g.strokeStyle = "#0B1220"; g.lineWidth = 6;        // dark halo for contrast
        g.beginPath();
        g.moveTo(x - q, y - q); g.lineTo(x + q, y + q);
        g.moveTo(x + q, y - q); g.lineTo(x - q, y + q);
        g.stroke();
        g.strokeStyle = "#EF4444"; g.lineWidth = 3;
        g.stroke();
      }
    }

    // Labelled pins on top of everything
    for (const m of markers) {
      const p = layout.nodes.get(m.node);
      if (!p) continue;
      const [x, y] = t.toScreen(p.x, p.y);
      g.beginPath(); g.arc(x, y, 11, 0, Math.PI * 2);
      g.fillStyle = m.color; g.fill();
      g.lineWidth = 2; g.strokeStyle = "#0B1220"; g.stroke();
      g.fillStyle = "#0B1220";
      g.font = "bold 12px Inter, system-ui, sans-serif";
      g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText(m.label, x, y + 0.5);
    }
  }, [layout, transform, scores, gatekeepers, disabledNodes, disabledEdges,
      routeOriginal, routeAlternate, selected, hover, showHealed, showCriticality, markers,
      backdropExtent, backdropOpacity]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => {
    const ro = new ResizeObserver(() => draw());
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, [draw]);

  /* ------------------------------------------------------ interaction */
  const pick = useCallback((ev: React.MouseEvent): number | null => {
    const t = transform();
    const cv = ref.current;
    if (!t || !layout || !cv) return null;
    const rect = cv.getBoundingClientRect();
    const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    let best: number | null = null, bestD = 13;
    for (const [id, p] of layout.nodes) {
      const [x, y] = t.toScreen(p.x, p.y);
      const d = Math.hypot(x - mx, y - my);
      if (d < bestD) { bestD = d; best = id; }
    }
    return best;
  }, [transform, layout]);

  return (
    <div ref={wrapRef} className={className}>
      <canvas
        ref={ref}
        className="w-full h-full block cursor-crosshair rounded-lg"
        onMouseDown={(e) => {
          drag.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty };
        }}
        onMouseUp={() => { drag.current = null; }}
        onMouseLeave={() => { drag.current = null; setHover(null); }}
        onMouseMove={(e) => {
          if (drag.current) {
            setView((v) => ({
              ...v,
              tx: drag.current!.tx + (e.clientX - drag.current!.x),
              ty: drag.current!.ty + (e.clientY - drag.current!.y),
            }));
          } else {
            setHover(pick(e));
          }
        }}
        onClick={(e) => {
          const n = pick(e);
          if (n !== null && onSelectNode) onSelectNode(n);
        }}
        onWheel={(e) => {
          const f = e.deltaY < 0 ? 1.12 : 1 / 1.12;
          setView((v) => ({ ...v, k: Math.min(Math.max(v.k * f, 0.4), 14) }));
        }}
      />
    </div>
  );
}
