/**
 * Landing-page scene: a night-side Earth on the left (drag to rotate), two
 * satellites on orbit (Sentinel-2 and ResourceSat-2A / LISS-IV, the sensors the
 * project uses), and a large glowing map of India on the right with a road web
 * and lit cities, joined to India's position on the globe by two light lines.
 *
 * Pure canvas, no dependencies. Respects prefers-reduced-motion.
 * Artwork only: the road web is decorative, not project data.
 */
import { useEffect, useRef } from "react";
import { LAND_MASK } from "@/assets/landMask";
import { INDIA_MASK, INDIA_RINGS } from "@/assets/indiaMask";

type V3 = [number, number, number];
type Mask = { rows: number; cols: number; step: number; lat0: number; lon0: number; b64: string };
const DEG = Math.PI / 180;

const toV = (lat: number, lon: number): V3 =>
  [Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon)];

function maskTest(M: Mask, centred: boolean) {
  const bytes = Uint8Array.from(atob(M.b64), (c) => c.charCodeAt(0));
  return (lat: number, lon: number) => {
    const f = centred ? Math.floor : Math.round;
    const r = f((lat - M.lat0) / M.step);
    if (r < 0 || r >= M.rows) return false;
    let c = f((lon - M.lon0) / M.step);
    if (centred) { if (c < 0 || c >= M.cols) return false; }
    else c = ((c % M.cols) + M.cols) % M.cols;
    const i = r * M.cols + c;
    return ((bytes[i >> 3] >> (7 - (i & 7))) & 1) === 1;
  };
}
const isLand = maskTest(LAND_MASK, false);
const inIndia = maskTest(INDIA_MASK, true);

function rng(seed: number) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
}

// Major cities (lit with a warm glow) and other towns for the road web.
const METROS: [number, number][] = [
  [28.6, 77.2], [19.08, 72.88], [22.57, 88.36], [13.08, 80.27], [12.97, 77.59],
  [17.39, 78.49], [23.02, 72.57], [18.52, 73.86], [26.85, 80.95], [21.15, 79.09],
];
const TOWNS: [number, number][] = [
  [34.08, 74.8], [34.15, 77.58], [31.63, 74.87], [30.73, 76.78], [30.32, 78.03], [25.6, 85.1],
  [25.32, 82.97], [23.34, 85.31], [20.3, 85.82], [21.25, 81.63], [22.72, 75.86], [21.17, 72.83],
  [26.24, 73.02], [24.58, 73.71], [17.69, 83.22], [15.5, 73.83], [12.91, 74.86], [9.93, 76.27],
  [8.52, 76.94], [9.93, 78.12], [11.0, 76.96], [25.58, 91.89], [24.82, 93.94], [26.73, 88.4],
  [26.22, 78.18], [19.88, 75.34], [17.66, 75.91], [32.73, 74.86], [26.14, 91.74], [23.26, 77.41],
  [26.92, 75.79], [16.5, 80.64], [15.83, 78.04], [14.44, 79.99], [10.79, 78.7], [27.18, 78.01],
  [29.39, 76.97], [24.79, 85.0], [22.31, 73.18], [20.94, 77.75], [23.18, 79.95], [27.49, 94.9],
];

function meshLinks(nodes: [number, number][], k: number, maxDeg: number): [number, number][] {
  const seen = new Set<string>(), links: [number, number][] = [];
  nodes.forEach((a, i) => {
    const near: [number, number][] = [];
    nodes.forEach((b, j) => {
      if (j === i) return;
      const d = Math.hypot(a[0] - b[0], (a[1] - b[1]) * Math.cos(a[0] * DEG));
      if (d < maxDeg) near.push([j, d]);
    });
    near.sort((x, y) => x[1] - y[1]).slice(0, k).forEach(([j]) => {
      const key = i < j ? `${i}|${j}` : `${j}|${i}`;
      if (!seen.has(key)) { seen.add(key); links.push([i, j]); }
    });
  });
  return links;
}

export interface HeroAnchors {
  /** screen positions (CSS px) of the India map box, for HTML callouts */
  map: { x: number; y: number; w: number; h: number } | null;
}

export default function HeroScene({ className = "", onLayout }: {
  className?: string; onLayout?: (a: HeroAnchors) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current!;
    const g = cv.getContext("2d")!;
    const rand = rng(5);

    // ---------------------------------------------------------- globe data
    const world: { v: V3; city: boolean }[] = [];
    for (let lat = -56; lat <= 80; lat += 0.8) {
      const dl = 0.8 / Math.max(0.2, Math.cos(lat * DEG));
      for (let lon = -180; lon < 180; lon += dl) {
        if (isLand(lat, lon)) world.push({ v: toV(lat * DEG, lon * DEG), city: rand() < 0.08 });
      }
    }
    const globeIndia: V3[] = [];
    for (let lat = 6; lat <= 37; lat += 0.6) {
      const dl = 0.6 / Math.cos(lat * DEG);
      for (let lon = 68; lon <= 98; lon += dl) if (inIndia(lat, lon)) globeIndia.push(toV(lat * DEG, lon * DEG));
    }
    const INDIA_C = toV(22 * DEG, 79 * DEG);
    const grat: V3[][] = [];
    for (let lat = -60; lat <= 60; lat += 30) {
      const l: V3[] = []; for (let lon = -180; lon <= 180; lon += 4) l.push(toV(lat * DEG, lon * DEG)); grat.push(l);
    }
    for (let lon = -180; lon < 180; lon += 30) {
      const l: V3[] = []; for (let lat = -84; lat <= 84; lat += 4) l.push(toV(lat * DEG, lon * DEG)); grat.push(l);
    }
    const lmNodes: [number, number][] = [];
    { const r2 = rng(3);
      for (let lat = -54; lat <= 76; lat += 3.2) {
        const dl = 3.2 / Math.max(0.2, Math.cos(lat * DEG));
        for (let lon = -180; lon < 180; lon += dl) {
          const la = lat + (r2() - 0.5) * 2.6, lo = lon + (r2() - 0.5) * dl * 0.9;
          if (isLand(la, lo)) lmNodes.push([la, lo]);
        }
      } }
    const lmLinks = meshLinks(lmNodes, 3, 5);
    const lmV = lmNodes.map(([la, lo]) => toV(la * DEG, lo * DEG));

    // ---------------------------------------------------------- map data
    // fine texture dots inside India
    const mapDots: [number, number][] = [];
    for (let lat = 6.3; lat <= 37.2; lat += 0.34) {
      for (let lon = 68.2; lon <= 97.6; lon += 0.34 / Math.cos(lat * DEG) * Math.cos(22 * DEG)) {
        if (inIndia(lat, lon)) mapDots.push([lat, lon]);
      }
    }
    const webNodes: [number, number][] = [...METROS, ...TOWNS];
    const wr = rng(17);
    while (webNodes.length < 950) {
      const lat = 7 + wr() * 30, lon = 68.5 + wr() * 29;
      if (inIndia(lat, lon)) webNodes.push([lat, lon]);
    }
    const webLinks = meshLinks(webNodes, 4, 2.4);
    const trunk = meshLinks([...METROS, ...TOWNS], 3, 6.5);   // brighter long roads
    const trunkNodes = [...METROS, ...TOWNS];

    const stars = Array.from({ length: 260 }, () => ({
      x: rand(), y: rand(), r: rand() * 1.1 + 0.2, a: rand() * 0.55 + 0.1, tw: rand() * 6.28,
    }));

    // sprites
    const mk = (stops: [number, string][]) => {
      const c = document.createElement("canvas"); c.width = c.height = 64;
      const x = c.getContext("2d")!;
      const gr = x.createRadialGradient(32, 32, 0, 32, 32, 32);
      stops.forEach(([o, col]) => gr.addColorStop(o, col));
      x.fillStyle = gr; x.fillRect(0, 0, 64, 64);
      return c;
    };
    const glowCyan = mk([[0, "rgba(255,255,255,1)"], [0.2, "rgba(165,243,252,0.7)"], [1, "rgba(34,211,238,0)"]]);
    const glowGold = mk([[0, "rgba(255,255,240,1)"], [0.15, "rgba(253,224,140,0.9)"],
                         [0.45, "rgba(251,191,36,0.25)"], [1, "rgba(251,146,60,0)"]]);

    // ---------------------------------------------------------- interaction
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let W = 0, H = 0, dpr = 1, raf = 0;
    const HOME_TILT = 18 * DEG;
    let yaw = -76 * DEG, tilt = HOME_TILT;
    const IDLE = reduced ? 0 : -0.03;
    let vYaw = IDLE, vTilt = 0, dragging = false, lastX = 0, lastY = 0, lastT = 0, idleSince = -1e9;
    let geo = { cx: 0, cy: 0, R: 1 };
    let lastLayout = "";

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);
    const pos = (e: PointerEvent) => { const b = cv.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; };
    const over = (x: number, y: number) => Math.hypot(x - geo.cx, y - geo.cy) < geo.R;
    const onDown = (e: PointerEvent) => {
      const [x, y] = pos(e); if (!over(x, y)) return;
      dragging = true; lastX = x; lastY = y; lastT = performance.now(); vYaw = 0; vTilt = 0;
      cv.setPointerCapture(e.pointerId); cv.style.cursor = "grabbing";
    };
    const onMove = (e: PointerEvent) => {
      const [x, y] = pos(e);
      if (!dragging) { cv.style.cursor = over(x, y) ? "grab" : "default"; return; }
      const now = performance.now(), dt = Math.max(8, now - lastT) / 1000;
      const dY = ((x - lastX) / geo.R) * 0.9, dT = ((y - lastY) / geo.R) * 0.9;
      yaw += dY; tilt = Math.max(-70 * DEG, Math.min(70 * DEG, tilt + dT));
      vYaw = dY / dt; vTilt = dT / dt; lastX = x; lastY = y; lastT = now;
    };
    const onUp = (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false; idleSince = performance.now();
      try { cv.releasePointerCapture(e.pointerId); } catch { /* released */ }
      cv.style.cursor = "grab";
    };
    cv.addEventListener("pointerdown", onDown);
    cv.addEventListener("pointermove", onMove);
    cv.addEventListener("pointerup", onUp);
    cv.addEventListener("pointercancel", onUp);

    let prev = performance.now();
    const frame = (time: number) => {
      const dt = Math.min(0.05, (time - prev) / 1000); prev = time;
      if (!dragging) {
        yaw += vYaw * dt; tilt = Math.max(-70 * DEG, Math.min(70 * DEG, tilt + vTilt * dt));
        const k = Math.min(1, dt * 1.5); vYaw += (IDLE - vYaw) * k; vTilt += -vTilt * k;
        if (!reduced && time - idleSince > 3500) tilt += (HOME_TILT - tilt) * Math.min(1, dt * 0.6);
      }
      const t = reduced ? 0 : time / 1000;
      const wide = W > 1000;

      // layout: globe behind the text (left), India map on the right
      // The globe sits directly behind the text column (centred on it, and no
      // taller than the text), so it frames the words instead of competing.
      // Big, bright globe behind the text column (TCS-style): centred to the
      // right of the text so it shows around and through the glass panel.
      const R = wide ? Math.min(H * 0.47, W * 0.22, 440) : Math.min(W * 0.4, H * 0.28);
      const cx = wide ? W * 0.3 : W * 0.5, cy = wide ? H * 0.5 : H * 0.3;
      geo = { cx, cy, R };
      const mapLatSpan = 32, mapLonSpan = 30 * Math.cos(22 * DEG);
      // keep India clear of the text column (which ends about 790 px from the left)
      const mapH = wide ? Math.max(0, Math.min(H * 0.82, (W - 890) / (mapLonSpan / mapLatSpan))) : 0;
      const mapScale = mapH / mapLatSpan;
      const mapW = mapLonSpan * mapScale;
      const mapX = W - mapW - W * 0.02, mapY = (H - mapH) / 2 + H * 0.01;
      const M = (lat: number, lon: number): [number, number] =>
        [mapX + (lon - 68) * Math.cos(22 * DEG) * mapScale, mapY + (38 - lat) * mapScale];

      const layout = wide ? `${Math.round(mapX)},${Math.round(mapY)},${Math.round(mapW)},${Math.round(mapH)}` : "none";
      if (layout !== lastLayout && onLayout) {
        lastLayout = layout;
        onLayout({ map: wide ? { x: mapX, y: mapY, w: mapW, h: mapH } : null });
      }

      const cr = Math.cos(yaw), sr = Math.sin(yaw), ct = Math.cos(tilt), st = Math.sin(tilt);
      const PD = 3.2, SIL = Math.sqrt(PD * PD - 1), ZV = 1 / PD;
      const project = (p: V3) => {
        const x = p[0] * cr + p[2] * sr, z0 = -p[0] * sr + p[2] * cr;
        const y = p[1] * ct - z0 * st, z = p[1] * st + z0 * ct;
        const k = SIL / (PD - z);
        return { x: cx + x * k * R, y: cy - y * k * R, z: (z - ZV) / (1 - ZV) };
      };

      g.clearRect(0, 0, W, H);

      // stars
      for (const d of stars) {
        const a = d.a * (0.6 + 0.4 * Math.sin(d.tw + t * 0.9));
        g.fillStyle = `rgba(200,235,255,${a.toFixed(3)})`;
        g.beginPath(); g.arc(d.x * W, d.y * H, d.r, 0, Math.PI * 2); g.fill();
      }

      // map grid behind India
      if (wide) {
        g.strokeStyle = "rgba(56,189,248,0.06)"; g.lineWidth = 1;
        const gs = mapScale * 2;
        g.beginPath();
        for (let x = mapX - mapW * 0.15; x < W; x += gs) { g.moveTo(x, 0); g.lineTo(x, H); }
        for (let y = 0; y < H; y += gs) { g.moveTo(mapX - mapW * 0.15, y); g.lineTo(W, y); }
        g.stroke();
      }

      // ---------------------------------------------------------- globe
      const atm = g.createRadialGradient(cx, cy, R * 0.96, cx, cy, R * 1.25);
      atm.addColorStop(0, "rgba(56,189,248,0.55)"); atm.addColorStop(0.35, "rgba(34,211,238,0.16)");
      atm.addColorStop(1, "rgba(3,10,20,0)");
      g.fillStyle = atm; g.beginPath(); g.arc(cx, cy, R * 1.25, 0, Math.PI * 2); g.fill();

      // orbits: tilted ellipses around the globe; satellites move along them
      const ORB = [
        { rx: 1.3, ry: 0.42, rot: -0.32, sp: 0.06, ph: -0.55, name: "Sentinel-2" },
        { rx: 1.2, ry: 0.78, rot: 0.6, sp: 0.05, ph: 1.0, name: "ResourceSat-2A" },
      ];
      const orbPt = (o: (typeof ORB)[number], a: number) => {
        const ex = Math.cos(a) * o.rx * R, ey = Math.sin(a) * o.ry * R;
        return { x: cx + ex * Math.cos(o.rot) - ey * Math.sin(o.rot),
                 y: cy + ex * Math.sin(o.rot) + ey * Math.cos(o.rot), front: Math.sin(a) > 0 };
      };
      const orbit = (o: (typeof ORB)[number], front: boolean) => {
        g.strokeStyle = `rgba(165,243,252,${front ? 0.32 : 0.12})`; g.lineWidth = 1;
        g.beginPath(); let on = false;
        for (let i = 0; i <= 200; i++) {
          const q = orbPt(o, (i / 200) * Math.PI * 2);
          const vis = q.front || Math.hypot(q.x - cx, q.y - cy) > R;
          if (vis === front) { if (!on) { g.moveTo(q.x, q.y); on = true; } else g.lineTo(q.x, q.y); } else on = false;
        }
        g.stroke();
      };
      const sats = ORB.map((o) => {
        const a = o.ph + t * o.sp;
        const q = orbPt(o, a), q2 = orbPt(o, a + 0.01 * Math.sign(o.sp));
        return { q, q2, vis: q.front || Math.hypot(q.x - cx, q.y - cy) > R };
      });
      const sat = (x: (typeof sats)[number]) => {
        const { q, q2 } = x;
        const ang = Math.atan2(q2.y - q.y, q2.x - q.x);
        g.save(); g.globalAlpha = x.vis ? 1 : 0.25;
        g.globalCompositeOperation = "lighter";
        g.drawImage(glowCyan, q.x - 22, q.y - 22, 44, 44);
        g.globalCompositeOperation = "source-over";
        g.translate(q.x, q.y); g.rotate(ang + Math.PI / 2); g.scale(1.25, 1.25);
        g.fillStyle = "#1E3A8A"; g.strokeStyle = "#93C5FD"; g.lineWidth = 0.6;
        for (const x0 of [-21, 8]) {
          g.fillRect(x0, -4.5, 13, 9); g.strokeRect(x0, -4.5, 13, 9);
          g.beginPath(); for (let k = 1; k < 4; k++) { g.moveTo(x0 + k * 3.25, -4.5); g.lineTo(x0 + k * 3.25, 4.5); } g.stroke();
        }
        g.strokeStyle = "#CBD5E1"; g.lineWidth = 1.2;
        g.beginPath(); g.moveTo(-8, 0); g.lineTo(8, 0); g.stroke();
        g.fillStyle = "#E5E7EB"; g.fillRect(-6, -7, 12, 14);
        g.fillStyle = "#B45309"; g.fillRect(-6, -7, 12, 4);
        g.fillStyle = "#0F172A"; g.beginPath(); g.arc(0, 4, 2, 0, Math.PI * 2); g.fill();
        g.restore();
      };
      ORB.forEach((o) => orbit(o, false));
      sats.filter((x) => !x.vis).forEach(sat);

      // ocean
      const oc = g.createRadialGradient(cx - R * 0.35, cy - R * 0.4, R * 0.05, cx, cy, R);
      oc.addColorStop(0, "#17608F"); oc.addColorStop(0.55, "#0C3A60"); oc.addColorStop(1, "#072643");
      g.fillStyle = oc; g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fill();

      // graticule
      g.strokeStyle = "rgba(125,211,252,0.18)"; g.lineWidth = 0.7;
      for (const l of grat) {
        g.beginPath(); let on = false;
        for (const p of l) { const q = project(p); if (q.z <= 0) { on = false; continue; }
          if (!on) { g.moveTo(q.x, q.y); on = true; } else g.lineTo(q.x, q.y); }
        g.stroke();
      }
      // continent mesh
      const pm = lmV.map(project);
      g.strokeStyle = "rgba(125,230,255,0.36)"; g.lineWidth = 0.7;
      g.beginPath();
      for (const [i, j] of lmLinks) { const a = pm[i], b = pm[j];
        if (a.z <= 0.05 || b.z <= 0.05) continue; g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); }
      g.stroke();

      // land (night side: dim blue land with warm city lights)
      g.globalCompositeOperation = "lighter";
      const ds = Math.max(1.7, R / 150);
      for (const w of world) {
        const q = project(w.v); if (q.z <= 0) continue;
        if (w.city) {
          g.globalAlpha = 0.5 + 0.5 * q.z;
          g.drawImage(glowGold, q.x - 4, q.y - 4, 8, 8);
          g.globalAlpha = 1;
        } else {
          g.fillStyle = `rgba(140,225,255,${(0.6 + 0.4 * q.z).toFixed(3)})`;
          g.fillRect(q.x - ds / 2, q.y - ds / 2, ds, ds);
        }
      }
      for (const v of globeIndia) {
        const q = project(v); if (q.z <= 0) continue;
        g.fillStyle = `rgba(186,245,255,${(0.5 + 0.5 * q.z).toFixed(3)})`;
        g.fillRect(q.x - ds / 2, q.y - ds / 2, ds, ds);
      }
      g.globalCompositeOperation = "source-over";

      // terminator shading + rim
      const sh = g.createLinearGradient(cx - R, cy - R, cx + R * 0.6, cy + R);
      sh.addColorStop(0, "rgba(0,0,0,0)"); sh.addColorStop(1, "rgba(1,6,14,0.25)");
      g.fillStyle = sh; g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fill();
      const rim = g.createRadialGradient(cx, cy, R * 0.88, cx, cy, R);
      rim.addColorStop(0, "rgba(56,189,248,0)"); rim.addColorStop(1, "rgba(125,211,252,0.35)");
      g.fillStyle = rim; g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fill();
      g.strokeStyle = "rgba(165,243,252,0.6)"; g.lineWidth = 1.3;
      g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.stroke();

      const spec = g.createRadialGradient(cx - R * 0.45, cy - R * 0.5, 0, cx - R * 0.45, cy - R * 0.5, R);
      spec.addColorStop(0, "rgba(186,230,253,0.14)"); spec.addColorStop(1, "rgba(186,230,253,0)");
      g.fillStyle = spec; g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fill();
      // HUD rings
      g.save(); g.translate(cx, cy);
      g.strokeStyle = "rgba(165,243,252,0.35)"; g.lineWidth = 1;
      g.setLineDash([2, 6]); g.lineDashOffset = -t * 12;
      g.beginPath(); g.arc(0, 0, R * 1.08, 0, Math.PI * 2); g.stroke();
      g.setLineDash([]);
      g.strokeStyle = "rgba(34,211,238,0.55)"; g.lineWidth = 2;
      const ra = t * 0.25;
      g.beginPath(); g.arc(0, 0, R * 1.15, ra, ra + 0.9); g.stroke();
      g.beginPath(); g.arc(0, 0, R * 1.15, ra + Math.PI, ra + Math.PI + 0.5); g.stroke();
      g.restore();

      ORB.forEach((o) => orbit(o, true));
      sats.filter((x) => x.vis).forEach(sat);

      if (!wide) { raf = requestAnimationFrame(frame); return; }

      // ---------------------------------------------------------- connectors
      const iq = project(INDIA_C);
      const from = iq.z > 0 ? iq : { x: cx + R * 0.7, y: cy - R * 0.2 };
      const [tx1, ty1] = M(35, 75), [tx2, ty2] = M(9, 77);
      const lg = g.createLinearGradient(from.x, from.y, tx1, ty1);
      lg.addColorStop(0, "rgba(165,243,252,0.65)"); lg.addColorStop(1, "rgba(165,243,252,0.08)");
      g.strokeStyle = lg; g.lineWidth = 1;
      g.beginPath(); g.moveTo(from.x, from.y); g.lineTo(tx1, ty1);
      g.moveTo(from.x, from.y); g.lineTo(tx2, ty2); g.stroke();
      if (iq.z > 0) {
        g.globalCompositeOperation = "lighter";
        g.drawImage(glowCyan, from.x - 10, from.y - 10, 20, 20);
        g.globalCompositeOperation = "source-over";
      }

      // ---------------------------------------------------------- India map
      const outline = new Path2D();
      for (const ring of INDIA_RINGS) {
        for (let i = 0; i < ring.length; i += 2) {
          const [x, y] = M(ring[i + 1], ring[i]);
          if (i === 0) outline.moveTo(x, y); else outline.lineTo(x, y);
        }
        outline.closePath();
      }
      // glow around the border
      g.save();
      g.shadowColor = "rgba(34,211,238,0.9)"; g.shadowBlur = 28;
      g.strokeStyle = "rgba(103,232,249,0.5)"; g.lineWidth = 3;
      g.stroke(outline);
      g.restore();
      // translucent fill
      const fg = g.createLinearGradient(mapX, mapY, mapX + mapW, mapY + mapH);
      fg.addColorStop(0, "rgba(14,116,144,0.30)"); fg.addColorStop(1, "rgba(8,80,120,0.22)");
      g.fillStyle = fg; g.fill(outline);

      g.save();
      g.clip(outline);
      g.globalCompositeOperation = "lighter";
      // texture dots
      for (const [la, lo] of mapDots) {
        const [x, y] = M(la, lo);
        g.fillStyle = "rgba(103,232,249,0.35)";
        g.fillRect(x - 0.8, y - 0.8, 1.6, 1.6);
      }
      // dense road web
      g.strokeStyle = "rgba(125,230,255,0.32)"; g.lineWidth = 0.7;
      g.beginPath();
      for (const [i, j] of webLinks) {
        const [x1, y1] = M(...webNodes[i]), [x2, y2] = M(...webNodes[j]);
        g.moveTo(x1, y1); g.lineTo(x2, y2);
      }
      g.stroke();
      // highways
      g.shadowColor = "rgba(103,232,249,0.9)"; g.shadowBlur = 8;
      g.strokeStyle = "rgba(207,250,254,0.85)"; g.lineWidth = 1.3;
      g.beginPath();
      for (const [i, j] of trunk) {
        const [x1, y1] = M(...trunkNodes[i]), [x2, y2] = M(...trunkNodes[j]);
        g.moveTo(x1, y1); g.lineTo(x2, y2);
      }
      g.stroke();
      g.shadowBlur = 0;
      // junction points
      for (let i = 0; i < webNodes.length; i++) {
        const [x, y] = M(...webNodes[i]);
        g.fillStyle = "rgba(224,252,255,0.9)";
        g.fillRect(x - 0.9, y - 0.9, 1.8, 1.8);
        if (i % 4 === 0) { g.globalAlpha = 0.6; g.drawImage(glowCyan, x - 4, y - 4, 8, 8); g.globalAlpha = 1; }
      }
      g.restore();
      // crisp border on top
      g.strokeStyle = "rgba(186,245,255,0.95)"; g.lineWidth = 1.4;
      g.stroke(outline);
      g.globalCompositeOperation = "lighter";
      // pulses on trunk roads
      if (!reduced) {
        trunk.forEach(([i, j], k) => {
          if (k % 2) return;
          const [x1, y1] = M(...trunkNodes[i]), [x2, y2] = M(...trunkNodes[j]);
          const u = (t * 0.25 + k * 0.173) % 1;
          g.drawImage(glowCyan, x1 + (x2 - x1) * u - 4, y1 + (y2 - y1) * u - 4, 8, 8);
        });
      }
      // lit metros
      METROS.forEach(([la, lo], i) => {
        const [x, y] = M(la, lo);
        const pulse = reduced ? 1 : 0.85 + 0.15 * Math.sin(t * 1.6 + i);
        const s = (i < 7 ? 80 : 50) * pulse;
        g.drawImage(glowGold, x - s / 2, y - s / 2, s, s);
      });
      g.globalCompositeOperation = "source-over";

      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      cv.removeEventListener("pointerdown", onDown);
      cv.removeEventListener("pointermove", onMove);
      cv.removeEventListener("pointerup", onUp);
      cv.removeEventListener("pointercancel", onUp);
    };
  }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  return <canvas ref={ref} className={className} style={{ touchAction: "none" }}
                 role="img" aria-label="Earth with Sentinel-2 and ResourceSat-2A satellites and a map of India's road network; drag the globe to rotate" />;
}
