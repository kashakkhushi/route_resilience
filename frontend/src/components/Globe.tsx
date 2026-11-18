/**
 * Interactive globe for the landing page.
 *
 *  - Land as rows of small dots (even spacing on the sphere) in teal and green,
 *    on a dark ocean with a glowing rim, faint graticule and dashed orbit rings.
 *  - India drawn brightest: dense cyan dots, a glowing outline (official extent,
 *    including all of Jammu & Kashmir and Ladakh) and a web of roads between
 *    cities and towns with travelling light pulses.
 *  - Drag (mouse or touch) to rotate, with inertia; it then eases back to a
 *    slow spin with India facing the viewer.
 *
 * Pure canvas, no dependencies. Respects prefers-reduced-motion.
 */
import { useEffect, useRef } from "react";
import { LAND_MASK } from "@/assets/landMask";
import { INDIA_MASK } from "@/assets/indiaMask";

type V3 = [number, number, number];
type Mask = { rows: number; cols: number; step: number; lat0: number; lon0: number; b64: string };
const DEG = Math.PI / 180;

function toV(lat: number, lon: number): V3 {
  return [Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon)];
}

/** Lookup in a bit-mask grid; `centred` cells are sampled at their centre. */
function maskTest(M: Mask, centred: boolean) {
  const bytes = Uint8Array.from(atob(M.b64), (c) => c.charCodeAt(0));
  return (lat: number, lon: number) => {
    const f = centred ? Math.floor : Math.round;
    const r = f((lat - M.lat0) / M.step);
    const c = ((f((lon - M.lon0) / M.step) % M.cols) + M.cols) % M.cols;
    if (r < 0 || r >= M.rows) return false;
    if (centred && (lon - M.lon0 < 0 || lon - M.lon0 >= M.cols * M.step)) return false;
    const i = r * M.cols + c;
    return ((bytes[i >> 3] >> (7 - (i & 7))) & 1) === 1;
  };
}
const isLand = maskTest(LAND_MASK, false);
const inIndia = maskTest(INDIA_MASK, true);

/** Points on rows of latitude, evenly spaced along each row, where `keep` holds. */
function rowPoints(lat0: number, lat1: number, lon0: number, lon1: number, step: number,
                   keep: (lat: number, lon: number) => boolean): [number, number][] {
  const out: [number, number][] = [];
  for (let lat = lat0; lat <= lat1; lat += step) {
    const dl = step / Math.max(0.15, Math.cos(lat * DEG));
    for (let lon = lon0; lon < lon1; lon += dl) if (keep(lat, lon)) out.push([lat, lon]);
  }
  return out;
}

// Seeded random numbers, so the road web looks the same on every load.
function rng(seed: number) {
  return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
}

const CITIES: [number, number][] = [
  [28.6, 77.2], [19.1, 72.9], [12.97, 77.6], [13.08, 80.27], [22.57, 88.36], [17.4, 78.5],
  [16.5, 80.5], [26.1, 91.7], [23.0, 72.6], [26.9, 75.8], [18.5, 73.9], [26.8, 80.9],
  [23.26, 77.41], [21.15, 79.09], [34.08, 74.8], [34.15, 77.58], [31.63, 74.87], [30.73, 76.78],
  [30.32, 78.03], [25.6, 85.1], [25.32, 82.97], [23.34, 85.31], [20.3, 85.82], [21.25, 81.63],
  [22.72, 75.86], [21.17, 72.83], [26.24, 73.02], [24.58, 73.71], [17.69, 83.22], [15.5, 73.83],
  [12.91, 74.86], [9.93, 76.27], [8.52, 76.94], [9.93, 78.12], [11.0, 76.96], [25.58, 91.89],
  [24.82, 93.94], [26.73, 88.4], [26.22, 78.18], [19.88, 75.34], [17.66, 75.91], [32.73, 74.86],
];

/** Join each node to its k nearest neighbours closer than maxDeg (unique pairs). */
function meshLinks(nodes: [number, number][], k: number, maxDeg: number): [number, number][] {
  const dist = (a: [number, number], b: [number, number]) => {
    let dl = Math.abs(a[1] - b[1]); if (dl > 180) dl = 360 - dl;
    return Math.hypot(a[0] - b[0], dl * Math.cos(((a[0] + b[0]) / 2) * DEG));
  };
  const seen = new Set<string>(), links: [number, number][] = [];
  nodes.forEach((a, i) => {
    const near: [number, number][] = [];
    nodes.forEach((b, j) => {
      if (j === i || Math.abs(a[0] - b[0]) > maxDeg) return;
      const d = dist(a, b);
      if (d < maxDeg) near.push([j, d]);
    });
    near.sort((x, y) => x[1] - y[1]).slice(0, k).forEach(([j]) => {
      const key = i < j ? `${i}|${j}` : `${j}|${i}`;
      if (!seen.has(key)) { seen.add(key); links.push([i, j]); }
    });
  });
  return links;
}

/** Faint mesh over the other continents: jittered land nodes joined to neighbours. */
function landMesh() {
  const rand = rng(3);
  const nodes: [number, number][] = [];
  for (let lat = -54; lat <= 78; lat += 2.6) {
    const dl = 2.6 / Math.max(0.2, Math.cos(lat * DEG));
    for (let lon = -180; lon < 180; lon += dl) {
      const la = lat + (rand() - 0.5) * 2.2, lo = lon + (rand() - 0.5) * dl * 0.9;
      if (isLand(la, lo) && !inIndia(la, lo)) nodes.push([la, lo]);
    }
  }
  return { nodes, links: meshLinks(nodes, 3, 4.2) };
}

/** Road web: cities plus seeded towns inside India, each joined to its nearest neighbours. */
function roadWeb() {
  const rand = rng(7);
  const nodes: [number, number][] = [...CITIES];
  while (nodes.length < 260) {
    const lat = 8 + rand() * 29, lon = 68.5 + rand() * 28;
    if (inIndia(lat, lon)) nodes.push([lat, lon]);
  }
  return { nodes, links: meshLinks(nodes, 4, 3.2), cities: CITIES.length };
}

export default function Globe({ className = "" }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current!;
    const g = cv.getContext("2d")!;

    // ---------------------------------------------------------------- scene
    const world = rowPoints(-56, 82, -180, 180, 0.72, (la, lo) => isLand(la, lo) && !inIndia(la, lo))
      .map(([la, lo]) => ({ v: toV(la * DEG, lo * DEG), green: Math.sin(la * 0.17 + lo * 0.11) + 0.6 * Math.sin(lo * 0.05 - la * 0.09) > 0.35 }));
    const indiaFill = rowPoints(6, 38, 68, 98, 0.3, inIndia).map(([la, lo]) => toV(la * DEG, lo * DEG));
    const s = INDIA_MASK.step;
    const indiaEdge = rowPoints(6, 38, 68, 98, 0.18, (la, lo) =>
      inIndia(la, lo) && (!inIndia(la + s, lo) || !inIndia(la - s, lo) || !inIndia(la, lo + s) || !inIndia(la, lo - s)),
    ).map(([la, lo]) => toV(la * DEG, lo * DEG));
    const web = roadWeb();
    const mesh = landMesh();
    const meshV = mesh.nodes.map(([la, lo]) => toV(la * DEG, lo * DEG));
    const webV = web.nodes.map(([la, lo]) => toV(la * DEG, lo * DEG));
    const grat: V3[][] = [];
    for (let lat = -60; lat <= 60; lat += 30) {
      const l: V3[] = [];
      for (let lon = -180; lon <= 180; lon += 4) l.push(toV(lat * DEG, lon * DEG));
      grat.push(l);
    }
    for (let lon = -180; lon < 180; lon += 30) {
      const l: V3[] = [];
      for (let lat = -84; lat <= 84; lat += 4) l.push(toV(lat * DEG, lon * DEG));
      grat.push(l);
    }
    const rand = rng(11);
    const stars = Array.from({ length: 220 }, () => ({
      x: rand(), y: rand(), r: rand() * 1.1 + 0.25, a: rand() * 0.5 + 0.1, tw: rand() * 6.28,
    }));
    // orbit rings in screen space: [radius x, radius y, tilt, dash phase speed, dot count]
    const rings = [
      { rx: 1.42, ry: 0.36, rot: -0.32, sp: 0.05, dots: 3 },
      { rx: 1.30, ry: 0.62, rot: 0.55, sp: -0.035, dots: 2 },
      { rx: 1.55, ry: 0.22, rot: 0.18, sp: 0.025, dots: 2 },
    ];

    // soft glow sprite for additive light
    const sprite = document.createElement("canvas");
    sprite.width = sprite.height = 32;
    {
      const c = sprite.getContext("2d")!;
      const gr = c.createRadialGradient(16, 16, 0, 16, 16, 16);
      gr.addColorStop(0, "rgba(255,255,255,1)");
      gr.addColorStop(0.3, "rgba(255,255,255,0.45)");
      gr.addColorStop(1, "rgba(255,255,255,0)");
      c.fillStyle = gr; c.fillRect(0, 0, 32, 32);
    }

    // ---------------------------------------------------------- interaction
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let W = 0, H = 0, dpr = 1, raf = 0;
    const HOME_YAW = -79 * DEG, HOME_TILT = 12 * DEG;
    let yaw = HOME_YAW, tilt = HOME_TILT;
    const IDLE = reduced ? 0 : -0.035;
    let vYaw = IDLE, vTilt = 0;
    let dragging = false, lastX = 0, lastY = 0, lastT = 0, idleSince = -1e9;
    let geo = { cx: 0, cy: 0, R: 1 };

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const overGlobe = (x: number, y: number) => Math.hypot(x - geo.cx, y - geo.cy) < geo.R * 1.08;
    const pos = (e: PointerEvent) => {
      const b = cv.getBoundingClientRect();
      return [e.clientX - b.left, e.clientY - b.top];
    };
    const onDown = (e: PointerEvent) => {
      const [x, y] = pos(e);
      if (!overGlobe(x, y)) return;
      dragging = true; lastX = x; lastY = y; lastT = performance.now();
      vYaw = 0; vTilt = 0;
      cv.setPointerCapture(e.pointerId);
      cv.style.cursor = "grabbing";
    };
    const onMove = (e: PointerEvent) => {
      const [x, y] = pos(e);
      if (!dragging) { cv.style.cursor = overGlobe(x, y) ? "grab" : "default"; return; }
      const now = performance.now();
      const dt = Math.max(8, now - lastT) / 1000;
      const dYaw = ((x - lastX) / geo.R) * 0.9, dTilt = ((y - lastY) / geo.R) * 0.9;
      yaw += dYaw;
      tilt = Math.max(-70 * DEG, Math.min(70 * DEG, tilt + dTilt));
      vYaw = dYaw / dt; vTilt = dTilt / dt;
      lastX = x; lastY = y; lastT = now;
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

    // ---------------------------------------------------------------- frame
    let prev = performance.now();
    const frame = (time: number) => {
      const dt = Math.min(0.05, (time - prev) / 1000);
      prev = time;
      if (!dragging) {
        yaw += vYaw * dt;
        tilt = Math.max(-70 * DEG, Math.min(70 * DEG, tilt + vTilt * dt));
        const k = Math.min(1, dt * 1.5);
        vYaw += (IDLE - vYaw) * k; vTilt += -vTilt * k;
        if (!reduced && time - idleSince > 3500) tilt += (HOME_TILT - tilt) * Math.min(1, dt * 0.6);
      }
      const t = reduced ? 0 : time / 1000;

      const cr = Math.cos(yaw), sr = Math.sin(yaw), ct = Math.cos(tilt), st = Math.sin(tilt);
      const wide = W > 900;
      const R = wide ? Math.min(H * 0.49, W * 0.3) : Math.min(W * 0.46, H * 0.3);
      const cx = wide ? W * 0.7 : W * 0.5;
      const cy = wide ? H * 0.5 : R * 1.35;
      geo = { cx, cy, R };
      // Mild perspective: the side facing us is magnified (up to 1.4x), the limb
      // stays on the rim, so the region in front (India) reads large and close.
      // True perspective from a camera PD globe-radii away, scaled so the
      // silhouette is exactly R. `z` is shifted so that z > 0 means "visible".
      const PD = 3.0, SIL = Math.sqrt(PD * PD - 1), ZV = 1 / PD;
      const project = (p: V3) => {
        const x = p[0] * cr + p[2] * sr;
        const z0 = -p[0] * sr + p[2] * cr;
        const y = p[1] * ct - z0 * st, z = p[1] * st + z0 * ct;
        const k = SIL / (PD - z);
        return { x: cx + x * k * R, y: cy - y * k * R, z: (z - ZV) / (1 - ZV) };
      };

      g.clearRect(0, 0, W, H);

      // stars
      for (const d of stars) {
        const a = d.a * (0.6 + 0.4 * Math.sin(d.tw + t * 0.9));
        g.fillStyle = `rgba(190,240,255,${a.toFixed(3)})`;
        g.beginPath(); g.arc(d.x * W, d.y * H, d.r, 0, Math.PI * 2); g.fill();
      }

      // atmosphere
      const atm = g.createRadialGradient(cx, cy, R * 0.95, cx, cy, R * 1.45);
      atm.addColorStop(0, "rgba(45,212,191,0.45)");
      atm.addColorStop(0.3, "rgba(34,211,238,0.10)");
      atm.addColorStop(1, "rgba(4,20,28,0)");
      g.fillStyle = atm;
      g.beginPath(); g.arc(cx, cy, R * 1.45, 0, Math.PI * 2); g.fill();

      // orbit rings, back halves (behind the globe)
      const ring = (o: (typeof rings)[number], front: boolean) => {
        g.save();
        g.translate(cx, cy); g.rotate(o.rot);
        g.strokeStyle = `rgba(125,240,255,${front ? 0.45 : 0.18})`;
        g.lineWidth = 1;
        g.setLineDash([3, 7]);
        g.lineDashOffset = -t * 30 * Math.sign(o.sp);
        g.beginPath();
        if (front) g.ellipse(0, 0, o.rx * R, o.ry * R, 0, 0, Math.PI);
        else g.ellipse(0, 0, o.rx * R, o.ry * R, 0, Math.PI, Math.PI * 2);
        g.stroke();
        g.setLineDash([]);
        for (let i = 0; i < o.dots; i++) {
          const a = t * o.sp * 6 + (i * Math.PI * 2) / o.dots;
          const isFront = Math.sin(a) > 0;
          if (isFront !== front) continue;
          const x = Math.cos(a) * o.rx * R, y = Math.sin(a) * o.ry * R;
          g.globalCompositeOperation = "lighter";
          g.globalAlpha = front ? 0.9 : 0.35;
          g.drawImage(sprite, x - 9, y - 9, 18, 18);
          g.globalAlpha = 1; g.globalCompositeOperation = "source-over";
        }
        g.restore();
      };
      rings.forEach((o) => ring(o, false));

      // ocean
      const ocean = g.createRadialGradient(cx - R * 0.3, cy - R * 0.35, R * 0.1, cx, cy, R);
      ocean.addColorStop(0, "#0E4252");
      ocean.addColorStop(0.65, "#072B38");
      ocean.addColorStop(1, "#052029");
      g.fillStyle = ocean;
      g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fill();

      // graticule
      g.strokeStyle = "rgba(94,234,212,0.07)"; g.lineWidth = 0.8;
      for (const l of grat) {
        g.beginPath();
        let on = false;
        for (const p of l) {
          const q = project(p);
          if (q.z <= 0) { on = false; continue; }
          if (!on) { g.moveTo(q.x, q.y); on = true; } else g.lineTo(q.x, q.y);
        }
        g.stroke();
      }

      // world land
      g.globalCompositeOperation = "lighter";
      const ds = Math.max(1.5, R / 210);
      for (const w of world) {
        const q = project(w.v);
        if (q.z <= 0) continue;
        const a = Math.min(1, 0.4 + 0.75 * q.z);
        g.fillStyle = w.green ? `rgba(110,231,183,${(a * 0.8).toFixed(3)})`
                              : `rgba(56,224,230,${(a * 0.85).toFixed(3)})`;
        g.fillRect(q.x - ds / 2, q.y - ds / 2, ds, ds);
      }

      // continent mesh
      const pm = meshV.map(project);
      g.lineWidth = 0.7;
      g.strokeStyle = "rgba(74,222,128,0.42)";
      g.beginPath();
      for (const [i, j] of mesh.links) {
        const a = pm[i], b = pm[j];
        if (a.z <= 0.05 || b.z <= 0.05) continue;
        g.moveTo(a.x, a.y); g.lineTo(b.x, b.y);
      }
      g.stroke();
      for (const q of pm) {
        if (q.z <= 0.05) continue;
        g.fillStyle = `rgba(187,247,208,${(0.25 + 0.6 * q.z).toFixed(3)})`;
        g.fillRect(q.x - 0.9, q.y - 0.9, 1.8, 1.8);
      }

      // India: glow, fill, outline
      const di = Math.max(1.4, R / 260);
      for (const p of indiaFill) {
        const q = project(p);
        if (q.z <= 0) continue;
        const a = 0.35 + 0.65 * q.z;
        g.globalAlpha = 0.025 * a;
        g.drawImage(sprite, q.x - 8, q.y - 8, 16, 16);
        g.globalAlpha = 1;
        g.fillStyle = `rgba(34,211,238,${(0.9 * a).toFixed(3)})`;
        g.fillRect(q.x - di / 2, q.y - di / 2, di, di);
      }
      for (const p of indiaEdge) {
        const q = project(p);
        if (q.z <= 0) continue;
        g.globalAlpha = 0.07 * q.z;
        g.drawImage(sprite, q.x - 6, q.y - 6, 12, 12);
        g.globalAlpha = 1;
        g.fillStyle = `rgba(103,232,249,${(0.95 * q.z).toFixed(3)})`;
        g.fillRect(q.x - 0.9, q.y - 0.9, 1.8, 1.8);
      }

      // road web
      const pw = web.nodes.map((_, i) => project(webV[i]));
      g.lineWidth = 1;
      g.strokeStyle = "rgba(207,250,254,0.55)";
      g.beginPath();
      for (const [i, j] of web.links) {
        const a = pw[i], b = pw[j];
        if (a.z <= 0.02 || b.z <= 0.02) continue;
        g.moveTo(a.x, a.y); g.lineTo(b.x, b.y);
      }
      g.stroke();
      if (!reduced) {
        web.links.forEach(([i, j], k) => {
          if (k % 3) return;                                   // pulses on every third road
          const a = pw[i], b = pw[j];
          if (a.z <= 0.02 || b.z <= 0.02) return;
          const u = (t * 0.45 + k * 0.137) % 1;
          g.globalAlpha = 0.9;
          g.drawImage(sprite, a.x + (b.x - a.x) * u - 3, a.y + (b.y - a.y) * u - 3, 6, 6);
          g.globalAlpha = 1;
        });
      }
      pw.forEach((q, i) => {
        if (q.z <= 0.02) return;
        const big = i < web.cities;
        g.globalAlpha = big ? 0.9 : 0.5;
        g.drawImage(sprite, q.x - (big ? 6 : 3.5), q.y - (big ? 6 : 3.5), big ? 12 : 7, big ? 12 : 7);
        g.globalAlpha = 1;
        g.fillStyle = big ? "#ECFEFF" : "rgba(207,250,254,0.8)";
        g.fillRect(q.x - (big ? 1.1 : 0.7), q.y - (big ? 1.1 : 0.7), big ? 2.2 : 1.4, big ? 2.2 : 1.4);
      });
      g.globalCompositeOperation = "source-over";

      // night-side shading and rim
      const shade = g.createRadialGradient(cx - R * 0.25, cy - R * 0.3, R * 0.55, cx, cy, R);
      shade.addColorStop(0, "rgba(0,0,0,0)");
      shade.addColorStop(1, "rgba(1,10,16,0.45)");
      g.fillStyle = shade;
      g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fill();
      const rim = g.createRadialGradient(cx, cy, R * 0.86, cx, cy, R);
      rim.addColorStop(0, "rgba(45,212,191,0)");
      rim.addColorStop(1, "rgba(94,234,212,0.28)");
      g.fillStyle = rim;
      g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fill();
      g.strokeStyle = "rgba(167,243,240,0.8)"; g.lineWidth = 1.6;
      g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.stroke();

      // orbit rings, front halves
      rings.forEach((o) => ring(o, true));

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
  }, []);

  return (
    <canvas ref={ref} className={className} style={{ touchAction: "none" }}
            role="img" aria-label="Rotating globe with India's road network highlighted; drag to rotate" />
  );
}
