/**
 * Decorative rotating globe for the landing page.
 *
 * A dotted Earth (land cells from a coarse mask), a glowing rim, an orbit ring,
 * drifting stars, and a few glowing "road links" between Indian cities with
 * pulses travelling along them. Pure canvas, no dependencies. Respects
 * prefers-reduced-motion by drawing one still frame.
 */
import { useEffect, useRef } from "react";
import { LAND_MASK } from "@/assets/landMask";

type V3 = [number, number, number];

const DEG = Math.PI / 180;

/** Unpack the bit mask into a list of unit vectors for land cells. */
function landPoints(): V3[] {
  const { rows, cols, step, lat0, lon0, b64 } = LAND_MASK;
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const pts: V3[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      if (!((bytes[i >> 3] >> (7 - (i & 7))) & 1)) continue;
      const lat = (lat0 + r * step) * DEG;
      const lon = (lon0 + c * step) * DEG;
      pts.push(toV(lat, lon));
    }
  }
  return pts;
}

function toV(lat: number, lon: number): V3 {
  return [Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon)];
}

/** Great-circle interpolation between two unit vectors. */
function slerp(a: V3, b: V3, t: number): V3 {
  const d = Math.min(1, Math.max(-1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]));
  const w = Math.acos(d);
  if (w < 1e-6) return a;
  const s = Math.sin(w);
  const k1 = Math.sin((1 - t) * w) / s, k2 = Math.sin(t * w) / s;
  return [a[0] * k1 + b[0] * k2, a[1] * k1 + b[1] * k2, a[2] * k1 + b[2] * k2];
}

// Indian cities: [lat, lon]
const CITIES: Record<string, [number, number]> = {
  delhi: [28.6, 77.2], mumbai: [19.1, 72.9], bengaluru: [12.97, 77.6],
  chennai: [13.08, 80.27], kolkata: [22.57, 88.36], hyderabad: [17.4, 78.5],
  amaravati: [16.5, 80.5], guwahati: [26.1, 91.7], ahmedabad: [23.0, 72.6],
  jaipur: [26.9, 75.8], pune: [18.5, 73.9], lucknow: [26.8, 80.9],
};
const LINKS: [string, string][] = [
  ["delhi", "mumbai"], ["delhi", "kolkata"], ["mumbai", "bengaluru"],
  ["bengaluru", "chennai"], ["chennai", "kolkata"], ["hyderabad", "delhi"],
  ["amaravati", "hyderabad"], ["amaravati", "chennai"], ["kolkata", "guwahati"],
  ["ahmedabad", "mumbai"], ["jaipur", "delhi"], ["pune", "hyderabad"],
  ["lucknow", "kolkata"], ["jaipur", "ahmedabad"], ["bengaluru", "hyderabad"],
];

export default function Globe({ className = "" }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current!;
    const g = cv.getContext("2d")!;
    const land = landPoints();
    const cityV = Object.fromEntries(
      Object.entries(CITIES).map(([k, [la, lo]]) => [k, toV(la * DEG, lo * DEG)]),
    ) as Record<string, V3>;
    const arcs = LINKS.map(([a, b]) => {
      const pts: V3[] = [];
      for (let i = 0; i <= 24; i++) pts.push(slerp(cityV[a], cityV[b], i / 24));
      return pts;
    });

    // Background stars, in normalised screen space.
    const stars = Array.from({ length: 140 }, () => ({
      x: Math.random(), y: Math.random(), r: Math.random() * 1.3 + 0.2,
      a: Math.random() * 0.5 + 0.15, v: Math.random() * 0.00004 + 0.00001,
    }));
    // Orbit ring particles.
    const ring = Array.from({ length: 70 }, (_, i) => ({
      t: (i / 70) * Math.PI * 2, s: Math.random() * 1.8 + 0.6,
    }));

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let W = 0, H = 0, dpr = 1, raf = 0;

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const tilt = 22 * DEG;                // view centred near 20°N so India faces us
    const ct = Math.cos(tilt), st = Math.sin(tilt);

    const frame = (time: number) => {
      const rot = -78 * DEG - (reduced ? 0 : time * 0.00003); // start over India
      const cr = Math.cos(rot), sr = Math.sin(rot);
      const wide = W > 900;
      const R = Math.min(H * 0.46, W * (wide ? 0.33 : 0.48));
      const cx = wide ? W * 0.68 : W * 0.5;
      const cy = H * 0.5;

      const project = (p: V3) => {
        // rotate around Y, then tilt around X
        const x = p[0] * cr + p[2] * sr;
        const z0 = -p[0] * sr + p[2] * cr;
        const y = p[1] * ct - z0 * st;
        const z = p[1] * st + z0 * ct;
        return { x: cx + x * R, y: cy - y * R, z };
      };

      g.clearRect(0, 0, W, H);

      // stars
      for (const s of stars) {
        if (!reduced) s.x = (s.x + s.v) % 1;
        g.fillStyle = `rgba(150,200,255,${s.a})`;
        g.beginPath(); g.arc(s.x * W, s.y * H, s.r, 0, Math.PI * 2); g.fill();
      }

      // atmosphere glow
      const glow = g.createRadialGradient(cx, cy, R * 0.85, cx, cy, R * 1.35);
      glow.addColorStop(0, "rgba(34,211,238,0.28)");
      glow.addColorStop(0.4, "rgba(14,116,144,0.12)");
      glow.addColorStop(1, "rgba(5,11,22,0)");
      g.fillStyle = glow;
      g.beginPath(); g.arc(cx, cy, R * 1.35, 0, Math.PI * 2); g.fill();

      // sphere body
      const body = g.createRadialGradient(cx - R * 0.35, cy - R * 0.35, R * 0.1, cx, cy, R);
      body.addColorStop(0, "rgba(14,55,90,0.95)");
      body.addColorStop(1, "rgba(4,18,36,0.98)");
      g.fillStyle = body;
      g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fill();

      // land dots
      const ds = Math.max(1.2, R / 190);
      for (const p of land) {
        const q = project(p);
        if (q.z <= 0) continue;
        const a = 0.35 + 0.65 * q.z;
        g.fillStyle = `rgba(103,232,249,${a.toFixed(3)})`;
        g.fillRect(q.x - ds / 2, q.y - ds / 2, ds, ds);
      }

      // rim
      g.save();
      g.shadowColor = "rgba(34,211,238,0.9)"; g.shadowBlur = 25;
      g.strokeStyle = "rgba(125,240,255,0.55)"; g.lineWidth = 1.5;
      g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.stroke();
      g.restore();

      // road links between cities, with travelling pulses
      g.lineCap = "round";
      arcs.forEach((pts, i) => {
        const proj = pts.map((p) => {
          const lift = 1.02;
          return project([p[0] * lift, p[1] * lift, p[2] * lift]);
        });
        g.strokeStyle = "rgba(56,189,248,0.55)"; g.lineWidth = 1.2;
        g.beginPath();
        let on = false;
        for (const q of proj) {
          if (q.z <= 0.02) { on = false; continue; }
          if (!on) { g.moveTo(q.x, q.y); on = true; } else g.lineTo(q.x, q.y);
        }
        g.stroke();
        const t = reduced ? 0.5 : ((time * 0.00035 + i * 0.37) % 1);
        const q = proj[Math.floor(t * (proj.length - 1))];
        if (q.z > 0.02) {
          g.save();
          g.shadowColor = "#a5f3fc"; g.shadowBlur = 12;
          g.fillStyle = "#e0faff";
          g.beginPath(); g.arc(q.x, q.y, 2.2, 0, Math.PI * 2); g.fill();
          g.restore();
        }
      });

      // city nodes
      for (const v of Object.values(cityV)) {
        const q = project(v);
        if (q.z <= 0.02) continue;
        g.save();
        g.shadowColor = "#22d3ee"; g.shadowBlur = 14;
        g.fillStyle = "#cffafe";
        g.beginPath(); g.arc(q.x, q.y, 2.6, 0, Math.PI * 2); g.fill();
        g.restore();
      }

      // orbit ring (tilted ellipse), back half dimmer
      const ra = R * 1.28, rb = R * 0.32, rt = -0.35;
      g.strokeStyle = "rgba(103,232,249,0.12)"; g.lineWidth = 1;
      g.beginPath(); g.ellipse(cx, cy, ra, rb, rt, 0, Math.PI * 2); g.stroke();
      for (const o of ring) {
        const t = o.t + (reduced ? 0 : time * 0.00012);
        const ex = Math.cos(t) * ra, ey = Math.sin(t) * rb;
        const x = cx + ex * Math.cos(rt) - ey * Math.sin(rt);
        const y = cy + ex * Math.sin(rt) + ey * Math.cos(rt);
        const front = Math.sin(t) > 0;
        if (!front && Math.hypot(x - cx, y - cy) < R) continue; // hidden behind globe
        g.fillStyle = front ? "rgba(165,243,252,0.85)" : "rgba(103,232,249,0.35)";
        g.beginPath(); g.arc(x, y, o.s, 0, Math.PI * 2); g.fill();
      }

      if (!reduced) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", resize); };
  }, []);

  return <canvas ref={ref} className={className} aria-hidden="true" />;
}
