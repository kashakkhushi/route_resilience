/**
 * Trip planner on a real map.
 *
 * A and B are coordinates (typed, or clicked on the map). The backend downloads
 * the OpenStreetMap road network, finds the newest Sentinel-2 image that is
 * clear over the area, runs the fine-tuned Sentinel-2 model, and scores every
 * road: verified (the satellite sees it), unverified (it does not), or not
 * checkable (too narrow for 10 m, or under cloud). Then:
 *   - fastest route A -> B
 *   - the roads this trip depends on most (closing each one in turn)
 *   - what-if: close roads by clicking, a flood zone, or avoid unverified roads
 *
 * Nothing on this page is computed in the browser except drawing.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  Crosshair, Droplets, Loader2, MapPin, Navigation, RotateCcw, Satellite, ShieldAlert, XCircle,
} from "lucide-react";
import clsx from "clsx";
import { api } from "@/services/api";
import { Chip, ErrorNote, Metric, Note, Panel } from "@/components/ui";
import type { PlanResponse, RerouteResponse, TripCore } from "@/types/api";

type Click = "a" | "b" | "close" | "flood";
type LL = { lat: number; lon: number };

const COLORS = {
  verified: "#34D399",
  bridged: "#BEF264",
  bridge: "#F8FAFC",
  unverified: "#F59E0B",
  not_checkable: "#64748B",
  closed: "#EF4444",
  route: "#38BDF8",
  normal: "#94A3B8",
  alternate: "#FB923C",
  newRoad: "#C084FC",
};

// Example trip in Hyderabad, the city the Sentinel-2 model was fine-tuned on.
const PRESETS: { label: string; a: LL; b: LL }[] = [
  { label: "Hyderabad: Gachibowli → HITEC City",
    a: { lat: 17.4401, lon: 78.3489 }, b: { lat: 17.4504, lon: 78.3810 } },
  { label: "Hyderabad: Ameerpet → Begumpet",
    a: { lat: 17.4375, lon: 78.4483 }, b: { lat: 17.4440, lon: 78.4690 } },
];

const km = (m?: number | null) => (m == null ? null : (m / 1000).toFixed(2));

export default function TripPlannerSection() {
  const [a, setA] = useState<LL | null>(PRESETS[0].a);
  const [b, setB] = useState<LL | null>(PRESETS[0].b);
  const [satellite, setSatellite] = useState(true);
  const [optimise, setOptimise] = useState<"time" | "distance">("time");
  const [click, setClick] = useState<Click>("a");

  const [plan, setPlan] = useState<PlanResponse | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<{ title: string; detail?: string } | null>(null);

  const [closed, setClosed] = useState<number[]>([]);
  const [flood, setFlood] = useState<LL | null>(null);
  const [floodR, setFloodR] = useState(250);
  const [avoid, setAvoid] = useState(false);
  const [what, setWhat] = useState<RerouteResponse | null>(null);
  const [showImg, setShowImg] = useState(false);
  const [showModel, setShowModel] = useState(false);
  const [showNew, setShowNew] = useState(true);

  const conditionsOn = closed.length > 0 || flood !== null || avoid;
  const cur: TripCore | null = conditionsOn ? what : plan;

  /* ------------------------------------------------------------ backend */
  async function runPlan(refresh = false) {
    if (!a || !b) return;
    setBusy(satellite
      ? "Downloading roads, finding the latest clear Sentinel-2 image and running the model…"
      : "Downloading OpenStreetMap roads…");
    setErr(null);
    try {
      const r = await api.plan({ a, b, satellite, refresh, optimise });
      setPlan(r); setClosed([]); setFlood(null); setAvoid(false); setWhat(null);
      fitTo(r);
    } catch (e: any) {
      setErr({ title: "Could not plan this trip", detail: e?.message });
    } finally {
      setBusy(null);
    }
  }

  // Re-plan under the current conditions whenever they change.
  useEffect(() => {
    if (!plan || !conditionsOn) { setWhat(null); return; }
    let live = true;
    setBusy("Re-planning…");
    api.reroute(plan.plan_id, {
      closed, avoid_unverified: avoid,
      flood: flood ? { ...flood, radius_m: floodR } : null,
    }).then((r) => { if (live) setWhat(r); })
      .catch((e) => { if (live) setErr({ title: "Re-planning failed", detail: e?.message }); })
      .finally(() => { if (live) setBusy(null); });
    return () => { live = false; };
  }, [plan, closed, flood, floodR, avoid]);   // eslint-disable-line react-hooks/exhaustive-deps

  /* ---------------------------------------------------------------- map */
  const mapEl = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const layers = useRef<{ base?: L.LayerGroup; top?: L.LayerGroup; img?: L.ImageOverlay;
                          model?: L.ImageOverlay }>({});
  const clickRef = useRef(click);
  clickRef.current = click;

  useEffect(() => {
    if (!mapEl.current || map.current) return;
    const m = L.map(mapEl.current, { zoomControl: true, attributionControl: true, preferCanvas: true })
      .setView([PRESETS[0].a.lat, (PRESETS[0].a.lon + PRESETS[0].b.lon) / 2], 14);
    // Base maps that need no API key. "Map (dark)" is the standard OpenStreetMap
    // tiles darkened with a CSS filter to match the dashboard; "Satellite" is
    // Esri World Imagery. (CARTO's free tiles now require a key.)
    const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19, className: "rr-dark-tiles",
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    });
    const osmLight = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    });
    const esri = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 19,
      attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    });
    osm.addTo(m);
    L.control.layers({ "Map (dark)": osm, "Map (light)": osmLight, "Satellite (Esri)": esri },
                     undefined, { position: "topright" }).addTo(m);
    layers.current.base = L.layerGroup().addTo(m);
    layers.current.top = L.layerGroup().addTo(m);
    m.on("click", (e: L.LeafletMouseEvent) => onMapClick(e.latlng));
    map.current = m;
    // The page stays mounted while other tabs are open (so a planned trip is
    // kept); when it becomes visible again Leaflet must re-measure its box.
    const ro = new ResizeObserver(() => m.invalidateSize());
    ro.observe(mapEl.current);
    return () => { ro.disconnect(); m.remove(); map.current = null; };
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  function onMapClick(ll: L.LatLng) {
    const p = { lat: +ll.lat.toFixed(6), lon: +ll.lng.toFixed(6) };
    const c = clickRef.current;
    if (c === "a") { setA(p); setClick("b"); }
    else if (c === "b") { setB(p); }
    else if (c === "flood") { setFlood(p); }
  }

  function fitTo(r: PlanResponse) {
    const [w, s, e, n] = r.bbox;
    map.current?.fitBounds([[s, w], [n, e]], { padding: [10, 10] });
  }

  const closedSet = useMemo(() => new Set(closed), [closed]);
  const blockedSet = useMemo(() => new Set(cur?.blocked_edges ?? []), [cur]);

  // Roads layer (and satellite overlays): redrawn when the plan or closures change.
  useEffect(() => {
    const m = map.current, g = layers.current.base;
    if (!m || !g) return;
    g.clearLayers();
    layers.current.img?.remove(); layers.current.model?.remove();
    if (!plan) return;
    const ov = plan.satellite.overlays;
    if (ov && showImg) layers.current.img = L.imageOverlay(ov.satellite.image, ov.satellite.bounds,
                                                           { opacity: 0.9 }).addTo(m);
    if (ov && showModel) layers.current.model = L.imageOverlay(ov.model_roads.image,
                                                               ov.model_roads.bounds).addTo(m);
    L.geoJSON(plan.roads as any, {
      style: (f: any) => {
        const pr = f.properties, eid = pr.eid as number;
        // red = closed by the user or under the flood; roads merely avoided
        // because the satellite did not see them keep their amber colour
        const isClosed = closedSet.has(eid)
          || (blockedSet.has(eid) && !(avoid && pr.status === "unverified"));
        return {
          color: isClosed ? COLORS.closed : (COLORS as any)[pr.status] ?? COLORS.not_checkable,
          weight: isClosed ? 5 : pr.status === "not_checkable" ? 2 : 3.5,
          opacity: pr.status === "not_checkable" && !isClosed ? 0.55 : 0.95,
          dashArray: isClosed ? "6 5" : undefined,
        };
      },
      onEachFeature: (f: any, layer) => {
        const pr = f.properties;
        layer.bindTooltip(
          `<b>${pr.name ?? "(unnamed)"}</b> · ${pr.class}<br/>` +
          `${(pr.length_m / 1000).toFixed(2)} km · ${pr.status.replace("_", " ")}` +
          (pr.support != null ? ` (seen on ${Math.round(pr.support * 100)}% of its length)` : ""),
          { sticky: true });
        layer.on("click", (e: any) => {
          if (clickRef.current !== "close") return;
          L.DomEvent.stopPropagation(e);
          const eid = pr.eid as number;
          setClosed((c) => (c.includes(eid) ? c.filter((x) => x !== eid) : [...c, eid]));
        });
      },
    }).addTo(g);
    // Roads the satellite sees but OpenStreetMap does not have: candidates for
    // unmapped roads, shown for review only, never used for routing.
    // Gaps the healing step joined on the satellite mask (tree crowns, shadows).
    const hg = plan.satellite.healing?.geojson;
    if (hg && hg.features.length > 0)
      L.geoJSON(hg as any, {
        style: () => ({ color: COLORS.bridge, weight: 4, opacity: 0.95 }),
        onEachFeature: (_f: any, layer) => layer.bindTooltip(
          "<b>Gap joined by healing</b><br/>The model saw the road on both sides; " +
          "something (tree, shadow) hid this stretch.", { sticky: true }),
      }).addTo(g);
    const nr = plan.satellite.new_roads;
    if (showNew && nr && nr.count > 0)
      L.geoJSON(nr.geojson as any, {
        style: () => ({ color: COLORS.newRoad, weight: 3, opacity: 0.9, dashArray: "2 5" }),
        onEachFeature: (f: any, layer) => layer.bindTooltip(
          `<b>Possible unmapped road</b><br/>${(f.properties.length_m / 1000).toFixed(2)} km seen by the ` +
          `satellite model, not in OpenStreetMap.<br/>Not used for routing.`, { sticky: true }),
      }).addTo(g);
  }, [plan, closedSet, blockedSet, showImg, showModel, avoid, showNew]);

  // Route, markers, flood circle.
  const [showWorst, setShowWorst] = useState(false);
  useEffect(() => {
    const g = layers.current.top;
    if (!g) return;
    g.clearLayers();
    // detour first, so the shared part of the trip still reads as the route
    if (showWorst && cur?.worst_case?.alternate_coords)
      L.polyline(cur.worst_case.alternate_coords, { color: COLORS.alternate, weight: 5,
                                                   dashArray: "8 6" }).addTo(g);
    if (cur?.route?.reachable && cur.route.coords) {
      if (conditionsOn && what?.normal?.reachable && what.normal.coords)
        L.polyline(what.normal.coords, { color: COLORS.normal, weight: 4, opacity: 0.6,
                                         dashArray: "4 6" }).addTo(g);
      L.polyline(cur.route.coords, { color: "#0B1220", weight: 9, opacity: 0.7 }).addTo(g);
      L.polyline(cur.route.coords, { color: COLORS.route, weight: 5 }).addTo(g);
    }
    if (showWorst && cur?.worst_case) {
      const wc = cur.worst_case;
      const road = cur.critical_roads.find((c) => c.edge === wc.edge);
      if (road) L.marker(road.midpoint, { icon: L.divIcon({
        className: "", html: `<div style="color:#EF4444;font:700 22px/1 sans-serif;text-shadow:0 0 4px #000">✕</div>`,
        iconSize: [22, 22], iconAnchor: [11, 11] }) }).addTo(g);
    }
    if (flood) L.circle([flood.lat, flood.lon], { radius: floodR, color: "#60A5FA",
                                                 fillColor: "#3B82F6", fillOpacity: 0.25 }).addTo(g);
    const pin = (p: LL, label: string, color: string) =>
      L.marker([p.lat, p.lon], { icon: L.divIcon({
        className: "",
        html: `<div style="background:${color};color:#0B1220;font:700 12px/22px sans-serif;width:22px;height:22px;border-radius:50%;text-align:center;border:2px solid #0B1220;box-shadow:0 0 0 2px ${color}">${label}</div>`,
        iconSize: [22, 22], iconAnchor: [11, 11] }) }).addTo(g);
    if (a) pin(a, "A", "#34D399");
    if (b) pin(b, "B", "#F472B6");
  }, [cur, a, b, flood, floodR, showWorst, conditionsOn, what]);

  /* ----------------------------------------------------------------- UI */
  const r = cur?.route;
  const routeLen = r?.length_m ?? 0;
  const cnt = plan?.satellite.counts;
  const worst = cur?.critical_roads?.[0];
  const closeWorst = () => {
    if (!worst) return;
    setClosed((c) => (c.includes(worst.edge) ? c : [...c, worst.edge]));
    setShowWorst(false);
  };
  const resetConditions = () => { setClosed([]); setFlood(null); setAvoid(false); };

  return (
    <div className="flex-1 grid grid-cols-12 gap-3 min-h-0">
      <Panel className="col-span-12 lg:col-span-8" title="Map"
        actions={
          <div className="flex items-center gap-1.5">
            {(["a", "b", "close", "flood"] as Click[]).map((k) => (
              <button key={k} onClick={() => setClick(k)}
                      disabled={(k === "close" || k === "flood") && !plan}
                      className={clsx("px-2 py-1 rounded text-[11px] border transition-colors disabled:opacity-35",
                        click === k ? "border-accent/50 bg-accent/15 text-accent" : "border-line text-ink-300 hover:bg-base-700")}>
                {k === "a" ? "Click sets A" : k === "b" ? "Click sets B" : k === "close" ? "Click closes road" : "Click places flood"}
              </button>
            ))}
          </div>
        }>
        <div className="relative w-full h-full min-h-[520px] rounded-md overflow-hidden">
          <div ref={mapEl} className="absolute inset-0" />
          {busy && (
            <div className="absolute top-3 left-1/2 -translate-x-1/2 z-[500] flex items-center gap-2 px-3 py-2 rounded-md bg-base-900/90 border border-line text-[12px] text-ink-100">
              <Loader2 size={14} className="animate-spin text-accent" /> {busy}
            </div>
          )}
          {plan && (
            <div className="absolute bottom-3 left-3 z-[500] px-3 py-2 rounded-md bg-base-900/90 border border-line text-[11px] space-y-1">
              <Legend color={COLORS.route} label="Route" wide />
              {conditionsOn && <Legend color={COLORS.normal} label="Normal route" dashed />}
              {showWorst && <Legend color={COLORS.alternate} label="Detour if ✕ closes" dashed />}
              <Legend color={COLORS.verified} label="Satellite sees road" />
              {(plan.satellite.counts?.bridged ?? 0) > 0 &&
                <Legend color={COLORS.bridged} label="Hidden (tree / shadow), joined by healing" />}
              {(plan.satellite.healing?.gaps_joined ?? 0) > 0 &&
                <Legend color={COLORS.bridge} label="Gap joined by healing" />}
              <Legend color={COLORS.unverified} label="Not seen even after healing (check it)" />
              <Legend color={COLORS.not_checkable} label="Not checkable (narrow / cloud)" />
              {showNew && (plan.satellite.new_roads?.count ?? 0) > 0 &&
                <Legend color={COLORS.newRoad} label="Seen by satellite, not in OSM" dashed />}
              <Legend color={COLORS.closed} label="Closed" dashed />
            </div>
          )}
        </div>
      </Panel>

      <div className="col-span-12 lg:col-span-4 flex flex-col gap-3 overflow-auto">
        <Panel title="1 · Trip" dense className="shrink-0">
          <div className="space-y-2.5">
            <PointInput label="A" color="#34D399" v={a} onChange={setA} onPick={() => setClick("a")} />
            <PointInput label="B" color="#F472B6" v={b} onChange={setB} onPick={() => setClick("b")} />
            <select className="input w-full text-[12px]" value=""
                    onChange={(e) => { const p = PRESETS[+e.target.value]; if (p) { setA(p.a); setB(p.b); } }}>
              <option value="">Example trips…</option>
              {PRESETS.map((p, i) => <option key={i} value={i}>{p.label}</option>)}
            </select>
            <div className="flex items-center justify-between text-[12px]">
              <label className="flex items-center gap-2 text-ink-300">
                <input type="checkbox" checked={satellite} onChange={(e) => setSatellite(e.target.checked)} />
                Check roads with latest Sentinel-2
              </label>
              <select className="input text-[12px] py-1" value={optimise}
                      onChange={(e) => setOptimise(e.target.value as any)}>
                <option value="time">Fastest</option>
                <option value="distance">Shortest</option>
              </select>
            </div>
            <button className="btn-primary w-full justify-center" disabled={!a || !b || !!busy}
                    onClick={() => runPlan(false)}>
              <Navigation size={14} /> Plan trip
            </button>
            <p className="text-[10px] text-ink-500 leading-relaxed">
              Needs internet the first time an area is used. With the satellite check, the first
              run for an area takes about 1–3 minutes on a laptop CPU (up to ~6 minutes for trips near the 10 km limit); after that it is cached.
            </p>
            {err && <ErrorNote title={err.title} detail={err.detail} />}
          </div>
        </Panel>

        {plan && (
          <Panel title="2 · Route" dense className="shrink-0"
            actions={conditionsOn ? <Chip tone="warn">with closures</Chip> : null}>
            {!r?.reachable ? (
              <Note tone="warn"><strong>No route: A and B are cut off</strong> under these conditions.
                Every way between them is closed.</Note>
            ) : (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <Metric size="sm" label="Distance" value={km(r.length_m)} unit="km" />
                  <Metric size="sm" label="Time" value={r.time_min} unit="min"
                          hint="Length ÷ typical speed for each road class. No live traffic." />
                </div>
                {conditionsOn && what?.compared_to_normal && (
                  <Note tone="warn">Detour: +{what.compared_to_normal.extra_min} min,
                    +{km(what.compared_to_normal.extra_m)} km compared with the normal route.</Note>
                )}
                {plan.satellite.available && routeLen > 0 && (
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-ink-500 mb-1">
                      Satellite check along this route
                    </div>
                    <Bar parts={[
                      [r.verified_length_m ?? 0, COLORS.verified],
                      [r.bridged_length_m ?? 0, COLORS.bridged],
                      [r.unverified_length_m ?? 0, COLORS.unverified],
                      [routeLen - (r.verified_length_m ?? 0) - (r.bridged_length_m ?? 0)
                        - (r.unverified_length_m ?? 0), COLORS.not_checkable],
                    ]} total={routeLen} />
                    <div className="flex justify-between gap-2 text-[11px] text-ink-300 mt-1">
                      <span>seen {km(r.verified_length_m)} km</span>
                      {(r.bridged_length_m ?? 0) > 0 &&
                        <span>joined by healing {km(r.bridged_length_m)} km</span>}
                      <span>not seen {km(r.unverified_length_m)} km</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </Panel>
        )}

        {plan && (
          <Panel title="3 · What if roads fail?" dense className="shrink-0"
            actions={conditionsOn && (
              <button className="btn-ghost text-xs" onClick={resetConditions}><RotateCcw size={12} /> Reset</button>
            )}>
            <div className="space-y-3">
              {cur?.critical_roads?.length ? (
                <>
                  <div className="text-[10px] uppercase tracking-wider text-ink-500">
                    Roads this trip depends on most
                  </div>
                  <ul className="space-y-1">
                    {cur.critical_roads.map((c, i) => (
                      <li key={c.edge} className="flex items-center justify-between gap-2 text-[12px]">
                        <button className="text-left text-ink-300 hover:text-ink-100 truncate"
                                onClick={() => map.current?.setView(c.midpoint, 16)}>
                          #{i + 1} {c.name ?? `unnamed ${c.class}`}
                        </button>
                        {c.cuts_trip
                          ? <span className="text-risk-crit font-medium shrink-0">cuts the trip</span>
                          : <span className="font-mono text-ink-100 shrink-0">+{c.extra_min} min</span>}
                      </li>
                    ))}
                  </ul>
                  <div className="grid grid-cols-2 gap-2">
                    <button className="btn-ghost justify-center text-[12px]" onClick={() => setShowWorst((v) => !v)}>
                      <Crosshair size={13} /> {showWorst ? "Hide" : "Show"} #1
                    </button>
                    <button className="btn-danger justify-center text-[12px]" onClick={closeWorst}>
                      <XCircle size={13} /> Close #1
                    </button>
                  </div>
                  {showWorst && worst && (
                    <Note tone={worst.cuts_trip ? "warn" : "info"}>
                      {worst.cuts_trip
                        ? <>If <strong>{worst.name ?? "this road"}</strong> (✕) closes, there is <strong>no other way</strong> from A to B.</>
                        : <>If <strong>{worst.name ?? "this road"}</strong> (✕) closes, the orange detour takes
                            <strong> +{worst.extra_min} min</strong> (+{km(worst.extra_m)} km).</>}
                    </Note>
                  )}
                </>
              ) : r?.reachable ? <p className="text-[12px] text-ink-500">No roads to test.</p> : null}

              <div className="pt-2 border-t border-line-soft space-y-2">
                <button className={clsx("w-full justify-center text-[12px]", click === "close" ? "btn-primary" : "btn-ghost")}
                        onClick={() => setClick("close")}>
                  <ShieldAlert size={13} /> Close roads by clicking them ({closed.length} closed)
                </button>
                <button className={clsx("w-full justify-center text-[12px]", click === "flood" ? "btn-primary" : "btn-ghost")}
                        onClick={() => setClick("flood")}>
                  <Droplets size={13} /> {flood ? "Move the flood zone (click map)" : "Place a flood zone (click map)"}
                </button>
                {flood && (
                  <label className="flex items-center gap-2 text-[11px] text-ink-300">
                    Radius
                    <input type="range" min={50} max={1500} step={50} value={floodR}
                           onChange={(e) => setFloodR(+e.target.value)} className="flex-1" />
                    <span className="font-mono w-14 text-right">{floodR} m</span>
                    <button className="text-ink-500 hover:text-ink-100" onClick={() => setFlood(null)}>✕</button>
                  </label>
                )}
                <label className={clsx("flex items-center gap-2 text-[12px]",
                                       plan.satellite.available ? "text-ink-300" : "text-ink-500")}>
                  <input type="checkbox" checked={avoid} disabled={!plan.satellite.available}
                         onChange={(e) => setAvoid(e.target.checked)} />
                  Avoid roads the satellite could not see
                </label>
              </div>
            </div>
          </Panel>
        )}

        {plan && (
          <Panel title="4 · Satellite check" dense className="shrink-0"
            actions={plan.satellite.available && (
              <button className="btn-ghost text-xs" disabled={!!busy} onClick={() => runPlan(true)}
                      title="Search again for the newest clear image">
                <Satellite size={12} /> Refresh
              </button>
            )}>
            {plan.satellite.available ? (
              <div className="space-y-2.5 text-[12px]">
                <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-ink-300">
                  <span className="text-ink-500">Image date</span><span className="font-mono">{plan.satellite.scene?.date}</span>
                  <span className="text-ink-500">Cloud over area</span><span className="font-mono">{plan.satellite.scene?.box_unusable_pct}%</span>
                  <span className="text-ink-500">Model</span>
                  <span className="font-mono truncate" title={plan.satellite.model?.checkpoint}>
                    {plan.satellite.model?.architecture ?? "—"} (Sentinel-2)
                  </span>
                  <span className="text-ink-500">Roads checked</span>
                  <span className="font-mono">{(cnt?.verified ?? 0) + (cnt?.bridged ?? 0) + (cnt?.unverified ?? 0)} of {plan.road_count}</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  <Chip tone="good">{cnt?.verified ?? 0} seen</Chip>
                  <Chip tone="accent">{cnt?.bridged ?? 0} joined by healing</Chip>
                  <Chip tone="warn">{cnt?.unverified ?? 0} not seen</Chip>
                  <Chip>{(cnt?.not_checkable ?? 0) + (cnt?.outside_image ?? 0)} not checkable</Chip>
                </div>
                {plan.satellite.healing && (
                  <p className="text-[10px] text-ink-500 leading-relaxed">
                    Healing (the same step as the image pipeline) joined{" "}
                    <span className="font-mono text-ink-100">{plan.satellite.healing.gaps_joined}</span>{" "}
                    gaps of up to {plan.satellite.healing.max_gap_m} m where the model saw a road on
                    both sides - typically a tree crown or a shadow. Roads still not seen after
                    healing are the ones worth checking; one image cannot prove a road is closed.
                  </p>
                )}
                {plan.satellite.new_roads && (
                  <div className="pt-1 space-y-1">
                    <label className="flex items-center gap-1.5 text-[11px] text-ink-300">
                      <input type="checkbox" checked={showNew} onChange={(e) => setShowNew(e.target.checked)} />
                      Show roads the satellite sees but OSM does not
                      <span className="font-mono text-ink-100">
                        ({plan.satellite.new_roads.count}, {plan.satellite.new_roads.length_km} km)
                      </span>
                    </label>
                    <p className="text-[10px] text-ink-500 leading-relaxed">
                      Candidates for new or unmapped roads (≥ 200 m). The 10 m model also picks up
                      canals, field edges and roofs, so check them on the image. They are not used
                      for routing.
                    </p>
                  </div>
                )}
                <div className="flex gap-3 text-[11px] text-ink-300">
                  <label className="flex items-center gap-1.5">
                    <input type="checkbox" checked={showImg} onChange={(e) => setShowImg(e.target.checked)} /> Show image
                  </label>
                  <label className="flex items-center gap-1.5">
                    <input type="checkbox" checked={showModel} onChange={(e) => setShowModel(e.target.checked)} /> Show model's roads
                  </label>
                </div>
                <p className="text-[10px] text-ink-500">
                  {plan.satellite.scene?.id}
                  {plan.satellite.from_cache ? ` · cached (fetched ${plan.satellite.fetched_on})` : ""}
                </p>
              </div>
            ) : (
              <Note tone="warn">{plan.satellite.reason} The route uses OpenStreetMap only.</Note>
            )}
          </Panel>
        )}

        {plan && (
          <Panel title="How to read this" dense className="shrink-0">
            <div className="space-y-2">
              {plan.warnings.map((w, i) => <Note key={i} tone="warn">{w}</Note>)}
              <p className="text-[10px] text-ink-500 leading-relaxed">{plan.note}</p>
            </div>
          </Panel>
        )}
      </div>
    </div>
  );
}

function PointInput({ label, color, v, onChange, onPick }: {
  label: string; color: string; v: LL | null; onChange: (p: LL | null) => void; onPick: () => void;
}) {
  const [txt, setTxt] = useState(v ? `${v.lat}, ${v.lon}` : "");
  useEffect(() => { setTxt(v ? `${v.lat}, ${v.lon}` : ""); }, [v]);
  const commit = () => {
    const m = txt.split(/[ ,]+/).filter(Boolean).map(Number);
    if (m.length === 2 && m.every(Number.isFinite) && Math.abs(m[0]) <= 85 && Math.abs(m[1]) <= 180)
      onChange({ lat: m[0], lon: m[1] });
  };
  return (
    <div className="flex items-center gap-2">
      <span className="w-6 h-6 rounded-full grid place-items-center text-[11px] font-bold text-base-900 shrink-0"
            style={{ background: color }}>{label}</span>
      <input className="input flex-1 font-mono text-[12px]" value={txt} placeholder="lat, lon"
             onChange={(e) => setTxt(e.target.value)} onBlur={commit}
             onKeyDown={(e) => { if (e.key === "Enter") commit(); }} />
      <button className="btn-ghost px-2" title="Pick on the map" onClick={onPick}><MapPin size={13} /></button>
    </div>
  );
}

function Legend({ color, label, dashed, wide }: { color: string; label: string; dashed?: boolean; wide?: boolean }) {
  return (
    <div className="flex items-center gap-2 text-ink-300">
      <span className="inline-block w-5" style={{ borderTop: `${wide ? 4 : 3}px ${dashed ? "dashed" : "solid"} ${color}` }} />
      {label}
    </div>
  );
}

function Bar({ parts, total }: { parts: [number, string][]; total: number }) {
  return (
    <div className="flex h-2 rounded overflow-hidden bg-base-900">
      {parts.map(([v, c], i) => (
        <div key={i} style={{ width: `${Math.max(0, (100 * v) / Math.max(total, 1))}%`, background: c }} />
      ))}
    </div>
  );
}
