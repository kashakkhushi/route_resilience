/** Imagery: load a demo city or upload a raster; inspect layers and metadata. */
import { useRef, useState } from "react";
import { Image as ImageIcon, Upload } from "lucide-react";
import { Empty, Note, Panel } from "@/components/ui";
import { Row } from "@/App";

const CITIES = [
  { id: "grid_river",  label: "Grid + river",  note: "Few crossings \u2192 real gatekeepers" },
  { id: "dense_urban", label: "Dense urban",   note: "High redundancy" },
  { id: "sparse_rural",label: "Sparse rural",  note: "Tree-like, low redundancy" },
];

export default function ImagerySection({ p }: any) {
  const { state } = p;
  const [city, setCity] = useState("grid_river");
  const [size, setSize] = useState(768);
  const [seed, setSeed] = useState(1);
  const [layer, setLayer] = useState("original");
  const fileRef = useRef<HTMLInputElement>(null);

  const layers = state.imagery?.layers ?? {};
  const names = Object.keys(layers);

  return (
    <div className="flex-1 grid grid-cols-12 gap-3 min-h-0">
      <Panel className="col-span-12 lg:col-span-8" title="Imagery"
        actions={names.length > 0 && (
          <div className="flex gap-1">
            {names.map((n) => (
              <button key={n} onClick={() => setLayer(n)}
                className={`chip ${layer === n
                  ? "border-accent/40 text-accent bg-accent/10"
                  : "border-line text-ink-500 bg-base-700"}`}>
                {n.replace(/_/g, " ")}
              </button>
            ))}
          </div>
        )}>
        {state.imagery ? (
          <div className="h-full flex items-center justify-center bg-base-900 rounded-md overflow-hidden">
            <img src={layers[layer] ?? layers[names[0]]} alt={layer}
                 className="max-w-full max-h-[520px] object-contain" />
          </div>
        ) : (
          <Empty icon={ImageIcon} title="No imagery loaded"
                 hint="Load a demo city for a deterministic scene with known ground truth, or upload your own raster." />
        )}
      </Panel>

      <div className="col-span-12 lg:col-span-4 flex flex-col gap-3 overflow-auto">
        <Panel title="Demo study area" dense>
          <div className="space-y-3">
            <div className="space-y-1.5">
              {CITIES.map((c) => (
                <button key={c.id} onClick={() => setCity(c.id)}
                  className={`w-full text-left px-2.5 py-2 rounded-md border text-[12px] transition-colors ${
                    city === c.id ? "border-accent/40 bg-accent/10" : "border-line hover:bg-base-700"}`}>
                  <div className="font-medium">{c.label}</div>
                  <div className="text-[10px] text-ink-500 mt-0.5">{c.note}</div>
                </button>
              ))}
            </div>
            <div className="grid grid-cols-2 gap-2">
              <label className="text-[11px] text-ink-300">
                Size
                <input type="number" value={size} min={256} max={1536} step={128}
                       onChange={(e) => setSize(+e.target.value)}
                       className="input w-full mt-1" />
              </label>
              <label className="text-[11px] text-ink-300">
                Seed
                <input type="number" value={seed} min={0} max={9999}
                       onChange={(e) => setSeed(+e.target.value)}
                       className="input w-full mt-1" />
              </label>
            </div>
            <button className="btn-primary w-full justify-center" disabled={!!state.busy}
                    onClick={() => p.loadDemo(city, size, seed)}>
              Load demonstration city
            </button>
            <Note>
              Synthetic and deterministic: the same seed always reproduces the same
              city. Because we generate the ground truth, segmentation and healing
              quality can be <em>measured</em> here rather than asserted.
            </Note>
          </div>
        </Panel>

        <Panel title="Upload raster" dense>
          <input ref={fileRef} type="file" accept=".tif,.tiff,.png,.jpg,.jpeg"
                 className="hidden"
                 onChange={(e) => e.target.files?.[0] && p.upload(e.target.files[0])} />
          <button className="btn-ghost w-full justify-center" disabled={!!state.busy}
                  onClick={() => fileRef.current?.click()}>
            <Upload size={14} /> Choose GeoTIFF or image
          </button>
          <p className="text-[10px] text-ink-500 mt-2 leading-relaxed">
            A georeferenced GeoTIFF enables real-world lengths and OSM reference
            comparison. Ungeoreferenced images still work, but distances are in
            pixels scaled by an assumed GSD.
          </p>
        </Panel>

        {state.imagery && (
          <Panel title="Raster metadata" dense>
            <div className="space-y-1.5 text-[11px]">
              <Row k="Source" v={state.imagery.meta.source} />
              <Row k="Dimensions" v={`${state.imagery.meta.width} \u00d7 ${state.imagery.meta.height}`} />
              <Row k="Bands" v={state.imagery.meta.bands} />
              <Row k="Dtype" v={state.imagery.meta.dtype} />
              <Row k="GSD" v={`${state.imagery.meta.pixel_size_m} m/px`} />
              <Row k="CRS" v={state.imagery.meta.crs ?? "\u2014"} />
              <Row k="Georeferenced" v={state.imagery.meta.is_georeferenced ? "yes" : "no"} />
              {state.imagery.meta.bounds && (
                <Row k="Bounds" v={state.imagery.meta.bounds.map((b: number) => b.toFixed(3)).join(", ")} />
              )}
            </div>
          </Panel>
        )}
      </div>
    </div>
  );
}
