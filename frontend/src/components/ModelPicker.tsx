/**
 * Pick which trained checkpoint runs extraction (DeepGlobe, Sentinel-2,
 * LISS-IV ...). The list comes from GET /api/models - only checkpoints that
 * actually exist on the server are offered.
 */
import { useEffect, useState } from "react";
import { api } from "@/services/api";
import type { ModelInfo } from "@/types/api";

export default function ModelPicker({ value, onChange, disabled }: {
  value: string | null;
  onChange: (id: string | null, info: ModelInfo | null) => void;
  disabled?: boolean;
}) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [def, setDef] = useState<string | null>(null);

  useEffect(() => {
    api.models().then((r) => { setModels(r.models); setDef(r.default); }).catch(() => {});
  }, []);

  if (models.length < 2) return null;              // nothing to choose
  const cur = models.find((m) => m.id === (value ?? def)) ?? null;
  return (
    <label className="block text-[11px] text-ink-300 space-y-1">
      <span>Model (match it to the imagery)</span>
      <select className="input w-full" disabled={disabled} value={value ?? def ?? ""}
              onChange={(e) => {
                const m = models.find((x) => x.id === e.target.value) ?? null;
                onChange(m?.id ?? null, m);
              }}>
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.sensor}{m.id === def ? " (default)" : ""} - {m.id}
          </option>
        ))}
      </select>
      {cur && <span className="block text-[10px] text-ink-500">{cur.note}. Input pixel size {cur.input_gsd_m} m.</span>}
    </label>
  );
}
