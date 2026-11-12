/**
 * Shared primitives.
 *
 * `ProvenanceBanner` is the load-bearing one. The backend refuses to return a
 * mask without provenance; this component makes sure the user sees it. A
 * classical-CV result and a trained-model result must never look alike on
 * screen, so the banner colour and wording change with `is_learned`.
 */
import {
  AlertTriangle, Brain, CheckCircle2, FlaskConical, Info, Loader2, Map,
} from "lucide-react";
import clsx from "clsx";
import type { Provenance } from "@/types/api";

/* ------------------------------------------------------------------ panel */
export function Panel({
  title, actions, children, className, dense,
}: {
  title?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  dense?: boolean;
}) {
  return (
    <section className={clsx("panel flex flex-col min-h-0", className)}>
      {title && (
        <header className="panel-hd">
          <h2 className="label">{title}</h2>
          {actions}
        </header>
      )}
      <div className={clsx("min-h-0 flex-1", dense ? "p-3" : "p-4")}>{children}</div>
    </section>
  );
}

/* ------------------------------------------------------------- provenance */
const PROV_STYLE: Record<string, { cls: string; Icon: typeof Brain }> = {
  TRAINED_MODEL:      { cls: "border-accent/40 bg-accent/10 text-accent",       Icon: Brain },
  CLASSICAL_BASELINE: { cls: "border-risk-med/40 bg-risk-med/10 text-risk-med", Icon: FlaskConical },
  DEMO_DATA:          { cls: "border-risk-high/40 bg-risk-high/10 text-risk-high", Icon: FlaskConical },
  OSM_REFERENCE:      { cls: "border-risk-low/40 bg-risk-low/10 text-risk-low", Icon: Map },
};

export function ProvenanceBanner({ p, compact }: { p: Provenance; compact?: boolean }) {
  const st = PROV_STYLE[p.mode] ?? {
    cls: "border-line bg-base-700 text-ink-300",
    Icon: Info,
  };
  const { Icon } = st;
  return (
    <div className={clsx("rounded-md border px-3 py-2 flex gap-2.5", st.cls)}>
      <Icon size={15} className="mt-0.5 shrink-0" />
      <div className="min-w-0">
        <div className="text-[11px] font-semibold tracking-wide">{p.label}</div>
        {!compact && (
          <p className="text-[11px] leading-relaxed mt-1 text-ink-300">{p.description}</p>
        )}
        {!p.is_learned && !compact && (
          <p className="text-[11px] mt-1.5 font-medium">
            This output is not a deep-learning prediction.
          </p>
        )}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- metrics */
export function Metric({
  label, value, unit, hint, tone = "default", size = "md",
}: {
  label: string;
  value: string | number | null | undefined;
  unit?: string;
  hint?: string;
  tone?: "default" | "good" | "warn" | "bad" | "accent";
  size?: "sm" | "md" | "lg";
}) {
  const toneCls = {
    default: "text-ink-100", good: "text-risk-low", warn: "text-risk-med",
    bad: "text-risk-crit", accent: "text-accent",
  }[tone];
  const sizeCls = { sm: "text-sm", md: "text-lg", lg: "text-2xl" }[size];
  const shown = value === null || value === undefined || value === ""
    ? "\u2014" : value;

  return (
    <div className="min-w-0" title={hint}>
      <div className="label truncate">{label}</div>
      <div className={clsx("metric font-semibold mt-0.5", sizeCls, toneCls)}>
        {shown}
        {unit && shown !== "\u2014" && (
          <span className="text-ink-500 text-xs font-normal ml-1">{unit}</span>
        )}
      </div>
    </div>
  );
}

/** Renders a dash plus the backend's reason. Never invents a number. */
export function UnavailableMetric({ label, reason }: { label: string; reason?: string | null }) {
  return (
    <div className="min-w-0" title={reason ?? undefined}>
      <div className="label truncate">{label}</div>
      <div className="metric text-lg font-semibold mt-0.5 text-ink-500">&mdash;</div>
    </div>
  );
}

/* ----------------------------------------------------------------- states */
export function Loading({ text = "Computing\u2026" }: { text?: string }) {
  return (
    <div className="flex items-center gap-2 text-ink-300 text-sm py-6 justify-center">
      <Loader2 size={15} className="animate-spin" />
      {text}
    </div>
  );
}

export function Empty({ title, hint, icon: Icon = Info }: {
  title: string; hint?: string; icon?: typeof Info;
}) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-10 px-6 gap-2">
      <Icon size={22} className="text-ink-500" />
      <p className="text-sm text-ink-300">{title}</p>
      {hint && <p className="text-xs text-ink-500 max-w-xs leading-relaxed">{hint}</p>}
    </div>
  );
}

export function ErrorNote({ title, detail, hint }: {
  title: string; detail?: string; hint?: string;
}) {
  return (
    <div className="rounded-md border border-risk-crit/40 bg-risk-crit/10 px-3 py-2.5 flex gap-2.5">
      <AlertTriangle size={15} className="text-risk-crit mt-0.5 shrink-0" />
      <div className="min-w-0">
        <div className="text-[12px] font-semibold text-risk-crit">{title}</div>
        {detail && <p className="text-[11px] text-ink-300 mt-1 leading-relaxed">{detail}</p>}
        {hint && <p className="text-[11px] text-ink-500 mt-1 leading-relaxed">{hint}</p>}
      </div>
    </div>
  );
}

export function Note({ children, tone = "info" }: {
  children: React.ReactNode; tone?: "info" | "warn";
}) {
  const cls = tone === "warn"
    ? "border-risk-med/40 bg-risk-med/10 text-risk-med"
    : "border-line bg-base-700/60 text-ink-300";
  const Icon = tone === "warn" ? AlertTriangle : Info;
  return (
    <div className={clsx("rounded-md border px-3 py-2 flex gap-2.5 text-[11px] leading-relaxed", cls)}>
      <Icon size={13} className="mt-0.5 shrink-0" />
      <div className="min-w-0">{children}</div>
    </div>
  );
}

/* ------------------------------------------------------------------ misc */
export function Chip({ children, tone = "neutral" }: {
  children: React.ReactNode;
  tone?: "neutral" | "good" | "warn" | "bad" | "accent";
}) {
  const cls = {
    neutral: "border-line text-ink-300 bg-base-700",
    good: "border-risk-low/40 text-risk-low bg-risk-low/10",
    warn: "border-risk-med/40 text-risk-med bg-risk-med/10",
    bad: "border-risk-crit/40 text-risk-crit bg-risk-crit/10",
    accent: "border-accent/40 text-accent bg-accent/10",
  }[tone];
  return <span className={clsx("chip", cls)}>{children}</span>;
}

export function StatusDot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] text-ink-300">
      <span className={clsx("w-1.5 h-1.5 rounded-full",
        ok ? "bg-risk-low" : "bg-risk-med")} />
      {label}
    </span>
  );
}

export function Ok({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] text-risk-low">
      <CheckCircle2 size={12} /> {children}
    </span>
  );
}

/** Shared risk ramp. Criticality colour must mean the same thing everywhere. */
export function riskColor(v: number): string {
  if (v >= 0.75) return "#EF4444";
  if (v >= 0.5) return "#F97316";
  if (v >= 0.25) return "#EAB308";
  return "#22C55E";
}

export function riskLabel(v: number): "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" {
  if (v >= 0.75) return "CRITICAL";
  if (v >= 0.5) return "HIGH";
  if (v >= 0.25) return "MEDIUM";
  return "LOW";
}

export const fmt = {
  n: (v: number | null | undefined, d = 2) =>
    v === null || v === undefined || !isFinite(v) ? null : v.toFixed(d),
  pct: (v: number | null | undefined, d = 1) =>
    v === null || v === undefined || !isFinite(v) ? null : `${(v * 100).toFixed(d)}%`,
  km: (m: number | null | undefined) =>
    m === null || m === undefined ? null : `${(m / 1000).toFixed(2)}`,
};
