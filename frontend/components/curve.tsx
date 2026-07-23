"use client";

// The M6 improvement curve: two stacked single-axis panels (quality %, cost ¢/query)
// sharing the week axis — never a dual-axis chart. Promotion weeks get a vertical
// marker. Series colors come from the theme tokens (--color-quality / --color-cost).

import { useState } from "react";
import type { WeeklyPoint } from "@/app/lib/api";
import { cn } from "@/lib/utils";

const W = 640;
const PANEL_H = 118;
const PAD = { l: 40, r: 14, t: 16, b: 8 };
const AXIS_H = 22;

function Panel({
  label,
  values,
  weekly,
  colorVar,
  fmt,
  domainMax,
  hover,
  setHover,
  showAxis,
}: {
  label: string;
  values: number[];
  weekly: WeeklyPoint[];
  colorVar: string;
  fmt: (v: number) => string;
  domainMax: number;
  hover: number | null;
  setHover: (i: number | null) => void;
  showAxis: boolean;
}) {
  const h = PANEL_H + (showAxis ? AXIS_H : 0);
  const iw = W - PAD.l - PAD.r;
  const ih = PANEL_H - PAD.t - PAD.b;
  const x = (i: number) =>
    PAD.l + (weekly.length > 1 ? (i / (weekly.length - 1)) * iw : iw / 2);
  const y = (v: number) => PAD.t + ih - (v / domainMax) * ih;
  const line = values.map((v, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(v)}`).join(" ");
  const area = `${line} L${x(values.length - 1)},${PAD.t + ih} L${x(0)},${PAD.t + ih} Z`;
  const gridVals = [0.5, 1].map((f) => domainMax * f);

  return (
    <svg
      viewBox={`0 0 ${W} ${h}`}
      className="w-full"
      role="img"
      aria-label={`${label}: ${weekly.map((w, i) => `week ${w.week} ${fmt(values[i])}`).join(", ")}`}
      onMouseLeave={() => setHover(null)}
    >
      <text x={PAD.l} y={11} className="fill-muted-foreground text-[11px] font-medium">
        {label}
      </text>
      {gridVals.map((g) => (
        <g key={g}>
          <line
            x1={PAD.l}
            x2={W - PAD.r}
            y1={y(g)}
            y2={y(g)}
            className="stroke-border"
            strokeDasharray="2 4"
          />
          <text
            x={PAD.l - 6}
            y={y(g) + 3}
            textAnchor="end"
            className="fill-muted-foreground/70 text-[10px] tabular-nums"
          >
            {fmt(g)}
          </text>
        </g>
      ))}
      {weekly.map(
        (w, i) =>
          w.cycle.promoted && (
            <line
              key={`p${i}`}
              x1={x(i)}
              x2={x(i)}
              y1={PAD.t - 2}
              y2={PAD.t + ih}
              className="stroke-foreground/30"
              strokeDasharray="4 3"
            />
          ),
      )}
      <path d={area} fill={`var(${colorVar})`} opacity={0.08} />
      <path d={line} fill="none" stroke={`var(${colorVar})`} strokeWidth={2} />
      {values.map((v, i) => (
        <g key={i}>
          {/* invisible wide hit target per point */}
          <rect
            x={x(i) - iw / (2 * values.length)}
            y={0}
            width={iw / values.length}
            height={PANEL_H}
            fill="transparent"
            onMouseEnter={() => setHover(i)}
          />
          <circle
            cx={x(i)}
            cy={y(v)}
            r={hover === i ? 5 : 3.5}
            fill={`var(${colorVar})`}
            className="stroke-card"
            strokeWidth={2}
            pointerEvents="none"
          />
        </g>
      ))}
      {hover !== null && (
        <line
          x1={x(hover)}
          x2={x(hover)}
          y1={PAD.t - 2}
          y2={PAD.t + ih}
          className="stroke-foreground/15"
          pointerEvents="none"
        />
      )}
      {showAxis &&
        weekly.map((w, i) => (
          <text
            key={i}
            x={x(i)}
            y={h - 7}
            textAnchor="middle"
            className={cn(
              "text-[11px] tabular-nums",
              hover === i ? "fill-foreground" : "fill-muted-foreground/70",
            )}
          >
            w{w.week}
          </text>
        ))}
    </svg>
  );
}

export function ImprovementCurve({ weekly }: { weekly: WeeklyPoint[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const quality = weekly.map((w) => (100 * w.grounded) / w.n_queries);
  const cost = weekly.map((w) => (100 * w.cost_usd) / w.n_queries);
  const promoted = weekly.filter((w) => w.cycle.promoted);
  const hovered = hover !== null ? weekly[hover] : null;

  return (
    <div className="flex flex-col">
      <Panel
        label="Quality — grounded answers"
        values={quality}
        weekly={weekly}
        colorVar="--color-quality"
        fmt={(v) => `${v.toFixed(0)}%`}
        domainMax={100}
        hover={hover}
        setHover={setHover}
        showAxis={false}
      />
      <Panel
        label="Cost per query"
        values={cost}
        weekly={weekly}
        colorVar="--color-cost"
        fmt={(v) => `${v.toFixed(1)}¢`}
        domainMax={Math.max(...cost) * 1.15}
        hover={hover}
        setHover={setHover}
        showAxis={true}
      />
      <div className="mt-2 flex min-h-[2rem] flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {hovered ? (
          <>
            <span className="font-medium text-foreground">week {hovered.week}</span>
            <span style={{ color: "var(--color-quality)" }}>
              {hovered.grounded}/{hovered.n_queries} grounded
            </span>
            <span style={{ color: "var(--color-cost)" }}>
              {((100 * hovered.cost_usd) / hovered.n_queries).toFixed(1)}¢/query
            </span>
            <span className="font-mono">
              {Object.entries(hovered.tier_mix)
                .map(([t, n]) => `${t}:${n}`)
                .join(" ")}
            </span>
            <span className="truncate">{hovered.router_version}</span>
          </>
        ) : (
          <>
            <span className="inline-flex items-center gap-1.5">
              <span
                className="h-0.5 w-4 rounded"
                style={{ background: "var(--color-quality)" }}
              />
              quality
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span
                className="h-0.5 w-4 rounded"
                style={{ background: "var(--color-cost)" }}
              />
              cost
            </span>
            {promoted.length > 0 && (
              <span className="inline-flex items-center gap-1.5">
                <span className="h-3 border-l border-dashed border-foreground/40" />
                promotion at week {promoted.map((w) => w.week).join(", ")}
              </span>
            )}
          </>
        )}
      </div>
    </div>
  );
}
