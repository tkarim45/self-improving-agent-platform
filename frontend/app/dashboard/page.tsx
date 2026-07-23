"use client";

// The admin console: live traffic summary, the M6 improvement curve, promotion history,
// the golden gate, and recent traces. Read-only — it renders the same artifacts the CLIs
// write, so what you see here is exactly what the repo measures.

import { useEffect, useState } from "react";
import {
  api,
  type GoldenReport,
  type Promotions,
  type Summary,
  type TraceRow,
  type WeeklyPoint,
} from "../lib/api";

// One measure per panel, shared x — never a dual-axis chart.
const QUALITY = "#2a78d6";
const COST = "#eb6834";

function Curve({ weekly }: { weekly: WeeklyPoint[] }) {
  const W = 560;
  const H = 120;
  const PAD = { l: 44, r: 12, t: 10, b: 22 };
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;
  const promoWeeks = weekly
    .map((w, i) => (w.cycle.promoted ? i : -1))
    .filter((i) => i >= 0);

  const x = (i: number) =>
    PAD.l + (weekly.length > 1 ? (i / (weekly.length - 1)) * iw : iw / 2);

  function panel(
    label: string,
    values: number[],
    color: string,
    fmt: (v: number) => string,
  ) {
    const max = Math.max(...values) * 1.1 || 1;
    const y = (v: number) => PAD.t + ih - (v / max) * ih;
    const path = values.map((v, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(v)}`).join(" ");
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label}>
        <text x={PAD.l} y={PAD.t + 2} className="fill-neutral-500 text-[10px]">
          {label}
        </text>
        {/* promotion marker(s) */}
        {promoWeeks.map((i) => (
          <line
            key={i}
            x1={x(i)}
            x2={x(i)}
            y1={PAD.t}
            y2={PAD.t + ih}
            stroke="currentColor"
            className="text-neutral-300 dark:text-neutral-700"
            strokeDasharray="3 3"
          />
        ))}
        <path d={path} fill="none" stroke={color} strokeWidth={2} />
        {values.map((v, i) => (
          <g key={i}>
            <circle cx={x(i)} cy={y(v)} r={4} fill={color} />
            <title>{`week ${weekly[i].week}: ${fmt(v)} (${weekly[i].router_version})`}</title>
          </g>
        ))}
        {values.map((_, i) => (
          <text
            key={i}
            x={x(i)}
            y={H - 6}
            textAnchor="middle"
            className="fill-neutral-400 text-[10px]"
          >
            w{weekly[i].week}
          </text>
        ))}
        <text
          x={PAD.l - 6}
          y={y(values[0]) + 3}
          textAnchor="end"
          className="fill-neutral-400 text-[10px]"
        >
          {fmt(values[0])}
        </text>
        <text
          x={W - PAD.r + 2}
          y={y(values[values.length - 1]) + 3}
          className="fill-neutral-500 text-[10px]"
        >
          {fmt(values[values.length - 1])}
        </text>
      </svg>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      {panel(
        "quality — grounded answers (%)",
        weekly.map((w) => (100 * w.grounded) / w.n_queries),
        QUALITY,
        (v) => `${v.toFixed(0)}%`,
      )}
      {panel(
        "cost per query (¢)",
        weekly.map((w) => (100 * w.cost_usd) / w.n_queries),
        COST,
        (v) => `${v.toFixed(1)}¢`,
      )}
      {promoWeeks.length > 0 && (
        <p className="text-xs text-neutral-400">
          dashed line = automated promotion (week {promoWeeks.map((i) => weekly[i].week).join(", ")})
        </p>
      )}
    </div>
  );
}

function Card({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
      <div className="text-xs uppercase tracking-wide text-neutral-400">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-neutral-500">{sub}</div>}
    </div>
  );
}

export default function Dashboard() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [weekly, setWeekly] = useState<WeeklyPoint[] | null>(null);
  const [promos, setPromos] = useState<Promotions | null>(null);
  const [golden, setGolden] = useState<GoldenReport | null>(null);
  const [traces, setTraces] = useState<TraceRow[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api.summary().then(setSummary).catch((e) => setError(String(e)));
    api.weekly().then(setWeekly).catch(() => setWeekly(null)); // 404 until a sim has run
    api.promotions().then(setPromos).catch(() => null);
    api.golden().then(setGolden).catch(() => null);
    api.traces(15).then(setTraces).catch(() => null);
  }, []);

  return (
    <div className="flex flex-col gap-8">
      <section>
        <h1 className="text-xl font-semibold">Admin console</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Traffic, the improvement curve, promotion history and the eval gate — read from the
          same artifacts the CLIs write.
        </p>
        {error && <p className="mt-2 text-sm text-red-600">backend unreachable: {error}</p>}
      </section>

      {summary && (
        <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Card label="requests" value={String(summary.n)} />
          <Card
            label="grounded rate"
            value={`${(100 * summary.grounded_rate).toFixed(0)}%`}
          />
          <Card
            label="total cost"
            value={`$${summary.total_cost.toFixed(3)}`}
            sub={`mean $${summary.mean_cost.toFixed(4)}/query`}
          />
          <Card
            label="guardrail actions"
            value={String(summary.blocks + summary.redactions)}
            sub={`${summary.blocks} blocked · ${summary.redactions} redacted`}
          />
        </section>
      )}

      <section>
        <h2 className="mb-2 font-medium">The improvement curve (M6)</h2>
        {weekly ? (
          <Curve weekly={weekly} />
        ) : (
          <p className="text-sm text-neutral-400">
            No simulation artifact yet — run <code>make sim</code> in backend/.
          </p>
        )}
      </section>

      <section className="grid gap-6 md:grid-cols-2">
        <div>
          <h2 className="mb-2 font-medium">Promotion history (M5)</h2>
          {promos && (
            <>
              <p className="mb-2 text-sm text-neutral-500">
                active router:{" "}
                <code className="rounded bg-neutral-100 px-1 dark:bg-neutral-900">
                  {promos.active.router.version}
                </code>
              </p>
              <ul className="flex flex-col gap-1 text-sm">
                {promos.entries.length === 0 && (
                  <li className="text-neutral-400">no cycles recorded</li>
                )}
                {promos.entries.map((e, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <span
                      className={`mt-0.5 rounded px-1.5 text-xs font-medium ${
                        e.promoted
                          ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                          : "bg-neutral-100 text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400"
                      }`}
                    >
                      {e.promoted ? "PROMOTE" : "reject"}
                    </span>
                    <span className="text-neutral-600 dark:text-neutral-300">
                      <span className="text-neutral-400">{e.ts.slice(0, 10)}</span>{" "}
                      {e.reason}
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>

        <div>
          <h2 className="mb-2 font-medium">Golden gate (M4)</h2>
          {golden ? (
            <>
              <p className="text-sm">
                <span
                  className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                    golden.passed
                      ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                      : "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300"
                  }`}
                >
                  {golden.passed ? "GREEN" : "RED"}
                </span>{" "}
                {(100 * golden.score).toFixed(0)}% vs threshold{" "}
                {(100 * golden.threshold).toFixed(0)}%
              </p>
              <ul className="mt-2 grid grid-cols-2 gap-x-4 text-sm text-neutral-500">
                {Object.entries(golden.by_kind).map(([kind, [p, n]]) => (
                  <li key={kind}>
                    {kind}: {p}/{n}
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="text-sm text-neutral-400">no golden records</p>
          )}
        </div>
      </section>

      <section>
        <h2 className="mb-2 font-medium">Recent traces (M3)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase tracking-wide text-neutral-400">
              <tr>
                <th className="py-1 pr-3">ts</th>
                <th className="py-1 pr-3">query</th>
                <th className="py-1 pr-3">tier</th>
                <th className="py-1 pr-3">grounded</th>
                <th className="py-1 pr-3">guard</th>
                <th className="py-1 pr-3 text-right">cost</th>
              </tr>
            </thead>
            <tbody>
              {traces.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-2 text-neutral-400">
                    no traces yet — ask something in Chat
                  </td>
                </tr>
              )}
              {traces.map((t) => (
                <tr
                  key={t.trace_id}
                  className="border-t border-neutral-100 dark:border-neutral-800"
                >
                  <td className="py-1.5 pr-3 whitespace-nowrap text-neutral-400">
                    {t.ts}
                  </td>
                  <td className="max-w-xs truncate py-1.5 pr-3" title={t.query}>
                    {t.query}
                  </td>
                  <td className="py-1.5 pr-3">{t.model_tier.includes("sonnet") ? "strong" : "cheap"}</td>
                  <td className="py-1.5 pr-3">{t.grounded ? "✓" : "✗"}</td>
                  <td className="py-1.5 pr-3">{t.guard_action}</td>
                  <td className="py-1.5 pr-3 text-right tabular-nums">
                    ${t.cost_usd.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
