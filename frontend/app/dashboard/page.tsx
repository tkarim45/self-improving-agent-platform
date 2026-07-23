"use client";

// Admin console. Evidence first: the M6 curve, the promotion log, the golden gate and
// the trace table are the heroes; everything reads from the same artifacts the CLIs
// write. Loading gets skeletons, absence gets an empty state that teaches the fix.

import { useEffect, useState } from "react";
import { ArrowUpCircle, MinusCircle, TerminalSquare } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { StatusBadge } from "@/components/status-badge";
import { ImprovementCurve } from "@/components/curve";
import {
  api,
  type GoldenReport,
  type Promotions,
  type Summary,
  type TraceRow,
  type WeeklyPoint,
} from "../lib/api";

type Load<T> = { state: "loading" } | { state: "ready"; data: T } | { state: "absent" };

function useLoad<T>(fn: () => Promise<T>): Load<T> {
  const [v, setV] = useState<Load<T>>({ state: "loading" });
  useEffect(() => {
    fn()
      .then((data) => setV({ state: "ready", data }))
      .catch(() => setV({ state: "absent" }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return v;
}

function Stat({
  label,
  value,
  detail,
}: {
  label: string;
  value: React.ReactNode;
  detail?: string;
}) {
  return (
    <div className="flex flex-col gap-0.5 rounded-lg border bg-card px-4 py-3">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-xl font-semibold tabular-nums">{value}</span>
      {detail && <span className="text-xs text-muted-foreground">{detail}</span>}
    </div>
  );
}

function EmptyHint({ command, children }: { command: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col items-start gap-2 rounded-lg border border-dashed p-6 text-sm text-muted-foreground">
      <TerminalSquare className="h-4 w-4" />
      <p>{children}</p>
      <code className="rounded bg-secondary px-2 py-1 font-mono text-xs">{command}</code>
    </div>
  );
}

export default function Dashboard() {
  const summary = useLoad<Summary>(api.summary);
  const weekly = useLoad<WeeklyPoint[]>(api.weekly);
  const promos = useLoad<Promotions>(api.promotions);
  const golden = useLoad<GoldenReport>(api.golden);
  const traces = useLoad<TraceRow[]>(() => api.traces(15));

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Admin console</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Traffic, the improvement curve, promotion history and the eval gate — read from
          the same artifacts the CLIs write.
        </p>
      </div>

      {/* traffic summary */}
      <section aria-label="Traffic summary">
        {summary.state === "loading" && (
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-[74px] rounded-lg" />
            ))}
          </div>
        )}
        {summary.state === "absent" && (
          <EmptyHint command="cd backend && make api">
            Backend unreachable — start it, then reload.
          </EmptyHint>
        )}
        {summary.state === "ready" && (
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat label="Requests traced" value={summary.data.n} />
            <Stat
              label="Grounded rate"
              value={`${(100 * summary.data.grounded_rate).toFixed(0)}%`}
              detail={`${(100 * summary.data.mean_citation_rate).toFixed(0)}% of claims cited`}
            />
            <Stat
              label="Total cost"
              value={`$${summary.data.total_cost.toFixed(3)}`}
              detail={`mean $${summary.data.mean_cost.toFixed(4)} / query`}
            />
            <Stat
              label="Guardrail actions"
              value={summary.data.blocks + summary.data.redactions}
              detail={`${summary.data.blocks} blocked · ${summary.data.redactions} redacted · ${summary.data.escalations} escalations`}
            />
          </div>
        )}
      </section>

      {/* the curve */}
      <Card>
        <CardHeader>
          <CardTitle>The improvement curve</CardTitle>
          <CardDescription>
            Six unattended simulated weeks on real Bedrock — quality held while the
            flywheel cut cost 54% at the week-3 promotion. $1.77 total.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {weekly.state === "loading" && <Skeleton className="h-64 w-full" />}
          {weekly.state === "absent" && (
            <EmptyHint command="cd backend && make sim">
              No simulation artifact yet. Run the six-week simulation (spends ~$2 on
              Bedrock) to produce weekly.json.
            </EmptyHint>
          )}
          {weekly.state === "ready" && <ImprovementCurve weekly={weekly.data} />}
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* promotions */}
        <Card>
          <CardHeader>
            <CardTitle>Promotion history</CardTitle>
            {promos.state === "ready" && (
              <CardDescription>
                active router:{" "}
                <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-xs">
                  {promos.data.active.router.version}
                </code>
              </CardDescription>
            )}
          </CardHeader>
          <CardContent>
            {promos.state === "loading" && <Skeleton className="h-40 w-full" />}
            {promos.state === "absent" && (
              <EmptyHint command="cd backend && make flywheel-cycle">
                No cycles recorded yet.
              </EmptyHint>
            )}
            {promos.state === "ready" && (
              <ol className="relative flex flex-col gap-4 border-l pl-5">
                {promos.data.entries.length === 0 && (
                  <li className="text-sm text-muted-foreground">no cycles recorded</li>
                )}
                {promos.data.entries.map((e, i) => (
                  <li key={i} className="relative">
                    <span className="absolute -left-[27px] top-0.5 bg-card">
                      {e.promoted ? (
                        <ArrowUpCircle className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                      ) : (
                        <MinusCircle className="h-4 w-4 text-muted-foreground/60" />
                      )}
                    </span>
                    <div className="flex items-baseline gap-2">
                      <span className="text-sm font-medium">
                        {e.promoted ? "Promoted" : "Rejected"}
                      </span>
                      <span className="font-mono text-xs text-muted-foreground">
                        {e.ts.slice(0, 10)}
                      </span>
                    </div>
                    <p className="mt-0.5 max-w-[60ch] text-sm text-muted-foreground">
                      {e.reason}
                    </p>
                  </li>
                ))}
              </ol>
            )}
          </CardContent>
        </Card>

        {/* golden gate */}
        <Card>
          <CardHeader>
            <CardTitle>Golden gate</CardTitle>
            <CardDescription>
              Deterministic replay of the committed golden records — the same gate CI runs.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {golden.state === "loading" && <Skeleton className="h-40 w-full" />}
            {golden.state === "absent" && (
              <EmptyHint command="cd backend && make golden-live">
                No golden records yet.
              </EmptyHint>
            )}
            {golden.state === "ready" && (
              <>
                <div className="flex items-center gap-3">
                  <StatusBadge tone={golden.data.passed ? "good" : "bad"}>
                    {golden.data.passed ? "GREEN" : "RED"}
                  </StatusBadge>
                  <span className="text-sm tabular-nums">
                    {(100 * golden.data.score).toFixed(0)}% vs threshold{" "}
                    {(100 * golden.data.threshold).toFixed(0)}%
                  </span>
                </div>
                <div className="flex flex-col gap-2">
                  {Object.entries(golden.data.by_kind).map(([kind, [p, n]]) => (
                    <div key={kind} className="flex items-center gap-3">
                      <span className="w-20 text-sm text-muted-foreground">{kind}</span>
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-secondary">
                        <div
                          className="h-full rounded-full bg-primary"
                          style={{ width: `${(100 * p) / n}%` }}
                        />
                      </div>
                      <span className="w-10 text-right font-mono text-xs tabular-nums text-muted-foreground">
                        {p}/{n}
                      </span>
                    </div>
                  ))}
                </div>
                <details className="text-sm">
                  <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                    per-case results
                  </summary>
                  <ul className="mt-2 flex flex-col gap-1">
                    {golden.data.cases.map((c) => (
                      <li key={c.id} className="flex items-start gap-2">
                        <StatusBadge tone={c.passed ? "good" : "bad"} className="mt-0.5">
                          {c.passed ? "pass" : "fail"}
                        </StatusBadge>
                        <span className="font-mono text-xs text-muted-foreground">
                          {c.id}
                        </span>
                        <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground" title={c.detail}>
                          {c.detail}
                        </span>
                      </li>
                    ))}
                  </ul>
                </details>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* traces */}
      <Card>
        <CardHeader>
          <CardTitle>Recent traces</CardTitle>
          <CardDescription>
            Every request persisted to SQLite — redacted before it was written, never after.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {traces.state === "loading" && <Skeleton className="h-48 w-full" />}
          {traces.state === "absent" && (
            <EmptyHint command="cd backend && make api">Backend unreachable.</EmptyHint>
          )}
          {traces.state === "ready" && traces.data.length === 0 && (
            <p className="py-4 text-sm text-muted-foreground">
              No traces yet — ask something in Chat and it will show up here.
            </p>
          )}
          {traces.state === "ready" && traces.data.length > 0 && (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Time</TableHead>
                  <TableHead>Query</TableHead>
                  <TableHead>Tier</TableHead>
                  <TableHead>Grounded</TableHead>
                  <TableHead>Guard</TableHead>
                  <TableHead className="text-right">Cost</TableHead>
                  <TableHead className="text-right">Latency</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {traces.data.map((t) => (
                  <TableRow key={t.trace_id}>
                    <TableCell className="whitespace-nowrap font-mono text-xs text-muted-foreground">
                      {t.ts.replace("T", " ")}
                    </TableCell>
                    <TableCell className="max-w-[28ch] truncate" title={t.query}>
                      {t.query}
                    </TableCell>
                    <TableCell>
                      <StatusBadge tone="neutral" icon="none">
                        {t.model_tier.includes("sonnet") ? "strong" : "cheap"}
                      </StatusBadge>
                    </TableCell>
                    <TableCell>
                      <StatusBadge tone={t.grounded ? "good" : "warn"}>
                        {t.grounded ? "yes" : "no"}
                      </StatusBadge>
                    </TableCell>
                    <TableCell>
                      {t.guard_action === "allow" ? (
                        <span className="text-xs text-muted-foreground">allow</span>
                      ) : (
                        <StatusBadge tone="bad" icon="guard">
                          {t.guard_action}
                        </StatusBadge>
                      )}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs tabular-nums">
                      ${t.cost_usd.toFixed(4)}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs tabular-nums">
                      <Tooltip>
                        <TooltipTrigger render={<span />}>
                          {Math.round(t.latency_ms)} ms
                        </TooltipTrigger>
                        <TooltipContent>
                          {t.escalated ? "escalated to strong tier" : "served by first route"}
                        </TooltipContent>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
