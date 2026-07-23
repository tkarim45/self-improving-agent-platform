// Typed client for the FastAPI backend (backend/src/api). All calls go through the
// Next.js rewrite in next.config.ts, so the browser only ever talks same-origin.

export interface CitationReport {
  grounded: boolean;
  cited_ids: string[];
  invalid_ids: string[];
  citation_rate: number;
  n_claims: number;
  n_uncited: number;
}

export interface QueryResponse {
  trace_id: string;
  answer: string;
  citations: CitationReport;
  grounded: boolean;
  tier: string;
  escalated: boolean;
  routing_reason: string;
  tools: string[];
  iterations: number;
  guard_action: string;
  guard_events: { stage: string; signals?: string[] }[];
  cost: { total_usd: number; calls: number; input_tokens: number; output_tokens: number };
  latency_ms: number;
  live: boolean;
  fabricated: boolean;
}

export interface TraceRow {
  trace_id: string;
  ts: string;
  tenant: string;
  query: string;
  answer: string;
  model_tier: string;
  cost_usd: number;
  latency_ms: number;
  grounded: number;
  citation_rate: number;
  invalid_cites: number;
  escalated: number;
  guard_action: string;
}

export interface Summary {
  n: number;
  total_cost: number;
  mean_cost: number;
  mean_latency: number;
  grounded_rate: number;
  mean_citation_rate: number;
  escalations: number;
  blocks: number;
  redactions: number;
  [k: string]: unknown;
}

export interface WeeklyPoint {
  week: number;
  router_version: string;
  n_queries: number;
  grounded: number;
  escalations: number;
  cost_usd: number;
  tier_mix: Record<string, number>;
  cycle: { ran: boolean; promoted: boolean; reason?: string };
}

export interface Promotions {
  active: { router: { kind: string; artifact: string | null; version: string } };
  entries: {
    ts: string;
    component: string;
    promoted: boolean;
    reason: string;
    candidate?: string;
  }[];
}

export interface GoldenReport {
  score: number;
  passed: boolean;
  threshold: number;
  by_kind: Record<string, [number, number]>;
  cases: { id: string; kind: string; passed: boolean; detail: string }[];
}

export interface Health {
  status: string;
  cheap: string;
  strong: string;
  pricing_as_of: string;
  live_enabled: boolean;
  active: Promotions["active"];
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const api = {
  health: () => get<Health>("/api/health"),
  summary: () => get<Summary>("/api/summary"),
  traces: (limit = 20) => get<TraceRow[]>(`/api/traces?limit=${limit}`),
  trace: (id: string) => get<TraceRow & { payload: string }>(`/api/traces/${id}`),
  weekly: () => get<WeeklyPoint[]>("/api/sim/weekly"),
  promotions: () => get<Promotions>("/api/promotions"),
  golden: () => get<GoldenReport>("/api/golden"),
  query: async (question: string, live: boolean): Promise<QueryResponse> => {
    const r = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, live }),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(detail.detail ?? `query failed: ${r.status}`);
    }
    return r.json();
  },
};
