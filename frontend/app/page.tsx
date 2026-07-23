"use client";

// The chat surface. Asks the grounded agent a question, renders the answer with its
// inline [citation] ids highlighted, and shows exactly what the run cost and how it was
// routed — the same honesty the CLIs have. Dry mode by default; live mode only works if
// the backend was started with SIAP_ALLOW_LIVE=1.

import { useState } from "react";
import { api, type QueryResponse } from "./lib/api";

const EXAMPLES = [
  "how do I filter the output of a window function",
  "what is the syntax for PIVOT",
  "match each trade to the most recent price recorded before it",
];

function AnswerText({ answer }: { answer: string }) {
  // Highlight [hexid] citations without pretending to be a markdown renderer.
  const parts = answer.split(/(\[[0-9a-f]{8,}\])/g);
  return (
    <p className="whitespace-pre-wrap leading-relaxed">
      {parts.map((p, i) =>
        /^\[[0-9a-f]{8,}\]$/.test(p) ? (
          <code
            key={i}
            className="mx-0.5 rounded bg-sky-100 px-1 py-0.5 text-xs text-sky-800 dark:bg-sky-950 dark:text-sky-300"
            title="citation — a chunk id the agent actually retrieved"
          >
            {p}
          </code>
        ) : (
          <span key={i}>{p}</span>
        ),
      )}
    </p>
  );
}

function Badge({
  children,
  tone,
}: {
  children: React.ReactNode;
  tone: "good" | "warn" | "bad" | "muted";
}) {
  const tones = {
    good: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
    warn: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
    bad: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
    muted: "bg-neutral-100 text-neutral-600 dark:bg-neutral-800 dark:text-neutral-300",
  } as const;
  return (
    <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}

interface Turn {
  question: string;
  response?: QueryResponse;
  error?: string;
}

export default function Chat() {
  const [question, setQuestion] = useState("");
  const [live, setLive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);

  async function ask(q: string) {
    if (!q.trim() || busy) return;
    setBusy(true);
    setQuestion("");
    setTurns((t) => [...t, { question: q }]);
    try {
      const response = await api.query(q, live);
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, response } : turn)));
    } catch (e) {
      setTurns((t) =>
        t.map((turn, i) => (i === t.length - 1 ? { ...turn, error: String(e) } : turn)),
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <section>
        <h1 className="text-xl font-semibold">Ask the DuckDB support agent</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Grounded answers over the DuckDB docs with inline citations, routed to the cheapest
          model tier the flywheel last promoted. Dry mode uses a fake provider (no spend,
          fabricated numbers — and labelled as such).
        </p>
      </section>

      <div className="flex flex-wrap gap-2 text-sm">
        {EXAMPLES.map((e) => (
          <button
            key={e}
            onClick={() => ask(e)}
            disabled={busy}
            className="rounded-full border border-neutral-300 px-3 py-1 text-neutral-600 hover:bg-neutral-100 disabled:opacity-50 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-900"
          >
            {e}
          </button>
        ))}
      </div>

      <div className="flex flex-col gap-4">
        {turns.map((t, i) => (
          <div key={i} className="flex flex-col gap-2">
            <div className="self-end rounded-2xl bg-neutral-100 px-4 py-2 dark:bg-neutral-900">
              {t.question}
            </div>
            {t.error && (
              <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
                {t.error}
              </div>
            )}
            {t.response && (
              <div className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
                <AnswerText answer={t.response.answer} />
                <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-neutral-100 pt-3 dark:border-neutral-800">
                  <Badge tone={t.response.grounded ? "good" : "warn"}>
                    {t.response.grounded ? "grounded" : "ungrounded"}
                  </Badge>
                  <Badge tone="muted">tier: {t.response.tier}</Badge>
                  {t.response.escalated && <Badge tone="warn">escalated</Badge>}
                  {t.response.guard_action !== "allow" && (
                    <Badge tone="bad">guard: {t.response.guard_action}</Badge>
                  )}
                  {t.response.citations.invalid_ids.length > 0 && (
                    <Badge tone="bad">
                      {t.response.citations.invalid_ids.length} invented source(s)
                    </Badge>
                  )}
                  <Badge tone="muted">
                    ${t.response.cost.total_usd.toFixed(4)}
                    {t.response.fabricated ? " (fabricated — dry run)" : ""}
                  </Badge>
                  <Badge tone="muted">{Math.round(t.response.latency_ms)} ms</Badge>
                  <span className="text-xs text-neutral-400">
                    {t.response.tools.join(", ") || "no tools"} · {t.response.iterations}{" "}
                    iterations
                  </span>
                </div>
              </div>
            )}
            {!t.response && !t.error && (
              <div className="animate-pulse text-sm text-neutral-400">thinking…</div>
            )}
          </div>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(question);
        }}
        className="sticky bottom-4 flex gap-2"
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about DuckDB…"
          className="flex-1 rounded-lg border border-neutral-300 bg-white px-4 py-2 outline-none focus:border-sky-500 dark:border-neutral-700 dark:bg-neutral-950"
        />
        <label className="flex items-center gap-1.5 text-sm text-neutral-500">
          <input
            type="checkbox"
            checked={live}
            onChange={(e) => setLive(e.target.checked)}
          />
          live (spends)
        </label>
        <button
          type="submit"
          disabled={busy || !question.trim()}
          className="rounded-lg bg-sky-600 px-4 py-2 text-white hover:bg-sky-700 disabled:opacity-50"
        >
          Ask
        </button>
      </form>
    </div>
  );
}
