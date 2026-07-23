"use client";

// Chat: ask the grounded agent, see the answer with its citations, and see exactly what
// the run cost and how it was routed. Dry-run answers are visually marked as fabricated —
// labelling the fabrication is a design principle here, not a footnote.

import { useEffect, useRef, useState } from "react";
import { ArrowUp, BookOpenCheck, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { StatusBadge } from "@/components/status-badge";
import { api, type QueryResponse } from "./lib/api";

const EXAMPLES = [
  "how do I filter the output of a window function",
  "what is the syntax for PIVOT",
  "match each trade to the most recent price recorded before it",
];

function AnswerText({ answer }: { answer: string }) {
  const parts = answer.split(/(\[[0-9a-f]{8,}\])/g);
  return (
    <div className="max-w-[72ch] text-[15px] leading-relaxed whitespace-pre-wrap">
      {parts.map((p, i) =>
        /^\[[0-9a-f]{8,}\]$/.test(p) ? (
          <Tooltip key={i}>
            <TooltipTrigger
              render={
                <code className="mx-0.5 inline-flex cursor-default items-center rounded border border-primary/25 bg-primary/10 px-1 font-mono text-[11px] text-primary" />
              }
            >
              {p.slice(1, 9)}
            </TooltipTrigger>
            <TooltipContent>
              Citation — chunk <span className="font-mono">{p.slice(1, -1)}</span>, verified
              against what the agent actually retrieved
            </TooltipContent>
          </Tooltip>
        ) : (
          <span key={i}>{p}</span>
        ),
      )}
    </div>
  );
}

function RunMeta({ r }: { r: QueryResponse }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 border-t pt-3">
      <StatusBadge tone={r.grounded ? "good" : "warn"}>
        {r.grounded ? "grounded" : "ungrounded"}
      </StatusBadge>
      <StatusBadge tone="neutral" icon="none">
        {r.tier}
      </StatusBadge>
      {r.escalated && (
        <StatusBadge tone="warn" icon="escalated">
          escalated
        </StatusBadge>
      )}
      {r.guard_action !== "allow" && (
        <StatusBadge tone="bad" icon="guard">
          guard: {r.guard_action}
        </StatusBadge>
      )}
      {r.citations.invalid_ids.length > 0 && (
        <StatusBadge tone="bad">
          {r.citations.invalid_ids.length} invented source
          {r.citations.invalid_ids.length > 1 ? "s" : ""}
        </StatusBadge>
      )}
      {r.fabricated && (
        <Tooltip>
          <TooltipTrigger render={<span />}>
            <StatusBadge tone="fabricated">dry run — numbers fabricated</StatusBadge>
          </TooltipTrigger>
          <TooltipContent>
            The fake provider answered. Cost and tokens are synthetic; start the backend
            with SIAP_ALLOW_LIVE=1 and toggle live for real Bedrock.
          </TooltipContent>
        </Tooltip>
      )}
      <span className="ml-auto font-mono text-xs text-muted-foreground tabular-nums">
        ${r.cost.total_usd.toFixed(4)} · {Math.round(r.latency_ms)} ms · {r.iterations} iter
        {r.tools.length > 0 && ` · ${r.tools.join(", ")}`}
      </span>
    </div>
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
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (turns.length) endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  async function ask(q: string) {
    if (!q.trim() || busy) return;
    setBusy(true);
    setQuestion("");
    setTurns((t) => [...t, { question: q }]);
    try {
      const response = await api.query(q, live);
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, response } : turn)));
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, error: msg } : turn)));
      toast.error("Query failed", { description: msg });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6">
      {turns.length === 0 ? (
        <div className="flex flex-col items-center gap-6 py-16 text-center">
          <div className="flex h-11 w-11 items-center justify-center rounded-lg border bg-card">
            <BookOpenCheck className="h-5 w-5 text-primary" />
          </div>
          <div className="space-y-1.5">
            <h1 className="text-xl font-semibold tracking-tight">
              Ask the DuckDB support agent
            </h1>
            <p className="mx-auto max-w-[52ch] text-sm text-muted-foreground">
              Answers are grounded in the DuckDB docs with inline citations, verified
              against what was actually retrieved, and routed to the cheapest model tier
              the flywheel last promoted.
            </p>
          </div>
          <div className="flex flex-col gap-2">
            {EXAMPLES.map((e) => (
              <Button
                key={e}
                variant="outline"
                size="sm"
                className="justify-start font-normal text-muted-foreground"
                disabled={busy}
                onClick={() => ask(e)}
              >
                {e}
              </Button>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-5" aria-live="polite">
          {turns.map((t, i) => (
            <div key={i} className="flex flex-col gap-2.5">
              <div className="max-w-[85%] self-end rounded-lg bg-secondary px-3.5 py-2 text-[15px]">
                {t.question}
              </div>
              {t.error && (
                <div className="rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
                  {t.error}
                </div>
              )}
              {t.response && (
                <div className="rounded-lg border bg-card p-4">
                  <AnswerText answer={t.response.answer} />
                  <div className="mt-3">
                    <RunMeta r={t.response} />
                  </div>
                </div>
              )}
              {!t.response && !t.error && (
                <div className="flex items-center gap-2 px-1 text-sm text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  retrieving, answering, checking citations…
                </div>
              )}
            </div>
          ))}
          <div ref={endRef} />
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(question);
        }}
        className="sticky bottom-4 rounded-xl border bg-card p-2 shadow-sm"
      >
        <div className="flex items-center gap-2">
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask about DuckDB…"
            aria-label="Question"
            className="border-0 shadow-none focus-visible:ring-0"
          />
          <Tooltip>
            <TooltipTrigger render={<div className="flex items-center gap-1.5 pr-1" />}>
              <Switch id="live" checked={live} onCheckedChange={setLive} />
              <Label htmlFor="live" className="text-xs text-muted-foreground">
                live
              </Label>
            </TooltipTrigger>
            <TooltipContent>
              Live spends real Bedrock money and needs SIAP_ALLOW_LIVE=1 on the backend.
              Off = fake provider, zero spend.
            </TooltipContent>
          </Tooltip>
          <Button type="submit" size="icon" disabled={busy || !question.trim()} aria-label="Send">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowUp className="h-4 w-4" />}
          </Button>
        </div>
      </form>
    </div>
  );
}
