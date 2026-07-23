// One badge vocabulary for the whole app. Status is never color-alone: every tone
// pairs with an icon (or explicit text) so it survives color blindness and grayscale.

import {
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  FlaskConical,
  ShieldAlert,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

type Tone = "good" | "warn" | "bad" | "neutral" | "fabricated";

const TONES: Record<Tone, string> = {
  good: "border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  warn: "border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400",
  bad: "border-red-600/25 bg-red-500/10 text-red-700 dark:text-red-400",
  neutral: "border-border bg-secondary/60 text-muted-foreground",
  fabricated: "border-dashed border-muted-foreground/40 text-muted-foreground",
};

const ICONS: Record<Tone, React.ComponentType<{ className?: string }> | null> = {
  good: CheckCircle2,
  warn: AlertTriangle,
  bad: XCircle,
  neutral: null,
  fabricated: FlaskConical,
};

export function StatusBadge({
  tone,
  icon,
  children,
  className,
}: {
  tone: Tone;
  icon?: "escalated" | "guard" | "none";
  children: React.ReactNode;
  className?: string;
}) {
  let Icon = ICONS[tone];
  if (icon === "escalated") Icon = ArrowUpRight;
  if (icon === "guard") Icon = ShieldAlert;
  if (icon === "none") Icon = null;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium whitespace-nowrap",
        TONES[tone],
        className,
      )}
    >
      {Icon && <Icon className="h-3 w-3 shrink-0" />}
      {children}
    </span>
  );
}
