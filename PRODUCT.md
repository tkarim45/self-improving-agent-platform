# Product

## Register

product

## Users

- **The owner (Taimour)** demoing the platform: runs it locally, walks recruiters/clients
  through the chat and the flywheel evidence.
- **A support engineer persona** (the product fiction): asks DuckDB questions, needs a
  grounded answer with citations they can check, fast.
- **An operator persona**: watches the admin console — traffic, cost, promotion history,
  the eval gate — to decide whether to trust the self-improvement loop.

## Product Purpose

The M7 product surface of a self-improving RAG-agent platform. Chat answers DuckDB
questions with inline citations and honest cost/routing metadata; the dashboard proves the
closed evaluation-and-retraining loop with the same artifacts the CLIs write (M6 curve,
promotion log, golden gate, traces). Success = a viewer trusts the numbers within 30
seconds and can tell dry-run (fabricated) data from live data at a glance.

## Brand Personality

Precise, technical, honest. Calm density over flash. Every number is measured, every
caveat is labelled (fabricated dry-run costs say so; a quality dip is reported, not
smoothed). The UI should feel like a well-built engineering tool — Linear/Grafana class,
not a marketing page.

## Anti-references

- Generic AI-SaaS dashboard: gradient heroes, purple-blue washes, big-number metric cards
  with decorative sparklines.
- Chatbot toy UI: bubbles with avatars, typing dots theater, emoji.
- Anything that hides uncertainty — unlabelled mock data is the cardinal sin here.

## Design Principles

1. **Evidence first.** Artifacts (curve, promotion log, gate verdict) are the heroes;
   chrome recedes.
2. **Label the fabrication.** Dry-run data is visually distinct everywhere it appears.
3. **Honest states.** Loading skeletons, real empty states that teach ("run make sim"),
   errors with the actual detail string.
4. **One vocabulary.** Same badge/table/card language on both screens; tabular numerals
   for every metric.
5. **Density without noise.** Operators get tables and small multiples, not scroll theater.

## Accessibility & Inclusion

WCAG AA: body text ≥4.5:1 in light and dark; status never color-alone (icon or text with
every badge); focus visible on all interactive elements; prefers-reduced-motion respected.
