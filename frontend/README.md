# frontend — M7 product surface

Next.js chat UI + admin console for the platform. Talks to the FastAPI backend
(`../backend/src/api`) through a same-origin rewrite (see `next.config.ts`).

```bash
# 1. start the backend (from ../backend; dry-only unless SIAP_ALLOW_LIVE=1)
make api

# 2. run the frontend
npm install
npm run dev        # http://localhost:3000
```

- `/` — chat: grounded answers with inline citation ids, tier/cost/grounding badges.
  Dry mode by default (fabricated numbers, labelled). The "live (spends)" toggle only
  works if the backend was started with `SIAP_ALLOW_LIVE=1`.
- `/dashboard` — admin console: traffic summary, the M6 improvement curve (two stacked
  single-axis panels, promotion annotated), promotion history, golden-gate status,
  recent traces.

Set `BACKEND_URL` to point the rewrite somewhere other than `http://127.0.0.1:8000`.
