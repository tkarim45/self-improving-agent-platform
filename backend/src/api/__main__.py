"""Run the API server.

    python -m src.api                  # dry-only: /api/query uses the fake provider
    SIAP_ALLOW_LIVE=1 python -m src.api  # allow live Bedrock queries (spends money)

Serves on http://127.0.0.1:8000 — the Next.js frontend (frontend/) proxies to it.
"""

from __future__ import annotations

import argparse


def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser(prog="python -m src.api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "src.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
