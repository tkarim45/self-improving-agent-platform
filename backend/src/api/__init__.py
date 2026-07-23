"""M7 — the FastAPI product surface.

Thin HTTP layer over subsystems that already exist: the grounded agent (M2), the trace
store (M3), the golden gate (M4), the promotion log (M5) and the simulation artifacts
(M6). It adds no new behaviour — every endpoint calls the same code the CLIs call, so the
API cannot drift from what the milestones measured.
"""

from src.api.app import create_app

__all__ = ["create_app"]
