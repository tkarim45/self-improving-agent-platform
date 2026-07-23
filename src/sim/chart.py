"""The headline curve: quality and cost per simulated week, promotions annotated.

Two stacked panels sharing the week axis — NOT a dual-axis chart. Two measures of different
scale never share y-axes; each gets its own panel, and the promotion event is a vertical
marker running through both so the eye links cause to effect.

Colors are the validated placeholder palette (categorical slots 1 and 2; CVD ΔE 24.7,
contrast ≥3:1 on the light surface — checked with the palette validator, not by eye). One
series per panel, so panel titles carry identity and no legend box is needed. Annotation
text wears text colors, never a series color.
"""

from __future__ import annotations

import json
from pathlib import Path

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
QUALITY = "#2a78d6"  # categorical slot 1
COST = "#eb6834"  # categorical slot 2


def render(weekly_path: Path, out_path: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weeks = json.loads(weekly_path.read_text())
    x = [w["week"] for w in weeks]
    quality = [w["grounded_rate"] * 100 for w in weeks]
    cost = [w["cost_per_query"] * 100 for w in weeks]  # cents/query reads better than $0.0x
    promo_weeks = [w["week"] for w in weeks if w["cycle"].get("promoted")]

    fig, (ax_q, ax_c) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True, facecolor=SURFACE,
        gridspec_kw={"hspace": 0.18},
    )

    for ax in (ax_q, ax_c):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_2, labelsize=9)

    ax_q.plot(x, quality, color=QUALITY, linewidth=2, marker="o", markersize=5)
    ax_q.set_ylabel("grounded answers (%)", color=TEXT, fontsize=10)
    ax_q.set_ylim(0, 105)

    ax_c.plot(x, cost, color=COST, linewidth=2, marker="o", markersize=5)
    ax_c.set_ylabel("cost per query (¢)", color=TEXT, fontsize=10)
    ax_c.set_xlabel("simulated week", color=TEXT, fontsize=10)
    ax_c.set_ylim(0, max(cost) * 1.25 if cost else 1)
    ax_c.set_xticks(x)

    for wk in promo_weeks:
        for ax in (ax_q, ax_c):
            ax.axvline(wk, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
        ax_q.annotate(
            "config promoted\n(learned router)",
            xy=(wk, 50), xytext=(wk + 0.15, 30),
            fontsize=8.5, color=TEXT_2,
        )

    fig.suptitle(
        "The flywheel, unattended: quality holds while cost falls at the promotion",
        color=TEXT, fontsize=12, x=0.5, y=0.98,
    )
    fig.text(
        0.5, 0.915,
        "6 simulated weeks of DuckDB support traffic on real Bedrock — "
        "no human in the retrain loop",
        ha="center", color=TEXT_2, fontsize=9,
    )

    # Selective direct labels: first, promotion-adjacent, and last points only.
    def label(ax, series, wk, fmt):
        ax.annotate(fmt, xy=(wk, series[wk]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.5, color=TEXT)

    if x:
        label(ax_c, cost, x[0], f"{cost[0]:.1f}¢")
        label(ax_c, cost, x[-1], f"{cost[-1]:.1f}¢")
        label(ax_q, quality, x[0], f"{quality[0]:.0f}%")
        label(ax_q, quality, x[-1], f"{quality[-1]:.0f}%")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return {
        "weeks": len(weeks),
        "promotions": promo_weeks,
        "quality_first_last": (quality[0], quality[-1]) if quality else None,
        "cost_first_last": (cost[0], cost[-1]) if cost else None,
        "out": str(out_path),
    }
