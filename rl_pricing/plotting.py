"""Visualisation helpers for the four-method experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_pricing.evaluation import MetricSummary


PALETTE = {
    "Fixed Pricing":   "#6B7280",
    "Rule-Based":      "#2F855A",
    "Q-Learning":      "#2B6CB0",
    "Policy Gradient": "#7C3AED",
}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    weights = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, weights, mode="valid")


def plot_experiment(results: dict, output_path: str | Path) -> Path:
    """Three-panel plot: training curves | profit bars | rate-acceptance scatter."""

    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ModuleNotFoundError:
        return plot_svg_fallback(results, Path(output_path).with_suffix(".svg"))

    summaries = [MetricSummary(**item) for item in results["summaries"]]
    train_runs = results.get("training", [])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#fafafa")
    for ax in axes:
        ax.set_facecolor("#fafafa")

    # ── Panel 1: Training curves ──────────────────────────────────────────────
    ax = axes[0]
    runs_by_method: dict[str, list[np.ndarray]] = {}
    for run in train_runs:
        method_key = run.get("method", "q_learning")
        display = "Q-Learning" if "q" in method_key else "Policy Gradient"
        runs_by_method.setdefault(display, []).append(
            np.array(run["train_rewards"], dtype=np.float64)
        )

    for display, curves in runs_by_method.items():
        min_len = min(len(c) for c in curves)
        stacked = np.stack([c[:min_len] for c in curves])
        window = max(5, min(80, min_len // 12))
        mean = moving_average(stacked.mean(axis=0), window)
        std  = moving_average(stacked.std(axis=0),  window)
        x = np.arange(len(mean))
        color = PALETTE.get(display, "#111827")
        ax.plot(x, mean, color=color, lw=2, label=display)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    for summary in summaries[:2]:   # Fixed & Rule-Based dashed baselines
        ax.axhline(
            summary.profit_mean,
            color=PALETTE.get(summary.method, "#111827"),
            ls="--", lw=1.2, alpha=0.85,
            label=summary.method,
        )

    ax.set_title("Training Curve", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Risk-adjusted profit / step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # ── Panel 2: Profit bar chart with non-zero baseline ─────────────────────
    # Using a non-zero y-axis floor makes the differences between methods visible.
    # The floor is set to 90% of the minimum value so bars are clearly comparable.
    ax = axes[1]
    names  = [s.method.replace(" ", "\n") for s in summaries]
    profits = [s.profit_mean for s in summaries]
    errs    = [s.profit_std  for s in summaries]
    colors  = [PALETTE.get(s.method, "#111827") for s in summaries]

    bars = ax.bar(names, profits, color=colors,
                  edgecolor="#111827", linewidth=0.6,
                  yerr=errs, capsize=4, error_kw={"linewidth": 1.0})

    # Add value labels on top of each bar
    for bar, val in zip(bars, profits):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(errs) * 1.1,
            f"{val:.5f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
        )

    # Non-zero floor: show the interesting part of the y-axis
    y_floor = min(profits) * 0.90
    y_ceil  = max(profits) + max(errs) * 3.5
    ax.set_ylim(y_floor, y_ceil)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.5f"))

    ax.set_title("Risk-Adjusted Profit / Step", fontweight="bold")
    ax.set_ylabel("Profit / step")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)

    # ── Panel 3: Rate-acceptance scatter ─────────────────────────────────────
    ax = axes[2]
    for summary in summaries:
        ax.scatter(
            summary.rate_mean,
            summary.accept_mean,
            s=150,
            color=PALETTE.get(summary.method, "#111827"),
            edgecolor="#111827", linewidth=0.8,
            label=summary.method,
            zorder=3,
        )
        # Annotate each point with method name
        ax.annotate(
            summary.method,
            (summary.rate_mean, summary.accept_mean),
            textcoords="offset points", xytext=(8, 4),
            fontsize=7.5,
        )

    ax.set_title("Rate–Acceptance Trade-off", fontweight="bold")
    ax.set_xlabel("Average offered rate (%)")
    ax.set_ylabel("Acceptance rate")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "RL Dynamic Pricing — Four-Method Comparison",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_svg_fallback(results: dict, output_path: str | Path) -> Path:
    """Dependency-free SVG fallback when Matplotlib is unavailable."""

    summaries = [MetricSummary(**item) for item in results["summaries"]]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    bar_w, gap = 52, 38
    profits = [s.profit_mean for s in summaries]
    y_floor = min(profits) * 0.90
    y_range = max(profits) - y_floor or 1e-6
    max_rate = max(s.rate_mean for s in summaries) + 0.5
    min_rate = min(s.rate_mean for s in summaries) - 0.5
    max_acc  = min(1.0, max(s.accept_mean for s in summaries) + 0.05)
    min_acc  = max(0.0, min(s.accept_mean for s in summaries) - 0.05)

    def color(name: str) -> str:
        return PALETTE.get(name, "#111827")

    def sx(rate: float) -> float:
        return 560 + (rate - min_rate) / (max_rate - min_rate) * 360

    def sy(acc: float) -> float:
        return 345 - (acc - min_acc) / (max_acc - min_acc) * 250

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="430" viewBox="0 0 980 430">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<text x="40" y="32" font-family="Arial" font-size="18" font-weight="700">RL Dynamic Pricing Results</text>',
        '<text x="40" y="65" font-family="Arial" font-size="14" font-weight="700">Risk-Adjusted Profit / Step</text>',
        '<line x1="40" y1="350" x2="430" y2="350" stroke="#111827"/>',
        '<line x1="40" y1="90"  x2="40"  y2="350" stroke="#111827"/>',
        '<text x="560" y="65" font-family="Arial" font-size="14" font-weight="700">Rate–Acceptance Trade-off</text>',
        '<line x1="560" y1="350" x2="920" y2="350" stroke="#111827"/>',
        '<line x1="560" y1="90"  x2="560" y2="350" stroke="#111827"/>',
    ]

    x = 58
    chart_h = 240
    for summary in summaries:
        h = (summary.profit_mean - y_floor) / y_range * chart_h
        h = max(h, 2)
        y = 350 - h
        parts.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" '
            f'fill="{color(summary.method)}" stroke="#111827" stroke-width="0.7"/>'
        )
        parts.append(
            f'<text x="{x + bar_w/2}" y="{y - 5:.1f}" text-anchor="middle" '
            f'font-family="Arial" font-size="10">{summary.profit_mean:.5f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w/2}" y="368" text-anchor="middle" '
            f'font-family="Arial" font-size="9">{summary.method}</text>'
        )
        x += bar_w + gap

    for summary in summaries:
        px, py = sx(summary.rate_mean), sy(summary.accept_mean)
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="9" '
            f'fill="{color(summary.method)}" stroke="#111827" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{px + 13:.1f}" y="{py + 4:.1f}" '
            f'font-family="Arial" font-size="10">{summary.method}</text>'
        )

    parts += [
        '<text x="735" y="392" text-anchor="middle" font-family="Arial" font-size="11">Avg offered rate (%)</text>',
        '<text x="515" y="230" transform="rotate(-90 515 230)" text-anchor="middle" font-family="Arial" font-size="11">Acceptance rate</text>',
        "</svg>",
    ]
    output.write_text("\n".join(parts), encoding="utf-8")
    return output