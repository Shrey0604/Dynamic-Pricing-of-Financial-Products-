"""Visualisation helpers for the four-method experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_pricing.evaluation import MetricSummary


# Colours for the four paper methods
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
    """Three-panel plot: training curves | gross profit bars | rate-acceptance scatter."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return plot_svg_fallback(results, Path(output_path).with_suffix(".svg"))

    summaries = [MetricSummary(**item) for item in results["summaries"]]
    train_runs = results.get("training", [])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # ── Panel 1: Training curves (one per RL method) ─────────────────────────
    ax = axes[0]
    # Group runs by method
    runs_by_method: dict[str, list[np.ndarray]] = {}
    for run in train_runs:
        method_key = run.get("method", "q_learning")
        # Map internal keys to display names
        display = "Q-Learning" if "q" in method_key else "Policy Gradient"
        runs_by_method.setdefault(display, []).append(
            np.array(run["train_rewards"], dtype=np.float64)
        )

    for display, curves in runs_by_method.items():
        min_len = min(len(c) for c in curves)
        stacked = np.stack([c[:min_len] for c in curves])
        window = max(5, min(100, min_len // 10))
        mean = moving_average(stacked.mean(axis=0), window)
        std = moving_average(stacked.std(axis=0), window)
        x = np.arange(len(mean))
        color = PALETTE.get(display, "#111827")
        ax.plot(x, mean, color=color, lw=2, label=display)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)

    # Baseline dashed lines (Fixed and Rule-Based only)
    for summary in summaries[:2]:
        ax.axhline(
            summary.profit_mean,
            color=PALETTE.get(summary.method, "#111827"),
            ls="--", lw=1.2,
            label=summary.method,
        )

    ax.set_title("Training Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Risk-adjusted profit / step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # ── Panel 2: Gross profit bar chart ──────────────────────────────────────
    ax = axes[1]
    names = [s.method.replace(" ", "\n") for s in summaries]
    gross = [s.gross_profit_mean for s in summaries]
    colors = [PALETTE.get(s.method, "#111827") for s in summaries]
    ax.bar(names, gross, color=colors, edgecolor="#111827", linewidth=0.6)
    ax.set_title("Gross Profit")
    ax.set_ylabel("Accepted margin / step")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)

    # ── Panel 3: Rate vs acceptance scatter ───────────────────────────────────
    ax = axes[2]
    for summary in summaries:
        ax.scatter(
            summary.rate_mean,
            summary.accept_mean,
            s=130,
            color=PALETTE.get(summary.method, "#111827"),
            edgecolor="#111827", linewidth=0.7,
            label=summary.method,
        )
    ax.set_title("Rate-Acceptance Trade-off")
    ax.set_xlabel("Average offered rate (%)")
    ax.set_ylabel("Acceptance rate")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_svg_fallback(results: dict, output_path: str | Path) -> Path:
    """Dependency-free SVG visualisation when Matplotlib is unavailable."""

    summaries = [MetricSummary(**item) for item in results["summaries"]]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    bar_w, gap = 52, 38
    max_gross = max(s.gross_profit_mean for s in summaries) or 1.0
    min_rate = min(s.rate_mean for s in summaries) - 0.5
    max_rate = max(s.rate_mean for s in summaries) + 0.5
    min_acc = max(0.0, min(s.accept_mean for s in summaries) - 0.05)
    max_acc = min(1.0, max(s.accept_mean for s in summaries) + 0.05)

    def color(name: str) -> str:
        return PALETTE.get(name, "#111827")

    def sx(rate: float) -> float:
        return 560 + (rate - min_rate) / (max_rate - min_rate) * 360

    def sy(acc: float) -> float:
        return 345 - (acc - min_acc) / (max_acc - min_acc) * 250

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="430" viewBox="0 0 980 430">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="32" font-family="Arial" font-size="20" font-weight="700">RL Dynamic Pricing Results</text>',
        '<text x="40" y="70" font-family="Arial" font-size="15" font-weight="700">Gross profit per step</text>',
        '<line x1="40" y1="350" x2="420" y2="350" stroke="#111827"/>',
        '<line x1="40" y1="95" x2="40" y2="350" stroke="#111827"/>',
        '<text x="560" y="70" font-family="Arial" font-size="15" font-weight="700">Rate-acceptance trade-off</text>',
        '<line x1="560" y1="350" x2="920" y2="350" stroke="#111827"/>',
        '<line x1="560" y1="95" x2="560" y2="350" stroke="#111827"/>',
    ]

    x = 58
    for summary in summaries:
        h = summary.gross_profit_mean / max_gross * 235
        y = 350 - h
        parts.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" '
            f'fill="{color(summary.method)}" stroke="#111827" stroke-width="0.7"/>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2}" y="{y - 6:.1f}" text-anchor="middle" '
            f'font-family="Arial" font-size="11">{summary.gross_profit_mean:.4f}</text>'
        )
        label = summary.method
        parts.append(
            f'<text x="{x + bar_w / 2}" y="372" text-anchor="middle" '
            f'font-family="Arial" font-size="10">{label}</text>'
        )
        x += bar_w + gap

    for summary in summaries:
        px, py = sx(summary.rate_mean), sy(summary.accept_mean)
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" '
            f'fill="{color(summary.method)}" stroke="#111827" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{px + 11:.1f}" y="{py + 4:.1f}" '
            f'font-family="Arial" font-size="11">{summary.method}</text>'
        )

    parts.extend([
        '<text x="735" y="392" text-anchor="middle" font-family="Arial" font-size="12">Average offered rate (%)</text>',
        '<text x="515" y="230" transform="rotate(-90 515 230)" text-anchor="middle" font-family="Arial" font-size="12">Acceptance rate</text>',
        "</svg>",
    ])
    output.write_text("\n".join(parts), encoding="utf-8")
    return output