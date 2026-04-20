"""Evaluation metrics and experiment summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from rl_pricing.config import PricingConfig
from rl_pricing.environment import LoanPricingEnv
from rl_pricing.policies import RateTrackingPolicy


AgentFactory = Callable[[int], RateTrackingPolicy]


@dataclass
class EpisodeMetrics:
    profit_per_step: float
    gross_profit_per_step: float
    accept_rate: float
    avg_rate: float
    avg_default_prob: float
    avg_risk_cost: float


@dataclass
class MetricSummary:
    method: str
    profit_mean: float
    profit_std: float
    gross_profit_mean: float
    gross_profit_std: float
    accept_mean: float
    accept_std: float
    rate_mean: float
    rate_std: float
    default_mean: float
    risk_cost_mean: float
    episodes: int
    seeds: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "profit_mean": self.profit_mean,
            "profit_std": self.profit_std,
            "gross_profit_mean": self.gross_profit_mean,
            "gross_profit_std": self.gross_profit_std,
            "accept_mean": self.accept_mean,
            "accept_std": self.accept_std,
            "rate_mean": self.rate_mean,
            "rate_std": self.rate_std,
            "default_mean": self.default_mean,
            "risk_cost_mean": self.risk_cost_mean,
            "episodes": self.episodes,
            "seeds": self.seeds,
        }


def run_episode(
    env: LoanPricingEnv,
    agent: RateTrackingPolicy,
    greedy: bool = True,
) -> EpisodeMetrics:
    obs = env.reset()
    agent.start_episode(obs)

    rewards: list[float] = []
    gross: list[float] = []
    accepts: list[int] = []
    rates: list[float] = []
    defaults: list[float] = []
    risks: list[float] = []

    done = False
    while not done:
        action = agent.act(obs, greedy=greedy)
        next_obs, reward, done = env.step(action)
        if not greedy:
            agent.observe(obs, action, reward, next_obs, done)

        if env.last_info is None:
            raise RuntimeError("environment did not populate last_info")
        info = env.last_info
        rewards.append(float(reward))
        gross.append(info.gross_profit)
        accepts.append(info.accepted)
        rates.append(info.offered_rate)
        defaults.append(info.default_prob)
        risks.append(info.risk_cost)
        obs = next_obs

    return EpisodeMetrics(
        profit_per_step=float(np.mean(rewards)),
        gross_profit_per_step=float(np.mean(gross)),
        accept_rate=float(np.mean(accepts)),
        avg_rate=float(np.mean(rates)),
        avg_default_prob=float(np.mean(defaults)),
        avg_risk_cost=float(np.mean(risks)),
    )


def evaluate_agent(
    method: str,
    factory: AgentFactory,
    config: PricingConfig,
    seeds: list[int],
    episodes: int | None = None,
    greedy: bool = True,
) -> tuple[MetricSummary, list[EpisodeMetrics]]:
    """Evaluate a policy over multiple seeds and episodes."""

    n_episodes = config.eval_episodes if episodes is None else episodes
    all_metrics: list[EpisodeMetrics] = []

    for seed in seeds:
        env = LoanPricingEnv(config=config, seed=10_000 + seed)
        agent = factory(seed)
        for _ in range(n_episodes):
            all_metrics.append(run_episode(env, agent, greedy=greedy))

    def arr(name: str) -> np.ndarray:
        return np.array([getattr(m, name) for m in all_metrics], dtype=np.float64)

    summary = MetricSummary(
        method=method,
        profit_mean=float(arr("profit_per_step").mean()),
        profit_std=float(arr("profit_per_step").std(ddof=0)),
        gross_profit_mean=float(arr("gross_profit_per_step").mean()),
        gross_profit_std=float(arr("gross_profit_per_step").std(ddof=0)),
        accept_mean=float(arr("accept_rate").mean()),
        accept_std=float(arr("accept_rate").std(ddof=0)),
        rate_mean=float(arr("avg_rate").mean() * 100.0),
        rate_std=float(arr("avg_rate").std(ddof=0) * 100.0),
        default_mean=float(arr("avg_default_prob").mean()),
        risk_cost_mean=float(arr("avg_risk_cost").mean()),
        episodes=len(all_metrics),
        seeds=list(seeds),
    )
    return summary, all_metrics


def format_summary_table(summaries: list[MetricSummary]) -> str:
    """Return a readable ASCII table for terminal output."""

    if not summaries:
        return "No evaluation results."

    headers = [
        "Method",
        "Profit/step",
        "Gross",
        "Accept",
        "Avg Rate",
        "Risk Cost",
    ]

    table_rows = []
    for s in summaries:
        table_rows.append(
            [
                s.method,
                f"{s.profit_mean:+.5f} +/- {s.profit_std:.4f}",
                f"{s.gross_profit_mean:.5f}",
                f"{s.accept_mean:.3f} +/- {s.accept_std:.3f}",
                f"{s.rate_mean:.2f}%",
                f"{s.risk_cost_mean:.5f}",
            ]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]

    def fmt_row(values: list[str]) -> str:
        return (
            f"{values[0]:<{widths[0]}}  "
            f"{values[1]:>{widths[1]}}  "
            f"{values[2]:>{widths[2]}}  "
            f"{values[3]:>{widths[3]}}  "
            f"{values[4]:>{widths[4]}}  "
            f"{values[5]:>{widths[5]}}"
        )

    divider = "-" * len(fmt_row(headers))
    lines = [
        "",
        "Evaluation Results",
        divider,
        fmt_row(headers),
        divider,
    ]
    lines.extend(fmt_row(row) for row in table_rows)
    lines.append(divider)
    return "\n".join(lines)
