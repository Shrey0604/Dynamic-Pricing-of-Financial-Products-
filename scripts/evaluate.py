"""Evaluate saved or built-in pricing policies without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_pricing.agents import ReplayQLearningAgent
from rl_pricing.config import PricingConfig
from rl_pricing.evaluation import evaluate_agent, format_summary_table
from rl_pricing.policies import (
    BalancedTargetRatePolicy,
    FixedRatePolicy,
    ProfitGreedyPolicy,
    RuleBasedPolicy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pricing policies")
    parser.add_argument("--episodes", type=int, default=PricingConfig.eval_episodes)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(PricingConfig.seeds))
    parser.add_argument("--episode-length", type=int, default=PricingConfig.episode_length)
    parser.add_argument("--model", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PricingConfig(
        episode_length=args.episode_length,
        eval_episodes=args.episodes,
        seeds=tuple(args.seeds),
    )

    specs = [
        ("Fixed Pricing", lambda seed: FixedRatePolicy(config=config)),
        ("Rule-Based", lambda seed: RuleBasedPolicy(config=config)),
        ("Profit-Greedy Planner", lambda seed: ProfitGreedyPolicy(config=config)),
        ("Balanced Planner", lambda seed: BalancedTargetRatePolicy(config=config)),
    ]
    if args.model is not None:
        specs.append(
            (
                "Replay Q-Learning",
                lambda seed: ReplayQLearningAgent.load(args.model, seed=seed),
            )
        )

    summaries = []
    for method, factory in specs:
        summary, _ = evaluate_agent(
            method=method,
            factory=factory,
            config=config,
            seeds=list(args.seeds),
            episodes=args.episodes,
            greedy=True,
        )
        summaries.append(summary)

    print(format_summary_table(summaries))


if __name__ == "__main__":
    main()
