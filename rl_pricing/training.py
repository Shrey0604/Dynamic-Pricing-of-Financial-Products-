"""Training orchestration for learning agents."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
import time

from rl_pricing.agents import ReplayQLearningAgent
from rl_pricing.config import DEFAULT_OUTPUT_DIR, PricingConfig
from rl_pricing.environment import LoanPricingEnv
from rl_pricing.evaluation import MetricSummary, evaluate_agent
from rl_pricing.policies import (
    BalancedTargetRatePolicy,
    FixedRatePolicy,
    ProfitGreedyPolicy,
    RuleBasedPolicy,
)


@dataclass
class TrainingResult:
    seed: int
    train_rewards: list[float]
    final_epsilon: float
    model_path: str


def train_replay_q_agent(
    seed: int,
    config: PricingConfig,
    train_episodes: int | None = None,
    output_dir: str | Path | None = None,
) -> tuple[ReplayQLearningAgent, TrainingResult]:
    """Train one replay Q-learning agent."""

    n_train = config.train_episodes if train_episodes is None else train_episodes
    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = LoanPricingEnv(config=config, seed=seed)
    agent = ReplayQLearningAgent(config=config, seed=seed)
    train_rewards: list[float] = []

    for _episode in range(n_train):
        obs = env.reset()
        agent.start_episode(obs)
        done = False
        rewards = []
        while not done:
            action = agent.act(obs, greedy=False)
            next_obs, reward, done = env.step(action)
            agent.observe(obs, action, reward, next_obs, done)
            rewards.append(float(reward))
            obs = next_obs
        agent.decay_epsilon()
        train_rewards.append(sum(rewards) / len(rewards))

    model_path = out_dir / f"replay_q_seed{seed}.pkl"
    agent.save(model_path)
    return agent, TrainingResult(
        seed=seed,
        train_rewards=train_rewards,
        final_epsilon=agent.epsilon,
        model_path=str(model_path),
    )


def run_full_experiment(
    config: PricingConfig,
    seeds: list[int] | None = None,
    train_episodes: int | None = None,
    eval_episodes: int | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Train RL agents, evaluate all methods, and return serializable results."""

    selected_seeds = list(config.seeds if seeds is None else seeds)
    n_train = config.train_episodes if train_episodes is None else train_episodes
    n_eval = config.eval_episodes if eval_episodes is None else eval_episodes
    config = replace(
        config,
        train_episodes=n_train,
        eval_episodes=n_eval,
        seeds=tuple(selected_seeds),
    )
    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    trained_agents: dict[int, ReplayQLearningAgent] = {}
    training_runs: list[TrainingResult] = []
    for seed in selected_seeds:
        agent, result = train_replay_q_agent(
            seed=seed,
            config=config,
            train_episodes=None,
            output_dir=out_dir,
        )
        trained_agents[seed] = agent
        training_runs.append(result)

    summaries: list[MetricSummary] = []

    baseline_specs = [
        ("Fixed Pricing", lambda seed: FixedRatePolicy(config=config)),
        ("Rule-Based", lambda seed: RuleBasedPolicy(config=config)),
        ("Profit-Greedy Planner", lambda seed: ProfitGreedyPolicy(config=config)),
        (
            "Balanced Planner",
            lambda seed: BalancedTargetRatePolicy(config=config),
        ),
        (
            "Replay Q-Learning",
            lambda seed: trained_agents[seed],
        ),
    ]

    for method, factory in baseline_specs:
        summary, _episodes = evaluate_agent(
            method=method,
            factory=factory,
            config=config,
            seeds=selected_seeds,
            episodes=None,
            greedy=True,
        )
        summaries.append(summary)

    result = {
        "config": config.__dict__,
        "runtime_seconds": time.time() - started,
        "training": [
            {
                "seed": run.seed,
                "final_epsilon": run.final_epsilon,
                "model_path": run.model_path,
                "train_rewards": run.train_rewards,
            }
            for run in training_runs
        ],
        "summaries": [summary.as_dict() for summary in summaries],
    }

    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["results_path"] = str(results_path)
    return result
