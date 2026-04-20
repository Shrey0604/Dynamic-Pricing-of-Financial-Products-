"""Contract-compliant loan-pricing simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_pricing.config import PricingConfig
from rl_pricing.mdp import (
    Observation,
    acceptance_probability,
    apply_action,
    default_probability,
    realized_reward,
)


@dataclass(frozen=True)
class StepInfo:
    """Diagnostics captured after each environment step."""

    t: int
    demand: int
    credit: int
    market_rate: float
    offered_rate: float
    accepted: int
    p_accept: float
    default_prob: float
    gross_profit: float
    risk_cost: float


class LoanPricingEnv:
    """Synthetic retail-lending environment.

    Public interface required by the reference document:

    - ``obs = env.reset()``
    - ``obs, reward, done = env.step(action)``
    - ``env.action_space_n == 5``

    The observation is ``(demand, credit, market_rate, rolling_acceptance)``.
    Diagnostics needed for evaluation are stored in ``env.last_info`` so agents
    can remain environment-agnostic.
    """

    def __init__(self, config: PricingConfig | None = None, seed: int | None = None):
        self.config = config or PricingConfig()
        self.action_space_n = self.config.n_actions
        self._rng = np.random.default_rng(seed)
        self._seed = seed

        self.t = 0
        self.market_rate = self.config.market_star
        self.current_rate = self.config.market_star
        self.demand = 1
        self.credit = 2
        self._acceptance_history: list[int] = []
        self.last_info: StepInfo | None = None

    def seed(self, seed: int | None) -> None:
        """Reset the simulator RNG."""

        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> Observation:
        """Start a new episode and return the initial observation."""

        cfg = self.config
        self.t = 0
        self.market_rate = float(
            self._rng.uniform(cfg.initial_market_low, cfg.initial_market_high)
        )
        self.current_rate = float(np.clip(self.market_rate, cfg.r_min, cfg.r_max))
        self.demand = int(self._rng.choice(cfg.demand_levels, p=cfg.demand_probs))
        self.credit = int(self._rng.choice(cfg.credit_categories, p=cfg.credit_probs))
        self._acceptance_history.clear()
        self.last_info = None
        return self._observation()

    def step(self, action: int) -> tuple[Observation, float, bool]:
        """Apply a pricing action and return ``(next_obs, reward, done)``."""

        cfg = self.config
        self.current_rate = apply_action(self.current_rate, action, cfg)

        p_accept = acceptance_probability(
            self.current_rate, self.credit, self.demand, cfg
        )
        accepted = int(self._rng.random() < p_accept)
        breakdown = realized_reward(accepted, self.current_rate, self.credit, cfg)
        self._acceptance_history.append(accepted)

        self.last_info = StepInfo(
            t=self.t,
            demand=self.demand,
            credit=self.credit,
            market_rate=self.market_rate,
            offered_rate=self.current_rate,
            accepted=accepted,
            p_accept=p_accept,
            default_prob=default_probability(self.credit, cfg),
            gross_profit=breakdown.gross_profit,
            risk_cost=breakdown.risk_cost,
        )

        self.t += 1
        done = self.t >= cfg.episode_length
        if not done:
            self._transition()
        return self._observation(), breakdown.reward, done

    def _observation(self) -> Observation:
        window = self._acceptance_history[-self.config.acceptance_window :]
        rolling_acceptance = float(np.mean(window)) if window else 0.5
        return (
            int(self.demand),
            int(self.credit),
            float(self.market_rate),
            rolling_acceptance,
        )

    def _transition(self) -> None:
        cfg = self.config
        noise = self._rng.normal(0.0, cfg.market_sigma)
        self.market_rate = float(
            np.clip(
                self.market_rate
                + cfg.market_eta * (cfg.market_star - self.market_rate)
                + noise,
                cfg.market_min,
                cfg.market_max,
            )
        )
        self.demand = int(self._rng.choice(cfg.demand_levels, p=cfg.demand_probs))
        self.credit = int(self._rng.choice(cfg.credit_categories, p=cfg.credit_probs))
