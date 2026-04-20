"""Baseline and model-based pricing policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_pricing.config import PricingConfig
from rl_pricing.mdp import (
    Observation,
    apply_action,
    closest_action,
    expected_reward,
    target_rate_for_acceptance,
)


class RateTrackingPolicy:
    """Small base class for policies that only observe the public state."""

    name = "rate_tracking_policy"

    def __init__(self, config: PricingConfig | None = None):
        self.config = config or PricingConfig()
        self.current_rate = self.config.market_star

    def start_episode(self, obs: Observation) -> None:
        self.current_rate = float(np.clip(obs[2], self.config.r_min, self.config.r_max))

    def act(self, obs: Observation, greedy: bool = True) -> int:
        action = self._select_action(obs)
        self.current_rate = apply_action(self.current_rate, action, self.config)
        return action

    def _select_action(self, obs: Observation) -> int:
        raise NotImplementedError

    def observe(
        self,
        obs: Observation,
        action: int,
        reward: float,
        next_obs: Observation,
        done: bool,
    ) -> None:
        """Hook for a shared evaluation loop. Stateless policies ignore it."""


class FixedRatePolicy(RateTrackingPolicy):
    """Move the offer toward a fixed rate."""

    name = "fixed"

    def __init__(self, rate: float | None = None, config: PricingConfig | None = None):
        super().__init__(config)
        self.rate = self.config.fixed_rate if rate is None else rate

    def _select_action(self, obs: Observation) -> int:
        return closest_action(self.current_rate, self.rate, self.config)


class RuleBasedPolicy(RateTrackingPolicy):
    """Credit-tier pricing heuristic from the report."""

    name = "rule_based"

    def __init__(
        self,
        tier_rates: dict[int, float] | None = None,
        config: PricingConfig | None = None,
    ):
        super().__init__(config)
        self.tier_rates = tier_rates or self.config.rule_based_rates

    def _select_action(self, obs: Observation) -> int:
        _demand, credit, _market_rate, _accept_rate = obs
        return closest_action(self.current_rate, self.tier_rates[int(credit)], self.config)


class ProfitGreedyPolicy(RateTrackingPolicy):
    """One-step model-based policy that maximizes expected net reward."""

    name = "profit_greedy"

    def _target_rate(self, obs: Observation) -> float:
        demand, credit, _market_rate, _accept_rate = obs
        grid = np.linspace(
            self.config.r_min, self.config.r_max, self.config.target_rate_grid_size
        )
        scores = [expected_reward(r, credit, demand, self.config).expected_reward for r in grid]
        return float(grid[int(np.argmax(scores))])

    def _select_action(self, obs: Observation) -> int:
        return closest_action(self.current_rate, self._target_rate(obs), self.config)


@dataclass(frozen=True)
class BalancedDecision:
    target_rate: float
    expected_reward: float
    expected_gross_profit: float
    accept_probability: float


class BalancedTargetRatePolicy(RateTrackingPolicy):
    """Model-based policy with an explicit acceptance floor.

    It picks the highest-profit rate among candidates that meet the configured
    target acceptance probability. If the target is infeasible for a state, it
    falls back to the rate with the highest acceptance probability.
    """

    name = "balanced_planner"

    def __init__(
        self,
        target_acceptance: float | None = None,
        config: PricingConfig | None = None,
    ):
        super().__init__(config)
        self.target_acceptance = (
            self.config.target_acceptance
            if target_acceptance is None
            else target_acceptance
        )
        self.last_decision: BalancedDecision | None = None

    def _decision(self, obs: Observation) -> BalancedDecision:
        demand, credit, _market_rate, _accept_rate = obs
        analytic_cap = target_rate_for_acceptance(
            self.target_acceptance, credit, demand, self.config
        )
        grid = np.linspace(
            self.config.r_min,
            min(self.config.r_max, max(self.config.r_min, analytic_cap)),
            max(3, self.config.target_rate_grid_size // 3),
        )

        candidates: list[tuple[float, float, float, float]] = []
        for rate in grid:
            e = expected_reward(rate, credit, demand, self.config)
            if e.accept_probability >= self.target_acceptance - 1e-9:
                candidates.append(
                    (
                        e.expected_reward,
                        e.expected_gross_profit,
                        e.accept_probability,
                        float(rate),
                    )
                )

        if not candidates:
            fallback = expected_reward(self.config.r_min, credit, demand, self.config)
            return BalancedDecision(
                target_rate=self.config.r_min,
                expected_reward=fallback.expected_reward,
                expected_gross_profit=fallback.expected_gross_profit,
                accept_probability=fallback.accept_probability,
            )

        best = max(candidates, key=lambda item: item[0])
        return BalancedDecision(
            target_rate=best[3],
            expected_reward=best[0],
            expected_gross_profit=best[1],
            accept_probability=best[2],
        )

    def _select_action(self, obs: Observation) -> int:
        self.last_decision = self._decision(obs)
        return closest_action(
            self.current_rate, self.last_decision.target_rate, self.config
        )
