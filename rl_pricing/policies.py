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
