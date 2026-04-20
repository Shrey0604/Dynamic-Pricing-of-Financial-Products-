"""Feature scaling and discretization for pricing agents."""

from __future__ import annotations

import numpy as np

from rl_pricing.config import PricingConfig
from rl_pricing.mdp import Observation


def market_bucket(market_rate: float) -> int:
    """Bucket market rates exactly as specified in the interface document."""

    if market_rate < 0.04:
        return 0
    if market_rate < 0.05:
        return 1
    if market_rate < 0.06:
        return 2
    if market_rate < 0.07:
        return 3
    return 4


def acceptance_bucket(acceptance_rate: float) -> int:
    """Bucket rolling acceptance into four bins."""

    if acceptance_rate < 0.25:
        return 0
    if acceptance_rate < 0.50:
        return 1
    if acceptance_rate < 0.75:
        return 2
    return 3


def rate_bucket(rate: float, cfg: PricingConfig, n_bins: int = 25) -> int:
    """Discretize the internally tracked offered rate."""

    clipped = float(np.clip(rate, cfg.r_min, cfg.r_max))
    scaled = (clipped - cfg.r_min) / (cfg.r_max - cfg.r_min)
    return int(np.clip(round(scaled * (n_bins - 1)), 0, n_bins - 1))


def discretize_observation(
    obs: Observation,
    tracked_rate: float,
    cfg: PricingConfig,
) -> tuple[int, int, int, int, int]:
    """Return a compact tabular state key.

    The public observation has four fields as required. Since actions adjust
    the current rate, a learning agent also tracks the rate it has produced.
    This restores the Markov information missing from the old implementation
    without changing the environment contract.
    """

    demand, credit, market_rate, rolling_acceptance = obs
    return (
        int(demand),
        int(credit) - 1,
        market_bucket(market_rate),
        acceptance_bucket(rolling_acceptance),
        rate_bucket(tracked_rate, cfg),
    )


def encode_features(
    obs: Observation,
    tracked_rate: float,
    cfg: PricingConfig,
) -> np.ndarray:
    """Continuous normalized feature vector for diagnostics or future models."""

    demand, credit, market_rate, rolling_acceptance = obs
    return np.array(
        [
            demand / 2.0,
            (credit - 1) / 2.0,
            (market_rate - cfg.market_min) / (cfg.market_max - cfg.market_min),
            rolling_acceptance,
            (tracked_rate - cfg.r_min) / (cfg.r_max - cfg.r_min),
        ],
        dtype=np.float64,
    )
