"""Pure MDP helpers: action transitions, probabilities, and rewards."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np

from rl_pricing.config import PricingConfig


Observation = tuple[int, int, float, float]


@dataclass(frozen=True)
class RewardBreakdown:
    """Realized reward details for one pricing decision."""

    reward: float
    gross_profit: float
    risk_cost: float
    accepted: int


@dataclass(frozen=True)
class ExpectedBreakdown:
    """Expected one-step economics for a candidate rate."""

    expected_reward: float
    expected_gross_profit: float
    expected_risk_cost: float
    accept_probability: float


def sigmoid(x: float) -> float:
    """Stable scalar sigmoid."""

    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def logit(p: float) -> float:
    """Inverse sigmoid with clipping for numerical safety."""

    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return log(p / (1.0 - p))


def apply_action(current_rate: float, action: int, cfg: PricingConfig) -> float:
    """Apply an action index to the current rate and clip to valid bounds."""

    if action < 0 or action >= cfg.n_actions:
        raise ValueError(f"action must be in [0, {cfg.n_actions - 1}], got {action}")
    return float(np.clip(current_rate + cfg.actions[action], cfg.r_min, cfg.r_max))


def closest_action(current_rate: float, target_rate: float, cfg: PricingConfig) -> int:
    """Return the action that moves the current rate closest to a target."""

    candidates = [apply_action(current_rate, a, cfg) for a in range(cfg.n_actions)]
    return int(np.argmin([abs(rate - target_rate) for rate in candidates]))


def acceptance_probability(
    rate: float,
    credit: int,
    demand: int,
    cfg: PricingConfig,
) -> float:
    """Customer acceptance probability under the calibrated logistic model."""

    demand_shift = cfg.demand_logit_shift[demand]
    log_odds = (
        cfg.accept_alpha
        - cfg.accept_beta * (100.0 * rate)
        + cfg.accept_delta * credit
        + demand_shift
    )
    return sigmoid(log_odds)


def default_probability(credit: int, cfg: PricingConfig) -> float:
    """Default probability by credit tier."""

    return sigmoid(-cfg.default_k * (credit - cfg.default_cbar))


def realized_reward(
    accepted: int,
    rate: float,
    credit: int,
    cfg: PricingConfig,
) -> RewardBreakdown:
    """Risk-adjusted profit.

    Risk cost is charged only for originated loans. Charging risk on rejected
    offers was the main reason the old implementation made every method look
    strongly negative, even when the pricing margin was sensible.
    """

    accepted = int(accepted)
    gross = accepted * (rate - cfg.cost_of_capital) * cfg.loan_amount
    risk = accepted * cfg.risk_loss_scale * default_probability(credit, cfg)
    return RewardBreakdown(
        reward=float(gross - risk),
        gross_profit=float(gross),
        risk_cost=float(risk),
        accepted=accepted,
    )


def expected_reward(
    rate: float,
    credit: int,
    demand: int,
    cfg: PricingConfig,
) -> ExpectedBreakdown:
    """Expected one-step reward and gross profit for a candidate rate."""

    p_accept = acceptance_probability(rate, credit, demand, cfg)
    margin = (rate - cfg.cost_of_capital) * cfg.loan_amount
    risk = cfg.risk_loss_scale * default_probability(credit, cfg)
    return ExpectedBreakdown(
        expected_reward=float(p_accept * (margin - risk)),
        expected_gross_profit=float(p_accept * margin),
        expected_risk_cost=float(p_accept * risk),
        accept_probability=float(p_accept),
    )


def target_rate_for_acceptance(
    target_acceptance: float,
    credit: int,
    demand: int,
    cfg: PricingConfig,
) -> float:
    """Analytic rate whose logistic acceptance probability equals target."""

    demand_shift = cfg.demand_logit_shift[demand]
    rate_pct = (
        cfg.accept_alpha
        + cfg.accept_delta * credit
        + demand_shift
        - logit(target_acceptance)
    ) / cfg.accept_beta
    return float(np.clip(rate_pct / 100.0, cfg.r_min, cfg.r_max))
