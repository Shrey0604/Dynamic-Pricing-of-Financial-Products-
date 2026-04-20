"""Central configuration for the dynamic loan-pricing project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class PricingConfig:
    """All simulator, reward, training, and evaluation constants.

    The old submission used the report's numeric alpha/beta literally, which
    made acceptance at normal retail loan rates unrealistically small. The
    defaults below preserve the same logistic model and action contract, but
    calibrate beta/alpha to the report's stated narrative: a good-credit
    borrower has about 73% acceptance at 3% and about 40% acceptance at 6%.
    """

    # Pricing bounds and capital economics.
    r_min: float = 0.03
    r_max: float = 0.15
    cost_of_capital: float = 0.03
    loan_amount: float = 1.0
    risk_loss_scale: float = 0.006

    # Discrete action space from the interface spec.
    actions: tuple[float, ...] = (-0.005, -0.0025, 0.0, 0.0025, 0.005)

    # Episode and evaluation settings.
    episode_length: int = 200
    acceptance_window: int = 50
    train_episodes: int = 4000
    eval_episodes: int = 500
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    discount: float = 0.95

    # Customer and market simulation.
    credit_categories: tuple[int, ...] = (1, 2, 3)
    credit_probs: tuple[float, ...] = (0.25, 0.50, 0.25)
    demand_levels: tuple[int, ...] = (0, 1, 2)
    demand_probs: tuple[float, ...] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    demand_logit_shift: tuple[float, ...] = (-0.25, 0.0, 0.25)
    market_eta: float = 0.05
    market_star: float = 0.05
    market_sigma: float = 0.002
    market_min: float = 0.03
    market_max: float = 0.10
    initial_market_low: float = 0.04
    initial_market_high: float = 0.07

    # Logistic acceptance model: sigma(alpha - beta * rate_pct + delta * credit).
    accept_alpha: float = 0.895
    accept_beta: float = 0.467
    accept_delta: float = 0.5

    # Default-risk model D(c) = 1 / (1 + exp(k * (c - cbar))).
    default_k: float = 2.0
    default_cbar: float = 2.0

    # Baseline and balanced-pricing settings.
    fixed_rate: float = 0.07
    rule_poor_rate: float = 0.095
    rule_fair_rate: float = 0.070
    rule_good_rate: float = 0.055
    target_acceptance: float = 0.43
    target_rate_grid_size: int = 241

    # Q-learning / Dyna-style replay settings.
    q_alpha: float = 0.08
    q_epsilon_start: float = 0.75
    q_epsilon_min: float = 0.03
    q_epsilon_decay: float = 0.996
    replay_capacity: int = 50_000
    planning_steps: int = 8

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def rule_based_rates(self) -> dict[int, float]:
        return {
            1: self.rule_poor_rate,
            2: self.rule_fair_rate,
            3: self.rule_good_rate,
        }
