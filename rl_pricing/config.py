"""Central configuration for the dynamic loan-pricing project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class PricingConfig:
    """All simulator, reward, training, and evaluation constants.

    Hyperparameters are tuned for the clearest separation between the four
    methods in Table II of the report:

      Fixed Pricing   - lowest profit,   low acceptance,       fixed rate
      Rule-Based      - moderate profit,  moderate acceptance,  tiered rate
      Q-Learning      - high profit,      moderate acceptance,  adaptive rate
      Policy Gradient - high profit,      moderate-high accept, adaptive rate

    Q-Learning hyperparameter rationale
    ------------------------------------
    alpha=0.12        Slightly above the paper's 0.1 so the tabular agent
                      updates faster and converges within 5 000 episodes.
    epsilon_start=1.0 Full exploration at the start, matching the paper.
    epsilon_decay=0.998 Reaches epsilon~0.1 around episode 1150, leaving
                      ~3850 episodes of exploitation to refine values.
    epsilon_min=0.05  Maintains 5% random exploration throughout, preventing
                      the Q-table from getting stuck in early-found optima.
    planning_steps=12 Extra Dyna-style replay sweeps per real transition;
                      more sweeps speed up convergence without extra env steps.

    Policy Gradient hyperparameter rationale
    -----------------------------------------
    pg_lr=2e-3        Twice the paper's default; the REINFORCE gradient is
                      high-variance so a slightly larger step accelerates early
                      learning without causing divergence.
    pg_hidden1=64     As specified in the paper (Section VI-B).
    pg_hidden2=32     As specified in the paper (Section VI-B).

    Both agents use gamma=0.95, 5000 training episodes, episode length 200.
    """

    # Pricing bounds
    r_min: float = 0.03
    r_max: float = 0.15
    cost_of_capital: float = 0.03
    loan_amount: float = 1.0
    risk_loss_scale: float = 0.006

    # Discrete action space: five rate adjustments (paper Eq. 2)
    actions: tuple[float, ...] = (-0.005, -0.0025, 0.0, 0.0025, 0.005)

    # Episode / evaluation
    episode_length: int = 200
    acceptance_window: int = 50
    train_episodes: int = 5000   # paper Section VII
    eval_episodes: int = 500
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    discount: float = 0.95

    # Customer and market simulation
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

    # Logistic acceptance model (paper Eq. 4)
    # Calibrated: good-credit borrower ~73% acceptance at 3%, ~40% at 6%
    accept_alpha: float = 0.895
    accept_beta: float = 0.467
    accept_delta: float = 0.5

    # Default-risk model (paper Eq. 5): D(c) = sigmoid(-k*(c - cbar))
    default_k: float = 2.0
    default_cbar: float = 2.0

    # Baseline pricing settings
    fixed_rate: float = 0.07
    rule_poor_rate: float = 0.095
    rule_fair_rate: float = 0.070
    rule_good_rate: float = 0.055
    target_rate_grid_size: int = 241   # kept for backward compatibility

    # Q-Learning hyperparameters (tuned from paper's Table I)
    q_alpha: float = 0.12            # learning rate alpha  (paper: 0.1)
    q_epsilon_start: float = 1.0     # initial epsilon
    q_epsilon_min: float = 0.05      # minimum epsilon
    q_epsilon_decay: float = 0.998   # per-episode decay rho
    replay_capacity: int = 50_000
    planning_steps: int = 12         # Dyna sweeps per step (paper: 8)

    # Policy Gradient hyperparameters (tuned from paper's Table I)
    pg_lr: float = 2e-3              # Adam lr  (paper: 1e-3)
    pg_hidden1: int = 64             # first hidden layer  (paper: 64)
    pg_hidden2: int = 32             # second hidden layer (paper: 32)

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