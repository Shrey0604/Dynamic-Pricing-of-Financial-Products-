# RL Dynamic Pricing of Financial Products — Group 31

Reinforcement learning agents that dynamically price retail loans by learning to balance profit maximisation against customer acceptance. The project implements and compares four pricing strategies from the accompanying report.

---

## Methods

| Method | Type | Description |
|---|---|---|
| **Fixed Pricing** | Baseline | Constant 7% rate offered to every customer |
| **Rule-Based** | Baseline | Credit-tier heuristic: 9.5% / 7.0% / 5.5% for poor / fair / good credit |
| **Q-Learning** | RL (tabular) | Off-policy Q-learning with Dyna-style replay planning |
| **Policy Gradient** | RL (neural) | REINFORCE with a two-layer softmax network (64 → 32 → 5) |

The two RL agents observe the full state vector `(demand, credit, market_rate, rolling_acceptance)` and adapt rates in real time. Baselines are static and serve as lower-bound references.

---

## Expected Results (Table II)

| Method | Profit / ep | Acceptance Rate | Avg Rate |
|---|---|---|---|
| Fixed Pricing | Lowest | Low | Fixed ~7% |
| Rule-Based | Moderate | Moderate | Tiered |
| Q-Learning | **High** | Moderate–high | Adaptive ~6.3–6.5% |
| Policy Gradient | **High** | Moderate–high | Adaptive ~5.9–6.3% |

**Key findings:**
- Both RL agents discover the ~6.3% optimal rate band where expected profit peaks, outperforming fixed 7% pricing.
- Q-Learning converges to the highest single-episode profit via off-policy bootstrapping.
- Policy Gradient achieves a **higher acceptance rate** at comparable profit because REINFORCE optimises full episode returns, naturally penalising long rejection streaks.
- Rule-Based pricing is surprisingly competitive with Fixed because credit-tier differentiation partially recovers margin from good-credit borrowers, but it remains blind to demand and market signals.

---

## Project Structure

```
.
├── configs/
│   └── default.json           # Default config values
├── outputs/                   # Training artefacts and result JSON
│   └── full/
│       ├── results.json
│       └── results_plot.png
├── rl_pricing/
│   ├── __init__.py
│   ├── agents.py              # ReplayQLearningAgent + PolicyGradientAgent
│   ├── config.py              # All hyperparameters (documented)
│   ├── environment.py         # LoanPricingEnv (reset / step / action_space_n)
│   ├── evaluation.py          # Metrics, episode runner, summary table
│   ├── mdp.py                 # Acceptance model, reward, action helpers
│   ├── plotting.py            # Three-panel result visualisation
│   ├── policies.py            # FixedRatePolicy, RuleBasedPolicy
│   ├── state.py               # Discretisation + feature normalisation
│   └── training.py            # train_q_learning_agent, train_policy_gradient_agent,
│                              #   run_full_experiment
├── scripts/
│   ├── evaluate.py            # Evaluate saved models without retraining
│   ├── smoke_test.py          # Fast contract check (< 5 s)
│   └── train.py               # Main entry point
├── tests/
│   └── test_contract.py
├── README.md
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Only `numpy` is required for training and evaluation. `matplotlib` is optional — the plotting module falls back to SVG if it is unavailable.

---

## Running the Experiment

### Quick contract check (< 5 s)

```bash
python scripts/smoke_test.py
```

### Fast test run (2 seeds, 100 episodes)

```bash
python scripts/train.py \
  --seeds 0 1 \
  --train-episodes 100 \
  --eval-episodes 20 \
  --episode-length 50 \
  --output-dir outputs/quick \
  --no-plot
```

### Full paper-scale experiment (5 seeds × 5 000 episodes)

```bash
python scripts/train.py \
  --seeds 0 1 2 3 4 \
  --train-episodes 5000 \
  --eval-episodes 500 \
  --episode-length 200 \
  --output-dir outputs/full
```

This takes 20–60 minutes depending on hardware. Results are saved to `outputs/full/results.json` and a three-panel PNG plot is generated automatically.

### Evaluate saved models without retraining

```bash
# Baselines only
python scripts/evaluate.py --episodes 200 --seeds 0 1 2

# With saved RL models
python scripts/evaluate.py \
  --q-model  outputs/full/q_learning_seed0.pkl \
  --pg-model outputs/full/policy_gradient_seed0.pkl \
  --episodes 200 --seeds 0
```

---

## Hyperparameters

All values live in `rl_pricing/config.py` and are documented inline. The most important knobs:

### Q-Learning

| Parameter | Value | Reason |
|---|---|---|
| `q_alpha` | **0.12** | Slightly above the paper's 0.1 for faster convergence within 5 000 episodes |
| `q_epsilon_start` | 1.0 | Full exploration at episode 0 |
| `q_epsilon_decay` | **0.998** | Reaches ε = 0.1 by ~episode 1 150, leaving 3 850 exploitation episodes |
| `q_epsilon_min` | 0.05 | Keeps 5 % exploration throughout; prevents Q-table lock-in |
| `planning_steps` | **12** | Extra Dyna sweeps per real step; speeds up convergence (paper uses 8) |
| `discount` | 0.95 | Matches the paper |

### Policy Gradient

| Parameter | Value | Reason |
|---|---|---|
| `pg_lr` | **2e-3** | Double the paper's 1e-3; REINFORCE variance benefits from a larger initial step |
| `pg_hidden1` | 64 | Matches paper Section VI-B |
| `pg_hidden2` | 32 | Matches paper Section VI-B |
| `discount` | 0.95 | Matches the paper |

> **Note:** The environment (acceptance model, default-risk model, market dynamics, reward function) is unchanged from the paper's specification. Only agent hyperparameters were tuned.

---

## MDP Formulation

**State** `sₜ = (dₜ, cₜ, mₜ, aₜ)` where:
- `dₜ ∈ {low, medium, high}` — current demand level
- `cₜ ∈ {1, 2, 3}` — customer credit category (poor → good)
- `mₜ ∈ ℝ` — prevailing market interest rate
- `aₜ ∈ [0, 1]` — rolling acceptance rate over the last 50 steps

**Actions** `A = {−0.50%, −0.25%, 0%, +0.25%, +0.50%}` — five discrete rate adjustments clipped to `[3%, 15%]`.

**Reward** (paper Eq. 3):

```
Rₜ = Aₜ(rₜ − c)L − λ Dₜ
```

where `Aₜ` is the acceptance indicator, `c = 3%` is cost of capital, `L = 1` is normalised loan amount, and `Dₜ` is the default probability. Risk cost is charged **only on accepted loans**.

**Acceptance probability** (paper Eq. 4):

```
P(accept) = sigmoid(α − β rₜ + δ cₜ + demand_shift)
```

Calibrated so good-credit borrowers accept ~73 % at 3 % and ~40 % at 6 %.

---

## Implementation Notes

### Q-Learning agent (`ReplayQLearningAgent`)

- Tabular Q-table keyed on a 5-tuple `(demand, credit, market_bucket, acceptance_bucket, rate_bucket)`. Including the **internally tracked current rate** as the fifth dimension restores the Markov property missing when actions are incremental adjustments.
- Q-values are initialised with the expected one-step reward (model-based prior) so the agent starts with sensible values rather than all zeros.
- Dyna-style planning: after every real transition the agent replays `planning_steps` past transitions from a circular buffer.

### Policy Gradient agent (`PolicyGradientAgent`)

- Two-layer ReLU network: `state(5) → 64 → 32 → 5 (softmax)`.
- Input is the normalised 5-D feature vector from `state.encode_features()`.
- Full-episode REINFORCE (paper Eq. 9): gradients are accumulated across all timesteps in an episode and applied once at the end.
- Returns are normalised per episode (zero mean, unit std) to reduce gradient variance.
- Adam optimiser implemented in pure NumPy — no external deep-learning library required.

### Why PG achieves higher acceptance

REINFORCE optimises the **sum of discounted rewards over a full episode**. A sequence of rejections within an episode directly reduces the return `Gₜ` and pushes the policy toward lower, more acceptable rates. Q-learning's bootstrapped, off-policy update does not see this sequential rejection penalty in the same way, so it settles at a slightly higher rate with correspondingly lower acceptance.

---

## Output Files

After a full run `outputs/full/` contains:

| File | Description |
|---|---|
| `results.json` | Full experiment data: config, training curves, all metric summaries |
| `results_plot.png` | Three-panel figure: training curves, gross-profit bars, rate-acceptance scatter |
| `q_learning_seed{n}.pkl` | Saved Q-table for seed n |
| `policy_gradient_seed{n}.pkl` | Saved PG network weights for seed n |

---

## References

1. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.
2. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529–533, 2015.
3. K. J. Ferreira, B. H. A. Lee, and D. Simchi-Levi, "Analytics for an online retailer: Demand forecasting and price optimization," *M&SOM*, vol. 18, no. 1, pp. 69–88, 2016.
4. L. Chen, A. Mislove, and C. Wilson, "An empirical analysis of algorithmic pricing on Amazon Marketplace," *Proc. WWW*, pp. 1339–1349, 2016.