# RL Dynamic Pricing for Financial Products

This is a rebuilt, submission-ready reinforcement learning project for Group 31's dynamic loan-pricing task. The old attempt is preserved under `provided/old_attempt/` only for reference; the runnable project is the new `rl_pricing/` package plus the `scripts/` entry points.

## What Changed

The rebuild fixes the main issues found in the submitted ZIP and in the interface specification:

- The environment now follows the required interface exactly: `reset()`, `step(action) -> (obs, reward, done)`, and `action_space_n == 5`.
- Observations are the raw four-field tuple from the spec: `(demand, credit, market_rate, rolling_acceptance)`.
- Agents track the current offered rate internally, because rate-adjustment actions are not Markov without that information.
- Risk cost is charged only on accepted loans, not on every rejected quote.
- The logistic acceptance model is calibrated to the report's stated behavior for good-credit customers: about 73% acceptance at 3% and about 40% at 6%.
- Evaluation reports risk-adjusted profit, gross accepted margin, acceptance rate, average rate, and risk cost.

## Project Structure

```text
.
|-- configs/
|   `-- default.json
|-- outputs/
|   |-- quick/
|   |   `-- results.json
|   `-- plot_smoke/
|       |-- results.json
|       `-- results_plot.svg
|-- provided/
|   |-- extracted_text/
|   `-- old_attempt/
|-- rl_pricing/
|   |-- __init__.py
|   |-- agents.py
|   |-- config.py
|   |-- environment.py
|   |-- evaluation.py
|   |-- mdp.py
|   |-- plotting.py
|   |-- policies.py
|   |-- state.py
|   `-- training.py
|-- scripts/
|   |-- evaluate.py
|   |-- smoke_test.py
|   `-- train.py
|-- tests/
|   `-- test_contract.py
|-- README.md
`-- requirements.txt
```

## Methods Included

| Method | Role |
|---|---|
| Fixed Pricing | Constant 7% benchmark. |
| Rule-Based | Credit-tier heuristic from the report. |
| Profit-Greedy Planner | Model-based policy that maximizes expected one-step risk-adjusted profit. |
| Balanced Planner | Model-based policy with an explicit acceptance floor. This is the recommended final strategy for the project objective. |
| Replay Q-Learning | Tabular Q-learning with replay planning and expected-reward initialization. |

The balanced planner is included because the task is not just "maximize rate margin"; it asks for a good profit and acceptance trade-off. The replay Q-learning agent is useful as the RL comparison point and can be trained/reloaded reproducibly.

## Setup

```bash
pip install -r requirements.txt
```

Only NumPy is required for training and evaluation. Matplotlib is used for PNG plots when installed. If Matplotlib is unavailable, the plotting code automatically writes an SVG fallback.

## Run

Fast contract check:

```bash
python scripts/smoke_test.py
```

Quick experiment:

```bash
python scripts/train.py --seeds 0 1 --train-episodes 500 --eval-episodes 50 --episode-length 100 --output-dir outputs/quick --no-plot
```

Paper-scale experiment:

```bash
python scripts/train.py --seeds 0 1 2 3 4 --train-episodes 4000 --eval-episodes 500 --episode-length 200 --output-dir outputs/full
```

Evaluate built-in policies without training:

```bash
python scripts/evaluate.py --episodes 500 --seeds 0 1 2 3 4
```

Evaluate a saved Replay Q-learning model:

```bash
python scripts/evaluate.py --model outputs/quick/replay_q_seed0.pkl --episodes 100 --seeds 0
```

## Quick Results

The following command was run in this workspace:

```bash
python scripts/train.py --seeds 0 1 --train-episodes 500 --eval-episodes 50 --episode-length 100 --output-dir outputs/quick --no-plot
```

| Method | Profit/step | Gross Profit | Accept Rate | Avg Rate | Risk Cost |
|---|---:|---:|---:|---:|---:|
| Fixed Pricing | +0.00793+/-0.0016 | 0.00849 | 0.213+/-0.042 | 6.99% | 0.00056 |
| Rule-Based | +0.00791+/-0.0015 | 0.00846 | 0.222+/-0.044 | 7.00% | 0.00054 |
| Profit-Greedy Planner | +0.00817+/-0.0013 | 0.00890 | 0.273+/-0.043 | 6.24% | 0.00073 |
| Balanced Planner | +0.00603+/-0.0008 | 0.00722 | 0.428+/-0.040 | 4.67% | 0.00119 |
| Replay Q-Learning | +0.00809+/-0.0013 | 0.00878 | 0.262+/-0.049 | 6.42% | 0.00069 |

Compared with the old reported table, the balanced planner gives positive risk-adjusted profit, improves gross profit over the old best Q-learning gross profit, and slightly improves acceptance over the old policy-gradient acceptance rate while keeping rates in a realistic low-to-mid retail range.

## Design Notes

The environment remains synthetic because the report did not provide a real loan dataset. The simulator keeps the original state variables, five rate-adjustment actions, market mean reversion, credit-tier default model, and rolling acceptance state.

The most important reward-design correction is that default risk is treated as an expected loss on originated loans. A rejected customer should not create default loss. This makes the reward align with real lending economics and avoids the old negative-reward artifact.

The current offered rate is not part of the public observation because the interface document fixes the observation length at four. Learning agents therefore maintain an internal rate tracker initialized from the reset market rate and updated using their own actions. This preserves the interface while giving the learner the Markov information it needs.

## Assumptions

- Rates are decimals internally, so 7% is represented as `0.07`.
- The action space remains exactly `[-0.50%, -0.25%, 0%, +0.25%, +0.50%]`.
- The acceptance model keeps the report's logistic form, but uses calibrated alpha/beta values to match the report's stated acceptance examples.
- Risk loss is normalized because loan amount is normalized to `L = 1.0`.
- The saved quick results are not a substitute for the full 5-seed, 4000-episode run, but they verify the final system end to end.
