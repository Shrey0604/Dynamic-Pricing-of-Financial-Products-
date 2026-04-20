"""Fast contract and training smoke test."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_pricing.config import PricingConfig
from rl_pricing.environment import LoanPricingEnv
from rl_pricing.training import run_full_experiment


def main() -> None:
    cfg = PricingConfig(episode_length=30, train_episodes=20, eval_episodes=5)
    env = LoanPricingEnv(config=cfg, seed=0)
    obs = env.reset()
    assert len(obs) == 4
    next_obs, reward, done = env.step(2)
    assert len(next_obs) == 4
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert env.action_space_n == 5

    result = run_full_experiment(
        config=cfg,
        seeds=[0],
        train_episodes=20,
        eval_episodes=5,
        output_dir=PROJECT_ROOT / "outputs" / "smoke",
    )
    assert result["summaries"]
    print("smoke test passed")


if __name__ == "__main__":
    main()
