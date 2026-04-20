from rl_pricing.config import PricingConfig
from rl_pricing.environment import LoanPricingEnv


def test_environment_contract() -> None:
    cfg = PricingConfig(episode_length=3)
    env = LoanPricingEnv(config=cfg, seed=123)

    obs = env.reset()
    assert len(obs) == 4
    assert env.action_space_n == 5

    done = False
    steps = 0
    while not done:
        obs, reward, done = env.step(2)
        steps += 1
        assert len(obs) == 4
        assert isinstance(reward, float)
        assert env.last_info is not None

    assert steps == 3
