"""Learning agents for the pricing environment."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Deque

import numpy as np

from rl_pricing.config import PricingConfig
from rl_pricing.mdp import Observation, apply_action, expected_reward
from rl_pricing.policies import RateTrackingPolicy
from rl_pricing.state import discretize_observation


StateKey = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class Transition:
    state: StateKey
    action: int
    reward: float
    next_state: StateKey
    done: bool


class ReplayQLearningAgent(RateTrackingPolicy):
    """Tabular Q-learning with replay planning and model-based initialization."""

    name = "replay_q_learning"

    def __init__(
        self,
        config: PricingConfig | None = None,
        seed: int = 0,
        alpha: float | None = None,
        epsilon: float | None = None,
        planning_steps: int | None = None,
        use_expected_prior: bool = True,
    ):
        super().__init__(config)
        self.rng = np.random.default_rng(seed)
        self.alpha = self.config.q_alpha if alpha is None else alpha
        self.gamma = self.config.discount
        self.epsilon = self.config.q_epsilon_start if epsilon is None else epsilon
        self.epsilon_min = self.config.q_epsilon_min
        self.epsilon_decay = self.config.q_epsilon_decay
        self.planning_steps = (
            self.config.planning_steps if planning_steps is None else planning_steps
        )
        self.use_expected_prior = use_expected_prior

        self.q: defaultdict[StateKey, np.ndarray] = defaultdict(self._zeros)
        self.visits: defaultdict[tuple[StateKey, int], int] = defaultdict(int)
        self.replay: Deque[Transition] = deque(maxlen=self.config.replay_capacity)
        self._last_state: StateKey | None = None

    def _zeros(self) -> np.ndarray:
        return np.zeros(self.config.n_actions, dtype=np.float64)

    def start_episode(self, obs: Observation) -> None:
        super().start_episode(obs)
        self._last_state = None

    def _state_key(self, obs: Observation, rate: float | None = None) -> StateKey:
        return discretize_observation(obs, self.current_rate if rate is None else rate, self.config)

    def _ensure_prior(self, key: StateKey, obs: Observation, rate: float) -> None:
        if key in self.q or not self.use_expected_prior:
            _ = self.q[key]
            return

        demand, credit, _market_rate, _acceptance = obs
        values = []
        for action in range(self.config.n_actions):
            next_rate = apply_action(rate, action, self.config)
            values.append(
                expected_reward(
                    next_rate, int(credit), int(demand), self.config
                ).expected_reward
            )
        self.q[key] = np.array(values, dtype=np.float64)

    def act(self, obs: Observation, greedy: bool = False) -> int:
        rate_before = self.current_rate
        key = self._state_key(obs, rate_before)
        self._ensure_prior(key, obs, rate_before)

        if not greedy and self.rng.random() < self.epsilon:
            action = int(self.rng.integers(self.config.n_actions))
        else:
            action = int(np.argmax(self.q[key]))

        self._last_state = key
        self.current_rate = apply_action(rate_before, action, self.config)
        return action

    def observe(
        self,
        obs: Observation,
        action: int,
        reward: float,
        next_obs: Observation,
        done: bool,
    ) -> None:
        if self._last_state is None:
            return
        next_key = self._state_key(next_obs, self.current_rate)
        self._ensure_prior(next_key, next_obs, self.current_rate)
        transition = Transition(
            state=self._last_state,
            action=int(action),
            reward=float(reward),
            next_state=next_key,
            done=bool(done),
        )
        self._learn_from_transition(transition)
        self.replay.append(transition)
        self._planning_updates()

    def _learn_from_transition(self, transition: Transition) -> None:
        q_values = self.q[transition.state]
        target = transition.reward
        if not transition.done:
            target += self.gamma * float(np.max(self.q[transition.next_state]))
        td_error = target - q_values[transition.action]
        q_values[transition.action] += self.alpha * td_error
        self.visits[(transition.state, transition.action)] += 1

    def _planning_updates(self) -> None:
        if not self.replay or self.planning_steps <= 0:
            return
        max_index = len(self.replay)
        for _ in range(self.planning_steps):
            transition = self.replay[int(self.rng.integers(max_index))]
            self._learn_from_transition(transition)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        payload = {
            "config": self.config,
            "q": dict(self.q),
            "visits": dict(self.visits),
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "planning_steps": self.planning_steps,
            "use_expected_prior": self.use_expected_prior,
        }
        with Path(path).open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "ReplayQLearningAgent":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        agent = cls(
            config=payload["config"],
            seed=seed,
            alpha=payload["alpha"],
            epsilon=payload["epsilon"],
            planning_steps=payload["planning_steps"],
            use_expected_prior=payload["use_expected_prior"],
        )
        agent.q.update(payload["q"])
        agent.visits.update(payload["visits"])
        return agent
