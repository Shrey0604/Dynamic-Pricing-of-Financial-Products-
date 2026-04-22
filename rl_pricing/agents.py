"""Learning agents for the pricing environment.

Two agents are implemented:
  - ReplayQLearningAgent  : tabular Q-learning with Dyna-style replay
  - PolicyGradientAgent   : REINFORCE with a two-layer softmax network
                            (64 → 32 → 5 units, Adam optimiser, NumPy only)
"""

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
from rl_pricing.state import discretize_observation, encode_features


StateKey = tuple[int, int, int, int, int]


# ─────────────────────────────────────────────────────────────────────────────
# Tabular Q-Learning
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Transition:
    state: StateKey
    action: int
    reward: float
    next_state: StateKey
    done: bool


class ReplayQLearningAgent(RateTrackingPolicy):
    """Tabular Q-learning with replay planning and model-based initialisation.

    Hyperparameters come from Table I of the paper:
    α = 0.1, ε₀ = 1.0, ρ = 0.995, εmin = 0.05, γ = 0.95.
    """

    name = "q_learning"

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


# ─────────────────────────────────────────────────────────────────────────────
# Policy Gradient (REINFORCE)
# ─────────────────────────────────────────────────────────────────────────────

class PolicyGradientAgent(RateTrackingPolicy):
    """REINFORCE policy gradient with a two-layer softmax network.

    Architecture (Section VI-B):  state(5) → 64 → 32 → 5 actions (softmax)
    Optimiser: Adam with lr = 1e-3
    Update rule (Eq. 9):  θ ← θ + α ∇θ log πθ(a|s) Gₜ

    The input state is the 5-dimensional normalised feature vector produced by
    ``encode_features()`` in state.py:
        [demand/2, (credit-1)/2, market_rate_norm, rolling_acceptance, rate_norm]

    The agent collects a full episode trajectory during training and performs
    a single batch REINFORCE update at the end of each episode.
    Pure NumPy implementation — no external deep-learning library required.
    """

    name = "policy_gradient"

    def __init__(
        self,
        config: PricingConfig | None = None,
        seed: int = 0,
        lr: float | None = None,
    ):
        super().__init__(config)
        self.rng = np.random.default_rng(seed)
        self.lr = self.config.pg_lr if lr is None else lr
        self.gamma = self.config.discount

        state_dim = 5  # encode_features output dimension
        h1 = self.config.pg_hidden1   # 64
        h2 = self.config.pg_hidden2   # 32
        n_act = self.config.n_actions  # 5

        # He initialisation for ReLU layers
        self.W1 = self.rng.standard_normal((state_dim, h1)) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(h1, dtype=np.float64)
        self.W2 = self.rng.standard_normal((h1, h2)) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2, dtype=np.float64)
        self.W3 = self.rng.standard_normal((h2, n_act)) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(n_act, dtype=np.float64)

        # Adam first/second moment estimates and step counter
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._t: int = 0
        self._init_adam()

        # Episode rollout buffer (filled during training, cleared each episode)
        self._ep_states: list[np.ndarray] = []
        self._ep_actions: list[int] = []
        self._ep_rewards: list[float] = []
        self._last_feat: np.ndarray | None = None
        self._last_action: int | None = None

    # ── Adam initialisation ──────────────────────────────────────────────────

    def _init_adam(self) -> None:
        for name, param in self._params():
            self._m[name] = np.zeros_like(param)
            self._v[name] = np.zeros_like(param)

    def _params(self):
        """Yield (name, array) pairs for all learnable parameters."""
        return [
            ("W1", self.W1), ("b1", self.b1),
            ("W2", self.W2), ("b2", self.b2),
            ("W3", self.W3), ("b3", self.b3),
        ]

    # ── Forward pass ─────────────────────────────────────────────────────────

    def _forward(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (h1, h2, action_probs) for a single state vector."""
        h1 = np.maximum(0.0, s @ self.W1 + self.b1)   # ReLU
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)  # ReLU
        logits = h2 @ self.W3 + self.b3
        logits = logits - logits.max()                  # numerical stability
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum()
        return h1, h2, probs

    # ── RateTrackingPolicy interface ─────────────────────────────────────────

    def start_episode(self, obs: Observation) -> None:
        super().start_episode(obs)
        self._ep_states.clear()
        self._ep_actions.clear()
        self._ep_rewards.clear()
        self._last_feat = None
        self._last_action = None

    def act(self, obs: Observation, greedy: bool = False) -> int:
        rate_before = self.current_rate
        s = encode_features(obs, rate_before, self.config)
        _, _, probs = self._forward(s)

        if greedy:
            action = int(np.argmax(probs))
        else:
            action = int(self.rng.choice(self.config.n_actions, p=probs))
            # Store for observe() — only needed during training
            self._last_feat = s
            self._last_action = action

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
        """Accumulate transition; update policy at episode end."""
        if self._last_feat is not None:
            self._ep_states.append(self._last_feat)
            self._ep_actions.append(self._last_action)  # type: ignore[arg-type]
            self._ep_rewards.append(reward)
            self._last_feat = None
            self._last_action = None

        if done and self._ep_rewards:
            self._reinforce_update()
            self._ep_states.clear()
            self._ep_actions.clear()
            self._ep_rewards.clear()

    # ── REINFORCE update ─────────────────────────────────────────────────────

    def _reinforce_update(self) -> None:
        """Batch REINFORCE gradient update over one full episode."""
        T = len(self._ep_rewards)
        rewards = np.array(self._ep_rewards, dtype=np.float64)

        # Compute discounted returns Gₜ = Σ_{k=t}^{T} γ^{k-t} R_k
        returns = np.zeros(T, dtype=np.float64)
        G = 0.0
        for t in range(T - 1, -1, -1):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        # Normalise returns for gradient stability
        std = returns.std()
        if std > 1e-8:
            returns = (returns - returns.mean()) / (std + 1e-8)

        # Accumulate gradients across all time-steps
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        dW3 = np.zeros_like(self.W3)
        db3 = np.zeros_like(self.b3)

        for t in range(T):
            s = self._ep_states[t]
            a = self._ep_actions[t]
            G_t = returns[t]

            h1, h2, probs = self._forward(s)

            # ∂ log π(a|s) / ∂ logits_i  =  1(i==a) – π_i
            d_logits = -probs.copy()
            d_logits[a] += 1.0
            d_logits *= G_t   # scale by discounted return

            # Backprop through output layer
            dW3 += np.outer(h2, d_logits)
            db3 += d_logits

            # Backprop through second hidden layer (ReLU)
            d_h2 = d_logits @ self.W3.T * (h2 > 0)
            dW2 += np.outer(h1, d_h2)
            db2 += d_h2

            # Backprop through first hidden layer (ReLU)
            d_h1 = d_h2 @ self.W2.T * (h1 > 0)
            dW1 += np.outer(s, d_h1)
            db1 += d_h1

        # Adam gradient ascent (we maximise J(θ), so we ADD the gradient)
        self._t += 1
        grads = {"W1": dW1, "b1": db1, "W2": dW2,
                 "b2": db2, "W3": dW3, "b3": db3}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name, param in self._params():
            g = grads[name]
            self._m[name] = beta1 * self._m[name] + (1 - beta1) * g
            self._v[name] = beta2 * self._v[name] + (1 - beta2) * g * g
            m_hat = self._m[name] / (1.0 - beta1 ** self._t)
            v_hat = self._v[name] / (1.0 - beta2 ** self._t)
            param += self.lr * m_hat / (np.sqrt(v_hat) + eps)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        payload = {
            "config": self.config,
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
            "lr": self.lr,
            "_t": self._t,
        }
        with Path(path).open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "PolicyGradientAgent":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        agent = cls(config=payload["config"], seed=seed, lr=payload["lr"])
        for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(agent, key, payload[key])
        agent._t = payload["_t"]
        agent._init_adam()   # reset moments (agent is in eval mode after load)
        return agent