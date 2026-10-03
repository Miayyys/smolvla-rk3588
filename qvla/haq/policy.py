"""Small NumPy recurrent policy-gradient controller for HAQ proposals.

The controller is a plumbing baseline, not a reproduction of HAQ's exact
training setup. It is deliberately separated from evaluation: only complete,
contract-scored policy results should be passed to ``update_batch``.
"""

from __future__ import annotations

# Allow direct execution as well as python -m qvla.<module>.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = next((parent for parent in _Path(__file__).absolute().parents
                  if (parent / "qvla/__init__.py").is_file()),
                 _Path(__file__).resolve().parents[2])
    _sys.path.insert(0, str(_root))


import json
from pathlib import Path
from typing import Any

import numpy as np


class RecurrentPolicyGradient:
    """Autoregressive masked categorical policy with episodic REINFORCE."""

    def __init__(self, space: dict[str, Any], hidden_size: int = 48, seed: int = 0):
        if not space.get("action_sites"):
            raise ValueError("Action space has no selectable sites")
        self.space = space
        self.sites = space["action_sites"]
        self.formats = space["format_vocab"]
        self.n_formats = len(self.formats)
        self.feature_size = len(space["feature_schema"])
        self.hidden_size = int(hidden_size)
        self.input_size = self.feature_size + self.n_formats
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")

        self.rng = np.random.default_rng(seed)
        self.params = {
            "wx": self.rng.normal(0, 1 / np.sqrt(self.input_size),
                                  (self.input_size, self.hidden_size)),
            "wh": self.rng.normal(0, 1 / np.sqrt(self.hidden_size),
                                  (self.hidden_size, self.hidden_size)),
            "bh": np.zeros(self.hidden_size, dtype=np.float64),
            "wo": self.rng.normal(0, 1 / np.sqrt(self.hidden_size),
                                  (self.hidden_size, self.n_formats)),
            "bo": np.zeros(self.n_formats, dtype=np.float64),
        }
        self.m = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.v = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.optimizer_step = 0
        self.reward_baseline: float | None = None

    def _masked_softmax(self, logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid = np.flatnonzero(mask)
        if not len(valid):
            raise ValueError("A search site has no allowed candidate")
        shifted = logits[valid] - np.max(logits[valid])
        values = np.exp(shifted)
        probs = np.zeros(self.n_formats, dtype=np.float64)
        probs[valid] = values / values.sum()
        return probs

    def sample(self) -> tuple[dict[str, str], list[dict[str, np.ndarray]]]:
        assignment: dict[str, str] = {}
        trajectory = []
        hidden = np.zeros(self.hidden_size, dtype=np.float64)
        previous = np.zeros(self.n_formats, dtype=np.float64)

        for site in self.sites:
            features = np.asarray(site["features"], dtype=np.float64)
            if features.shape != (self.feature_size,):
                raise ValueError(f"Bad feature vector for {site['module']}")
            x = np.concatenate((features, previous))
            hidden_previous = hidden
            hidden = np.tanh(x @ self.params["wx"] + hidden_previous @ self.params["wh"]
                             + self.params["bh"])
            logits = hidden @ self.params["wo"] + self.params["bo"]
            mask = np.zeros(self.n_formats, dtype=bool)
            for option in site["options"]:
                mask[int(option["format_id"])] = True
            probs = self._masked_softmax(logits, mask)
            action_id = int(self.rng.choice(self.n_formats, p=probs))
            selected = next(option["format"] for option in site["options"]
                            if int(option["format_id"]) == action_id)
            assignment[site["module"]] = selected
            trajectory.append({
                "x": x,
                "hidden_previous": hidden_previous.copy(),
                "hidden": hidden.copy(),
                "probs": probs,
                "mask": mask,
                "action_id": np.asarray(action_id, dtype=np.int64),
            })
            previous = np.zeros(self.n_formats, dtype=np.float64)
            previous[action_id] = 1.0
        return assignment, trajectory

    def update_batch(self, trajectories: list[list[dict[str, np.ndarray]]],
                     rewards: list[float], learning_rate: float = 1e-3,
                     entropy_weight: float = 0.002,
                     max_gradient_norm: float = 5.0) -> dict[str, float]:
        if len(trajectories) != len(rewards) or not trajectories:
            raise ValueError("Provide the same nonzero number of trajectories and rewards")
        values = np.asarray(rewards, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Rewards must be finite")
        if learning_rate <= 0 or entropy_weight < 0 or max_gradient_norm <= 0:
            raise ValueError("Invalid optimizer settings")
        if len(values) > 1:
            advantages = values - values.mean()
        elif self.reward_baseline is not None:
            advantages = values - self.reward_baseline
        else:
            raise ValueError("First policy update needs at least two scored configurations")

        grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        for trajectory, advantage in zip(trajectories, advantages):
            if len(trajectory) != len(self.sites):
                raise ValueError("Trajectory length does not match the action space")
            hidden_gradient_next = np.zeros(self.hidden_size, dtype=np.float64)
            for step in reversed(trajectory):
                probs = step["probs"]
                mask = step["mask"]
                action_id = int(step["action_id"])
                one_hot = np.zeros(self.n_formats, dtype=np.float64)
                one_hot[action_id] = 1.0
                log_probs = np.zeros_like(probs)
                log_probs[mask] = np.log(np.maximum(probs[mask], 1e-12))
                entropy = -float(np.sum(probs[mask] * log_probs[mask]))
                grad_logits = advantage * (probs - one_hot)
                grad_logits += entropy_weight * probs * (log_probs + entropy)

                grads["wo"] += np.outer(step["hidden"], grad_logits)
                grads["bo"] += grad_logits
                hidden_gradient = grad_logits @ self.params["wo"].T + hidden_gradient_next
                preactivation_gradient = hidden_gradient * (1 - step["hidden"] ** 2)
                grads["wx"] += np.outer(step["x"], preactivation_gradient)
                grads["wh"] += np.outer(step["hidden_previous"], preactivation_gradient)
                grads["bh"] += preactivation_gradient
                hidden_gradient_next = preactivation_gradient @ self.params["wh"].T

        scale = 1.0 / len(trajectories)
        for value in grads.values():
            value *= scale
        norm = float(np.sqrt(sum(np.sum(value * value) for value in grads.values())))
        if norm > max_gradient_norm:
            factor = max_gradient_norm / max(norm, 1e-12)
            for value in grads.values():
                value *= factor

        self.optimizer_step += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for name, parameter in self.params.items():
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * grads[name]
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (grads[name] ** 2)
            m_hat = self.m[name] / (1 - beta1 ** self.optimizer_step)
            v_hat = self.v[name] / (1 - beta2 ** self.optimizer_step)
            parameter -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        batch_mean = float(values.mean())
        self.reward_baseline = (batch_mean if self.reward_baseline is None
                                else 0.9 * self.reward_baseline + 0.1 * batch_mean)
        return {"reward_mean": batch_mean, "reward_std": float(values.std()),
                "gradient_norm_before_clip": norm}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "space_sha256": self.space["source"]["tables_sha256"],
            "hidden_size": self.hidden_size,
            "feature_size": self.feature_size,
            "formats": self.formats,
            "optimizer_step": self.optimizer_step,
            "reward_baseline": self.reward_baseline,
            "rng_state": self.rng.bit_generator.state,
        }
        arrays = {f"param_{name}": value for name, value in self.params.items()}
        arrays.update({f"m_{name}": value for name, value in self.m.items()})
        arrays.update({f"v_{name}": value for name, value in self.v.items()})
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(path, **arrays)

    def load(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            if metadata["space_sha256"] != self.space["source"]["tables_sha256"]:
                raise ValueError("Policy checkpoint was built for another hardware table")
            if metadata["hidden_size"] != self.hidden_size or metadata["formats"] != self.formats:
                raise ValueError("Policy checkpoint architecture does not match")
            for name in self.params:
                self.params[name][...] = archive[f"param_{name}"]
                self.m[name][...] = archive[f"m_{name}"]
                self.v[name][...] = archive[f"v_{name}"]
            self.optimizer_step = int(metadata["optimizer_step"])
            self.reward_baseline = metadata["reward_baseline"]
            self.rng.bit_generator.state = metadata["rng_state"]
