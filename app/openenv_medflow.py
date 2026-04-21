"""OpenEnv-compatible minimal medical workflow environment for MedFlow-OpenEnv.

This is a lightweight scaffold suitable for demos and local training. It provides:
- `OpenEnvMedFlow` class with `reset()` and `step(action)`
- `make_env()` convenience constructor

The environment is intentionally simple: an agent performs high-level actions
on a simulated patient case; rewards are heuristic for demo purposes.
"""
from typing import Any, Dict, List, Optional
import random


class OpenEnvMedFlow:
    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps
        self.step_count = 0
        self.done = False
        self.history: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {}

    def reset(self) -> Dict[str, Any]:
        self.step_count = 0
        self.done = False
        self.history = []
        # Simple patient case features
        self.state = {
            "patient_id": random.randint(1000, 9999),
            "complaint": random.choice(["headache", "chest pain", "fatigue"]),
            "tests_ordered": [],
            "notes": [],
        }
        obs = self._make_observation()
        return obs

    def step(self, action: str) -> (Dict[str, Any], float, bool, Dict[str, Any]):
        if self.done:
            raise RuntimeError("step() called after environment is done; call reset()")

        self.step_count += 1
        reward = self._evaluate_action(action)
        self.history.append({"step": self.step_count, "action": action, "reward": reward})

        # small stochastic effects
        if "order" in action:
            self.state["tests_ordered"].append(action)

        if self.step_count >= self.max_steps:
            self.done = True

        obs = self._make_observation()
        info = {"patient_id": self.state["patient_id"]}
        return obs, reward, self.done, info

    def render(self) -> None:
        print("--- Patient Case ---")
        print(f"ID: {self.state['patient_id']}")
        print(f"Complaint: {self.state['complaint']}")
        print("History:")
        for h in self.history:
            print(f"  {h['step']}: {h['action']} (+{h['reward']:.2f})")

    def _make_observation(self) -> Dict[str, Any]:
        return {
            "patient_id": self.state["patient_id"],
            "complaint": self.state["complaint"],
            "step": self.step_count,
            "notes": list(self.state.get("notes", [])),
            "tests_ordered": list(self.state.get("tests_ordered", [])),
        }

    def _evaluate_action(self, action: str) -> float:
        # Heuristic reward function for demo purposes
        a = action.lower()
        reward = 0.0
        if "gather" in a or "history" in a or "ask" in a:
            reward += 0.5
        if "order" in a or "test" in a:
            reward += 1.0
        if "schedule" in a or "followup" in a:
            reward += 0.8
        if "prescribe" in a or "treat" in a:
            reward += 1.5
        # discourage irrelevant actions
        if len(a.strip()) == 0:
            reward -= 1.0
        # small random noise
        reward += random.uniform(-0.1, 0.2)
        return float(reward)


def make_env(**kwargs) -> OpenEnvMedFlow:
    return OpenEnvMedFlow(**kwargs)


if __name__ == "__main__":
    # simple demo when run directly
    env = OpenEnvMedFlow()
    obs = env.reset()
    print("Reset obs:", obs)
    done = False
    while not done:
        action = random.choice(["gather_history", "order_blood_test", "schedule_followup", "prescribe_medication"])
        obs, r, done, info = env.step(action)
        print(f"Action={action} reward={r:.2f}")
    env.render()
