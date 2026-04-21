"""Simple grader for MedFlow-OpenEnv environments.

Provides functions to score individual actions and full episodes. This is a
lightweight, easily-extendable grader that can be improved with rule-based or
learned components later.
"""
from typing import Dict, List, Any


def grade_action(action: str, state: Dict[str, Any]) -> float:
    a = action.lower()
    score = 0.0
    # prefer information gathering first
    if any(k in a for k in ("gather", "history", "ask", "clarify")):
        score += 0.6
    if any(k in a for k in ("order", "test", "lab", "xray")):
        score += 1.0
    if any(k in a for k in ("prescribe", "treat", "medication")):
        score += 1.5
    if any(k in a for k in ("schedule", "followup")):
        score += 0.8
    # minor penalty for empty or irrelevant actions
    if not a.strip():
        score -= 1.0
    return float(score)


def grade_episode(history: List[Dict[str, Any]], final_state: Dict[str, Any]) -> Dict[str, Any]:
    total = sum([float(h.get("reward", 0.0)) for h in history])
    num_steps = len(history)
    # simple efficiency metric: reward per step
    efficiency = total / max(1, num_steps)

    # safety/quality proxy: presence of gather before prescribe
    actions = [h.get("action", "") for h in history]
    gathered = any("gather" in a or "history" in a for a in actions)
    prescribed = any("prescribe" in a or "treat" in a for a in actions)
    safety = 1.0
    if prescribed and not gathered:
        safety = 0.5  # penalize prescribing without gathering info

    return {
        "total_reward": float(total),
        "num_steps": num_steps,
        "efficiency": float(efficiency),
        "safety_score": float(safety),
    }
# Grading logic
class Grader:
    def __init__(self):
        self.total_reward = 0

    def add_reward(self, r):
        self.total_reward += r

    def get_score(self):
        return max(min(self.total_reward / 5, 1), 0)