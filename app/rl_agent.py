"""
RL agent for MedFlow-OpenEnv: replaces if-else decision logic with a structured policy.
"""

from app.models import HospitalAction
from app.env import HospitalQueueEnvironment


def calculate_reward(state, action):
    if state["severity"] == "high" and action == "ICU":
        return 10
    elif state["severity"] == "high" and action != "ICU":
        return -10
    return 2


class SmartDummyRLPolicy:
    """Quick-win policy with explicit state -> decision -> action flow."""

    def _state_from_obs(self, obs):
        waiting = obs.waiting_patients
        if any(p["priority"] == "emergency" for p in waiting):
            severity = "high"
        elif any(p["priority"] == "urgent" for p in waiting):
            severity = "medium"
        else:
            severity = "low"
        return {"severity": severity}

    def choose_action(self, state):
        # Simple heuristic policy to show structured agent behavior.
        if state["severity"] == "high":
            return "ICU"
        return "WAIT"

    def learn(self, state, action, reward):
        # Intentionally minimal learning hook for RL structure completeness.
        pass

    def select_action(self, obs):
        state = self._state_from_obs(obs)
        decision = self.choose_action(state)

        waiting = obs.waiting_patients
        free_docs = [d for d in obs.doctors if not d["busy"]]

        if decision == "ICU" and waiting and free_docs:
            # Pick highest priority patient first, then longest waiting.
            priority_rank = {"emergency": 0, "urgent": 1, "normal": 2}
            patient = sorted(
                waiting,
                key=lambda p: (priority_rank.get(p["priority"], 3), p.get("arrival_minute", 9999)),
            )[0]
            req_spec = patient["required_specialization"]
            matching = [d for d in free_docs if d["specialization"] == req_spec]
            general = [d for d in free_docs if d["specialization"] == "General"]
            doctor = (matching or general or free_docs)[0]
            return HospitalAction(action_type="assign", patient_id=patient["id"], doctor_id=doctor["id"])

        return HospitalAction(action_type="wait")

def run_rl_agent(env: HospitalQueueEnvironment, task_id: str = "easy_small_clinic", seed: int = 42) -> float:
    obs = env.reset(task_id=task_id, seed=seed)
    total_reward = 0.0
    policy = SmartDummyRLPolicy()
    while not obs.done:
        state = policy._state_from_obs(obs)
        decision = policy.choose_action(state)
        rl_reward = calculate_reward(state, decision)
        policy.learn(state, decision, rl_reward)

        action = policy.select_action(obs)
        obs = env.step(action)
        total_reward += obs.reward
    return round(total_reward, 3)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run RL agent on MedFlow tasks.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["easy_small_clinic", "medium_busy_opd", "hard_mass_casualty"],
        help="Task IDs to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reset seed for reproducibility.")
    args = parser.parse_args()
    env = HospitalQueueEnvironment()
    for task_id in args.tasks:
        score = run_rl_agent(env, task_id=task_id, seed=args.seed)
        print(f"[Task: {task_id}] → Reward: {score}")
        if score < 0:
            print("⚠️ Needs improvement")

if __name__ == "__main__":
    main()
