"""Baseline greedy agent for MedFlow-OpenEnv."""

import argparse

from app.models import HospitalAction
from app.env import HospitalQueueEnvironment


def run_baseline(
    env: HospitalQueueEnvironment,
    task_id: str = "easy_small_clinic",
    seed: int = 42,
) -> float:
    """
    Greedy baseline strategy:
    - Prioritize emergency patients (highest priority first)
    - Assign to available doctor with matching specialization
    - Fall back to any available doctor if no specialist free
    - Wait if no doctors available
    
    Returns: total accumulated reward over the episode.
    """
    obs = env.reset(task_id=task_id, seed=seed)
    total_reward = 0.0
    
    while not obs.done:
        waiting = obs.waiting_patients
        
        if not waiting:
            # No patients in queue, wait
            action = HospitalAction(action_type="wait")
        else:
            # Greedily pick first waiting patient (sorted by priority)
            patient = waiting[0]
            pid = patient["id"]
            
            # Find available doctors
            free_docs = [d for d in obs.doctors if not d["busy"]]
            
            if not free_docs:
                # No free doctors, wait for next step
                action = HospitalAction(action_type="wait")
            else:
                # Prefer doctor matching specialization; else any available
                req_spec = patient["required_specialization"]
                matching_spec = [d for d in free_docs if d["specialization"] == req_spec]
                general = [d for d in free_docs if d["specialization"] == "General"]
                
                if matching_spec:
                    doctor = matching_spec[0]
                elif general:
                    doctor = general[0]
                else:
                    doctor = free_docs[0]
                
                # Assign patient to doctor
                action = HospitalAction(
                    action_type="assign",
                    patient_id=pid,
                    doctor_id=doctor["id"]
                )
        
        # Step environment
        obs = env.step(action)
        total_reward += obs.reward
    
    return round(total_reward, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run greedy baseline on MedFlow tasks.")
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
        score = run_baseline(env, task_id=task_id, seed=args.seed)
        print(f"{task_id}: total_reward={score}")


if __name__ == "__main__":
    main()