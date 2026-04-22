"""Greedy policy runner for MedFlow-OpenEnv.

Runs multiple episodes using a simple greedy policy:
- Assign highest-priority waiting patient to any free doctor
- Discharge patients when doctors are free and patient still assigned
- Otherwise `wait` to advance time

Logs per-episode results to `outputs/eval_greedy.csv`.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Tuple

from app.env import HospitalQueueEnvironment
from app.models import HospitalAction


def run_episode(env: HospitalQueueEnvironment) -> Tuple[dict, object]:
    obs = env.reset()
    while not obs.done:
        # Try assign: first waiting patient to first free doctor
        assigned = False
        for d in obs.doctors:
            if (not d["busy"]) and (d["current_patient_id"] is None) and obs.waiting_patients:
                pid = obs.waiting_patients[0]["id"]
                act = HospitalAction(action_type="assign", patient_id=pid, doctor_id=d["id"])
                obs = env.step(act)
                assigned = True
                break
        if assigned:
            continue

        # Try discharge: doctors who are free but still have a current_patient_id
        discharged = False
        for d in obs.doctors:
            if (not d["busy"]) and (d["current_patient_id"] is not None):
                pid = d["current_patient_id"]
                act = HospitalAction(action_type="discharge", patient_id=pid)
                obs = env.step(act)
                discharged = True
                break
        if discharged:
            continue

        # Otherwise wait
        obs = env.step(HospitalAction(action_type="wait"))

    return {
        "episode_id": env.state.episode_id,
        "final_score": env.state.final_score or 0.0,
        "total_seen": env.state.total_patients_seen,
        "total_arrived": env.state.total_patients_arrived,
        "avg_wait_all": env.state.avg_wait_all,
        "submitted": env.state.submitted,
        "task_id": env.state.task_id,
        "steps": env.state.step_count if hasattr(env.state, "step_count") else None,
    }, obs


def main(episodes: int, outpath: str):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    headers = [
        "episode_id",
        "task_id",
        "final_score",
        "total_seen",
        "total_arrived",
        "avg_wait_all",
        "submitted",
    ]

    with open(outpath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()

        for i in range(episodes):
            env = HospitalQueueEnvironment()
            stats, obs = run_episode(env)
            writer.writerow({k: stats.get(k) for k in headers})
            print(f"Episode {i+1}/{episodes}: score={stats['final_score']:.3f} seen={stats['total_seen']}/{stats['total_arrived']}")

    print("Results saved to:", outpath)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    p.add_argument(
        "--out",
        type=str,
        default="outputs/eval_greedy.csv",
        help="CSV output path",
    )
    args = p.parse_args()
    main(args.episodes, args.out)
