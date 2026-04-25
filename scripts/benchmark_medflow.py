"""Benchmark MedFlow-OpenEnv with a few lightweight policies.

Usage:
    python scripts/benchmark_medflow.py --episodes 3

Outputs:
    - outputs/benchmarks/benchmark_summary.json
    - outputs/benchmarks/benchmark_summary.md

The benchmark compares a greedy policy against a random policy across all
defined tasks and records the environment's final score and episode stats.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.env import HospitalQueueEnvironment
from app.models import HospitalAction
from app.tasks import ALL_TASKS


def _choose_greedy_action(obs) -> HospitalAction:
    waiting = list(obs.waiting_patients)
    doctors = list(obs.doctors)
    free_docs = [d for d in doctors if not d.get("busy", False)]

    if waiting and free_docs:
        priority_rank = {"emergency": 0, "urgent": 1, "normal": 2}
        patient = sorted(
            waiting,
            key=lambda p: (
                priority_rank.get(p.get("priority", "normal"), 3),
                -int(p.get("severity_score", 0)),
                -int(p.get("wait_minutes", 0)),
                int(p.get("id", 0)),
            ),
        )[0]
        required = patient.get("required_specialization")
        matching = [d for d in free_docs if d.get("specialization") == required]
        general = [d for d in free_docs if d.get("specialization") == "General"]
        doctor = (matching or general or free_docs)[0]
        return HospitalAction(action_type="assign", patient_id=int(patient["id"]), doctor_id=int(doctor["id"]))

    if waiting:
        patient = waiting[0]
        return HospitalAction(action_type="prioritize", patient_id=int(patient["id"]))

    for d in doctors:
        if (not d.get("busy", False)) and d.get("current_patient_id") is not None:
            return HospitalAction(action_type="discharge", patient_id=int(d["current_patient_id"]))

    return HospitalAction(action_type="wait")


def _choose_random_action(obs) -> HospitalAction:
    waiting = list(obs.waiting_patients)
    doctors = list(obs.doctors)
    free_docs = [d for d in doctors if not d.get("busy", False)]
    if waiting and free_docs:
        patient = random.choice(waiting)
        doctor = random.choice(free_docs)
        return HospitalAction(action_type="assign", patient_id=int(patient["id"]), doctor_id=int(doctor["id"]))
    if waiting:
        return HospitalAction(action_type="prioritize", patient_id=int(random.choice(waiting)["id"]))
    for d in doctors:
        if d.get("current_patient_id") is not None:
            return HospitalAction(action_type="discharge", patient_id=int(d["current_patient_id"]))
    return HospitalAction(action_type="wait")


def run_episode(task_id: str, policy: str, seed: int = 42) -> Dict[str, object]:
    random.seed(seed)
    env = HospitalQueueEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    total_reward = 0.0
    steps = 0

    while not obs.done and steps < ALL_TASKS[task_id].max_steps:
        if policy == "greedy":
            action = _choose_greedy_action(obs)
        elif policy == "random":
            action = _choose_random_action(obs)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        obs = env.step(action)
        total_reward += float(obs.reward or 0.0)
        steps += 1

    state = env.state
    return {
        "task_id": task_id,
        "policy": policy,
        "final_score": float(state.final_score or 0.0),
        "total_reward": round(total_reward, 3),
        "steps": steps,
        "total_seen": state.total_patients_seen,
        "total_arrived": state.total_patients_arrived,
        "avg_wait_all": float(state.avg_wait_all),
        "critical_deaths": len(state.emergency_response_times),
        "submitted": bool(state.submitted),
    }


def _write_markdown(rows: List[Dict[str, object]], out_path: Path) -> None:
    lines = [
        "# MedFlow OpenEnv Benchmark Summary",
        "",
        "| task_id | policy | final_score | total_reward | steps | seen/arrived | avg_wait_all |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['policy']} | {row['final_score']:.3f} | {row['total_reward']:.3f} | {row['steps']} | {row['total_seen']}/{row['total_arrived']} | {row['avg_wait_all']:.2f} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_policy: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)

    summary: Dict[str, object] = {"policies": {}, "rows": rows}
    for policy, policy_rows in by_policy.items():
        summary["policies"][policy] = {
            "avg_final_score": round(mean(float(r["final_score"]) for r in policy_rows), 3),
            "avg_total_reward": round(mean(float(r["total_reward"]) for r in policy_rows), 3),
            "avg_steps": round(mean(int(r["steps"]) for r in policy_rows), 1),
            "success_rate": round(sum(1 for r in policy_rows if float(r["final_score"]) >= 0.10) / len(policy_rows), 3),
        }
    return summary


def main(episodes: int, outdir: Path, seed: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    rows: List[Dict[str, object]] = []
    for task_id in ALL_TASKS.keys():
        for policy in ("greedy", "random"):
            for episode in range(episodes):
                row = run_episode(task_id=task_id, policy=policy, seed=seed + episode)
                rows.append(row)
                print(
                    f"{task_id} | {policy} | score={row['final_score']:.3f} | reward={row['total_reward']:.3f} | steps={row['steps']}"
                )

    summary = _summarize(rows)
    summary_path = outdir / "benchmark_summary.json"
    markdown_path = outdir / "benchmark_summary.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(rows, markdown_path)

    print(f"Saved JSON summary to {summary_path}")
    print(f"Saved markdown summary to {markdown_path}")
    for policy, stats in summary["policies"].items():
        print(
            f"{policy}: avg_final_score={stats['avg_final_score']:.3f}, avg_total_reward={stats['avg_total_reward']:.3f}, success_rate={stats['success_rate']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/benchmarks"))
    args = parser.parse_args()
    main(episodes=args.episodes, outdir=args.outdir, seed=args.seed)