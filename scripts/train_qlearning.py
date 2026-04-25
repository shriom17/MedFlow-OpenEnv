"""Tabular Q-learning trainer for MedFlow-OpenEnv.

This script trains a simple discrete policy against `HospitalQueueEnvironment`
and saves judge-friendly artifacts:
- reward curve (`outputs/evals/qlearning_reward_curve.png`)
- loss curve (`outputs/evals/qlearning_loss_curve.png`)
- trained-vs-random comparison (`outputs/evals/qlearning_policy_compare.png`)
- per-episode metrics CSV (`outputs/evals/qlearning_training_metrics.csv`)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

# Allow running this file directly from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.env import HospitalQueueEnvironment
from app.models import HospitalAction, HospitalObservation


ActionKey = Tuple[str, int | None, int | None]
StateKey = Tuple[int, int, int, int, str, int]
QTable = DefaultDict[StateKey, Dict[ActionKey, float]]


def action_key(action: HospitalAction) -> ActionKey:
    return (action.action_type, action.patient_id, action.doctor_id)


def encode_state(obs: HospitalObservation) -> StateKey:
    waiting = list(obs.waiting_patients)
    doctors = list(obs.doctors)
    queue_bucket = min(int(obs.queue_length), 8)
    critical_bucket = min(int(obs.critical_untreated), 4)
    free_docs_bucket = min(sum(1 for d in doctors if not d.get("busy", False)), 4)
    beds_bucket = min(int(obs.beds_available), 4)

    if waiting:
        top = waiting[0]
        top_priority = str(top.get("priority", "none"))
        top_wait_bucket = min(int(top.get("wait_minutes", 0)) // 5, 12)
    else:
        top_priority = "none"
        top_wait_bucket = 0

    return (
        queue_bucket,
        critical_bucket,
        free_docs_bucket,
        beds_bucket,
        top_priority,
        top_wait_bucket,
    )


def _sorted_waiting(waiting: Sequence[dict]) -> List[dict]:
    priority_rank = {"emergency": 0, "urgent": 1, "normal": 2}
    return sorted(
        waiting,
        key=lambda p: (
            priority_rank.get(p.get("priority", "normal"), 3),
            -int(p.get("severity_score", 0)),
            -int(p.get("wait_minutes", 0)),
            int(p.get("id", 0)),
        ),
    )


def candidate_actions(obs: HospitalObservation) -> List[HospitalAction]:
    waiting = _sorted_waiting(obs.waiting_patients)
    doctors = list(obs.doctors)
    free_docs = [d for d in doctors if not d.get("busy", False)]
    busy_docs = [d for d in doctors if d.get("busy", False) and d.get("current_patient_id")]

    actions: List[HospitalAction] = [HospitalAction(action_type="wait")]

    # Prioritize top waiting patients.
    for patient in waiting[:2]:
        actions.append(HospitalAction(action_type="prioritize", patient_id=int(patient["id"])))

    # Assign top patients to best matching doctors when available.
    if waiting and free_docs:
        for patient in waiting[:2]:
            req = patient.get("required_specialization")
            matching = [d for d in free_docs if d.get("specialization") == req]
            general = [d for d in free_docs if d.get("specialization") == "General"]
            preferred = matching or general or free_docs
            for doctor in preferred[:2]:
                actions.append(
                    HospitalAction(
                        action_type="assign",
                        patient_id=int(patient["id"]),
                        doctor_id=int(doctor["id"]),
                    )
                )

    # Consider discharging currently treated patients.
    for doctor in busy_docs[:2]:
        pid = doctor.get("current_patient_id")
        if pid is not None:
            actions.append(HospitalAction(action_type="discharge", patient_id=int(pid)))

    # Deduplicate while preserving order.
    unique: List[HospitalAction] = []
    seen: set[ActionKey] = set()
    for action in actions:
        key = action_key(action)
        if key not in seen:
            seen.add(key)
            unique.append(action)
    return unique


def select_action(
    q_table: QTable,
    state: StateKey,
    actions: Sequence[HospitalAction],
    epsilon: float,
    rng: random.Random,
) -> HospitalAction:
    if not actions:
        return HospitalAction(action_type="wait")

    if rng.random() < epsilon:
        return rng.choice(list(actions))

    best = max(actions, key=lambda a: q_table[state].get(action_key(a), 0.0))
    return best


def run_episode(
    env: HospitalQueueEnvironment,
    q_table: QTable,
    task_id: str,
    alpha: float,
    gamma: float,
    epsilon: float,
    rng: random.Random,
) -> Tuple[float, float, int]:
    obs = env.reset(task_id=task_id)
    done = False
    total_reward = 0.0
    total_loss = 0.0
    steps = 0

    while not done:
        state = encode_state(obs)
        actions = candidate_actions(obs)
        action = select_action(q_table, state, actions, epsilon, rng)
        key = action_key(action)

        next_obs = env.step(action)
        reward = float(next_obs.reward)
        done = bool(next_obs.done)

        next_state = encode_state(next_obs)
        next_actions = candidate_actions(next_obs)
        max_next = (
            max(q_table[next_state].get(action_key(a), 0.0) for a in next_actions)
            if (next_actions and not done)
            else 0.0
        )

        current_q = q_table[state].get(key, 0.0)
        td_target = reward + gamma * max_next
        td_error = td_target - current_q
        q_table[state][key] = current_q + alpha * td_error

        total_reward += reward
        total_loss += td_error * td_error
        steps += 1
        obs = next_obs

    mean_loss = (total_loss / steps) if steps else 0.0
    return total_reward, mean_loss, steps


def evaluate_policy(
    q_table: QTable,
    tasks: Iterable[str],
    episodes_per_task: int,
    seed: int,
    random_policy: bool,
) -> float:
    rng = random.Random(seed)
    env = HospitalQueueEnvironment()
    rewards: List[float] = []

    for task_id in tasks:
        for _ in range(episodes_per_task):
            obs = env.reset(task_id=task_id)
            done = False
            total = 0.0
            while not done:
                actions = candidate_actions(obs)
                if random_policy:
                    action = rng.choice(actions) if actions else HospitalAction(action_type="wait")
                else:
                    state = encode_state(obs)
                    action = select_action(q_table, state, actions, epsilon=0.0, rng=rng)
                obs = env.step(action)
                total += float(obs.reward)
                done = bool(obs.done)
            rewards.append(total)

    return sum(rewards) / len(rewards) if rewards else 0.0


def save_training_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(outdir: str, rewards: List[float], losses: List[float], trained_avg: float, random_avg: float) -> None:
    episodes = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, rewards, color="#0B6E4F", linewidth=1.8)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Training Reward Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "qlearning_reward_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, losses, color="#9A031E", linewidth=1.8)
    plt.xlabel("Episode")
    plt.ylabel("Mean TD Error Squared (Loss)")
    plt.title("Q-learning Training Loss Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "qlearning_loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    labels = ["Trained Q-policy", "Random policy"]
    values = [trained_avg, random_avg]
    colors = ["#0B6E4F", "#6C757D"]
    plt.bar(labels, values, color=colors)
    plt.ylabel("Average Total Reward")
    plt.title("Policy Comparison (same tasks)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "qlearning_policy_compare.png"), dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning policy on MedFlow.")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=0.35)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--outdir", type=str, default="outputs/evals")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    tasks = ["easy_small_clinic", "medium_busy_opd", "hard_mass_casualty"]
    env = HospitalQueueEnvironment()
    rng = random.Random(args.seed)
    q_table: QTable = defaultdict(dict)

    rewards: List[float] = []
    losses: List[float] = []
    rows: List[dict] = []

    for episode in range(1, args.episodes + 1):
        task_id = tasks[(episode - 1) % len(tasks)]
        t = (episode - 1) / max(1, args.episodes - 1)
        epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * t

        total_reward, mean_loss, steps = run_episode(
            env=env,
            q_table=q_table,
            task_id=task_id,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=epsilon,
            rng=rng,
        )

        rewards.append(total_reward)
        losses.append(mean_loss)
        rows.append(
            {
                "episode": episode,
                "task_id": task_id,
                "epsilon": round(epsilon, 4),
                "steps": steps,
                "total_reward": round(total_reward, 4),
                "mean_td_loss": round(mean_loss, 6),
            }
        )

    trained_avg = evaluate_policy(
        q_table=q_table,
        tasks=tasks,
        episodes_per_task=args.eval_episodes,
        seed=args.seed,
        random_policy=False,
    )
    random_avg = evaluate_policy(
        q_table=q_table,
        tasks=tasks,
        episodes_per_task=args.eval_episodes,
        seed=args.seed + 1,
        random_policy=True,
    )

    csv_path = os.path.join(args.outdir, "qlearning_training_metrics.csv")
    save_training_csv(csv_path, rows)
    save_plots(args.outdir, rewards, losses, trained_avg, random_avg)

    summary = {
        "episodes": args.episodes,
        "tasks": tasks,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "trained_policy_avg_reward": round(trained_avg, 4),
        "random_policy_avg_reward": round(random_avg, 4),
        "artifacts": {
            "csv": csv_path,
            "reward_curve": os.path.join(args.outdir, "qlearning_reward_curve.png"),
            "loss_curve": os.path.join(args.outdir, "qlearning_loss_curve.png"),
            "comparison": os.path.join(args.outdir, "qlearning_policy_compare.png"),
        },
    }

    summary_path = os.path.join(args.outdir, "qlearning_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Training completed.")
    print(f"Trained policy avg reward: {trained_avg:.3f}")
    print(f"Random policy avg reward : {random_avg:.3f}")
    print(f"Artifacts saved in: {args.outdir}")


if __name__ == "__main__":
    main()
