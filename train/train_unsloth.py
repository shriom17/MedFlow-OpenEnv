"""Minimal training/demo runner for an OpenEnv-like environment.

Usage examples:
    python -m train.train_unsloth --env app.env --episodes 5 --demo
    python -m train.train_unsloth --env app.openenv_medflow --episodes 3

This script will import the environment module, run episodes with simple
heuristic/random actions, save per-episode JSON logs to `outputs/logs/` and
try to produce a reward curve in `outputs/evals/reward_curve.png` if matplotlib
is available.
"""
import argparse
import importlib
import json
import os
import random
from typing import Any, Dict, List, Optional


def _load_hospital_action_cls():
    try:
        from app.models import HospitalAction

        return HospitalAction
    except Exception:
        return None


def _is_structured_env(obs: Any) -> bool:
    return hasattr(obs, "waiting_patients") and hasattr(obs, "doctors")


def _action_to_text(action: Any) -> str:
    if hasattr(action, "action_type"):
        action_type = getattr(action, "action_type", "wait")
        patient_id = getattr(action, "patient_id", None)
        doctor_id = getattr(action, "doctor_id", None)
        if action_type == "assign":
            return f"assign(patient_id={patient_id},doctor_id={doctor_id})"
        if action_type == "prioritize":
            return f"prioritize(patient_id={patient_id})"
        if action_type == "discharge":
            return f"discharge(patient_id={patient_id})"
        return "wait"
    return str(action)


def _serialize_obs(obs: Any) -> Dict[str, Any]:
    if _is_structured_env(obs):
        return {
            "done": bool(getattr(obs, "done", False)),
            "reward": float(getattr(obs, "reward", 0.0) or 0.0),
            "step_feedback": str(getattr(obs, "step_feedback", "")),
            "queue_length": int(getattr(obs, "queue_length", 0) or 0),
            "beds_available": int(getattr(obs, "beds_available", 0) or 0),
            "current_time_minutes": int(getattr(obs, "current_time_minutes", 0) or 0),
        }
    return {
        "done": bool(obs[2]) if isinstance(obs, tuple) and len(obs) > 2 else False,
        "reward": float(obs[1]) if isinstance(obs, tuple) and len(obs) > 1 else 0.0,
        "step_feedback": str(obs[3]) if isinstance(obs, tuple) and len(obs) > 3 else "",
    }


def _choose_structured_action(env, obs, hospital_action_cls):
    waiting = list(getattr(obs, "waiting_patients", []))
    doctors = list(getattr(obs, "doctors", []))
    free_docs = [d for d in doctors if not d.get("busy", False)]

    if waiting and free_docs and hospital_action_cls is not None:
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
        return hospital_action_cls(action_type="assign", patient_id=patient["id"], doctor_id=doctor["id"])

    if waiting and hospital_action_cls is not None:
        patient = sorted(
            waiting,
            key=lambda p: (
                0 if p.get("priority") == "emergency" else 1 if p.get("priority") == "urgent" else 2,
                int(p.get("wait_minutes", 0)),
                int(p.get("id", 0)),
            ),
        )[0]
        return hospital_action_cls(action_type="prioritize", patient_id=patient["id"])

    return hospital_action_cls(action_type="wait") if hospital_action_cls is not None else "wait"


def import_env_module(env_path: str):
    # accept either module path (app.env) or file-like path (app/env)
    mod_path = env_path.replace("/", ".").rstrip(".py")
    return importlib.import_module(mod_path)


def run_demo(env_module, episodes: int = 5, demo: bool = True):
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/evals", exist_ok=True)

    total_rewards: List[float] = []
    hospital_action_cls = _load_hospital_action_cls()

    for ep in range(1, episodes + 1):
        # instantiate environment: prefer make_env()
        if hasattr(env_module, "make_env"):
            env = env_module.make_env()
        else:
            env_cls = getattr(env_module, "HospitalQueueEnvironment", None)
            if env_cls is None:
                env_cls = getattr(env_module, "OpenEnvMedFlow", None)
            if env_cls is None:
                raise AttributeError(
                    f"Could not find a usable environment class in {env_module.__name__}"
                )
            env = env_cls()

        obs = env.reset(task_id="easy_small_clinic") if hasattr(env, "reset") else env.reset()
        done = False
        ep_reward = 0.0
        steps = []

        structured = _is_structured_env(obs)
        # simple action set for the legacy toy env
        actions = [
            "gather_history",
            "order_blood_test",
            "order_xray",
            "schedule_followup",
            "prescribe_medication",
            "consult_specialist",
        ]

        while not done:
            if structured:
                a = _choose_structured_action(env, obs, hospital_action_cls)
                obs = env.step(a)
                r = float(getattr(obs, "reward", 0.0) or 0.0)
                done = bool(getattr(obs, "done", False))
                steps.append({"action": _action_to_text(a), "reward": r, "obs": _serialize_obs(obs)})
                ep_reward += r
                continue

            if demo:
                # deterministic-ish heuristic: pick by step
                a = actions[min(len(steps), len(actions) - 1)]
            else:
                a = random.choice(actions)
            obs, r, done, info = env.step(a)
            # if a grader is available, use it to override/adjust the reward
            try:
                from app import grader

                r = float(grader.grade_action(a, obs))
            except Exception:
                # no grader present or error; keep env reward
                pass
            steps.append({"action": a, "reward": r, "obs": _serialize_obs((obs, r, done, info))})
            ep_reward += float(r)

        total_rewards.append(ep_reward)

        # save episode log
        out_path = f"outputs/logs/episode_{ep:04d}.json"
        # compute episode-level metrics if grader available
        episode_meta = {}
        try:
            from app import grader

            episode_meta = grader.grade_episode(steps, getattr(env, "state", {}))
        except Exception:
            episode_meta = {"total_reward": ep_reward, "num_steps": len(steps)}

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"episode": ep, "total_reward": ep_reward, "steps": steps, "meta": episode_meta}, fh, indent=2)

        print(f"Episode {ep} finished — total_reward={ep_reward:.2f} saved to {out_path}")

    # try to plot reward curve
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(range(1, len(total_rewards) + 1), total_rewards, marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward Curve")
        png_path = "outputs/evals/reward_curve.png"
        plt.savefig(png_path)
        plt.close()
        print(f"Saved reward curve to {png_path}")
    except Exception:
        txt_path = "outputs/evals/reward_curve.txt"
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join([str(r) for r in total_rewards]))
        print(f"matplotlib not available; wrote rewards to {txt_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, help="Environment module path (module or slash path)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    env_module = import_env_module(args.env)
    run_demo(env_module, episodes=args.episodes, demo=args.demo)


if __name__ == "__main__":
    main()
