"""Minimal training/demo runner for an OpenEnv-like environment.

Usage examples:
  python -m train.train_unsloth --env app.openenv_medflow --episodes 5 --demo
  python -m train.train_unsloth --env app/openenv_medflow --episodes 3

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
from typing import List


def import_env_module(env_path: str):
    # accept either module path (app.openenv_medflow) or file-like path (app/openenv_medflow)
    mod_path = env_path.replace("/", ".").rstrip(".py")
    return importlib.import_module(mod_path)


def run_demo(env_module, episodes: int = 5, demo: bool = True):
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/evals", exist_ok=True)

    total_rewards: List[float] = []

    for ep in range(1, episodes + 1):
        # instantiate environment: prefer make_env()
        if hasattr(env_module, "make_env"):
            env = env_module.make_env()
        else:
            env = getattr(env_module, env_module.__name__.split(".")[-1].title(), None)
            if env is None:
                # fallback: try class OpenEnvMedFlow
                env = getattr(env_module, "OpenEnvMedFlow")()
            else:
                env = env()

        obs = env.reset()
        done = False
        ep_reward = 0.0
        steps = []

        # simple action set for demo
        actions = [
            "gather_history",
            "order_blood_test",
            "order_xray",
            "schedule_followup",
            "prescribe_medication",
            "consult_specialist",
        ]

        while not done:
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
            steps.append({"action": a, "reward": r, "obs": obs})
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
