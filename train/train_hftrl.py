"""HF-TRL training script for MedFlow-OpenEnv.

This script performs a lightweight, judge-rerunnable training pipeline:
1. Build a supervised dataset from environment rollouts (heuristic teacher)
2. Fine-tune a small causal LM with HF-TRL SFTTrainer
3. Evaluate the trained policy directly in the environment
4. Save real training artifacts (loss/reward plots + CSV + summary JSON)

Example:
    python -m train.train_hftrl --model sshleifer/tiny-gpt2 --epochs 1 --outdir outputs/evals

Generate/update Colab markdown helper:
    python -m train.train_hftrl --write-colab --out hftrl_colab.md
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from app.env import HospitalQueueEnvironment
from app.models import HospitalAction, HospitalObservation


ALLOWED_ACTIONS = {"assign", "prioritize", "discharge", "wait"}
TASKS = ["easy_small_clinic", "medium_busy_opd", "hard_mass_casualty"]


def val(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_action(data: Dict[str, Any]) -> HospitalAction:
    action_type = str(data.get("action_type", "wait")).lower()
    if action_type not in ALLOWED_ACTIONS:
        action_type = "wait"

    patient_id = _to_int_or_none(data.get("patient_id"))
    doctor_id = _to_int_or_none(data.get("doctor_id"))

    if action_type == "assign" and (patient_id is None or doctor_id is None):
        action_type, patient_id, doctor_id = "wait", None, None
    elif action_type in {"prioritize", "discharge"} and patient_id is None:
        action_type, patient_id, doctor_id = "wait", None, None
    elif action_type == "wait":
        patient_id, doctor_id = None, None

    return HospitalAction(action_type=action_type, patient_id=patient_id, doctor_id=doctor_id)


def heuristic_action(obs: HospitalObservation, rng: random.Random) -> HospitalAction:
    waiting = list(val(obs, "waiting_patients", []) or [])
    doctors = list(val(obs, "doctors", []) or [])
    free_docs = [d for d in doctors if not d.get("busy", False)]

    # Tiny exploration for more diverse supervised data.
    if rng.random() < 0.08:
        return HospitalAction(action_type="wait")

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
        return HospitalAction(
            action_type="assign",
            patient_id=int(patient["id"]),
            doctor_id=int(doctor["id"]),
        )

    if waiting:
        patient = sorted(
            waiting,
            key=lambda p: (
                0 if p.get("priority") == "emergency" else 1 if p.get("priority") == "urgent" else 2,
                -int(p.get("wait_minutes", 0)),
                int(p.get("id", 0)),
            ),
        )[0]
        return HospitalAction(action_type="prioritize", patient_id=int(patient["id"]))

    return HospitalAction(action_type="wait")


def make_prompt(obs: HospitalObservation) -> str:
    waiting = list(val(obs, "waiting_patients", []) or [])
    doctors = list(val(obs, "doctors", []) or [])

    waiting_small = [
        {
            "id": p.get("id"),
            "priority": p.get("priority"),
            "severity": p.get("severity_score"),
            "wait": p.get("wait_minutes"),
            "required_specialization": p.get("required_specialization"),
        }
        for p in waiting[:6]
    ]
    doctors_small = [
        {
            "id": d.get("id"),
            "specialization": d.get("specialization"),
            "busy": d.get("busy"),
            "current_patient_id": d.get("current_patient_id"),
        }
        for d in doctors[:6]
    ]

    payload = {
        "queue_length": int(val(obs, "queue_length", 0) or 0),
        "beds_available": int(val(obs, "beds_available", 0) or 0),
        "critical_untreated": int(val(obs, "critical_untreated", 0) or 0),
        "current_time_minutes": int(val(obs, "current_time_minutes", 0) or 0),
        "waiting_patients": waiting_small,
        "doctors": doctors_small,
    }

    return (
        "You are a hospital triage agent.\n"
        "Choose exactly one next action as JSON with keys: action_type, patient_id, doctor_id.\n"
        "Allowed action_type: assign, prioritize, discharge, wait.\n"
        "Observation:\n"
        f"{json.dumps(payload, ensure_ascii=True)}\n"
        "Return JSON only."
    )


def action_to_json(action: HospitalAction) -> str:
    data = {
        "action_type": action.action_type,
        "patient_id": action.patient_id,
        "doctor_id": action.doctor_id,
    }
    return json.dumps(data, ensure_ascii=True)


def build_supervised_rows(episodes_per_task: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []

    for task_id in TASKS:
        for _ in range(episodes_per_task):
            env = HospitalQueueEnvironment()
            obs = env.reset(task_id=task_id)
            done = False
            steps = 0
            max_steps = 70

            while not done and steps < max_steps:
                teacher = heuristic_action(obs, rng)
                prompt = make_prompt(obs)
                target = action_to_json(teacher)
                rows.append({"text": f"{prompt}\nAction:\n{target}"})

                obs = env.step(teacher)
                done = bool(val(obs, "done", False))
                steps += 1

    return rows


def _build_sft_trainer(model, tokenizer, dataset, args) -> Any:
    # Imported lazily so --write-colab works without ML dependencies.
    from transformers import TrainingArguments
    from trl import SFTTrainer

    errors: List[str] = []

    # Newer TRL path.
    try:
        from trl import SFTConfig

        sft_args = SFTConfig(
            output_dir=os.path.join(args.outdir, "trl_checkpoints"),
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            packing=False,
        )
        try:
            return SFTTrainer(model=model, train_dataset=dataset, args=sft_args, processing_class=tokenizer)
        except TypeError:
            return SFTTrainer(model=model, train_dataset=dataset, args=sft_args, tokenizer=tokenizer)
    except Exception as exc:
        errors.append(f"SFTConfig path failed: {exc}")

    # Backward-compatible TRL path.
    try:
        targs = TrainingArguments(
            output_dir=os.path.join(args.outdir, "trl_checkpoints"),
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
        )
        return SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=targs,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            packing=False,
        )
    except Exception as exc:
        errors.append(f"TrainingArguments path failed: {exc}")

    raise RuntimeError("Could not initialize SFTTrainer. " + " | ".join(errors))


def train_with_trl(rows: List[Dict[str, str]], args) -> Tuple[Any, Any, List[Tuple[int, float]]]:
    # Imported lazily so --write-colab does not require these packages.
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)

    dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)

    trainer = _build_sft_trainer(model, tokenizer, dataset, args)
    trainer.train()

    losses: List[Tuple[int, float]] = []
    for item in trainer.state.log_history:
        if "loss" in item and "step" in item:
            losses.append((int(item["step"]), float(item["loss"])))

    return model, tokenizer, losses


def parse_action_from_text(text: str) -> HospitalAction:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return HospitalAction(action_type="wait")

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return HospitalAction(action_type="wait")

    if not isinstance(data, dict):
        return HospitalAction(action_type="wait")

    return normalize_action(data)


def predict_action(model, tokenizer, obs: HospitalObservation, max_new_tokens: int) -> HospitalAction:
    prompt = make_prompt(obs) + "\nAction:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_tokens = generated[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return parse_action_from_text(text)


def evaluate_model_rewards(model, tokenizer, eval_episodes: int, max_new_tokens: int) -> List[float]:
    rewards: List[float] = []

    for task_id in TASKS:
        for _ in range(eval_episodes):
            env = HospitalQueueEnvironment()
            obs = env.reset(task_id=task_id)
            done = False
            total_reward = 0.0
            steps = 0
            max_steps = 90

            while not done and steps < max_steps:
                action = predict_action(model, tokenizer, obs, max_new_tokens=max_new_tokens)
                next_obs = env.step(action)
                total_reward += float(val(next_obs, "reward", 0.0) or 0.0)
                done = bool(val(next_obs, "done", False))
                obs = next_obs
                steps += 1

            rewards.append(total_reward)

    return rewards


def save_loss_curve(path: str, losses: List[Tuple[int, float]]) -> None:
    plt.figure(figsize=(8, 4.5))
    if losses:
        xs = [x for x, _ in losses]
        ys = [y for _, y in losses]
        plt.plot(xs, ys, color="#9A031E", linewidth=1.8)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("HF-TRL SFT Loss Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_reward_curve(path: str, rewards: List[float]) -> None:
    plt.figure(figsize=(8, 4.5))
    if rewards:
        xs = list(range(1, len(rewards) + 1))
        plt.plot(xs, rewards, marker="o", color="#0B6E4F", linewidth=1.8)
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Total Reward")
    plt.title("HF-TRL Policy Reward Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_loss_csv(path: str, losses: List[Tuple[int, float]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("step,loss\n")
        for step, loss in losses:
            fh.write(f"{step},{loss:.8f}\n")


def write_summary(path: str, args, rows_count: int, losses: List[Tuple[int, float]], rewards: List[float]) -> None:
    summary = {
        "trainer": "hf-trl-sft",
        "model": args.model,
        "dataset_rows": rows_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss_points": len(losses),
        "reward_episodes": len(rewards),
        "reward_avg": (sum(rewards) / len(rewards)) if rewards else 0.0,
        "artifacts": {
            "loss_curve": os.path.join(args.outdir, "trl_loss_curve.png"),
            "reward_curve": os.path.join(args.outdir, "trl_reward_curve.png"),
            "metrics_csv": os.path.join(args.outdir, "trl_training_metrics.csv"),
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def write_colab_template(output: Path) -> None:
    md = """
# MedFlow HF-TRL Training (Colab)

## 1) Install dependencies

```bash
pip install -q openenv-core==0.2.3 transformers datasets accelerate trl matplotlib
```

## 2) Clone repository

```bash
git clone <your-repo-url>
cd MedFlow-OpenEnv
```

## 3) Run training

```bash
python -m train.train_hftrl --model sshleifer/tiny-gpt2 --epochs 1 --outdir outputs/evals
```

## 4) Generated artifacts

- `outputs/evals/trl_loss_curve.png`
- `outputs/evals/trl_reward_curve.png`
- `outputs/evals/trl_training_metrics.csv`
- `outputs/evals/trl_summary.json`
"""
    output.write_text(md.strip() + "\n", encoding="utf-8")
    print(f"Wrote Colab template to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MedFlow policy with HF-TRL")
    parser.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--episodes-per-task", type=int, default=6)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="outputs/evals")
    parser.add_argument("--write-colab", action="store_true", help="Write/update colab markdown helper and exit")
    parser.add_argument("--out", type=Path, default=Path("hftrl_colab.md"), help="Output path for --write-colab")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.write_colab:
        write_colab_template(args.out)
        return

    os.makedirs(args.outdir, exist_ok=True)

    rows = build_supervised_rows(episodes_per_task=args.episodes_per_task, seed=args.seed)
    print(f"Built supervised dataset rows: {len(rows)}")

    model, tokenizer, losses = train_with_trl(rows, args)
    rewards = evaluate_model_rewards(model, tokenizer, eval_episodes=args.eval_episodes, max_new_tokens=args.max_new_tokens)

    loss_curve_path = os.path.join(args.outdir, "trl_loss_curve.png")
    reward_curve_path = os.path.join(args.outdir, "trl_reward_curve.png")
    metrics_csv_path = os.path.join(args.outdir, "trl_training_metrics.csv")
    summary_path = os.path.join(args.outdir, "trl_summary.json")

    save_loss_curve(loss_curve_path, losses)
    save_reward_curve(reward_curve_path, rewards)
    save_loss_csv(metrics_csv_path, losses)
    write_summary(summary_path, args, rows_count=len(rows), losses=losses, rewards=rewards)

    print("HF-TRL training finished.")
    print(f"Saved: {loss_curve_path}")
    print(f"Saved: {reward_curve_path}")
    print(f"Saved: {metrics_csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
