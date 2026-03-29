"""OpenAI-model baseline runner for MedFlow-OpenEnv.

This script runs an LLM policy against all tasks using the in-process environment.
It is deterministic at the environment level via reset seed and low-temperature decoding.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from app.env import HospitalQueueEnvironment
from app.models import HospitalAction, HospitalObservation

TASK_IDS = ["easy_small_clinic", "medium_busy_opd", "hard_mass_casualty"]


def _extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _build_prompt(obs: HospitalObservation) -> str:
    top_waiting = obs.waiting_patients[:8]
    doctors = obs.doctors
    return (
        "You are controlling a hospital triage simulator. Return ONLY one JSON object with keys "
        'action_type, patient_id, doctor_id.\\n\\n'
        "Valid action_type values: assign, prioritize, discharge, wait.\\n"
        "Rules:\\n"
        "- Use assign when a doctor is free and a patient is waiting.\\n"
        "- Use discharge for in-treatment patients who can free capacity.\\n"
        "- Use wait when no better action is possible.\\n"
        "- Never output prose.\\n\\n"
        f"Current time: {obs.current_time_minutes}\\n"
        f"Beds available: {obs.beds_available}\\n"
        f"Queue length: {obs.queue_length}\\n"
        f"Top waiting patients: {json.dumps(top_waiting)}\\n"
        f"Doctors: {json.dumps(doctors)}\\n"
    )


def choose_action(client: OpenAI, model: str, obs: HospitalObservation) -> HospitalAction:
    prompt = _build_prompt(obs)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    text = response.output_text or ""
    payload = _extract_json_object(text)

    action_type = str(payload.get("action_type", "wait")).lower()
    if action_type not in {"assign", "prioritize", "discharge", "wait"}:
        action_type = "wait"

    patient_id = payload.get("patient_id")
    doctor_id = payload.get("doctor_id")

    try:
        patient_id = int(patient_id) if patient_id is not None else None
    except (TypeError, ValueError):
        patient_id = None

    try:
        doctor_id = int(doctor_id) if doctor_id is not None else None
    except (TypeError, ValueError):
        doctor_id = None

    return HospitalAction(action_type=action_type, patient_id=patient_id, doctor_id=doctor_id)


def run_episode(client: OpenAI, model: str, task_id: str, seed: int, max_steps: int = 128) -> Dict[str, Any]:
    env = HospitalQueueEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)

    shaped_total = 0.0
    steps = 0

    while not obs.done and steps < max_steps:
        action = choose_action(client, model, obs)
        obs = env.step(action)
        shaped_total += float(obs.reward or 0.0)
        steps += 1

    final_score = float(obs.reward or 0.0) if obs.done else 0.0
    return {
        "task_id": task_id,
        "steps": steps,
        "done": obs.done,
        "final_score": round(final_score, 3),
        "shaped_total_reward": round(shaped_total, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI baseline on MedFlow tasks")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", default=TASK_IDS)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    client = OpenAI(api_key=api_key)

    results: List[Dict[str, Any]] = []
    for task_id in args.tasks:
        result = run_episode(client, args.model, task_id=task_id, seed=args.seed)
        results.append(result)
        print(json.dumps(result, ensure_ascii=True))

    avg = sum(r["final_score"] for r in results) / len(results)
    summary = {
        "model": args.model,
        "seed": args.seed,
        "task_count": len(results),
        "avg_final_score": round(avg, 3),
    }
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
