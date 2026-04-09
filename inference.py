"""Submission inference entrypoint with strict structured stdout logs.

Required env vars:
- API_BASE_URL
- MODEL_NAME
- API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

from app.env import HospitalQueueEnvironment
from app.models import HospitalAction, HospitalObservation


DEFAULT_TASKS = ["easy_small_clinic", "medium_busy_opd", "hard_mass_casualty"]
BENCHMARK_NAME = "medflow-openenv"
MAX_STEPS = 128
SUCCESS_SCORE_THRESHOLD = 0.10
PRIORITY_RANK = {"emergency": 0, "urgent": 1, "normal": 2}

load_dotenv()


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
        "Return ONLY one JSON object with keys action_type, patient_id, doctor_id.\\n"
        "Valid action_type: assign, prioritize, discharge, wait.\\n"
        "Current observation:\\n"
        f"time={obs.current_time_minutes}, beds_available={obs.beds_available}, queue_length={obs.queue_length}\\n"
        f"waiting_patients={json.dumps(top_waiting)}\\n"
        f"doctors={json.dumps(doctors)}"
    )


def _choose_action(client: OpenAI, model: str, obs: HospitalObservation) -> tuple[HospitalAction, str, Optional[str]]:
    prompt = _build_prompt(obs)
    try:
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
    except Exception as exc:
        # Fall back to wait while still keeping the run alive.
        return HospitalAction(action_type="wait"), "wait", str(exc)

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

    action = HospitalAction(action_type=action_type, patient_id=patient_id, doctor_id=doctor_id)
    if action_type == "assign":
        action_str = f"assign(patient_id={patient_id},doctor_id={doctor_id})"
    elif action_type == "prioritize":
        action_str = f"prioritize(patient_id={patient_id})"
    elif action_type == "discharge":
        action_str = f"discharge(patient_id={patient_id})"
    else:
        action_str = "wait"

    return action, action_str, None


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _single_line(text: Optional[str]) -> str:
    if not text:
        return "null"
    return str(text).replace("\n", " ").replace("\r", " ").strip() or "null"


def _action_to_str(action: HospitalAction) -> str:
    if action.action_type == "assign":
        return f"assign(patient_id={action.patient_id},doctor_id={action.doctor_id})"
    if action.action_type == "prioritize":
        return f"prioritize(patient_id={action.patient_id})"
    if action.action_type == "discharge":
        return f"discharge(patient_id={action.patient_id})"
    return "wait"


def _waiting_sorted(obs: HospitalObservation) -> List[Dict[str, Any]]:
    waiting = list(obs.waiting_patients)
    waiting.sort(
        key=lambda p: (
            PRIORITY_RANK.get(p.get("priority", "normal"), 2),
            -int(p.get("severity_score", 0)),
            -int(p.get("wait_minutes", 0)),
            int(p.get("id", 0)),
        )
    )
    return waiting


def _free_doctors(obs: HospitalObservation) -> List[Dict[str, Any]]:
    return [d for d in obs.doctors if not d.get("busy", False)]


def _completed_patient_ids(obs: HospitalObservation) -> List[int]:
    done_ids: List[int] = []
    for d in obs.doctors:
        pid = d.get("current_patient_id")
        if (not d.get("busy", False)) and pid is not None:
            done_ids.append(int(pid))
    return done_ids


def _best_doctor_id(patient: Dict[str, Any], free_docs: List[Dict[str, Any]]) -> Optional[int]:
    if not free_docs:
        return None
    required = patient.get("required_specialization")
    exact = [d for d in free_docs if d.get("specialization") == required]
    general = [d for d in free_docs if d.get("specialization") == "General"]
    pick = exact[0] if exact else (general[0] if general else free_docs[0])
    return int(pick["id"])


def _has_high_priority_waiting(obs: HospitalObservation) -> bool:
    for p in obs.waiting_patients:
        if p.get("priority") in {"emergency", "urgent"}:
            return True
    return False


def _preemptive_discharge_pid(obs: HospitalObservation) -> Optional[int]:
    """Pick a busy doctor's patient to discharge early when queue pressure is high."""
    busy_with_patient = [
        d for d in obs.doctors
        if d.get("busy", False) and d.get("current_patient_id") is not None
    ]
    if not busy_with_patient:
        return None
    # Prefer freeing the doctor that is farthest from becoming free.
    candidate = max(busy_with_patient, key=lambda d: int(d.get("free_in_minutes", 0)))
    return int(candidate["current_patient_id"])


def _heuristic_action(obs: HospitalObservation) -> HospitalAction:
    waiting = _waiting_sorted(obs)
    free_docs = _free_doctors(obs)
    idle_free_docs = [d for d in free_docs if d.get("current_patient_id") is None]
    completed_ids = _completed_patient_ids(obs)

    # Use truly idle free doctors first; this avoids losing track of completed patients.
    if waiting and idle_free_docs and obs.beds_available > 0:
        patient = waiting[0]
        doctor_id = _best_doctor_id(patient, idle_free_docs)
        if doctor_id is not None:
            return HospitalAction(
                action_type="assign",
                patient_id=int(patient["id"]),
                doctor_id=doctor_id,
            )

    # Discharge completed treatments before reusing those doctors for new assignments.
    if completed_ids:
        return HospitalAction(action_type="discharge", patient_id=completed_ids[0])

    # If there are no completed carry-overs, assign with any free doctor.
    if waiting and free_docs and obs.beds_available > 0:
        patient = waiting[0]
        doctor_id = _best_doctor_id(patient, free_docs)
        if doctor_id is not None:
            return HospitalAction(
                action_type="assign",
                patient_id=int(patient["id"]),
                doctor_id=doctor_id,
            )

    # Queue-pressure relief: avoid passive wait when high-priority cases are blocked.
    if waiting and not free_docs and _has_high_priority_waiting(obs):
        pid = _preemptive_discharge_pid(obs)
        if pid is not None:
            return HospitalAction(action_type="discharge", patient_id=pid)

    return HospitalAction(action_type="wait")


def _is_action_valid(obs: HospitalObservation, action: HospitalAction) -> bool:
    waiting_ids = {int(p["id"]) for p in obs.waiting_patients}
    free_doc_ids = {int(d["id"]) for d in obs.doctors if not d.get("busy", False)}
    completed_ids = set(_completed_patient_ids(obs))

    if action.action_type == "assign":
        return (
            action.patient_id in waiting_ids
            and action.doctor_id in free_doc_ids
            and obs.beds_available > 0
        )
    if action.action_type == "prioritize":
        return action.patient_id in waiting_ids
    if action.action_type == "discharge":
        return action.patient_id in completed_ids
    if action.action_type == "wait":
        return True
    return False


def _feedback_error(step_feedback: str) -> Optional[str]:
    feedback = (step_feedback or "").strip()
    if not feedback:
        return None
    error_markers = [
        "requires",
        "not found",
        "is not waiting",
        "cannot be discharged",
        "No beds available",
        "Unknown action_type",
        "busy",
        "Invalid",
    ]
    if any(m.lower() in feedback.lower() for m in error_markers):
        return feedback
    return None


def _run_task(task_id: str, seed: int, client: OpenAI, model_name: str) -> None:
    env = HospitalQueueEnvironment()
    rewards: List[float] = []
    steps = 0
    success = False
    score = 0.0

    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={model_name}", flush=True)

    try:
        obs = env.reset(task_id=task_id, seed=seed)
        done = bool(obs.done)

        while not done and steps < MAX_STEPS:
            action = _heuristic_action(obs)
            llm_action, _llm_action_str, llm_error = _choose_action(client, model_name, obs)
            model_error: Optional[str] = llm_error
            if _is_action_valid(obs, llm_action) and llm_action.action_type != "wait":
                action = llm_action

            action_str = _action_to_str(action)
            obs = env.step(action)

            reward = float(obs.reward or 0.0)
            done = bool(obs.done)
            error = model_error if model_error else _feedback_error(obs.step_feedback)

            steps += 1
            rewards.append(reward)

            error_text = _single_line(error)
            print(
                f"[STEP] step={steps} action={action_str} reward={reward:.2f} "
                f"done={_format_bool(done)} error={error_text}",
                flush=True,
            )

        final_reward = float(obs.reward or 0.0) if steps > 0 else 0.0
        score = max(0.0, min(1.0, final_reward))
        success = bool(done) and (score >= SUCCESS_SCORE_THRESHOLD)
    except Exception:
        success = False
    finally:
        try:
            env.close()
        except Exception:
            pass

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={_format_bool(success)} steps={steps} score={score:.2f} rewards={rewards_str}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedFlow inference with strict stdout format")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    args = parser.parse_args()

    api_base_url = os.environ["API_BASE_URL"]
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    local_image_name = os.getenv("LOCAL_IMAGE_NAME")
    api_key = os.environ["API_KEY"]

    # Kept for compatibility with templates that provide docker image metadata.
    _ = local_image_name

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    for task_id in args.tasks:
        _run_task(
            task_id=task_id,
            seed=args.seed,
            client=client,
            model_name=model_name,
        )


if __name__ == "__main__":
    main()
