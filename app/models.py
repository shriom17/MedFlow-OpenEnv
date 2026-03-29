# Data models
"""
MedFlow-OpenEnv — OpenEnv typed models.

Action  : what the agent decides each step
Observation : what the agent sees after each decision
State   : full internal episode state (via client.state())
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation, State


# ─────────────────────────────────────────────
# ACTION
# ─────────────────────────────────────────────
class HospitalAction(Action):
    """
    One decision the agent makes per step.

    action_type
    -----------
    "assign"   — assign the next patient in triage to a doctor/bed
    "prioritize" — re-order the queue by moving a patient_id to front
    "discharge"  — mark a patient as done and free the bed/doctor
    "wait"       — do nothing this step (costs time, may be strategic)

    Fields
    ------
    patient_id : int | None
        Target patient (required for assign / prioritize / discharge).
    doctor_id : int | None
        Target doctor (required for assign).
    """
    action_type: str = "assign"      # assign | prioritize | discharge | wait
    patient_id: Optional[int] = None
    doctor_id: Optional[int] = None


# ─────────────────────────────────────────────
# OBSERVATION
# ─────────────────────────────────────────────
class HospitalObservation(Observation):
    """
    What the agent sees after each step.

    Fields
    ------
    waiting_patients : list[dict]
        Each dict: {id, name, age, priority, condition, wait_minutes, severity_score}
    doctors : list[dict]
        Each dict: {id, name, specialization, busy, current_patient_id, patients_seen}
    beds_available : int
    current_time_minutes : int   — minutes elapsed in the episode
    step_feedback : str          — human-readable result of last action
    queue_length : int
    avg_wait_minutes : float
    critical_untreated : int     — emergency patients still waiting
    task_id : str
    task_description : str
    progress_score : float       — 0.0–1.0 partial progress signal
    """
    waiting_patients: List[Dict[str, Any]] = []
    doctors: List[Dict[str, Any]] = []
    beds_available: int = 0
    current_time_minutes: int = 0
    step_feedback: str = ""
    queue_length: int = 0
    avg_wait_minutes: float = 0.0
    critical_untreated: int = 0
    task_id: str = ""
    task_description: str = ""
    progress_score: float = 0.0


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
class HospitalState(State):
    """Full internal episode state."""
    task_id: str = ""
    total_patients_seen: int = 0
    total_patients_arrived: int = 0
    emergency_response_times: List[float] = []
    avg_wait_all: float = 0.0
    submitted: bool = False
    final_score: Optional[float] = None
    task_description: str = ""