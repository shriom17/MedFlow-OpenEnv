# Data models
"""
MedFlow-OpenEnv — OpenEnv typed models.

Action  : what the agent decides each step
Observation : what the agent sees after each decision
State   : full internal episode state (via client.state())
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
@dataclass
class HospitalAction:
    """One decision the agent makes per step."""

    action_type: str = "assign"      # assign | prioritize | discharge | wait
    patient_id: Optional[int] = None
    doctor_id: Optional[int] = None


# ─────────────────────────────────────────────
@dataclass
class HospitalObservation:
    """What the agent sees after each step."""

    done: bool = False
    reward: float = 0.0
    waiting_patients: List[Dict[str, Any]] = field(default_factory=list)
    doctors: List[Dict[str, Any]] = field(default_factory=list)
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
@dataclass
class HospitalState:
    """Full internal episode state."""

    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""
    total_patients_seen: int = 0
    total_patients_arrived: int = 0
    emergency_response_times: List[float] = field(default_factory=list)
    avg_wait_all: float = 0.0
    submitted: bool = False
    final_score: Optional[float] = None
    task_description: str = ""
