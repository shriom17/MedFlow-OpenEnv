# Data models
"""
MedFlow-OpenEnv — OpenEnv typed models.

Action  : what the agent decides each step
Observation : what the agent sees after each decision
State   : full internal episode state (via client.state())
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
@dataclass
class HospitalAction:
    """One decision the agent makes per step."""

    action_type: str = "assign"      # assign | prioritize | discharge | wait
    patient_id: Optional[int] = None
    doctor_id: Optional[int] = None
    
    def model_dump(self, *args, **kwargs) -> dict:
        """Compatibility shim: return a serializable dict like pydantic's model_dump."""
        return asdict(self)

    @classmethod
    def model_validate(cls, obj):
        """Compatibility shim: accept dicts or similar and return a HospitalAction."""
        if isinstance(obj, cls):
            return obj
        if obj is None:
            return cls()
        if isinstance(obj, dict):
            return cls(**obj)
        if hasattr(obj, "model_dump"):
            return cls(**obj.model_dump())
        raise TypeError(f"Cannot validate object of type {type(obj)} as {cls.__name__}")


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
    
    def model_dump(self, *args, **kwargs) -> dict:
        """Compatibility shim: return a serializable dict like pydantic's model_dump."""
        return asdict(self)


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

    def model_dump(self, *args, **kwargs) -> dict:
        """Compatibility shim: return a serializable dict like pydantic's model_dump."""
        return asdict(self)
