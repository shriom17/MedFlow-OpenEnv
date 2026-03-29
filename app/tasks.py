"""
Task catalogue for MedFlow-OpenEnv.

Task 1 (easy)   — Small clinic, 5 patients, 2 doctors, minimize wait time
Task 2 (medium) — Busy OPD, mixed priorities, resource constraints
Task 3 (hard)   — Mass casualty event, dynamic arrivals, specialist routing
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class Doctor:
    id: int
    name: str
    specialization: str          # General | Cardiologist | Ortho | Paediatrician | Surgeon
    busy: bool = False
    current_patient_id: Optional[int] = None
    patients_seen: int = 0
    time_free_at: int = 0        # sim-minute when this doctor becomes free


@dataclass
class Patient:
    id: int
    name: str
    age: int
    priority: str                # emergency | urgent | normal
    condition: str               # free-text condition description
    severity_score: int          # 1–10 (10 = most critical)
    required_specialization: str # General | Cardiologist | Ortho | Paediatrician | Surgeon
    arrival_minute: int          # sim-minute when patient arrived
    assigned_minute: Optional[int] = None
    discharged_minute: Optional[int] = None
    assigned_doctor_id: Optional[int] = None
    status: str = "waiting"      # waiting | in_treatment | discharged


@dataclass
class TaskConfig:
    task_id: str
    difficulty: str
    description: str
    initial_patients: List[Dict[str, Any]]
    doctors: List[Dict[str, Any]]
    total_beds: int
    max_steps: int
    dynamic_arrivals: List[Dict[str, Any]]   # patients that arrive mid-episode
    grader: Callable                         # (episode_stats) -> float
    hints: List[str]


# ─────────────────────────────────────────────
# GRADERS
# ─────────────────────────────────────────────

def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def easy_grader(stats: Dict[str, Any]) -> float:
    """
    Score based on:
      40% — all patients seen (throughput)
      40% — avg wait time (target: ≤ 10 min)
      20% — no emergency left waiting > 5 min
    """
    total = stats.get("total_arrived", 1)
    seen = stats.get("total_seen", 0)
    throughput = seen / total

    avg_wait = stats.get("avg_wait_minutes", 999)
    wait_score = _clamp(1.0 - (avg_wait / 20.0))   # perfect ≤0 min, zero ≥20 min

    emerg_violations = stats.get("emergency_wait_violations", 0)
    emerg_score = _clamp(1.0 - emerg_violations * 0.3)

    score = 0.4 * throughput + 0.4 * wait_score + 0.2 * emerg_score
    return round(_clamp(score), 3)


def medium_grader(stats: Dict[str, Any]) -> float:
    """
    Score based on:
      30% — throughput
      30% — avg wait (target: ≤ 15 min)
      25% — emergency response (target: ≤ 5 min)
      15% — specialization match rate
    """
    total = stats.get("total_arrived", 1)
    seen = stats.get("total_seen", 0)
    throughput = seen / total

    avg_wait = stats.get("avg_wait_minutes", 999)
    wait_score = _clamp(1.0 - (avg_wait / 30.0))

    emerg_times = stats.get("emergency_response_times", [])
    if emerg_times:
        avg_emerg = sum(emerg_times) / len(emerg_times)
        emerg_score = _clamp(1.0 - (avg_emerg / 10.0))
    else:
        emerg_score = 1.0

    spec_match = stats.get("specialization_match_rate", 1.0)

    score = (0.30 * throughput + 0.30 * wait_score
             + 0.25 * emerg_score + 0.15 * spec_match)
    return round(_clamp(score), 3)


def hard_grader(stats: Dict[str, Any]) -> float:
    """
    Score based on:
      25% — throughput under surge
      25% — avg wait (target: ≤ 20 min under surge)
      30% — emergency/critical response (target: ≤ 3 min)
      20% — zero critical deaths (severity≥9 untreated > 15 min)
    """
    total = stats.get("total_arrived", 1)
    seen = stats.get("total_seen", 0)
    throughput = seen / total

    avg_wait = stats.get("avg_wait_minutes", 999)
    wait_score = _clamp(1.0 - (avg_wait / 40.0))

    emerg_times = stats.get("emergency_response_times", [])
    if emerg_times:
        avg_emerg = sum(emerg_times) / len(emerg_times)
        emerg_score = _clamp(1.0 - (avg_emerg / 6.0))
    else:
        emerg_score = 0.5   # penalise if no emergencies processed

    critical_deaths = stats.get("critical_deaths", 0)
    death_score = _clamp(1.0 - critical_deaths * 0.25)

    score = (0.25 * throughput + 0.25 * wait_score
             + 0.30 * emerg_score + 0.20 * death_score)
    return round(_clamp(score), 3)


# ─────────────────────────────────────────────
# TASK 1 — EASY
# ─────────────────────────────────────────────

TASK_EASY = TaskConfig(
    task_id="easy_small_clinic",
    difficulty="easy",
    description=(
        "You are managing a small outpatient clinic. "
        "5 patients have arrived. You have 2 general doctors and 4 beds. "
        "Assign patients to available doctors, then discharge them when done. "
        "Goal: treat all patients as quickly as possible, "
        "ensuring emergency cases are seen within 5 minutes."
    ),
    initial_patients=[
        dict(id=1, name="Rahul Das",    age=34, priority="normal",    condition="fever and cough",      severity_score=3, required_specialization="General",  arrival_minute=0),
        dict(id=2, name="Priya Sen",    age=62, priority="emergency", condition="chest pain",            severity_score=9, required_specialization="General",  arrival_minute=0),
        dict(id=3, name="Amit Roy",     age=28, priority="normal",    condition="minor laceration",      severity_score=2, required_specialization="General",  arrival_minute=0),
        dict(id=4, name="Meena Ghosh",  age=45, priority="urgent",    condition="high fever 104°F",      severity_score=6, required_specialization="General",  arrival_minute=0),
        dict(id=5, name="Sanjay Paul",  age=71, priority="urgent",    condition="difficulty breathing",  severity_score=7, required_specialization="General",  arrival_minute=0),
    ],
    doctors=[
        dict(id=1, name="Dr. A. Roy",   specialization="General", treatment_minutes=8),
        dict(id=2, name="Dr. S. Ghosh", specialization="General", treatment_minutes=8),
    ],
    total_beds=4,
    max_steps=20,
    dynamic_arrivals=[],
    grader=easy_grader,
    hints=[
        "Assign emergency patients first — they are highest priority.",
        "Both doctors can see patients simultaneously.",
        "Discharge a patient once assigned to free up the doctor for the next one.",
    ],
)


# ─────────────────────────────────────────────
# TASK 2 — MEDIUM
# ─────────────────────────────────────────────

TASK_MEDIUM = TaskConfig(
    task_id="medium_busy_opd",
    difficulty="medium",
    description=(
        "You manage a busy OPD with mixed specializations. "
        "10 patients are waiting; 3 more arrive mid-shift. "
        "You have 4 doctors (General ×2, Cardiologist, Orthopaedist) and 6 beds. "
        "Route each patient to the correct specialist when possible. "
        "Minimize average wait time and ensure emergencies are seen within 5 minutes."
    ),
    initial_patients=[
        dict(id=1,  name="Alice Bose",    age=55, priority="emergency", condition="heart attack symptoms",   severity_score=10, required_specialization="Cardiologist", arrival_minute=0),
        dict(id=2,  name="Bob Mukherjee", age=40, priority="normal",    condition="back pain",               severity_score=3,  required_specialization="Ortho",        arrival_minute=0),
        dict(id=3,  name="Carol Dey",     age=28, priority="normal",    condition="sore throat",             severity_score=2,  required_specialization="General",       arrival_minute=0),
        dict(id=4,  name="David Pal",     age=67, priority="urgent",    condition="irregular heartbeat",     severity_score=7,  required_specialization="Cardiologist", arrival_minute=0),
        dict(id=5,  name="Eve Chakra",    age=35, priority="normal",    condition="knee injury",             severity_score=4,  required_specialization="Ortho",         arrival_minute=0),
        dict(id=6,  name="Frank Biswas",  age=50, priority="urgent",    condition="high blood pressure",     severity_score=6,  required_specialization="General",       arrival_minute=0),
        dict(id=7,  name="Grace Nath",    age=22, priority="normal",    condition="rash and itching",        severity_score=2,  required_specialization="General",       arrival_minute=0),
        dict(id=8,  name="Hari Banerjee", age=78, priority="urgent",    condition="shortness of breath",     severity_score=8,  required_specialization="Cardiologist", arrival_minute=0),
        dict(id=9,  name="Indira Mitra",  age=45, priority="normal",    condition="fractured wrist",         severity_score=5,  required_specialization="Ortho",         arrival_minute=0),
        dict(id=10, name="Jay Sarkar",    age=30, priority="normal",    condition="stomach ache",            severity_score=3,  required_specialization="General",       arrival_minute=0),
    ],
    doctors=[
        dict(id=1, name="Dr. B. Sen",       specialization="Cardiologist", treatment_minutes=12),
        dict(id=2, name="Dr. R. Bose",      specialization="Ortho",        treatment_minutes=10),
        dict(id=3, name="Dr. A. Roy",       specialization="General",      treatment_minutes=8),
        dict(id=4, name="Dr. S. Ghosh",     specialization="General",      treatment_minutes=8),
    ],
    total_beds=6,
    max_steps=40,
    dynamic_arrivals=[
        dict(id=11, name="Kamal Das",    age=60, priority="emergency", condition="stroke symptoms",       severity_score=10, required_specialization="General", arrival_minute=15),
        dict(id=12, name="Lata Roy",     age=33, priority="normal",    condition="mild fever",            severity_score=2,  required_specialization="General", arrival_minute=20),
        dict(id=13, name="Mohan Singh",  age=48, priority="urgent",    condition="severe abdominal pain", severity_score=7,  required_specialization="General", arrival_minute=25),
    ],
    grader=medium_grader,
    hints=[
        "Match patients to their required specialization for best outcomes.",
        "Emergency patients (severity 9–10) must be seen immediately.",
        "A General doctor can treat any patient if the specialist is busy.",
        "Discharge patients promptly to free beds for new arrivals.",
    ],
)


# ─────────────────────────────────────────────
# TASK 3 — HARD
# ─────────────────────────────────────────────

TASK_HARD = TaskConfig(
    task_id="hard_mass_casualty",
    difficulty="hard",
    description=(
        "A road accident has caused a mass casualty event. "
        "8 critical patients arrive immediately; 10 more arrive in waves. "
        "You have 5 doctors (General ×2, Surgeon, Cardiologist, Orthopaedist) and 8 beds. "
        "Severity ≥ 9 patients who wait > 15 minutes without treatment are counted as "
        "critical deaths — each one penalises your score by 25%. "
        "Triage ruthlessly: prioritize by severity, match specializations, "
        "and discharge patients as soon as treatment completes."
    ),
    initial_patients=[
        dict(id=1,  name="Victim 1",  age=32, priority="emergency", condition="internal bleeding",        severity_score=10, required_specialization="Surgeon",      arrival_minute=0),
        dict(id=2,  name="Victim 2",  age=45, priority="emergency", condition="cardiac arrest",            severity_score=10, required_specialization="Cardiologist", arrival_minute=0),
        dict(id=3,  name="Victim 3",  age=28, priority="emergency", condition="traumatic brain injury",    severity_score=9,  required_specialization="Surgeon",      arrival_minute=0),
        dict(id=4,  name="Victim 4",  age=60, priority="emergency", condition="multiple fractures",        severity_score=8,  required_specialization="Ortho",        arrival_minute=0),
        dict(id=5,  name="Victim 5",  age=19, priority="emergency", condition="spinal injury",             severity_score=9,  required_specialization="Surgeon",      arrival_minute=0),
        dict(id=6,  name="Victim 6",  age=55, priority="urgent",    condition="deep lacerations",          severity_score=7,  required_specialization="General",      arrival_minute=0),
        dict(id=7,  name="Victim 7",  age=38, priority="urgent",    condition="broken ribs",               severity_score=6,  required_specialization="General",      arrival_minute=0),
        dict(id=8,  name="Victim 8",  age=72, priority="urgent",    condition="concussion",                severity_score=6,  required_specialization="General",      arrival_minute=0),
    ],
    doctors=[
        dict(id=1, name="Dr. K. Sharma",   specialization="Surgeon",      treatment_minutes=15),
        dict(id=2, name="Dr. B. Sen",      specialization="Cardiologist", treatment_minutes=12),
        dict(id=3, name="Dr. R. Bose",     specialization="Ortho",        treatment_minutes=10),
        dict(id=4, name="Dr. A. Roy",      specialization="General",      treatment_minutes=8),
        dict(id=5, name="Dr. S. Ghosh",    specialization="General",      treatment_minutes=8),
    ],
    total_beds=8,
    max_steps=60,
    dynamic_arrivals=[
        dict(id=9,  name="Victim 9",  age=25, priority="emergency", condition="hemorrhagic shock",     severity_score=10, required_specialization="Surgeon",      arrival_minute=5),
        dict(id=10, name="Victim 10", age=40, priority="emergency", condition="heart arrhythmia",      severity_score=9,  required_specialization="Cardiologist", arrival_minute=5),
        dict(id=11, name="Victim 11", age=33, priority="urgent",    condition="pelvic fracture",       severity_score=7,  required_specialization="Ortho",        arrival_minute=10),
        dict(id=12, name="Victim 12", age=50, priority="urgent",    condition="severe burns",          severity_score=8,  required_specialization="General",      arrival_minute=10),
        dict(id=13, name="Victim 13", age=22, priority="normal",    condition="minor fracture",        severity_score=4,  required_specialization="Ortho",        arrival_minute=15),
        dict(id=14, name="Victim 14", age=65, priority="emergency", condition="aortic rupture",        severity_score=10, required_specialization="Surgeon",      arrival_minute=20),
        dict(id=15, name="Victim 15", age=48, priority="urgent",    condition="crush injury",          severity_score=7,  required_specialization="Surgeon",      arrival_minute=20),
        dict(id=16, name="Victim 16", age=30, priority="normal",    condition="lacerations",           severity_score=3,  required_specialization="General",      arrival_minute=25),
        dict(id=17, name="Victim 17", age=55, priority="urgent",    condition="respiratory distress",  severity_score=8,  required_specialization="General",      arrival_minute=30),
        dict(id=18, name="Victim 18", age=42, priority="emergency", condition="penetrating trauma",    severity_score=9,  required_specialization="Surgeon",      arrival_minute=35),
    ],
    grader=hard_grader,
    hints=[
        "Severity ≥ 9 + wait > 15 min = critical death penalty. Prioritize these above all.",
        "A Surgeon can handle most trauma; match specialization when possible.",
        "Discharge patients as soon as they finish treatment to free doctors for incoming critical cases.",
        "Use 'prioritize' action to move the most critical arriving patient to the front of the queue.",
        "When all specialist doctors are busy, assign a General doctor — partial treatment is better than none.",
    ],
)


ALL_TASKS: Dict[str, TaskConfig] = {
    t.task_id: t
    for t in [TASK_EASY, TASK_MEDIUM, TASK_HARD]
}