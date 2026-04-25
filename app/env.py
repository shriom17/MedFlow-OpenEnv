"""
MedFlow-OpenEnv — server-side Environment.

Simulation clock
----------------
Each call to step() advances the sim clock by SIM_MINUTES_PER_STEP (2 min).
Doctors free up automatically when their treatment time elapses.
Dynamic patients arrive when sim_clock reaches their arrival_minute.

Reward shaping
--------------
Step rewards come from a composable rubric plus small invalid-action penalties.
The default rubric rewards specialization matches and healthy bed utilization,
and penalizes emergency waiting time.
Final  — grader(episode_stats) on episode end
"""
from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from app.models import HospitalAction, HospitalObservation, HospitalState
from app.tasks import ALL_TASKS, Doctor, OpenEnvRubric, Patient, TaskConfig, default_rubric

SIM_MINUTES_PER_STEP = 2


class HospitalQueueEnvironment(Environment):
    """Hospital queue triage environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._episode_id: str = ""
        self._step_count: int = 0
        self._sim_clock: int = 0          # minutes elapsed in simulation
        self._current_task_id: str = "easy_small_clinic"
        self._task: Optional[TaskConfig] = None

        self._patients: Dict[int, Patient] = {}
        self._doctors: Dict[int, Doctor] = {}
        self._beds_total: int = 0
        self._beds_used: int = 0
        self._rubric: OpenEnvRubric = default_rubric()

        self._pending_arrivals: List[Dict] = []  # dynamic patients not yet arrived

        # episode statistics
        self._total_seen: int = 0
        self._emergency_response_times: List[float] = []
        self._wait_times: List[float] = []
        self._specialization_matches: List[bool] = []
        self._critical_deaths: int = 0
        self._submitted: bool = False
        self._final_score: Optional[float] = None

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _action_value(action: Any, field_name: str, default: Any = None) -> Any:
        if isinstance(action, dict):
            return action.get(field_name, default)
        return getattr(action, field_name, default)

    def _waiting(self) -> List[Patient]:
        q = [p for p in self._patients.values() if p.status == "waiting"]
        q.sort(key=lambda p: (
            0 if p.priority_boosted else 1,
            0 if p.priority == "emergency" else 1 if p.priority == "urgent" else 2,
            p.arrival_minute,
        ))
        return q

    def _free_doctors(self) -> List[Doctor]:
        return [d for d in self._doctors.values() if not d.busy]

    def _beds_available(self) -> int:
        return max(0, self._beds_total - self._beds_used)

    def _advance_clock(self) -> None:
        """Move clock forward; free doctors whose treatment has finished."""
        self._sim_clock += SIM_MINUTES_PER_STEP
        for doc in self._doctors.values():
            if doc.busy and self._sim_clock >= doc.time_free_at:
                doc.busy = False
                # patient is still in bed until agent discharges
                # (bed freed on discharge action)

        # dynamic arrivals
        newly = [a for a in self._pending_arrivals if a["arrival_minute"] <= self._sim_clock]
        for arr in newly:
            p = Patient(**{k: arr[k] for k in Patient.__dataclass_fields__ if k in arr})
            p.status = "waiting"
            self._patients[p.id] = p
        self._pending_arrivals = [a for a in self._pending_arrivals if a not in newly]

    def _check_critical_deaths(self) -> None:
        """Severity ≥ 9 patients waiting > 15 min without assignment = death."""
        for p in self._patients.values():
            if (p.status == "waiting"
                    and p.severity_score >= 9
                    and (self._sim_clock - p.arrival_minute) > 15):
                p.status = "discharged"   # remove from queue (counts as death)
                p.discharged_minute = self._sim_clock
                self._critical_deaths += 1

    def _specialization_ok(self, patient: Patient, doctor: Doctor) -> bool:
        return (patient.required_specialization == doctor.specialization
                or doctor.specialization == "General"
                or patient.required_specialization == "General")

    def _build_observation(
        self,
        reward: float,
        done: bool,
        feedback: str,
        progress: float,
    ) -> HospitalObservation:
        waiting = self._waiting()
        avg_wait = (
            sum(self._sim_clock - p.arrival_minute for p in waiting) / len(waiting)
            if waiting else 0.0
        )
        critical_untreated = sum(
            1 for p in waiting if p.priority == "emergency"
        )
        return HospitalObservation(
            done=done,
            reward=round(reward, 4),
            waiting_patients=[
                dict(
                    id=p.id,
                    name=p.name,
                    age=p.age,
                    priority=p.priority,
                    condition=p.condition,
                    wait_minutes=self._sim_clock - p.arrival_minute,
                    severity_score=p.severity_score,
                    required_specialization=p.required_specialization,
                )
                for p in waiting
            ],
            doctors=[
                dict(
                    id=d.id,
                    name=d.name,
                    specialization=d.specialization,
                    busy=d.busy,
                    current_patient_id=d.current_patient_id,
                    patients_seen=d.patients_seen,
                    free_in_minutes=max(0, d.time_free_at - self._sim_clock),
                )
                for d in self._doctors.values()
            ],
            beds_available=self._beds_available(),
            current_time_minutes=self._sim_clock,
            step_feedback=feedback,
            queue_length=len(waiting),
            avg_wait_minutes=round(avg_wait, 1),
            critical_untreated=critical_untreated,
            task_id=self._task.task_id,
            task_description=self._task.description,
            progress_score=round(progress, 3),
        )

    def _episode_stats(self) -> Dict[str, Any]:
        all_discharged = [p for p in self._patients.values() if p.status == "discharged" and p.assigned_minute is not None]
        avg_wait = (
            sum(p.assigned_minute - p.arrival_minute for p in all_discharged) / len(all_discharged)
            if all_discharged else 999.0
        )
        emerg = [p for p in all_discharged if p.priority == "emergency"]
        emerg_times = [p.assigned_minute - p.arrival_minute for p in emerg]
        emerg_violations = sum(1 for t in emerg_times if t > 5)
        spec_matches = [
            p.required_specialization == self._doctors[p.assigned_doctor_id].specialization
            or self._doctors[p.assigned_doctor_id].specialization == "General"
            for p in all_discharged
            if p.assigned_doctor_id
        ]
        spec_rate = sum(spec_matches) / len(spec_matches) if spec_matches else 1.0
        total_arrived = len(self._patients)
        return dict(
            total_arrived=total_arrived,
            total_seen=self._total_seen,
            avg_wait_minutes=avg_wait,
            emergency_response_times=emerg_times,
            emergency_wait_violations=emerg_violations,
            specialization_match_rate=spec_rate,
            critical_deaths=self._critical_deaths,
        )

    def _estimate_progress(self) -> float:
        total = len(self._patients) + len(self._pending_arrivals)
        if total == 0:
            return 1.0
        seen = self._total_seen
        base = seen / total
        # penalise critical untreated
        critical = sum(1 for p in self._patients.values()
                       if p.status == "waiting" and p.priority == "emergency")
        penalty = critical * 0.05
        return max(0.0, min(1.0, base - penalty))

    # ─────────────────────────────────────────
    # OpenEnv API
    # ─────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = "easy_small_clinic",
        **kwargs: Any,
    ) -> HospitalObservation:
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._sim_clock = 0
        self._total_seen = 0
        self._emergency_response_times = []
        self._wait_times = []
        self._specialization_matches = []
        self._critical_deaths = 0
        self._submitted = False
        self._final_score = None
        self._beds_used = 0

        # pick task
        selected_task_id = task_id or "easy_small_clinic"
        self._current_task_id = selected_task_id
        self._task = ALL_TASKS.get(selected_task_id)
        if self._task is None:
            raise ValueError("Invalid task_id")
        self._rubric = self._task.rubric or default_rubric()

        # build patients
        self._patients = {}
        for pd in self._task.initial_patients:
            p = Patient(**pd)
            self._patients[p.id] = p

        # dynamic arrivals (not yet here)
        self._pending_arrivals = copy.deepcopy(self._task.dynamic_arrivals)

        # build doctors
        self._doctors = {}
        for dd in self._task.doctors:
            d = Doctor(
                id=dd["id"],
                name=dd["name"],
                specialization=dd["specialization"],
            )
            d._treatment_minutes = dd["treatment_minutes"]  # store per-doctor
            self._doctors[d.id] = d

        self._beds_total = self._task.total_beds

        feedback = (
            f"Episode started — Task: {self._task.task_id} "
            f"(difficulty={self._task.difficulty}). "
            f"{len(self._patients)} patients waiting, "
            f"{len(self._doctors)} doctors available, "
            f"{self._beds_total} beds total."
        )
        return self._build_observation(0.0, False, feedback, 0.0)

    def step(
        self,
        action: HospitalAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HospitalObservation:
        if self._submitted:
            return self._build_observation(
                -0.1, True, "Episode already finished. Call reset().", self._final_score or 0.0
            )

        # Restore task context if it was lost unexpectedly.
        if self._task is None:
            self._task = ALL_TASKS.get(self._current_task_id)
            if self._task is None:
                raise ValueError("Invalid task_id")

        # If step() is called before reset(), initialize episode state from task.
        if not self._patients and not self._pending_arrivals:
            self.reset(task_id=self._task.task_id, episode_id=self._episode_id or None)

        self._step_count += 1
        self._advance_clock()
        self._check_critical_deaths()

        atype = str(self._action_value(action, "action_type", "wait")).lower()
        reward = 0.0
        feedback = ""
        spec_match = None
        action_valid = False

        current_emergency_wait = [
            self._sim_clock - p.arrival_minute
            for p in self._waiting()
            if p.priority == "emergency"
        ]

        rubric_state = {
            "emergency_wait_minutes": current_emergency_wait,
            "bed_utilization": (self._beds_used / self._beds_total) if self._beds_total else 0.0,
        }

        if atype == "assign":
            pid = self._action_value(action, "patient_id")
            did = self._action_value(action, "doctor_id")
            patient = self._patients.get(pid) if pid is not None else None
            doctor = self._doctors.get(did) if did is not None else None
            spec_match = self._specialization_ok(patient, doctor) if patient and doctor else False
            rubric_state["specialization_match"] = spec_match
            reward_delta, feedback, action_valid = self._do_assign(action)
            reward += reward_delta
        elif atype == "prioritize":
            reward_delta, feedback, action_valid = self._do_prioritize(action)
            reward += reward_delta
        elif atype == "discharge":
            reward_delta, feedback, action_valid = self._do_discharge(action)
            reward += reward_delta
        elif atype == "wait":
            feedback = f"Waiting. Sim clock: {self._sim_clock} min. {len(self._waiting())} patients in queue."
            action_valid = True
        else:
            feedback = f"Unknown action_type '{atype}'. Use assign | prioritize | discharge | wait."
            reward -= 0.05

        if action_valid:
            reward += self._rubric.score(rubric_state)

        # ── check episode end ──
        all_tasks_done = (
            not self._pending_arrivals
            and all(p.status == "discharged" for p in self._patients.values())
        )
        over_steps = self._step_count >= self._task.max_steps

        if all_tasks_done or over_steps:
            stats = self._episode_stats()
            final = self._task.grader(stats)
            self._final_score = final
            self._submitted = True
            feedback += (
                f"\n\nEpisode ended. Final score: {final:.3f}. "
                f"Seen: {stats['total_seen']}/{stats['total_arrived']}, "
                f"avg wait: {stats['avg_wait_minutes']:.1f} min, "
                f"critical deaths: {stats['critical_deaths']}."
            )
            return self._build_observation(final, True, feedback, final)

        progress = self._estimate_progress()
        return self._build_observation(reward, False, feedback, progress)

    @property
    def state(self) -> HospitalState:
        all_assigned = [p for p in self._patients.values() if p.assigned_minute is not None]
        avg_wait = (
            sum(p.assigned_minute - p.arrival_minute for p in all_assigned) / len(all_assigned)
            if all_assigned else 0.0
        )
        emerg_times = [
            p.assigned_minute - p.arrival_minute
            for p in all_assigned if p.priority == "emergency" and p.assigned_minute is not None
        ]
        return HospitalState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task.task_id if self._task else "",
            total_patients_seen=self._total_seen,
            total_patients_arrived=len(self._patients),
            emergency_response_times=emerg_times,
            avg_wait_all=round(avg_wait, 2),
            submitted=self._submitted,
            final_score=self._final_score,
            task_description=self._task.description if self._task else "",
        )

    def close(self) -> None:
        pass

    # ─────────────────────────────────────────
    # Action handlers
    # ─────────────────────────────────────────

    def _do_assign(self, action: HospitalAction):
        pid = self._action_value(action, "patient_id")
        did = self._action_value(action, "doctor_id")

        if pid is None or did is None:
            return -0.05, "assign requires patient_id and doctor_id.", False

        patient = self._patients.get(pid)
        if patient is None:
            return -0.05, f"Patient {pid} not found.", False
        if patient.status != "waiting":
            return -0.05, f"Patient {pid} is not waiting (status={patient.status}).", False

        doctor = self._doctors.get(did)
        if doctor is None:
            return -0.05, f"Doctor {did} not found.", False
        if doctor.busy:
            return -0.05, f"Doctor {doctor.name} is busy. Wait or choose another doctor.", False

        if self._beds_available() <= 0:
            return -0.05, "No beds available. Discharge a patient first.", False

        # assign
        wait_time = self._sim_clock - patient.arrival_minute
        spec_ok = self._specialization_ok(patient, doctor)
        treatment_mins = getattr(doctor, "_treatment_minutes", 8)

        patient.status = "in_treatment"
        patient.assigned_minute = self._sim_clock
        patient.assigned_doctor_id = did
        doctor.busy = True
        doctor.current_patient_id = pid
        doctor.time_free_at = self._sim_clock + treatment_mins
        doctor.patients_seen += 1
        self._beds_used += 1

        self._specialization_matches.append(spec_ok)

        # compute reward
        reward = 0.0
        if patient.priority == "emergency":
            if wait_time <= 5:
                reward += 0.15
            self._emergency_response_times.append(wait_time)
        elif patient.priority == "urgent":
            if wait_time <= 10:
                reward += 0.10
        else:
            reward += 0.05

        if not spec_ok:
            reward -= 0.10
            spec_msg = f" ⚠️  Specialization mismatch ({patient.required_specialization} needed, {doctor.specialization} assigned)."
        else:
            spec_msg = ""

        feedback = (
            f"Assigned {patient.name} ({patient.priority}, severity {patient.severity_score}) "
            f"→ {doctor.name} ({doctor.specialization}). "
            f"Wait was {wait_time} min.{spec_msg} "
            f"Treatment finishes in {treatment_mins} min."
        )
        return reward, feedback, True

    def _do_prioritize(self, action: HospitalAction):
        pid = self._action_value(action, "patient_id")
        if pid is None:
            return -0.05, "prioritize requires patient_id.", False
        patient = self._patients.get(pid)
        if patient is None:
            return -0.05, f"Patient {pid} not found.", False
        if patient.status != "waiting":
            return -0.05, f"Patient {pid} is not waiting.", False
        # Bump to emergency priority AND set arrival to very early
        patient.priority = "emergency"
        patient.priority_boosted = True
        feedback = f"Patient {patient.name} boosted to emergency priority and moved to front of queue."
        return 0.0, feedback, True

    def _do_discharge(self, action: HospitalAction):
        pid = self._action_value(action, "patient_id")
        if pid is None:
            return -0.05, "discharge requires patient_id.", False
        patient = self._patients.get(pid)
        if patient is None:
            return -0.05, f"Patient {pid} not found.", False
        if patient.status not in ("in_treatment", "waiting"):
            return -0.05, f"Patient {pid} cannot be discharged (status={patient.status}).", False

        # free the doctor if still assigned
        if patient.assigned_doctor_id:
            doc = self._doctors.get(patient.assigned_doctor_id)
            if doc and doc.current_patient_id == pid:
                doc.busy = False
                doc.current_patient_id = None

        patient.status = "discharged"
        patient.discharged_minute = self._sim_clock
        self._total_seen += 1
        self._beds_used = max(0, self._beds_used - 1)

        wait = (patient.assigned_minute or self._sim_clock) - patient.arrival_minute
        self._wait_times.append(wait)

        feedback = (
            f"Discharged {patient.name}. "
            f"Bed freed. Doctor is now available. "
            f"Total discharged: {self._total_seen}."
        )
        return 0.05, feedback, True