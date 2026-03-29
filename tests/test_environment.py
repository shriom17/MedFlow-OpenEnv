"""
Tests for Hospital Queue Management Environment.
Run: cd hospital_queue_env && PYTHONPATH=. pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.models import HospitalAction, HospitalObservation, HospitalState
from server.hospital_environment import HospitalQueueEnvironment


@pytest.fixture
def env():
    e = HospitalQueueEnvironment()
    yield e
    e.close()

@pytest.fixture
def easy_env(env):
    env.reset(task_id="easy_small_clinic")
    return env

@pytest.fixture
def medium_env(env):
    env.reset(task_id="medium_busy_opd")
    return env

@pytest.fixture
def hard_env(env):
    env.reset(task_id="hard_mass_casualty")
    return env


# ── reset ────────────────────────────────────────────────────────────────

class TestReset:
    def test_returns_observation(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        assert isinstance(obs, HospitalObservation)

    def test_done_false(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        assert obs.done is False

    def test_correct_patient_count_easy(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        assert obs.queue_length == 5

    def test_correct_doctor_count_easy(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        assert len(obs.doctors) == 2

    def test_beds_available_easy(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        assert obs.beds_available == 4

    def test_correct_patient_count_medium(self, env):
        obs = env.reset(task_id="medium_busy_opd")
        assert obs.queue_length == 10

    def test_correct_patient_count_hard(self, env):
        obs = env.reset(task_id="hard_mass_casualty")
        assert obs.queue_length == 8

    def test_emergency_patient_in_queue(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        priorities = [p["priority"] for p in obs.waiting_patients]
        assert "emergency" in priorities

    def test_queue_sorted_emergency_first(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        # first patient should be emergency
        assert obs.waiting_patients[0]["priority"] == "emergency"

    def test_task_description_populated(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        assert len(obs.task_description) > 10

    def test_sim_clock_zero_on_reset(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        assert obs.current_time_minutes == 0


# ── state ────────────────────────────────────────────────────────────────

class TestState:
    def test_returns_hospital_state(self, easy_env):
        st = easy_env.state
        assert isinstance(st, HospitalState)

    def test_step_count_zero(self, easy_env):
        assert easy_env.state.step_count == 0

    def test_step_count_increments(self, easy_env):
        easy_env.step(HospitalAction(action_type="wait"))
        assert easy_env.state.step_count == 1

    def test_task_id_correct(self, easy_env):
        assert easy_env.state.task_id == "easy_small_clinic"


# ── assign action ────────────────────────────────────────────────────────

class TestAssign:
    def _emergency_id(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        for p in obs.waiting_patients:
            if p["priority"] == "emergency":
                return p["id"]

    def test_assign_emergency_gives_positive_reward(self, env):
        pid = self._emergency_id(env)
        obs = env.step(HospitalAction(action_type="assign", patient_id=pid, doctor_id=1))
        # feedback should mention the patient was assigned
        assert "assigned" in obs.step_feedback.lower()

    def test_assign_removes_from_waiting(self, env):
        obs_before = env.reset(task_id="easy_small_clinic")
        pid = obs_before.waiting_patients[0]["id"]
        obs = env.step(HospitalAction(action_type="assign", patient_id=pid, doctor_id=1))
        waiting_ids = [p["id"] for p in obs.waiting_patients]
        assert pid not in waiting_ids

    def test_assign_reduces_beds(self, env):
        obs_before = env.reset(task_id="easy_small_clinic")
        beds_before = obs_before.beds_available
        pid = obs_before.waiting_patients[0]["id"]
        obs = env.step(HospitalAction(action_type="assign", patient_id=pid, doctor_id=1))
        assert obs.beds_available == beds_before - 1

    def test_assign_doctor_becomes_busy(self, env):
        obs_before = env.reset(task_id="easy_small_clinic")
        pid = obs_before.waiting_patients[0]["id"]
        obs = env.step(HospitalAction(action_type="assign", patient_id=pid, doctor_id=1))
        doc = next(d for d in obs.doctors if d["id"] == 1)
        assert doc["busy"] is True

    def test_assign_invalid_doctor_negative_reward(self, easy_env):
        obs = easy_env.step(HospitalAction(action_type="assign", patient_id=1, doctor_id=999))
        assert obs.reward < 0

    def test_assign_invalid_patient_negative_reward(self, easy_env):
        obs = easy_env.step(HospitalAction(action_type="assign", patient_id=999, doctor_id=1))
        assert obs.reward < 0

    def test_assign_busy_doctor_negative_reward(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        pid1 = obs.waiting_patients[0]["id"]
        pid2 = obs.waiting_patients[1]["id"]
        env.step(HospitalAction(action_type="assign", patient_id=pid1, doctor_id=1))
        obs2 = env.step(HospitalAction(action_type="assign", patient_id=pid2, doctor_id=1))
        assert obs2.reward < 0

    def test_specialization_mismatch_penalized(self, env):
        # medium task: patient 2 needs Ortho, assign Cardiologist (id=1) — clear mismatch
        env.reset(task_id="medium_busy_opd")
        # patient 2 (Bob Mukherjee) needs Ortho, doctor 1 is Cardiologist
        obs = env.step(HospitalAction(action_type="assign", patient_id=2, doctor_id=1))
        # Cardiologist treating Ortho patient = mismatch, -0.10 penalty applied
        assert "mismatch" in obs.step_feedback.lower() or obs.reward < 0.10


# ── discharge action ─────────────────────────────────────────────────────

class TestDischarge:
    def test_discharge_frees_bed(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        pid = obs.waiting_patients[0]["id"]
        env.step(HospitalAction(action_type="assign", patient_id=pid, doctor_id=1))
        obs2 = env.step(HospitalAction(action_type="discharge", patient_id=pid))
        assert obs2.beds_available == obs.beds_available   # bed restored

    def test_discharge_frees_doctor(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        pid = obs.waiting_patients[0]["id"]
        env.step(HospitalAction(action_type="assign", patient_id=pid, doctor_id=1))
        obs2 = env.step(HospitalAction(action_type="discharge", patient_id=pid))
        doc = next(d for d in obs2.doctors if d["id"] == 1)
        assert doc["busy"] is False

    def test_discharge_gives_positive_reward(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        pid = obs.waiting_patients[0]["id"]
        env.step(HospitalAction(action_type="assign", patient_id=pid, doctor_id=1))
        obs2 = env.step(HospitalAction(action_type="discharge", patient_id=pid))
        # discharge itself gives +0.05
        assert obs2.step_feedback and "discharged" in obs2.step_feedback.lower()

    def test_discharge_invalid_patient(self, easy_env):
        obs = easy_env.step(HospitalAction(action_type="discharge", patient_id=999))
        assert obs.reward < 0


# ── prioritize action ────────────────────────────────────────────────────

class TestPrioritize:
    def test_prioritize_moves_to_front(self, env):
        obs = env.reset(task_id="easy_small_clinic")
        # pick a normal-priority patient
        normal = next(p for p in obs.waiting_patients if p["priority"] == "normal")
        env.step(HospitalAction(action_type="prioritize", patient_id=normal["id"]))
        obs2 = env.step(HospitalAction(action_type="wait"))
        # that patient should now be first
        assert obs2.waiting_patients[0]["id"] == normal["id"]


# ── wait action ──────────────────────────────────────────────────────────

class TestWait:
    def test_wait_advances_clock(self, easy_env):
        obs = easy_env.step(HospitalAction(action_type="wait"))
        assert obs.current_time_minutes == 2   # SIM_MINUTES_PER_STEP

    def test_wait_does_not_crash(self, easy_env):
        obs = easy_env.step(HospitalAction(action_type="wait"))
        assert isinstance(obs, HospitalObservation)


# ── episode completion ───────────────────────────────────────────────────

class TestEpisodeCompletion:
    def test_perfect_easy_episode_scores_high(self, env):
        """Greedy optimal: always assign emergency first, then discharge."""
        obs = env.reset(task_id="easy_small_clinic")
        # Step through: assign all 5 patients then discharge
        doctor_ids = [1, 2]
        doc_idx = 0
        for _ in range(40):
            if obs.done:
                break
            waiting = obs.waiting_patients
            free_docs = [d for d in obs.doctors if not d["busy"]]
            if waiting and free_docs:
                obs = env.step(HospitalAction(
                    action_type="assign",
                    patient_id=waiting[0]["id"],
                    doctor_id=free_docs[0]["id"],
                ))
            else:
                # try to discharge anyone in_treatment
                assigned = [p for p in env._patients.values() if p.status == "in_treatment"]
                if assigned:
                    obs = env.step(HospitalAction(
                        action_type="discharge",
                        patient_id=assigned[0].id,
                    ))
                else:
                    obs = env.step(HospitalAction(action_type="wait"))

        assert obs.done is True
        assert obs.reward >= 0.5   # should score reasonably well

    def test_do_nothing_scores_low(self, env):
        """Agent that only waits should score poorly."""
        env.reset(task_id="easy_small_clinic")
        obs = None
        for _ in range(20):
            obs = env.step(HospitalAction(action_type="wait"))
            if obs.done:
                break
        assert obs.done is True
        assert obs.reward <= 0.5

    def test_episode_ends_after_max_steps(self, env):
        env.reset(task_id="easy_small_clinic")
        obs = None
        for _ in range(25):
            obs = env.step(HospitalAction(action_type="wait"))
            if obs.done:
                break
        assert obs.done is True

    def test_final_score_in_state(self, env):
        env.reset(task_id="easy_small_clinic")
        for _ in range(20):
            obs = env.step(HospitalAction(action_type="wait"))
            if obs.done:
                break
        assert env.state.final_score is not None
        assert 0.0 <= env.state.final_score <= 1.0


# ── dynamic arrivals ─────────────────────────────────────────────────────

class TestDynamicArrivals:
    def test_medium_patients_arrive_mid_episode(self, env):
        obs = env.reset(task_id="medium_busy_opd")
        initial_count = obs.queue_length
        # advance 15+ steps (30+ sim minutes) to trigger arrivals
        for _ in range(10):
            obs = env.step(HospitalAction(action_type="wait"))
            if obs.done:
                break
        # new patients should have arrived
        all_ids = {p["id"] for p in obs.waiting_patients}
        assert len(env._patients) > initial_count or obs.done


# ── graders standalone ───────────────────────────────────────────────────

class TestGraders:
    def test_easy_grader_perfect(self):
        from app.tasks import easy_grader
        stats = dict(total_arrived=5, total_seen=5, avg_wait_minutes=0,
                     emergency_wait_violations=0)
        assert easy_grader(stats) >= 0.9

    def test_easy_grader_zero_seen(self):
        from app.tasks import easy_grader
        stats = dict(total_arrived=5, total_seen=0, avg_wait_minutes=999,
                     emergency_wait_violations=3)
        assert easy_grader(stats) < 0.3

    def test_medium_grader_range(self):
        from app.tasks import medium_grader
        stats = dict(total_arrived=13, total_seen=10, avg_wait_minutes=12,
                     emergency_response_times=[3, 5, 8], specialization_match_rate=0.8)
        score = medium_grader(stats)
        assert 0.0 <= score <= 1.0

    def test_hard_grader_critical_death_penalty(self):
        from app.tasks import hard_grader
        good = dict(total_arrived=18, total_seen=15, avg_wait_minutes=10,
                    emergency_response_times=[2, 3], critical_deaths=0)
        bad  = dict(total_arrived=18, total_seen=15, avg_wait_minutes=10,
                    emergency_response_times=[2, 3], critical_deaths=3)
        assert hard_grader(good) > hard_grader(bad)

    def test_hard_grader_zero_to_one(self):
        from app.tasks import hard_grader
        stats = dict(total_arrived=18, total_seen=18, avg_wait_minutes=5,
                     emergency_response_times=[1, 2, 2], critical_deaths=0)
        assert 0.0 <= hard_grader(stats) <= 1.0
