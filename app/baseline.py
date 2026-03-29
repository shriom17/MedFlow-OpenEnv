"""Baseline greedy agent for Hospital Queue Management."""

from app.models import HospitalAction
from app.env import HospitalQueueEnvironment


def run_baseline(env: HospitalQueueEnvironment) -> float:
    """
    Greedy baseline strategy:
    - Prioritize emergency patients (highest priority first)
    - Assign to available doctor with matching specialization
    - Fall back to any available doctor if no specialist free
    - Wait if no doctors available
    
    Returns: total accumulated reward over the episode.
    """
    obs = env.reset()
    total_reward = 0.0
    
    while not obs.done:
        waiting = obs.waiting_patients
        
        if not waiting:
            # No patients in queue, wait
            action = HospitalAction(action_type="wait")
        else:
            # Greedily pick first waiting patient (sorted by priority)
            patient = waiting[0]
            pid = patient["id"]
            
            # Find available doctors
            free_docs = [d for d in obs.doctors if not d["busy"]]
            
            if not free_docs:
                # No free doctors, wait for next step
                action = HospitalAction(action_type="wait")
            else:
                # Prefer doctor matching specialization; else any available
                req_spec = patient["required_specialization"]
                matching_spec = [d for d in free_docs if d["specialization"] == req_spec]
                general = [d for d in free_docs if d["specialization"] == "General"]
                
                if matching_spec:
                    doctor = matching_spec[0]
                elif general:
                    doctor = general[0]
                else:
                    doctor = free_docs[0]
                
                # Assign patient to doctor
                action = HospitalAction(
                    action_type="assign",
                    patient_id=pid,
                    doctor_id=doctor["id"]
                )
        
        # Step environment
        obs = env.step(action)
        total_reward += obs.reward
    
    return round(total_reward, 3)