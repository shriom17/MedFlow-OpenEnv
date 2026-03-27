# Baseline implementations
def run_baseline(env):
    state = env.reset()
    total_reward = 0

    done = False
    while not done:
        patient = max(state.patients, key=lambda x: x.severity)
        action = {"select_patient_id": patient.id}

        state, reward, done = env.step(action)
        total_reward += reward

    return total_reward