# Environment configuration
def step(self, action):
    patient = next((p for p in self.state.patients if p.id == action.select_patient_id), None)

    if not patient:
        return self.state, 0, False   # invalid action safe

    reward = self.calculate_reward(patient)

    self.state.patients.remove(patient)
    self.state.time += 1

    for p in self.state.patients:
        p.waiting_time += 1

    done = len(self.state.patients) == 0

    return self.state, reward, done