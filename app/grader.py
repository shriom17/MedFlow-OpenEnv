# Grading logic
class Grader:
    def __init__(self):
        self.total_reward = 0

    def add_reward(self, r):
        self.total_reward += r

    def get_score(self):
        return max(min(self.total_reward / 5, 1), 0)