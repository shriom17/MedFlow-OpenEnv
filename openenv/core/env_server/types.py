from typing import TypedDict, Any


class Action(TypedDict, total=False):
    # minimal placeholder for action shape
    action_type: str
    payload: Any


class Observation(TypedDict, total=False):
    # placeholder fields used by the project
    done: bool
    reward: float
    step_feedback: str


class State(TypedDict, total=False):
    episode_id: str
    step_count: int
