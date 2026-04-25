from typing import Generic, TypeVar, Optional

TAction = TypeVar("TAction")
TObs = TypeVar("TObs")
TState = TypeVar("TState")


class EnvClient(Generic[TAction, TObs, TState]):
    """Minimal EnvClient shim used by MedFlow-OpenEnv when `openenv` is absent.

    Intended only to satisfy imports and act as a lightweight base class for
    the project's `HospitalQueueEnv` client wrapper.
    """

    _action_cls = None
    _observation_cls = None
    _state_cls = None

    def __init__(self, *args, **kwargs):
        # shim doesn't implement remote communication — override in real client
        self._connected = False

    def connect(self, *args, **kwargs) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def send_action(self, action: TAction) -> TObs:
        raise NotImplementedError("EnvClient shim does not implement send_action")

    def get_state(self) -> Optional[TState]:
        return None
