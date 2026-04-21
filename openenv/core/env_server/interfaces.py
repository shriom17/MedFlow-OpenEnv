from typing import Protocol, Any


class Environment(Protocol):
    """Minimal Environment protocol used by the project when OpenEnv isn't installed.

    Implementations should provide `reset()`, `step(action)` and `close()`.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def reset(self, *args, **kwargs) -> Any:  # pragma: no cover - shim
        ...

    def step(self, action: Any, *args, **kwargs) -> Any:  # pragma: no cover - shim
        ...

    def close(self) -> None:  # pragma: no cover - shim
        ...
