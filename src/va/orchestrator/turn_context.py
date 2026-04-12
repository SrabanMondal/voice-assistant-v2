import atexit
import multiprocessing as mp
import threading
from dataclasses import dataclass, field
from multiprocessing.managers import SyncManager
from typing import Protocol


class CancellationEventLike(Protocol):
    def set(self) -> None:
        ...

    def is_set(self) -> bool:
        ...


_MANAGER_LOCK = threading.Lock()
_MANAGER: SyncManager | None = None


def _get_manager() -> SyncManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = mp.Manager()
    return _MANAGER


def _shutdown_manager() -> None:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            return
        try:
            _MANAGER.shutdown()
        except Exception:
            pass
        finally:
            _MANAGER = None


def _new_cancel_event() -> CancellationEventLike:
    # Manager-backed event proxy is queue-picklable on spawn platforms.
    return _get_manager().Event()


atexit.register(_shutdown_manager)


@dataclass
class TurnContext:
    turn_id: int
    cancelled: CancellationEventLike = field(default_factory=_new_cancel_event)
