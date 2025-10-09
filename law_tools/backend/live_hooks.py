"""Asynchronous event hooks bridging the RCC solver and dashboards."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional


@dataclass
class LawEvent:
    """Simple event payload describing a solver update."""

    name: str
    payload: Dict[str, float]


class LiveHookBus:
    """Thread-safe publish/subscribe message bus for solver events."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[LawEvent]" = queue.Queue()
        self._subscribers: list[Callable[[LawEvent], None]] = []
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

    def publish(self, event: LawEvent) -> None:
        self._queue.put(event)

    def subscribe(self, callback: Callable[[LawEvent], None]) -> None:
        self._subscribers.append(callback)

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        def _run() -> None:
            while not self._stop.is_set():
                try:
                    event = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                for callback in list(self._subscribers):
                    callback(event)

        self._worker = threading.Thread(target=_run, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)


@dataclass
class LiveLawState:
    """Mutable container storing the latest law tensor snapshot."""

    gamma_trace: list[float] = field(default_factory=list)
    torsion_noise: float | None = None
    hysteresis_depth: float | None = None

    def update_from_tokens(self, tokens: Dict[str, float]) -> None:
        self.gamma_trace.append(tokens.get("LAW_GAMMA_LEVEL", 0.0))
        self.torsion_noise = tokens.get("LAW_TORSION_NOISE", self.torsion_noise)
        self.hysteresis_depth = tokens.get("LAW_HYSTERESIS_DEPTH", self.hysteresis_depth)


__all__ = ["LawEvent", "LiveHookBus", "LiveLawState"]
