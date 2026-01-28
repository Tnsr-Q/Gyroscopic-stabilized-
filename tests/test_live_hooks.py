import threading

import pytest

try:
    from law_tools.backend.live_hooks import LawEvent, LiveHookBus
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name == "h5py":
        pytest.skip("h5py is required to import live hook bus", allow_module_level=True)
    raise


def test_live_hook_bus_handles_callback_errors():
    bus = LiveHookBus()

    received = []
    done = threading.Event()

    def failing_callback(event):
        raise RuntimeError("boom")

    def successful_callback(event):
        received.append(event.name)
        done.set()

    bus.subscribe(failing_callback)
    bus.subscribe(successful_callback)

    bus.start()
    try:
        bus.publish(LawEvent(name="update", payload={}))
        assert done.wait(timeout=1.0)
    finally:
        bus.stop()

    assert received == ["update"]
