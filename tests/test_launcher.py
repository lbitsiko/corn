from __future__ import annotations

from corn.launcher import LauncherController, LauncherState, reconcile_process_status


class FakeSupervisor:
    def __init__(self):
        self.shutdown_calls = 0
        self.restart_calls = 0
        self.started = False
        self.url = "http://127.0.0.1:8123"

    def start(self):
        self.started = True
        return self.url

    def shutdown(self):
        self.shutdown_calls += 1

    def restart_worker(self):
        self.restart_calls += 1


def test_launcher_state_reaches_running_when_both_services_run() -> None:
    state = LauncherState().transition("worker", "running")
    state = state.transition("ui", "running")

    assert state.phase == "running"
    assert state.worker_status == "running"
    assert state.ui_status == "running"


def test_launcher_state_reports_worker_failure_without_losing_ui_state() -> None:
    state = LauncherState(ui_status="running", worker_status="running", phase="running")

    state = state.transition("worker", "failed", "Provider worker stopped")

    assert state.phase == "worker_failed"
    assert state.ui_status == "running"
    assert state.latest_error == "Provider worker stopped"


def test_launcher_state_reports_ui_failure() -> None:
    state = LauncherState(worker_status="running")

    state = state.transition("ui", "failed", "UI failed to start")

    assert state.phase == "ui_failed"
    assert state.latest_error == "UI failed to start"


def test_launcher_state_reaches_stopped_when_both_services_stop() -> None:
    state = LauncherState(ui_status="running", worker_status="running", phase="running")
    state = state.transition("ui", "stopped")
    state = state.transition("worker", "stopped")

    assert state.phase == "stopped"


def test_reconcile_process_status_detects_unexpected_worker_exit() -> None:
    state = LauncherState(
        phase="running",
        ui_status="running",
        worker_status="running",
        url="http://127.0.0.1:8123",
    )

    reconciled = reconcile_process_status(
        state,
        {"ui": "running", "worker": "failed"},
    )

    assert reconciled.phase == "worker_failed"
    assert reconciled.latest_error == "LLM worker exited unexpectedly"


def test_controller_starts_supervisor_and_reports_url() -> None:
    supervisor = FakeSupervisor()
    events = []
    controller = LauncherController(supervisor=supervisor, emit=events.append)

    controller.start_blocking()

    assert supervisor.started
    assert events[-1] == ("launcher", "running", supervisor.url)


def test_controller_stops_supervisor_only_once() -> None:
    supervisor = FakeSupervisor()
    events = []
    controller = LauncherController(supervisor=supervisor, emit=events.append)

    controller.stop()
    controller.stop()

    assert supervisor.shutdown_calls == 1
    assert events[-2:] == [
        ("ui", "stopped", None),
        ("worker", "stopped", None),
    ]


def test_controller_restarts_worker() -> None:
    supervisor = FakeSupervisor()
    events = []
    controller = LauncherController(supervisor=supervisor, emit=events.append)

    controller.restart_worker_blocking()

    assert supervisor.restart_calls == 1
    assert events[-1] == ("worker", "running", None)
