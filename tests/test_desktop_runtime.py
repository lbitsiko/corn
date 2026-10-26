from __future__ import annotations

from pathlib import Path
from urllib.error import URLError

import pytest

from corn.app_paths import AppPaths
from corn.desktop_runtime import (
    ProcessSupervisor,
    choose_local_port,
    desktop_environment,
    internal_command,
    redacted_environment_status,
    wait_for_health,
)


class FakeResponse:
    def __init__(self, status: int):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeProcess:
    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_choose_local_port_returns_available_loopback_port() -> None:
    port = choose_local_port()

    assert isinstance(port, int)
    assert 0 < port < 65536


def test_desktop_environment_does_not_mutate_or_expose_secrets(tmp_path: Path) -> None:
    source = {"GEMINI_API_KEY": "secret-value", "OTHER": "value"}

    result = desktop_environment(tmp_path, environ=source)

    assert source == {"GEMINI_API_KEY": "secret-value", "OTHER": "value"}
    assert result["CORN_DATA_DIR"] == str(tmp_path.resolve())
    assert redacted_environment_status(result) == {
        "GEMINI_API_KEY": True,
        "OPENROUTER_API_KEY": False,
    }
    assert "secret-value" not in repr(redacted_environment_status(result))


def test_internal_command_uses_python_module_in_source_mode(monkeypatch) -> None:
    monkeypatch.delattr("corn.desktop_runtime.sys.frozen", raising=False)
    monkeypatch.setattr("corn.desktop_runtime.sys.executable", "/python")

    command = internal_command("ui", "--port", "8501")

    assert command == [
        "/python",
        "-m",
        "corn.desktop",
        "--internal-ui",
        "--port",
        "8501",
    ]


def test_internal_command_uses_frozen_executable(monkeypatch) -> None:
    monkeypatch.setattr("corn.desktop_runtime.sys.frozen", True, raising=False)
    monkeypatch.setattr("corn.desktop_runtime.sys.executable", "/bundle/CORN")

    command = internal_command("worker", "--worker-id", "worker-1")

    assert command == [
        "/bundle/CORN",
        "--internal-worker",
        "--worker-id",
        "worker-1",
    ]


def test_wait_for_health_retries_connection_errors() -> None:
    responses = iter([URLError("not ready"), FakeResponse(503), FakeResponse(200)])
    sleeps = []

    def open_url(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    wait_for_health(
        "http://127.0.0.1:8501/_stcore/health",
        timeout=1,
        interval=0.01,
        urlopen=open_url,
        sleep=sleeps.append,
        monotonic=iter([0.0, 0.1, 0.2, 0.3]).__next__,
    )

    assert sleeps == [0.01, 0.01]


def test_wait_for_health_times_out() -> None:
    with pytest.raises(TimeoutError, match="did not become healthy"):
        wait_for_health(
            "http://127.0.0.1:8501/_stcore/health",
            timeout=0.1,
            interval=0,
            urlopen=lambda *args, **kwargs: (_ for _ in ()).throw(URLError("down")),
            sleep=lambda _: None,
            monotonic=iter([0.0, 0.2]).__next__,
        )


def _paths(tmp_path: Path) -> AppPaths:
    return AppPaths.from_roots(
        resource_dir=tmp_path / "resources",
        data_dir=tmp_path / "state",
    )


def test_supervisor_starts_worker_then_ui_and_opens_browser_after_health(
    tmp_path: Path,
) -> None:
    events = []
    processes = []

    def popen(command, **kwargs):
        events.append(
            (
                "process",
                command[3]
                if command[:3] == ["/python", "-m", "corn.desktop"]
                else command[1],
            )
        )
        process = FakeProcess(command, **kwargs)
        processes.append(process)
        return process

    supervisor = ProcessSupervisor(
        paths=_paths(tmp_path),
        popen=popen,
        health_waiter=lambda url, **kwargs: events.append(("healthy", url)),
        browser_open=lambda url: events.append(("browser", url)),
        port_selector=lambda: 8123,
        command_builder=lambda role, *args: [
            "/python",
            "-m",
            "corn.desktop",
            f"--internal-{role}",
            *args,
        ],
        environ={"GEMINI_API_KEY": "secret"},
    )

    url = supervisor.start()

    assert url == "http://127.0.0.1:8123"
    assert [event[0] for event in events] == [
        "process",
        "process",
        "healthy",
        "browser",
    ]
    assert events[0][1] == "--internal-worker"
    assert events[1][1] == "--internal-ui"
    assert processes[0].kwargs["env"]["CORN_DATA_DIR"] == str(
        (tmp_path / "state").resolve()
    )
    assert supervisor.status() == {"worker": "running", "ui": "running"}

    supervisor.shutdown()


def test_supervisor_cleans_up_partial_startup_failure(tmp_path: Path) -> None:
    processes = []

    def popen(command, **kwargs):
        process = FakeProcess(command, **kwargs)
        processes.append(process)
        return process

    supervisor = ProcessSupervisor(
        paths=_paths(tmp_path),
        popen=popen,
        health_waiter=lambda *args, **kwargs: (_ for _ in ()).throw(
            TimeoutError("not ready")
        ),
        browser_open=lambda _: None,
        port_selector=lambda: 8123,
        command_builder=lambda role, *args: [role, *args],
        environ={},
    )

    with pytest.raises(TimeoutError, match="not ready"):
        supervisor.start()

    assert all(process.terminated for process in processes)
    assert supervisor.status() == {"worker": "stopped", "ui": "stopped"}


def test_supervisor_restarts_only_worker_and_shutdown_is_idempotent(
    tmp_path: Path,
) -> None:
    processes = []

    def popen(command, **kwargs):
        process = FakeProcess(command, **kwargs)
        processes.append(process)
        return process

    supervisor = ProcessSupervisor(
        paths=_paths(tmp_path),
        popen=popen,
        health_waiter=lambda *args, **kwargs: None,
        browser_open=lambda _: None,
        port_selector=lambda: 8123,
        command_builder=lambda role, *args: [role, *args],
        environ={},
    )
    supervisor.start()
    original_worker, original_ui = processes

    supervisor.restart_worker()

    assert original_worker.terminated
    assert not original_ui.terminated
    assert len(processes) == 3

    supervisor.shutdown()
    supervisor.shutdown()

    assert original_ui.terminated
    assert processes[-1].terminated
