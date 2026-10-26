from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping
from urllib.error import URLError
from urllib.request import urlopen as default_urlopen
import webbrowser

from corn.app_paths import AppPaths, initialize_data_directory


PROVIDER_KEY_NAMES = ("GEMINI_API_KEY", "OPENROUTER_API_KEY")
_INTERNAL_ROLE_FLAGS = {
    "ui": "--internal-ui",
    "worker": "--internal-worker",
    "smoke-test": "--internal-smoke-test",
}


def choose_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def desktop_environment(
    data_dir: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    result = dict(os.environ if environ is None else environ)
    result["CORN_DATA_DIR"] = str(data_dir.expanduser().resolve())
    return result


def redacted_environment_status(environ: Mapping[str, str]) -> dict[str, bool]:
    return {name: bool(environ.get(name)) for name in PROVIDER_KEY_NAMES}


def internal_command(role: str, *args: str) -> list[str]:
    try:
        role_flag = _INTERNAL_ROLE_FLAGS[role]
    except KeyError as exc:
        raise ValueError(f"Unknown desktop role: {role}") from exc
    if getattr(sys, "frozen", False):
        return [sys.executable, role_flag, *args]
    return [sys.executable, "-m", "corn.desktop", role_flag, *args]


def wait_for_health(
    url: str,
    *,
    timeout: float = 30.0,
    interval: float = 0.2,
    urlopen: Callable[..., Any] = default_urlopen,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    deadline = monotonic() + timeout
    while True:
        try:
            with urlopen(url, timeout=min(1.0, timeout)) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        if monotonic() >= deadline:
            raise TimeoutError(
                f"CORN did not become healthy within {timeout:g} seconds"
            )
        sleep(interval)


class ProcessSupervisor:
    def __init__(
        self,
        *,
        paths: AppPaths,
        popen: Callable[..., Any] = subprocess.Popen,
        health_waiter: Callable[..., None] = wait_for_health,
        browser_open: Callable[[str], Any] = webbrowser.open,
        port_selector: Callable[[], int] = choose_local_port,
        command_builder: Callable[..., list[str]] = internal_command,
        environ: Mapping[str, str] | None = None,
        status_callback: Callable[[str, str], None] | None = None,
        startup_timeout: float = 30.0,
        shutdown_timeout: float = 5.0,
    ) -> None:
        self.paths = paths
        self._popen = popen
        self._health_waiter = health_waiter
        self._browser_open = browser_open
        self._port_selector = port_selector
        self._command_builder = command_builder
        self._source_environ = environ
        self._status_callback = status_callback or (lambda _service, _status: None)
        self._startup_timeout = startup_timeout
        self._shutdown_timeout = shutdown_timeout
        self._stop_event = threading.Event()
        self._worker_process: Any | None = None
        self._ui_process: Any | None = None
        self._worker_log: Any | None = None
        self._ui_log: Any | None = None
        self._environment: dict[str, str] | None = None
        self._url: str | None = None

    @property
    def url(self) -> str | None:
        return self._url

    @property
    def environment_status(self) -> dict[str, bool]:
        return redacted_environment_status(self._environment or {})

    def start(self) -> str:
        if self._ui_process is not None or self._worker_process is not None:
            raise RuntimeError("CORN processes are already started")
        initialize_data_directory(self.paths)
        self._stop_event.clear()
        self._environment = desktop_environment(
            self.paths.data_dir,
            environ=self._source_environ,
        )
        self._worker_log = (self.paths.logs_dir / "worker.log").open(
            "a", encoding="utf-8"
        )
        self._ui_log = (self.paths.logs_dir / "ui.log").open("a", encoding="utf-8")
        port = self._port_selector()
        self._url = f"http://127.0.0.1:{port}"
        try:
            self._start_worker()
            ui_command = self._command_builder("ui", "--port", str(port))
            self._ui_process = self._start_process(ui_command, self._ui_log)
            self._status_callback("ui", "starting")
            if self._ui_process.poll() is not None:
                raise RuntimeError("CORN UI exited during startup")
            self._health_waiter(
                f"{self._url}/_stcore/health",
                timeout=self._startup_timeout,
            )
            self._status_callback("ui", "running")
            self._browser_open(self._url)
            return self._url
        except BaseException:
            self.shutdown()
            raise

    def restart_worker(self) -> None:
        if self._ui_process is None or self._environment is None:
            raise RuntimeError("CORN is not running")
        self._stop_process(self._worker_process)
        self._worker_process = None
        self._start_worker()

    def status(self) -> dict[str, str]:
        return {
            "worker": self._process_status(self._worker_process),
            "ui": self._process_status(self._ui_process),
        }

    def shutdown(self) -> None:
        self._stop_event.set()
        for process in (self._ui_process, self._worker_process):
            self._stop_process(process)
        self._ui_process = None
        self._worker_process = None
        self._close_logs()
        self._status_callback("ui", "stopped")
        self._status_callback("worker", "stopped")

    def _start_worker(self) -> None:
        worker_id = f"desktop-{socket.gethostname()}-{os.getpid()}"
        command = self._command_builder("worker", "--worker-id", worker_id)
        self._worker_process = self._start_process(command, self._worker_log)
        if self._worker_process.poll() is not None:
            raise RuntimeError("CORN worker exited during startup")
        self._status_callback("worker", "running")

    def _start_process(self, command: list[str], log_handle: Any) -> Any:
        return self._popen(
            command,
            env=self._environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

    def _stop_process(self, process: Any | None) -> None:
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=self._shutdown_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=self._shutdown_timeout)

    @staticmethod
    def _process_status(process: Any | None) -> str:
        if process is None:
            return "stopped"
        return "running" if process.poll() is None else "failed"

    def _close_logs(self) -> None:
        for handle_name in ("_ui_log", "_worker_log"):
            handle = getattr(self, handle_name)
            if handle is not None and not handle.closed:
                handle.close()
            setattr(self, handle_name, None)
