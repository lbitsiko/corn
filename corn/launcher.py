from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Callable
import webbrowser

from corn.app_paths import build_app_paths
from corn.desktop_runtime import ProcessSupervisor, redacted_environment_status


LauncherEvent = tuple[str, str, str | None]


@dataclass(frozen=True)
class LauncherState:
    phase: str = "starting"
    ui_status: str = "starting"
    worker_status: str = "starting"
    url: str | None = None
    latest_error: str = ""

    def transition(
        self,
        service: str,
        status: str,
        message: str | None = None,
    ) -> LauncherState:
        updates: dict[str, str | None] = {}
        if service == "ui":
            updates["ui_status"] = status
        elif service == "worker":
            updates["worker_status"] = status
        elif service == "launcher" and status == "running":
            updates["url"] = message
        if status == "failed" and message:
            updates["latest_error"] = message
        state = replace(self, **updates)
        if state.ui_status == "failed":
            phase = "ui_failed"
        elif state.worker_status == "failed":
            phase = "worker_failed"
        elif state.ui_status == state.worker_status == "running":
            phase = "running"
        elif state.ui_status == state.worker_status == "stopped":
            phase = "stopped"
        else:
            phase = "starting"
        return replace(state, phase=phase)


def reconcile_process_status(
    state: LauncherState,
    statuses: dict[str, str],
) -> LauncherState:
    result = state
    current_statuses = {
        "ui": state.ui_status,
        "worker": state.worker_status,
    }
    failure_messages = {
        "ui": "CORN UI exited unexpectedly",
        "worker": "LLM worker exited unexpectedly",
    }
    for service in ("ui", "worker"):
        status = statuses.get(service)
        if status is None or status == current_statuses[service]:
            continue
        message = failure_messages[service] if status == "failed" else None
        result = result.transition(service, status, message)
    return result


class LauncherController:
    def __init__(
        self,
        *,
        supervisor: ProcessSupervisor,
        emit: Callable[[LauncherEvent], None],
    ) -> None:
        self.supervisor = supervisor
        self._emit = emit
        self._stopped = False
        self._stop_lock = threading.Lock()

    def start_blocking(self) -> None:
        try:
            url = self.supervisor.start()
        except Exception as exc:
            self._emit(("ui", "failed", str(exc)))
            return
        self._emit(("launcher", "running", url))

    def restart_worker_blocking(self) -> None:
        try:
            self._emit(("worker", "starting", None))
            self.supervisor.restart_worker()
        except Exception as exc:
            self._emit(("worker", "failed", str(exc)))
            return
        self._emit(("worker", "running", None))

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
        self.supervisor.shutdown()
        self._emit(("ui", "stopped", None))
        self._emit(("worker", "stopped", None))


def _start_background(target: Callable[[], None]) -> None:
    threading.Thread(target=target, daemon=True).start()


def _open_folder(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def run_launcher() -> int:
    import tkinter as tk
    from tkinter import ttk

    paths = build_app_paths()
    events: queue.Queue[LauncherEvent] = queue.Queue()
    supervisor = ProcessSupervisor(
        paths=paths,
        status_callback=lambda service, status: events.put((service, status, None)),
    )
    controller = LauncherController(supervisor=supervisor, emit=events.put)
    state = LauncherState()

    root = tk.Tk()
    root.title("CORN")
    root.minsize(460, 300)
    root.columnconfigure(0, weight=1)

    frame = ttk.Frame(root, padding=20)
    frame.grid(sticky="nsew")
    frame.columnconfigure(0, weight=1)

    ttk.Label(frame, text="CORN", font=("TkDefaultFont", 20, "bold")).grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(frame, text="Corpus-Oriented Relationship Notation").grid(
        row=1, column=0, sticky="w", pady=(0, 16)
    )

    url_var = tk.StringVar(value="Starting local application…")
    ui_var = tk.StringVar(value="UI: starting")
    worker_var = tk.StringVar(value="LLM worker: starting")
    error_var = tk.StringVar(value="")
    key_status = redacted_environment_status(os.environ)
    keys_var = tk.StringVar(
        value=" · ".join(
            f"{name.removesuffix('_API_KEY').title()}: "
            f"{'available' if available else 'not set'}"
            for name, available in key_status.items()
        )
    )

    ttk.Label(frame, textvariable=url_var).grid(row=2, column=0, sticky="w")
    ttk.Label(frame, textvariable=ui_var).grid(
        row=3, column=0, sticky="w", pady=(12, 0)
    )
    ttk.Label(frame, textvariable=worker_var).grid(row=4, column=0, sticky="w")
    ttk.Label(frame, textvariable=keys_var).grid(
        row=5, column=0, sticky="w", pady=(8, 0)
    )
    ttk.Label(frame, textvariable=error_var, foreground="#9b1c1c", wraplength=420).grid(
        row=6, column=0, sticky="w", pady=(8, 12)
    )

    buttons = ttk.Frame(frame)
    buttons.grid(row=7, column=0, sticky="ew")

    def open_browser() -> None:
        if supervisor.url:
            webbrowser.open(supervisor.url)

    def restart_worker() -> None:
        _start_background(controller.restart_worker_blocking)

    def stop() -> None:
        controller.stop()
        root.destroy()

    ttk.Button(buttons, text="Open Browser", command=open_browser).grid(
        row=0, column=0, padx=(0, 6)
    )
    ttk.Button(
        buttons, text="Open Logs", command=lambda: _open_folder(paths.logs_dir)
    ).grid(row=0, column=1, padx=6)
    ttk.Button(buttons, text="Restart Worker", command=restart_worker).grid(
        row=0, column=2, padx=6
    )
    ttk.Button(buttons, text="Stop CORN", command=stop).grid(
        row=0, column=3, padx=(6, 0)
    )

    def poll_events() -> None:
        nonlocal state
        try:
            while True:
                service, status, message = events.get_nowait()
                state = state.transition(service, status, message)
        except queue.Empty:
            pass
        if state.url and state.phase != "stopped":
            state = reconcile_process_status(state, supervisor.status())
        url_var.set(state.url or "Starting local application…")
        ui_var.set(f"UI: {state.ui_status}")
        worker_var.set(f"LLM worker: {state.worker_status}")
        error_var.set(
            f"{state.latest_error} See {paths.logs_dir} for details."
            if state.latest_error
            else ""
        )
        if state.phase != "stopped":
            root.after(100, poll_events)

    root.protocol("WM_DELETE_WINDOW", stop)
    _start_background(controller.start_blocking)
    poll_events()
    root.mainloop()
    return 0
