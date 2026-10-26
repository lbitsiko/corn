from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import socket
import subprocess
from typing import Callable, Sequence

from platformdirs import user_data_path

from corn.app_paths import build_app_paths, initialize_data_directory
from corn.desktop_runtime import (
    desktop_environment,
    internal_command,
    wait_for_health,
)
from corn.llm_worker import run_worker


@dataclass(frozen=True)
class DesktopArguments:
    role: str
    port: int | None = None
    worker_id: str | None = None


def parse_role(argv: Sequence[str] | None = None) -> DesktopArguments:
    parser = argparse.ArgumentParser(description="Run the CORN desktop application.")
    roles = parser.add_mutually_exclusive_group()
    roles.add_argument("--internal-ui", dest="role", action="store_const", const="ui")
    roles.add_argument(
        "--internal-worker", dest="role", action="store_const", const="worker"
    )
    roles.add_argument(
        "--internal-smoke-test",
        dest="role",
        action="store_const",
        const="smoke-test",
    )
    parser.set_defaults(role="launcher")
    parser.add_argument("--port", type=int)
    parser.add_argument("--worker-id")
    values = parser.parse_args(argv)

    if values.role == "ui" and values.port is None:
        parser.error("--internal-ui requires --port")
    if values.role == "worker" and not values.worker_id:
        parser.error("--internal-worker requires --worker-id")
    if values.role != "ui" and values.port is not None:
        parser.error("--port is valid only with --internal-ui")
    if values.role != "worker" and values.worker_id is not None:
        parser.error("--worker-id is valid only with --internal-worker")
    return DesktopArguments(
        role=values.role,
        port=values.port,
        worker_id=values.worker_id,
    )


def configure_desktop_data_directory() -> Path:
    configured = os.environ.get("CORN_DATA_DIR")
    data_dir = (
        Path(configured).expanduser()
        if configured
        else Path(user_data_path("CORN", appauthor=False))
    ).resolve()
    os.environ["CORN_DATA_DIR"] = str(data_dir)
    return data_dir


def run_ui(port: int) -> int:
    from streamlit.web import bootstrap

    paths = build_app_paths()
    initialize_data_directory(paths)
    streamlit_options = {
        "global_developmentMode": False,
        "server_address": "127.0.0.1",
        "server_port": port,
        "server_headless": True,
        "browser_gatherUsageStats": False,
    }
    bootstrap.load_config_options(streamlit_options)
    bootstrap.run(
        str(paths.resource_dir / "app.py"),
        False,
        [],
        streamlit_options,
    )
    return 0


def run_worker_role(worker_id: str) -> int:
    paths = build_app_paths()
    initialize_data_directory(paths)
    run_worker(
        registry_path=paths.registry_path,
        provider_settings_path=paths.provider_settings_path,
        worker_id=worker_id,
    )
    return 0


def run_smoke_test() -> int:
    paths = build_app_paths()
    initialize_data_directory(paths)
    required_resources = (
        paths.resource_dir / "app.py",
        paths.logos_dir,
        paths.resource_dir
        / "corn"
        / "components"
        / "document_selector"
        / "frontend"
        / "index.html",
        paths.resource_dir / "examples" / "demo" / "project.yaml",
        paths.resource_dir / "examples" / "demo" / "documents.csv",
    )
    missing = [str(path) for path in required_resources if not path.exists()]
    if missing:
        raise RuntimeError(f"Packaged CORN resources are missing: {', '.join(missing)}")

    run_worker(
        registry_path=paths.registry_path,
        provider_settings_path=paths.provider_settings_path,
        worker_id=f"smoke-{socket.gethostname()}-{os.getpid()}",
        once=True,
    )

    port = _choose_smoke_port()
    environment = desktop_environment(paths.data_dir)
    process = subprocess.Popen(
        internal_command("ui", "--port", str(port)),
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_health(
            f"http://127.0.0.1:{port}/_stcore/health",
            timeout=30,
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    return 0


def _choose_smoke_port() -> int:
    from corn.desktop_runtime import choose_local_port

    return choose_local_port()


def run_launcher() -> int:
    from corn.launcher import run_launcher as launch

    return launch()


def main(
    argv: Sequence[str] | None = None,
    *,
    launcher_handler: Callable[[], int] | None = None,
    ui_handler: Callable[[int], int] | None = None,
    worker_handler: Callable[[str], int] | None = None,
    smoke_handler: Callable[[], int] | None = None,
) -> int:
    arguments = parse_role(argv)
    configure_desktop_data_directory()
    launcher_handler = launcher_handler or run_launcher
    ui_handler = ui_handler or run_ui
    worker_handler = worker_handler or run_worker_role
    smoke_handler = smoke_handler or run_smoke_test

    if arguments.role == "ui":
        assert arguments.port is not None
        return ui_handler(arguments.port)
    if arguments.role == "worker":
        assert arguments.worker_id is not None
        return worker_handler(arguments.worker_id)
    if arguments.role == "smoke-test":
        return smoke_handler()
    return launcher_handler()


if __name__ == "__main__":
    raise SystemExit(main())
