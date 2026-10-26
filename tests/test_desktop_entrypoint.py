from __future__ import annotations

from pathlib import Path

import pytest

from corn.desktop import DesktopArguments, main, parse_role, run_ui


def test_parse_role_defaults_to_launcher() -> None:
    assert parse_role([]) == DesktopArguments(role="launcher")


def test_parse_role_accepts_internal_ui() -> None:
    assert parse_role(["--internal-ui", "--port", "8123"]) == DesktopArguments(
        role="ui",
        port=8123,
    )


def test_parse_role_accepts_internal_worker() -> None:
    assert parse_role(
        ["--internal-worker", "--worker-id", "desktop-test"]
    ) == DesktopArguments(role="worker", worker_id="desktop-test")


def test_parse_role_accepts_internal_smoke_test() -> None:
    assert parse_role(["--internal-smoke-test"]) == DesktopArguments(role="smoke-test")


@pytest.mark.parametrize(
    "argv",
    [
        ["--unknown"],
        ["--internal-ui"],
        ["--internal-worker"],
        ["--internal-ui", "--internal-worker", "--port", "8123"],
    ],
)
def test_parse_role_rejects_invalid_internal_arguments(argv) -> None:
    with pytest.raises(SystemExit):
        parse_role(argv)


def test_run_ui_loads_loopback_streamlit_options_before_start(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setenv("CORN_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(
        "streamlit.web.bootstrap.load_config_options",
        lambda options: calls.append(("config", options)),
    )
    monkeypatch.setattr(
        "streamlit.web.bootstrap.run",
        lambda *args: calls.append(("run", args)),
    )

    result = run_ui(8123)

    expected_options = {
        "global_developmentMode": False,
        "server_address": "127.0.0.1",
        "server_port": 8123,
        "server_headless": True,
        "browser_gatherUsageStats": False,
    }
    assert result == 0
    assert calls[0] == ("config", expected_options)
    assert calls[1][0] == "run"
    assert calls[1][1][3] == expected_options


def test_main_dispatches_to_selected_handler(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("CORN_DATA_DIR", str(tmp_path))

    result = main(
        ["--internal-ui", "--port", "8123"],
        launcher_handler=lambda: calls.append(("launcher", None)) or 10,
        ui_handler=lambda port: calls.append(("ui", port)) or 11,
        worker_handler=lambda worker_id: calls.append(("worker", worker_id)) or 12,
        smoke_handler=lambda: calls.append(("smoke-test", None)) or 13,
    )

    assert result == 11
    assert calls == [("ui", 8123)]


def test_main_configures_platform_data_directory_when_missing(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    monkeypatch.delenv("CORN_DATA_DIR", raising=False)
    monkeypatch.setattr(
        "corn.desktop.user_data_path",
        lambda *args, **kwargs: tmp_path / "CORN",
    )

    main(
        ["--internal-smoke-test"],
        launcher_handler=lambda: 0,
        ui_handler=lambda port: 0,
        worker_handler=lambda worker_id: 0,
        smoke_handler=lambda: calls.append("smoke") or 0,
    )

    assert calls == ["smoke"]
    assert __import__("os").environ["CORN_DATA_DIR"] == str(
        (tmp_path / "CORN").resolve()
    )
