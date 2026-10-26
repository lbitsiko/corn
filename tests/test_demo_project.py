from __future__ import annotations

from pathlib import Path

from corn.app_paths import AppPaths, initialize_data_directory
from corn.demo_project import DEMO_PROJECT_SLUG, install_demo_project
from corn.projects import get_project_status
from corn.registry import list_projects


def test_install_demo_project_creates_ready_project(tmp_path: Path) -> None:
    resource_dir = Path(__file__).resolve().parents[1]
    paths = AppPaths.from_roots(
        resource_dir=resource_dir,
        data_dir=tmp_path / "state",
    )
    initialize_data_directory(paths)

    entry = install_demo_project(paths)

    assert entry["slug"] == DEMO_PROJECT_SLUG
    assert get_project_status(paths.projects_dir / DEMO_PROJECT_SLUG) == "ready"
    assert [project["slug"] for project in list_projects(paths.registry_path)] == [
        DEMO_PROJECT_SLUG
    ]


def test_install_demo_project_refuses_to_replace_existing_project(
    tmp_path: Path,
) -> None:
    resource_dir = Path(__file__).resolve().parents[1]
    paths = AppPaths.from_roots(
        resource_dir=resource_dir,
        data_dir=tmp_path / "state",
    )
    initialize_data_directory(paths)
    install_demo_project(paths)

    try:
        install_demo_project(paths)
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Expected duplicate demo installation to fail")
