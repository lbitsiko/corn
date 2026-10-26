from pathlib import Path

from corn.registry import (
    archive_project,
    delete_project,
    list_projects,
    load_registry,
    register_project,
)


def test_load_registry_returns_empty_list_for_missing_file(tmp_path: Path) -> None:
    registry_path = tmp_path / "projects.yaml"

    assert load_registry(registry_path) == []


def test_register_project_persists_a_new_entry(tmp_path: Path) -> None:
    registry_path = tmp_path / "projects.yaml"

    entry = register_project(
        registry_path=registry_path,
        name="EU Qual",
        slug="eu-qual",
        project_path=tmp_path / "projects" / "eu-qual",
    )

    assert entry == {
        "name": "EU Qual",
        "slug": "eu-qual",
        "project_path": str(tmp_path / "projects" / "eu-qual"),
        "archived": False,
    }
    assert load_registry(registry_path) == [entry]


def test_archive_project_marks_entry_archived(tmp_path: Path) -> None:
    registry_path = tmp_path / "projects.yaml"
    register_project(
        registry_path=registry_path,
        name="EU Qual",
        slug="eu-qual",
        project_path=tmp_path / "projects" / "eu-qual",
    )

    archived = archive_project(registry_path, "eu-qual")

    assert archived["archived"]
    assert load_registry(registry_path)[0]["archived"]


def test_delete_project_removes_entry(tmp_path: Path) -> None:
    registry_path = tmp_path / "projects.yaml"
    register_project(
        registry_path=registry_path,
        name="EU Qual",
        slug="eu-qual",
        project_path=tmp_path / "projects" / "eu-qual",
    )

    delete_project(registry_path, "eu-qual")

    assert load_registry(registry_path) == []


def test_list_projects_excludes_archived_entries_by_default(tmp_path: Path) -> None:
    registry_path = tmp_path / "projects.yaml"
    active_entry = register_project(
        registry_path=registry_path,
        name="EU Qual",
        slug="eu-qual",
        project_path=tmp_path / "projects" / "eu-qual",
    )
    archived_entry = register_project(
        registry_path=registry_path,
        name="Archived Qual",
        slug="archived-qual",
        project_path=tmp_path / "projects" / "archived-qual",
    )
    archive_project(registry_path, "archived-qual")

    assert list_projects(registry_path) == [active_entry]
    assert list_projects(registry_path, include_archived=True) == [
        active_entry,
        {**archived_entry, "archived": True},
    ]
