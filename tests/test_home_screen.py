from __future__ import annotations

from app import _home_screen_state, _normalized_project_home_mode


def test_home_screen_state_returns_no_active_project_when_none_selected() -> None:
    state = _home_screen_state(_projects(), active_project_slug=None)

    assert state["show_home_screen"]
    assert state["active_project"] is None


def test_home_screen_state_returns_selected_active_project() -> None:
    state = _home_screen_state(_projects(), active_project_slug="beta")

    assert not state["show_home_screen"]
    assert state["active_project"] == {
        "name": "Beta",
        "slug": "beta",
        "project_path": "/tmp/projects/beta",
        "archived": False,
    }


def test_home_screen_state_hides_archived_projects_by_default() -> None:
    state = _home_screen_state(_projects(), active_project_slug=None)

    assert [project["slug"] for project in state["projects"]] == ["alpha", "beta"]


def test_normalized_project_home_mode_defaults_to_open() -> None:
    assert _normalized_project_home_mode(None) == "open"
    assert _normalized_project_home_mode("unknown") == "open"


def test_normalized_project_home_mode_accepts_known_entry_points() -> None:
    assert _normalized_project_home_mode("open") == "open"
    assert _normalized_project_home_mode("create") == "create"


def _projects() -> list[dict[str, object]]:
    return [
        {
            "name": "Alpha",
            "slug": "alpha",
            "project_path": "/tmp/projects/alpha",
            "archived": False,
        },
        {
            "name": "Beta",
            "slug": "beta",
            "project_path": "/tmp/projects/beta",
            "archived": False,
        },
        {
            "name": "Archived",
            "slug": "archived",
            "project_path": "/tmp/projects/archived",
            "archived": True,
        },
    ]
