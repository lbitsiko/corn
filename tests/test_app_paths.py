from corn.app_paths import AppPaths, build_app_paths, initialize_data_directory


def test_build_app_paths_uses_explicit_data_directory(tmp_path, monkeypatch):
    data_dir = tmp_path / "state"
    resource_dir = tmp_path / "resources"
    monkeypatch.setenv("CORN_DATA_DIR", str(data_dir))

    paths = build_app_paths(resource_dir=resource_dir)

    assert paths.resource_dir == resource_dir.resolve()
    assert paths.data_dir == data_dir.resolve()
    assert paths.projects_dir == data_dir.resolve() / "projects"
    assert paths.registry_path == data_dir.resolve() / "project_registry.yaml"
    assert paths.provider_settings_path == data_dir.resolve() / "provider_settings.json"
    assert paths.logs_dir == data_dir.resolve() / "logs"


def test_build_app_paths_defaults_to_repository_data_while_developing(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CORN_DATA_DIR", raising=False)

    paths = build_app_paths(resource_dir=tmp_path)

    assert paths.data_dir == tmp_path.resolve() / "data"


def test_initialize_data_directory_is_idempotent(tmp_path):
    paths = AppPaths.from_roots(
        resource_dir=tmp_path / "resources",
        data_dir=tmp_path / "state",
    )
    initialize_data_directory(paths)
    paths.registry_path.write_text("projects: []\n", encoding="utf-8")

    initialize_data_directory(paths)

    assert paths.registry_path.read_text(encoding="utf-8") == "projects: []\n"
    assert paths.projects_dir.is_dir()
    assert paths.guided_uploads_dir.is_dir()
    assert paths.logs_dir.is_dir()


def test_initialize_data_directory_does_not_create_legacy_files(tmp_path):
    paths = AppPaths.from_roots(
        resource_dir=tmp_path / "resources",
        data_dir=tmp_path / "state",
    )

    initialize_data_directory(paths)

    assert not paths.provider_settings_path.exists()
    assert not paths.legacy_annotations_db_path.exists()
