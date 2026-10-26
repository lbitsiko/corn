from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys


@dataclass(frozen=True)
class AppPaths:
    resource_dir: Path
    data_dir: Path
    logos_dir: Path
    projects_dir: Path
    registry_path: Path
    provider_settings_path: Path
    tie_options_path: Path
    raw_annotations_dir: Path
    legacy_annotations_db_path: Path
    guided_uploads_dir: Path
    logs_dir: Path

    @classmethod
    def from_roots(cls, *, resource_dir: Path, data_dir: Path) -> AppPaths:
        resource_root = resource_dir.expanduser().resolve()
        data_root = data_dir.expanduser().resolve()
        return cls(
            resource_dir=resource_root,
            data_dir=data_root,
            logos_dir=resource_root / "logos",
            projects_dir=data_root / "projects",
            registry_path=data_root / "project_registry.yaml",
            provider_settings_path=data_root / "provider_settings.json",
            tie_options_path=data_root / "tie_descriptions" / "tie_options.json",
            raw_annotations_dir=data_root / "raw_annotations",
            legacy_annotations_db_path=data_root / "annotations.sqlite",
            guided_uploads_dir=data_root / "_guided_uploads",
            logs_dir=data_root / "logs",
        )


def build_app_paths(*, resource_dir: Path | None = None) -> AppPaths:
    if resource_dir is None:
        resource_dir = Path(
            getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1])
        )
    resource_root = resource_dir.expanduser().resolve()
    configured_data_dir = os.environ.get("CORN_DATA_DIR")
    data_dir = (
        Path(configured_data_dir).expanduser()
        if configured_data_dir
        else resource_root / "data"
    )
    return AppPaths.from_roots(resource_dir=resource_root, data_dir=data_dir)


def initialize_data_directory(paths: AppPaths) -> None:
    for directory in (
        paths.data_dir,
        paths.projects_dir,
        paths.raw_annotations_dir,
        paths.guided_uploads_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if not paths.registry_path.exists():
        paths.registry_path.write_text("projects: []\n", encoding="utf-8")
