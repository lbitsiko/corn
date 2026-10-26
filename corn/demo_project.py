from __future__ import annotations

import yaml

from corn.app_paths import AppPaths
from corn.projects import create_and_register_project


DEMO_PROJECT_SLUG = "corn-demo"


def install_demo_project(paths: AppPaths) -> dict:
    demo_dir = paths.resource_dir / "examples" / "demo"
    config_path = demo_dir / "project.yaml"
    source_path = demo_dir / "documents.csv"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["source"] = {**config["source"], "path": str(source_path)}
    return create_and_register_project(
        paths.projects_dir,
        paths.registry_path,
        config,
    )
