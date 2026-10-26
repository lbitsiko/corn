from __future__ import annotations

from pathlib import Path

from corn.provider_settings import load_provider_settings, save_provider_settings


def test_provider_settings_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "provider_settings.json"

    save_provider_settings(
        path,
        {
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
            }
        },
    )

    assert load_provider_settings(path) == {
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
        }
    }
