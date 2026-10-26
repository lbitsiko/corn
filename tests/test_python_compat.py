from __future__ import annotations

from pathlib import Path


def test_quote_identifier_helpers_avoid_backslash_fstring_pattern() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    disallowed_fragment = 'identifier.replace("\\"", "\\"\\"")'

    for relative_path in ("corn/ingestion.py", "corn/querying.py"):
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        assert disallowed_fragment not in source
