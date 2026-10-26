from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules


ROOT = Path(SPECPATH).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corn.packaging_manifest import RESOURCE_PATHS, validate_resource_manifest

validate_resource_manifest(RESOURCE_PATHS)

streamlit_datas, streamlit_binaries, streamlit_hiddenimports = collect_all("streamlit")
project_datas = [
    (str(ROOT / "app.py"), "."),
    (str(ROOT / "logos"), "logos"),
    (
        str(ROOT / "corn" / "components" / "document_selector" / "frontend"),
        "corn/components/document_selector/frontend",
    ),
    (str(ROOT / "examples" / "demo"), "examples/demo"),
]

analysis = Analysis(
    [str(ROOT / "corn_desktop.py")],
    pathex=[str(ROOT)],
    binaries=streamlit_binaries,
    datas=[*streamlit_datas, *project_datas],
    hiddenimports=[*streamlit_hiddenimports, *collect_submodules("corn")],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="CORN",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="CORN",
)
