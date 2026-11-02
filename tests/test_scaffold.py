import importlib
import pytest

MODULES = ["data_io","geometry","gather_manager","velocity_analysis","picking",
           "qc","export","database","ui","visualization"]

@pytest.mark.parametrize("m", MODULES)
def test_subpackage_imports(m):
    importlib.import_module(f"velpick.{m}")

def test_core_packages_have_no_gui_dependency():
    """Checked in a clean interpreter (other tests may legitimately import Qt in this process)."""
    import pathlib
    import subprocess
    import sys
    import velpick
    src = str(pathlib.Path(velpick.__file__).resolve().parent.parent)
    code = (f"import sys; sys.path.insert(0, {src!r}); "
            "import velpick.velocity_analysis, velpick.data_io, velpick.geometry, velpick.gather_manager, "
            "velpick.picking, velpick.database, velpick.export; "
            "bad=[k for k in sys.modules if k.startswith(('PySide6','PyQt5','PyQt6','tkinter','pyqtgraph','matplotlib'))]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)
