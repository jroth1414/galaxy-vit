"""T0.1 smoke test — confirms pytest discovery works on the bare scaffold.

Real per-task tests land alongside their owning task (see DEVPLAN.md §3 for the
test-file layout). This file exists only so `pytest -q` exits 0 at T0.1.
"""

from __future__ import annotations

import importlib


def test_T0_1_package_importable() -> None:
    """The galaxy_vit package imports and exposes a __version__ string."""
    pkg = importlib.import_module("galaxy_vit")
    assert isinstance(pkg.__version__, str)
    assert pkg.__version__  # non-empty
