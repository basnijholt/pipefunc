from __future__ import annotations

from pathlib import Path

try:
    import versioningit
except ImportError:
    import importlib.metadata

    __version__ = importlib.metadata.version("pipefunc")
else:
    PROJECT_DIR = Path(__file__).parent.parent
    __version__ = versioningit.get_version(project_dir=PROJECT_DIR)
