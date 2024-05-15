"""pipefunc.map: Modules that handle MapSpecs and its runs."""

from pipefunc.map._adaptive import create_learners
from pipefunc.map._filearray import FileArray
from pipefunc.map._mapspec import MapSpec
from pipefunc.map._run import load_outputs, run

__all__ = ["MapSpec", "run", "FileArray", "create_learners", "load_outputs"]
