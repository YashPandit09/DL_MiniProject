"""Repository paths, so no module depends on the current working directory.

Everything a run writes (data/, runs/, reports/) goes below OUTPUT_ROOT. That is the repository
itself, unless the environment variable SPACEGEN_OUTPUT names another folder: a regeneration
(`python run.py all`, T46) writes there, starts from nothing and leaves the saved results alone.
The configuration files are always read from the repository.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUT_ROOT = Path(os.environ["SPACEGEN_OUTPUT"]).resolve() if os.environ.get("SPACEGEN_OUTPUT") else REPO_ROOT
DATA_DIR = OUTPUT_ROOT / "data"
REPORTS_DIR = OUTPUT_ROOT / "reports"
RUNS_DIR = OUTPUT_ROOT / "runs"
