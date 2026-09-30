"""Repository paths, so no module depends on the current working directory."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
RUNS_DIR = REPO_ROOT / "runs"
