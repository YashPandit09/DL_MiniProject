"""Project task runner: python run.py <task> [extra arguments for the task's command].

Windows has no `make`, so every reproducible step is a task here and the Makefile only
forwards to this script. Tasks are added as the development plan progresses (data, train,
experiments, figures, all).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

TASKS: dict[str, tuple[str, list[list[str]]]] = {
    "test": ("run the unit tests", [[PY, "-m", "pytest", "-q"]]),
    "check-env": (
        "versions, GPU and VRAM, determinism and evaluator-sized training speed",
        [[PY, "-m", "spacegen.env_check"]],
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="tasks:\n" + "\n".join(f"  {name:10s} {help}" for name, (help, _) in TASKS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="passed on to the task's last command")
    args = parser.parse_args(argv)

    commands = TASKS[args.task][1]
    for i, command in enumerate(commands):
        if i == len(commands) - 1:
            command = command + args.extra
        returncode = subprocess.run(command, cwd=ROOT).returncode
        if returncode != 0:
            return returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
