"""Engines used in the experiments: our version at the end of each session, and Cloudict."""
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSIONS = {"v1": "session-1", "v2": "session-2", "v3": "session-3", "v4": "session-4"}
VERSIONS_DIR = ROOT / "experiments" / "versions"  # Git worktrees of the tags (not committed).
CLOUDICT = {"Windows": "cloudict.exe", "Darwin": "cloudict.app", "Linux": "cloudict.linux"}


def version_dir(name):
    """Engine folder of a tagged version, checked out on first use."""
    path = VERSIONS_DIR / VERSIONS[name]
    if not path.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(path), VERSIONS[name]],
                       cwd=ROOT, check=True, capture_output=True)
    return path / "engine"


def spec(name):
    """(command, working directory) of an engine: v1..v4, 'current' or 'cloudict'."""
    if name == "cloudict":
        folder = ROOT / "gui" / "engines"
        return [str(folder / CLOUDICT[platform.system()])], folder
    folder = ROOT / "engine" if name == "current" else version_dir(name)
    return [sys.executable, "main.py"], folder


# Engines must not write bytecode into the version worktrees.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
