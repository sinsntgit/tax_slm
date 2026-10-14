"""
Resolves the ONE .env file this whole project reads from, anchored to the
repo root by file location - NOT by process working directory.

Root cause this fixes: every module in this codebase called plain
load_dotenv() (no path argument). python-dotenv's default behavior for that
is to search upward from os.getcwd() - the directory the process happened to
be LAUNCHED from, not the location of app.py or this repo. Any launch method
that doesn't cd into the repo root first (a shortcut, a different working
directory, a process manager, an IDE run configuration with its own default
cwd, etc.) makes EVERY load_dotenv() call in the app silently find no .env
file and fall back to defaults for every setting - which is exactly what
produced the "Langfuse keys are in .env but the app says they're missing"
report: the file the user edited was never being read at all.

Fix: resolve the path relative to this file's own location (stable
regardless of cwd) and pass it explicitly to every load_dotenv() call in the
project.
"""
import os
from pathlib import Path

# This file lives at framework/core/infra/dotenv_path.py - three parents up
# is the repo root, same depth used by env_writer.py/response_config.py/
# settings_store.py for their own repo-root-anchored paths.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = _REPO_ROOT / ".env"


def env_path() -> Path:
    """Allows an explicit override (ENV_FILE_PATH) for anyone running this
    app against a non-standard .env location, same override env var
    env_writer.py already recognizes - falls back to the repo-root .env."""
    override = os.environ.get("ENV_FILE_PATH", "").strip()
    return Path(override) if override else DEFAULT_ENV_PATH
