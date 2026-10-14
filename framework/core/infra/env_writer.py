"""
Small helper for updating specific keys in the project's .env file from
the webapp (currently just the Settings screen's "Enable Langfuse" form).

Deliberately narrow: set_env_values() only ever touches the exact keys it's
given - for each key, it replaces that key's existing `KEY=...` line in
place if one exists (preserving every other line and its order/comments
untouched), or appends a new `KEY=value` line if the key isn't present yet.
It never reads the file into an object that could end up echoed back to a
caller/log - only line-by-line matching against the key names given.
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = _REPO_ROOT / ".env"


def _env_path() -> Path:
    override = os.environ.get("ENV_FILE_PATH", "").strip()
    return Path(override) if override else DEFAULT_ENV_PATH


def set_env_values(values: dict) -> None:
    """values: {KEY: value_str}. Updates/appends each key's line in .env,
    then reloads process env vars (os.environ) so the change is visible to
    the current process immediately, without a restart."""
    path = _env_path()
    lines = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    remaining = dict(values)
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if "=" not in stripped or stripped.strip().startswith("#"):
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            lines[i] = f"{key}={remaining.pop(key)}\n"

    if remaining:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        for key, val in remaining.items():
            lines.append(f"{key}={val}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Make the change visible to this process right away (os.environ, not
    # just the file) - callers that then re-read via load_dotenv(override=True)
    # (e.g. observability.refresh_langfuse_config()) pick this up immediately.
    for key, val in values.items():
        os.environ[key] = val
