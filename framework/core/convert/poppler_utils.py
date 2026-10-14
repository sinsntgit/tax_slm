"""
Shared helper for locating Poppler's CLI tools (pdftohtml, pdftotext) -
PROMOTED unchanged from tax_kg_poc/scripts/poppler_utils.py. PDF-to-text
extraction via Poppler is a generic document-conversion concern (any domain
pack converting PDFs needs it), not tax-specific, so it lives in
framework/core/convert rather than a domain pack's converters/.

Poppler is a system dependency, not a Python package - `pip install` can't
provide it. Set POPPLER_BIN_DIR in .env if it isn't on PATH (common on
Windows) - see that env var's use below.
"""
import os

from dotenv import load_dotenv

from framework.core.infra.dotenv_path import env_path as _env_path

# Anchored to the repo root by file location, not process cwd - see
# framework.core.infra.dotenv_path's docstring.
load_dotenv(_env_path())


def poppler_bin(name: str) -> str:
    bin_dir = os.environ.get("POPPLER_BIN_DIR")
    if bin_dir:
        exe_name = name + (".exe" if os.name == "nt" else "")
        candidate = os.path.join(bin_dir, exe_name)
        if os.path.exists(candidate):
            return candidate
    return name
