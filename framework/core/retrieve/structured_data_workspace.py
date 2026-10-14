"""
Ephemeral, per-chat-session store for structured data files (.xlsx/.csv/
.json) attached to the Ask screen, per explicit request for a "structured
data query agent" that "queries the attached excel/csv/json". Same
architectural pattern as opinion_workspace.py's in-memory _WORKSPACES
registry - a module-level dict keyed by an opaque id, gone on process
restart, NEVER written to the permanent Neo4j graph or disk beyond a
transient parse pass.

WHY EPHEMERAL, SESSION-SCOPED (same reasoning as opinion_workspace.py's
module docstring): an attached spreadsheet is very plausibly real client
data (a transaction ledger, an asset schedule) - keeping it in-process
memory, keyed to one chat session, and never persisting it anywhere durable
is the same confidentiality-conscious tradeoff already made for uploaded
documents in the Opinion workspace. A user closing the chat or the app
restarting loses the attachment - that is the deliberate tradeoff, not a bug.

Parsing (pandas/openpyxl) happens ONCE, at attach time - the resulting
DataFrame(s) are kept in memory and reused for every question asked in that
chat afterward, not re-parsed per question.
"""
import json
import time
import uuid
from pathlib import Path

import pandas as pd

from framework.core.infra import step_logger

_ATTACHMENTS: dict[str, dict] = {}

# Hard cap on rows loaded into memory per attached file - a defensive limit,
# not a product decision about "how big a file can a user have" (a genuinely
# huge spreadsheet should be trimmed/pre-aggregated by the user before
# attaching it to a QUESTION-ANSWERING flow, which is not the same use case
# as a data warehouse). Keeps the in-memory footprint and the schema-summary
# prompt sent to the LLM bounded regardless of what gets uploaded.
MAX_ROWS_PER_SHEET = 50_000
MAX_SCHEMA_SAMPLE_ROWS = 5

SUPPORTED_EXTENSIONS = (".xlsx", ".xls", ".csv", ".json")


def _parse_file(filename: str, content: bytes) -> dict[str, "pd.DataFrame"]:
    """Returns {sheet_name: DataFrame} - a single-entry dict keyed "data" for
    CSV/JSON (which have no sheet concept), or one entry per sheet for
    Excel. Raises ValueError with a clear message for an unsupported
    extension or a file that fails to parse - the caller (the /api/ask/
    attach_file route) turns that into a 400, not a 500, since a bad upload
    is a user-input problem, not a server bug."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type {suffix!r} - only .xlsx/.xls/.csv/.json are supported.")

    import io
    buf = io.BytesIO(content)

    try:
        if suffix in (".xlsx", ".xls"):
            sheets = pd.read_excel(buf, sheet_name=None)  # dict of {sheet_name: DataFrame}
        elif suffix == ".csv":
            sheets = {"data": pd.read_csv(buf)}
        else:  # .json
            raw = json.loads(content.decode("utf-8", errors="replace"))
            # A JSON file may be a list of records ([{...}, {...}]) or a
            # single object with list-valued fields (json_normalize handles
            # both reasonably) - pd.json_normalize is more forgiving than
            # pd.read_json for hand-authored/export-tool JSON shapes.
            sheets = {"data": pd.json_normalize(raw)}
    except Exception as e:
        raise ValueError(f"Could not parse {filename!r} as {suffix}: {e}") from e

    if not sheets or all(df.empty for df in sheets.values()):
        raise ValueError(f"{filename!r} parsed but contains no data.")

    truncated_sheets = {}
    for name, df in sheets.items():
        if len(df) > MAX_ROWS_PER_SHEET:
            step_logger.log(f"[structured_data_workspace] {filename!r} sheet {name!r} has {len(df)} rows - "
                             f"truncated to {MAX_ROWS_PER_SHEET} for in-memory storage.")
            df = df.head(MAX_ROWS_PER_SHEET)
        truncated_sheets[name] = df
    return truncated_sheets


def _build_schema_summary(filename: str, sheets: dict[str, "pd.DataFrame"]) -> str:
    """A compact, LLM-readable description of the attached data's shape -
    column names/dtypes/sample rows per sheet - same role graphrag.py's
    get_schema_summary() plays for Cypher generation, but for tabular data.
    Deliberately text (not JSON) since this gets embedded directly into the
    structured_data_query prompt's system message."""
    parts = [f"File: {filename}"]
    for sheet_name, df in sheets.items():
        parts.append(f"\nSheet/table: {sheet_name!r} ({len(df)} rows, {len(df.columns)} columns)")
        parts.append("Columns: " + ", ".join(f"{c} ({df[c].dtype})" for c in df.columns))
        sample = df.head(MAX_SCHEMA_SAMPLE_ROWS)
        parts.append(f"Sample rows:\n{sample.to_string(index=False, max_colwidth=40)}")
    return "\n".join(parts)


def attach_file(session_id: str | None, filename: str, content: bytes) -> dict:
    """Parses and stores an uploaded structured data file against
    `session_id`, REPLACING any previously-attached file for that same
    session (one attachment per chat, not a growing list - a fresh
    attach_file() call is treated as the user swapping out the file, not
    adding a second one, since the structured agent works over exactly one
    file's data at a time).

    `session_id` is a plain opaque token, NOT necessarily the Neo4j chat
    history session id from chat_history.py - a chat session there is only
    created lazily on the FIRST question/answer (create_session() requires
    a question), so a file attached before any question has been asked
    would have no chat session id to key against yet. Same reasoning/
    pattern as opinion_workspace.create_workspace(): if `session_id` is
    None/falsy, this module self-generates a fresh token here (mirroring
    uuid.uuid4().hex[:12]) and the caller (the Flask route) hands that
    token back to the frontend, which holds it in memory for the lifetime
    of that Ask-screen chat and passes it on every subsequent attach/detach/
    ask call - independent of whether a chat has been saved to history yet.

    Returns a summary dict (session_id, filename, sheet names, row counts,
    schema_summary) - the caller returns this to the frontend so it can
    confirm what was attached and remember the session_id; does NOT return
    the actual DataFrame(s) (kept server-side only, in _ATTACHMENTS)."""
    if not session_id:
        session_id = uuid.uuid4().hex[:12]
    sheets = _parse_file(filename, content)
    schema_summary = _build_schema_summary(filename, sheets)
    _ATTACHMENTS[session_id] = {
        "filename": filename,
        "sheets": sheets,  # {sheet_name: DataFrame} - kept in memory only
        "schema_summary": schema_summary,
        "attached_at": time.time(),
    }
    step_logger.log(f"[structured_data_workspace] session={session_id} attached {filename!r} "
                     f"({len(sheets)} sheet(s), {sum(len(df) for df in sheets.values())} total rows)")
    return {
        "session_id": session_id,
        "filename": filename,
        "sheets": [{"name": name, "rows": len(df), "columns": list(df.columns)} for name, df in sheets.items()],
        "schema_summary": schema_summary,
    }


def get_attachment(session_id: str) -> dict | None:
    """Returns the raw attachment dict ({filename, sheets, schema_summary,
    attached_at}) for `session_id`, or None if nothing is attached - the
    shape _run_agentic_ask()'s attached_file_context parameter expects.
    Callers needing just the DataFrames should read result["sheets"]."""
    return _ATTACHMENTS.get(session_id)


def detach_file(session_id: str) -> bool:
    removed = _ATTACHMENTS.pop(session_id, None) is not None
    if removed:
        step_logger.log(f"[structured_data_workspace] session={session_id} detached its file")
    return removed
