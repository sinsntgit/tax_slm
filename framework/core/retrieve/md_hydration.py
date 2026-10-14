"""
Live text hydration from converted .md source files - the "chunkless" half
of the semantic_graph retrieval mode (see Semantic_Graph_Retrieval_
Architecture.md). Given an entity's stored `sourceMdPath` (and, for
line-range-parsed document types, its `line_start`/`line_end`), reads the
actual current text straight from the source file at QUERY time, instead of
only ever returning whatever text a Chunk node captured back at ingestion.

Two hydration shapes, chosen automatically by whether line_start/line_end
are present:

  1. WHOLE-FILE (CaseLaw, Circular, TaxTreaty): these document types are
     extracted via a single LLM read of the FULL text (extract_case_law.py/
     extract_circular.py/extract_treaty.py) - the resulting entity properties
     (holding, reasoning, hasText...) are LLM-composed/paraphrased, not a
     verbatim positional slice of the source file, so there is no correct
     line_start:line_end for them. Instead: read the WHOLE source .md file.
     This corpus's case-law/circular/treaty .md files run well under 1MB
     each (confirmed against the real converted files: case-law files
     9KB-550KB, so a whole-file read costs single-digit milliseconds, not a
     real latency concern.

  2. LINE-RANGE (StatutoryProvision/Section, and anything else chunking.py
     deterministically parses): chunking.py already tracks exact
     line_start/line_end per Section/SubSection while parsing the Act's .md
     file - reusing that instead of reading the ~3MB whole Act file per
     matched Section.

No caching: file sizes here are small enough (see above) that a fresh read
per query is cheap, and caching risks silently serving stale text after a
re-ingestion updates the underlying .md file while the webapp process keeps
running - not worth the risk for the sub-10ms this saves.
"""
from pathlib import Path

DEFAULT_WHOLE_FILE_MAX_CHARS = 4_000_000  # safety rail - see module docstring; real files are far smaller


def hydrate_text(source_md_path: str, line_start: int | None = None, line_end: int | None = None,
                  whole_file_max_chars: int = DEFAULT_WHOLE_FILE_MAX_CHARS) -> tuple[str, str]:
    """Returns (text, note). `text` is "" (never None/raises) on any failure
    - missing sourceMdPath, missing file, out-of-range lines - with `note`
    explaining why, so a caller can fall back to whatever stored Chunk text
    it has rather than crash the whole retrieval call over one bad pointer.
    """
    if not source_md_path:
        return "", "no sourceMdPath stored on this entity"

    path = Path(source_md_path)
    if not path.exists():
        return "", f"source file not found: {source_md_path}"

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return "", f"could not read source file: {e}"

    if line_start is None or line_end is None:
        # WHOLE-FILE hydration (case law / circular / treaty - see module docstring).
        if len(raw) > whole_file_max_chars:
            return raw[:whole_file_max_chars], (
                f"whole file exceeded {whole_file_max_chars} chars and was truncated - "
                f"unexpected for this corpus, see module docstring's size assumptions"
            )
        return raw, ""

    # LINE-RANGE hydration (Act Section/SubSection - see module docstring).
    lines = raw.split("\n")
    n = len(lines)
    start = max(1, int(line_start))
    end = min(n, int(line_end))
    if start > n or end < start:
        return "", f"line_start={line_start}/line_end={line_end} out of range for a {n}-line file"
    # line_start/line_end are 1-indexed (chunking.py's own convention) - slice is 0-indexed.
    return "\n".join(lines[start - 1:end]), ""


def hydrate_entity(entity_props: dict) -> tuple[str, str]:
    """Convenience wrapper reading the three relevant properties directly
    off an entity's property dict (as returned by a Cypher RETURN n) -
    dispatches to hydrate_text() above."""
    return hydrate_text(
        entity_props.get("sourceMdPath", ""),
        entity_props.get("line_start"),
        entity_props.get("line_end"),
    )
