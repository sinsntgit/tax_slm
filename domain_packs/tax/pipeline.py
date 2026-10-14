"""
Ingestion-UI orchestrator for the tax domain pack: upload -> convert -> extract
-> ingest -> embed, dispatched by file type, with a simple background-thread
job registry so the webapp can return immediately and poll for progress
(conversion + LLM extraction of a large document can take minutes).

Dispatch table (requirement 2a - "based on the file type use the appropriate
code"):
  act_pdf     -> converters/pdf_to_md_act.py      -> converters/extract_entities.py
                 (deterministic Section/SubSection/... parse + narrow LLM semantic layer)
  caselaw_pdf -> converters/pdf_to_md_caselaw.py   -> converters/extract_case_law.py
                 (one-call-per-judgment LLM structured extraction)
  generic     -> converters/pdf_to_md_generic.py   -> converters/extract_generic.py
                 (paragraph chunking + LLM enrichment, sized to the selected
                 LLM's context window - see extract_generic.py's
                 MODEL_CONTEXT_WINDOWS/model_chunk_budget())
  circular_pdf -> converters/pdf_to_md_generic.py  -> converters/extract_circular.py
                 (CBDT circular - one-call LLM extraction, classifies
                 Explanatory/Extension/General, resolves cited Act sections)
  treaty_pdf  -> converters/pdf_to_md_generic.py   -> converters/extract_treaty.py
                 (DTAA/protocol/TIEA - deterministic Article-by-Article split
                 + one short LLM call for the notification preamble)
  firm_opinion -> converters/pdf_to_md_generic.py  -> converters/extract_opinion.py
                 (2026-10-14: firm opinion - fully deterministic frontmatter +
                 section parse, no LLM call; cited authorities become
                 cited_reference stubs - see that module's docstring)

circular_pdf/treaty_pdf reuse pdf_to_md_generic.py for conversion (2026-08-27:
added per explicit request to wire these into the Ingest UI) rather than
getting their own dedicated converter, same reasoning as "generic" above -
see extract_circular.py/extract_treaty.py's own module docstrings for why a
generic pdftotext -layout pass is the right (and only necessary) conversion
step for these two types.

Conversion and extraction are run as SUBPROCESSES of the exact same CLI
scripts a user could run by hand (see each converter's own module docstring
for the equivalent command) - not reimplemented here - so the UI's "Run
Ingestion" button and the command line always do exactly the same thing.
Graph loading and embedding (framework.core.ingest.pipeline_ops) run
in-process instead, since those have no sys.path/import quirks to route
around and benefit from surfacing real exceptions rather than parsed
subprocess output.
"""
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from framework.core.ingest import pipeline_ops
from framework.core.retrieve import graphrag as graphrag_module

BASE_DIR = Path(__file__).resolve().parent                    # domain_packs/tax
CONVERTERS_DIR = BASE_DIR / "converters"
SCHEMA_PATH = BASE_DIR / "ontology" / "schema.json"
REVISED_ROOT = BASE_DIR.parent.parent                          # tax_kg_poc_revised/

DATA_DIR = REVISED_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MD_DIR = DATA_DIR / "converted_md"
GRAPH_DIR = DATA_DIR / "graphs"
for _d in (UPLOAD_DIR, MD_DIR, GRAPH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- File-type registry, drives the UI's dropdown + default metadata --------
FILE_TYPES = {
    "act_pdf": {
        "label": "Income Tax Act (PDF)",
        "accept": ".pdf",
        "description": "The Act itself - deterministic Section/SubSection/Clause/Proviso/Explanation "
                        "parsing (scripts/chunking.py) plus a narrow LLM semantic layer for defined "
                        "terms, illustrations, and memoranda.",
        "defaults": {"source_type": "Act", "authority_level": "Statute", "source_class": "StatutoryProvisionsSource"},
    },
    "caselaw_pdf": {
        "label": "Case Law (PDF)",
        "accept": ".pdf",
        "description": "A court/tribunal judgment or order - full LLM structured extraction "
                        "(facts, arguments, holding, cited sections) in one call per case.",
        "defaults": {"source_type": "CaseLaw", "authority_level": "Judicial", "source_class": "CaseLawsSource"},
    },
    "generic": {
        "label": "Generic Document (PDF / TXT / MD)",
        "accept": ".pdf,.txt,.md",
        "description": "Notifications, memos, firm opinions, and anything else without a dedicated "
                        "converter below - paragraph-level chunking plus best-effort LLM enrichment "
                        "against the full ontology, with extraction call-groups sized to the selected "
                        "LLM's context window (the whole document in one call if it fits).",
        "defaults": {"source_type": "KnowledgeRepositoryItem", "authority_level": "Unspecified",
                      "source_class": "KnowledgeRepositoryItem"},
    },
    # 2026-08-27: NEW - CBDT Circular / International Agreement, wired into
    # the Ingest UI per explicit request. See extract_circular.py/
    # extract_treaty.py's own module docstrings for the extraction design.
    "circular_pdf": {
        "label": "CBDT Circular (PDF)",
        "accept": ".pdf,.txt,.md",
        "description": "A CBDT circular/order - one-call LLM extraction of circular number/date/"
                        "subject/subtype (Explanatory/Extension/General), with cited Act sections "
                        "resolved onto the already-ingested Act graph and superseded/withdrawn "
                        "circulars linked as stubs.",
        "defaults": {"source_type": "Circular", "authority_level": "Delegated", "source_class": "RegulatoryPublication"},
    },
    "treaty_pdf": {
        "label": "International Agreement / DTAA (PDF)",
        "accept": ".pdf,.txt,.md",
        "description": "A bilateral tax agreement (DTAA/protocol/TIEA) notification - deterministic "
                        "Article-by-Article split (handles consolidated/amended-text redlines) plus "
                        "one short LLM read of the notification preamble for country, dates, and "
                        "every notification that has given the treaty legal effect.",
        "defaults": {"source_type": "TaxTreaty", "authority_level": "Treaty", "source_class": "TreatyNetworkSource"},
    },
    # 2026-10-14: NEW (phase0/opinions) - firm opinions, the customer's own
    # proprietary knowledge. .md/.txt are read as-is by pdf_to_md_generic.py
    # (so YAML frontmatter survives); a .pdf goes through the same generic
    # converter and extract_opinion.py falls back to parsing the letter body.
    # No entry-point store refresh is needed (absent from
    # _ENTRY_POINT_REFRESH below on purpose): opinion retrieval
    # (opinion_agent.py, the opinion_expert_views category) runs on the
    # Chunk vector index, which stage [4/4] embed_chunks_json() already
    # ensures and fills.
    "firm_opinion": {
        "label": "Firm Opinion (MD / TXT / PDF)",
        "accept": ".md,.txt,.pdf",
        "description": "A formal firm tax opinion - deterministic parse of frontmatter + body sections "
                        "(facts, issue, analysis, conclusion, caveats, reliance) into one FirmOpinion node "
                        "with per-section chunks; cited sections/cases/treaties/circulars become "
                        "provenance-marked stubs that join the real ingested authorities.",
        "defaults": {"source_type": "FirmOpinion", "authority_level": "Firm", "source_class": "KnowledgeRepositoryItem"},
    },
}

# 2026-10-13: NEW - which semantic_graph entry-point store (see
# Semantic_Graph_Retrieval_Architecture.md / local_vector_store.py) each file
# type feeds, as (module under framework.core.retrieve, embed function name).
# BUGFIX for: ingestion never rebuilt these stores - they were only ever
# built by hand via framework/cli/embed_*.py - so newly ingested documents
# were invisible to retrieval_mode="semantic_graph" (confirmed: three
# circulars ingested, but no circular_retrieval_index.npz existed at all,
# only the case-law and section stores). "generic" has no entry-point store,
# so it's deliberately absent here.
_ENTRY_POINT_REFRESH = {
    "act_pdf": ("section_entry_points", "embed_section_nodes"),
    "caselaw_pdf": ("case_law_entry_points", "embed_case_law_nodes"),
    "circular_pdf": ("circular_entry_points", "embed_circular_nodes"),
    "treaty_pdf": ("treaty_entry_points", "embed_treaty_nodes"),
}


def refresh_entry_point_index(file_type: str) -> str:
    """Re-embeds every node of this file type's entity label into its local
    semantic_graph entry-point store (embed_nodes() upserts via
    merge_and_save(), so earlier entries are kept and re-ingested ones are
    refreshed). Same call the matching framework/cli/embed_*.py script makes.

    Best-effort, never raises: returns a one-line status for the job log. A
    failure here must not mark an otherwise-successful ingestion as failed -
    the graph and Chunk embeddings are already loaded, and this store can
    always be rebuilt later with the CLI. Re-embeds the WHOLE label rather
    than just this document's nodes - cheap for case law/circulars/treaties
    (tens of nodes); for the Act it's a few thousand Section texts, which is
    small next to the Act's own multi-hour extraction."""
    target = _ENTRY_POINT_REFRESH.get(file_type)
    if target is None:
        return f"No semantic_graph entry-point index for file_type={file_type!r} - nothing to refresh."
    module_name, fn_name = target
    try:
        import importlib
        from framework.core.infra import neo4j_client
        module = importlib.import_module(f"framework.core.retrieve.{module_name}")
        with neo4j_client.get_session() as session:
            n = getattr(module, fn_name)(session)
        return f"Semantic-graph entry-point index refreshed ({module_name}): {n} node(s) embedded."
    except Exception as e:
        return (f"WARNING: semantic-graph entry-point index refresh failed ({module_name}): {e} - "
                f"ingestion itself succeeded; rebuild later with the matching "
                f"`python -m framework.cli.embed_*` command.")


MODEL_CONTEXT_WINDOWS = {"CLAUDE": 200_000, "GEMINI": 1_000_000, "GPT": 128_000}


def llm_chunk_info() -> dict:
    """Small helper for the UI to show "here's what the currently selected
    LLM's chunk budget looks like" before the user even uploads anything.
    Resolves the INGESTION-purpose model (USE_MODEL_INGESTION if set, else
    plain USE_MODEL) - this is describing extraction call sizing, so it
    should reflect whichever provider actually runs extraction, not
    whatever the retrieval side happens to be configured for."""
    from framework.core.infra import llm_client
    llm_client.refresh_model_config(purpose="ingestion")
    window = MODEL_CONTEXT_WINDOWS.get(llm_client.USE_MODEL, 100_000)
    budget = max(window - 1_500 - 2_000, 2_000)
    return {
        "provider": llm_client.USE_MODEL, "model": llm_client.MODEL_NAME,
        "context_window_tokens": window, "extraction_call_budget_tokens": budget,
    }


def save_upload(filename: str, content: bytes) -> Path:
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(filename).name}"
    path = UPLOAD_DIR / safe_name
    path.write_bytes(content)
    return path


def save_uploads_batch(files: list[tuple[str, bytes]]) -> list[Path]:
    """Same as save_upload(), for many files in one call (batch ingestion -
    e.g. a folder of case-law PDFs). Each still gets its own random-prefixed
    path, so same-named files from different folders never collide."""
    return [save_upload(filename, content) for filename, content in files]


def _run_subprocess(cmd: list[str], timeout: int) -> tuple[int, list[str]]:
    # encoding="utf-8" is NOT optional here: subprocess's `text=True` alone
    # decodes stdout/stderr using the platform's default locale encoding,
    # which on Windows is cp1252, not UTF-8 - and Act/case-law/circular text
    # routinely contains characters (smart quotes, section-symbol variants,
    # etc.) that aren't valid cp1252, which previously crashed the READER
    # THREAD mid-capture with UnicodeDecodeError (silently, in a background
    # thread) and left proc.stdout as None/truncated, surfacing later as a
    # confusing unrelated AttributeError. errors="replace" is a second
    # safety net - so a still-undecodable byte becomes U+FFFD instead of a
    # crash, since this is captured LOG TEXT, not something anything parses
    # byte-exactly.
    proc = subprocess.run(cmd, cwd=str(CONVERTERS_DIR), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    log_lines = (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines()
    return proc.returncode, log_lines


def convert_document(upload_path: Path, file_type: str) -> tuple[Path, dict, list[str]]:
    if file_type not in FILE_TYPES:
        raise ValueError(f"Unknown file_type {file_type!r}. Valid: {list(FILE_TYPES)}")
    md_path = MD_DIR / f"{upload_path.stem}.md"

    if file_type == "act_pdf":
        cmd = [sys.executable, str(CONVERTERS_DIR / "pdf_to_md_act.py"),
               "--pdf", str(upload_path), "--out", str(md_path), "--validate"]
    elif file_type == "caselaw_pdf":
        cmd = [sys.executable, str(CONVERTERS_DIR / "pdf_to_md_caselaw.py"),
               "--pdf", str(upload_path), "--out", str(md_path)]
    else:  # generic, circular_pdf, treaty_pdf - all use the generic converter
           # (see extract_circular.py/extract_treaty.py docstrings for why
           # neither needs its own dedicated PDF converter)
        cmd = [sys.executable, str(CONVERTERS_DIR / "pdf_to_md_generic.py"),
               "--in", str(upload_path), "--out", str(md_path)]

    returncode, log_lines = _run_subprocess(cmd, timeout=600)
    if returncode != 0 or not md_path.exists():
        raise RuntimeError("Conversion failed:\n" + "\n".join(log_lines[-50:]))

    stats = {"md_path": str(md_path), "md_bytes": md_path.stat().st_size}
    return md_path, stats, log_lines


def run_extraction(md_path: Path, file_type: str, doc_meta: dict) -> tuple[Path, Path | None, dict, list[str]]:
    doc_id = doc_meta.get("document_id") or md_path.stem
    graph_path = GRAPH_DIR / f"{doc_id}_graph.json"
    chunks_path = GRAPH_DIR / f"{doc_id}_chunks.json"

    if file_type == "act_pdf":
        cmd = [sys.executable, str(CONVERTERS_DIR / "extract_entities.py"),
               "--md", str(md_path), "--schema", str(SCHEMA_PATH),
               "--out", str(graph_path), "--chunks-out", str(chunks_path),
               "--document-id", doc_id, "--document-name", doc_meta.get("document_name", doc_id),
               "--source-type", doc_meta.get("source_type", "Act"),
               "--authority-level", doc_meta.get("authority_level", "Statute"),
               "--jurisdiction", doc_meta.get("jurisdiction", "India"),
               "--file-type", file_type]
        if doc_meta.get("effective_from"):
            cmd += ["--effective-from", doc_meta["effective_from"]]
        # act_namespace/act_year/act_number (2026-08-27): lets the webapp Ingest
        # flow give two different Acts distinct id-namespaces so their
        # same-numbered Sections don't collide - see extract_entities.py's
        # --act-namespace help text. Defaults to --document-id if omitted, so
        # this is optional (every upload already gets SOME distinct namespace
        # via its own doc_id) but worth setting explicitly to a clean value
        # like "ITA2025"/"ITA1961" when ingesting more than one Act.
        if doc_meta.get("act_namespace"):
            cmd += ["--act-namespace", doc_meta["act_namespace"]]
        if doc_meta.get("act_year"):
            cmd += ["--act-year", str(doc_meta["act_year"])]
        if doc_meta.get("act_number"):
            cmd += ["--act-number", str(doc_meta["act_number"])]
    elif file_type == "caselaw_pdf":
        cmd = [sys.executable, str(CONVERTERS_DIR / "extract_case_law.py"),
               "--md", str(md_path), "--out", str(graph_path), "--chunks-out", str(chunks_path),
               "--schema", str(SCHEMA_PATH), "--file-type", file_type,
               "--document-id", doc_id, "--document-name", doc_meta.get("document_name", doc_id),
               "--source-type", doc_meta.get("source_type", "CaseLaw"),
               "--authority-level", doc_meta.get("authority_level", "Judicial"),
               "--jurisdiction", doc_meta.get("jurisdiction", "India")]
        # cited_act_namespace: which Act's id-namespace this batch of case
        # law's section citations resolve against - defaults inside
        # extract_case_law.py to DEFAULT_CITED_ACT_NAMESPACE (the OLD Act,
        # correct for the currently-ingested corpus) if not given here.
        if doc_meta.get("cited_act_namespace"):
            cmd += ["--cited-act-namespace", doc_meta["cited_act_namespace"]]
    elif file_type == "circular_pdf":
        cmd = [sys.executable, str(CONVERTERS_DIR / "extract_circular.py"),
               "--md", str(md_path), "--schema", str(SCHEMA_PATH),
               "--out", str(graph_path), "--chunks-out", str(chunks_path),
               "--document-id", doc_id, "--document-name", doc_meta.get("document_name", doc_id),
               "--source-type", doc_meta.get("source_type", "Circular"),
               "--authority-level", doc_meta.get("authority_level", "Delegated"),
               "--jurisdiction", doc_meta.get("jurisdiction", "India"),
               "--file-type", file_type]
        # act_1961_namespace/act_2025_namespace: which already-ingested Act's
        # id-namespace a circular's cited sections resolve against, per Act
        # named in its own text - see extract_circular.py's "ACT-NAMESPACE
        # RESOLUTION" docstring section. Defaults inside extract_circular.py
        # (ITA1961/ITA2025) if not given here.
        if doc_meta.get("act_1961_namespace"):
            cmd += ["--act-1961-namespace", doc_meta["act_1961_namespace"]]
        if doc_meta.get("act_2025_namespace"):
            cmd += ["--act-2025-namespace", doc_meta["act_2025_namespace"]]
    elif file_type == "treaty_pdf":
        cmd = [sys.executable, str(CONVERTERS_DIR / "extract_treaty.py"),
               "--md", str(md_path), "--schema", str(SCHEMA_PATH),
               "--out", str(graph_path), "--chunks-out", str(chunks_path),
               "--document-id", doc_id, "--document-name", doc_meta.get("document_name", doc_id),
               "--source-type", doc_meta.get("source_type", "TaxTreaty"),
               "--authority-level", doc_meta.get("authority_level", "Treaty"),
               "--jurisdiction", doc_meta.get("jurisdiction", "India"),
               "--file-type", file_type]
    elif file_type == "firm_opinion":
        # 2026-10-14: NEW - see FILE_TYPES["firm_opinion"].
        cmd = [sys.executable, str(CONVERTERS_DIR / "extract_opinion.py"),
               "--md", str(md_path), "--schema", str(SCHEMA_PATH),
               "--out", str(graph_path), "--chunks-out", str(chunks_path),
               "--document-id", doc_id, "--document-name", doc_meta.get("document_name", doc_id),
               "--source-type", doc_meta.get("source_type", "FirmOpinion"),
               "--authority-level", doc_meta.get("authority_level", "Firm"),
               "--jurisdiction", doc_meta.get("jurisdiction", "India"),
               "--file-type", file_type]
        if doc_meta.get("cited_act_namespace"):
            cmd += ["--cited-act-namespace", doc_meta["cited_act_namespace"]]
    else:  # generic
        cmd = [sys.executable, str(CONVERTERS_DIR / "extract_generic.py"),
               "--md", str(md_path), "--schema", str(SCHEMA_PATH),
               "--out", str(graph_path), "--chunks-out", str(chunks_path),
               "--document-id", doc_id, "--document-name", doc_meta.get("document_name", doc_id),
               "--source-type", doc_meta.get("source_type", "KnowledgeRepositoryItem"),
               "--authority-level", doc_meta.get("authority_level", "Unspecified"),
               "--jurisdiction", doc_meta.get("jurisdiction", "India"),
               "--file-type", file_type]

    # 7200s (2hr) headroom - the Act path's LLM-call count dropped ~22x after
    # chunking.group_chunks_for_llm() stopped forcing a new group at every
    # Section boundary (see its docstring), so a full-Act run should now
    # finish in well under an hour, but this stays generous rather than
    # tight since a slow/rate-limited provider is still a real possibility.
    returncode, log_lines = _run_subprocess(cmd, timeout=7200)
    if returncode != 0 or not graph_path.exists():
        raise RuntimeError("Extraction failed:\n" + "\n".join(log_lines[-80:]))

    stats = {"graph_path": str(graph_path), "chunks_path": str(chunks_path) if chunks_path.exists() else None}
    return graph_path, (chunks_path if chunks_path.exists() else None), stats, log_lines


# --- background job registry -------------------------------------------------
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _log(job_id: str, line: str):
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["log"].append(line)


def start_ingestion_job(upload_path: Path, file_type: str, doc_meta: dict, mode: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "running", "log": [], "result": None, "error": None,
            "started_at": time.time(), "finished_at": None,
        }
    thread = threading.Thread(target=_run_ingestion_job, args=(job_id, upload_path, file_type, doc_meta, mode), daemon=True)
    thread.start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        return dict(_JOBS[job_id]) if job_id in _JOBS else None


def _run_ingestion_job(job_id: str, upload_path: Path, file_type: str, doc_meta: dict, mode: str):
    try:
        from domain_packs.tax.config import DOMAIN_CONFIG

        _log(job_id, f"[1/4] Converting {upload_path.name} to Markdown (file_type={file_type})...")
        md_path, convert_stats, convert_log = convert_document(upload_path, file_type)
        for line in convert_log:
            _log(job_id, line)
        _log(job_id, f"Converted -> {md_path}")

        _log(job_id, f"[2/4] Extracting entities/relationships/chunks...")
        graph_path, chunks_path, extract_stats, extract_log = run_extraction(md_path, file_type, doc_meta)
        for line in extract_log:
            _log(job_id, line)
        _log(job_id, f"Extraction complete -> {graph_path}")

        _log(job_id, f"[3/4] Loading graph into Neo4j (mode={mode})...")
        ingest_stats = pipeline_ops.ingest_graph_json(DOMAIN_CONFIG, str(graph_path),
                                                        str(chunks_path) if chunks_path else None, mode)
        _log(job_id, f"Graph loaded: {ingest_stats}")

        embed_stats = {"chunks_embedded": 0}
        if chunks_path:
            _log(job_id, "[4/4] Embedding chunks...")
            embed_stats = pipeline_ops.embed_chunks_json(DOMAIN_CONFIG, str(chunks_path))
            _log(job_id, f"Embedding complete: {embed_stats}")
        else:
            _log(job_id, "[4/4] No chunks file produced - skipping embedding.")

        # 2026-10-13: NEW - keep the semantic_graph entry-point store for
        # this file type in sync with what was just loaded (see
        # refresh_entry_point_index()). Best-effort; logs, never raises.
        _log(job_id, refresh_entry_point_index(file_type))

        # New labels (e.g. this document's file_type being ingested for the
        # very first time) only just landed in Neo4j above - force the
        # webapp's "Seed by label" dropdown to see them on its next
        # /api/schema call instead of whatever was cached at process start.
        graphrag_module.invalidate_schema_cache()

        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "done"
            _JOBS[job_id]["finished_at"] = time.time()
            _JOBS[job_id]["result"] = {
                "convert": convert_stats, "extract": extract_stats,
                "ingest": ingest_stats, "embed": embed_stats,
                "md_path": str(md_path), "graph_path": str(graph_path),
                "chunks_path": str(chunks_path) if chunks_path else None,
            }
    except Exception as e:
        _log(job_id, f"ERROR: {e}")
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["error"] = str(e)
            _JOBS[job_id]["finished_at"] = time.time()


# --- batch ingestion (many files, one file_type, one background job) --------
# Same 4-stage pipeline as a single-document job (convert -> extract -> load
# -> embed), run once per uploaded file, SEQUENTIALLY in one background
# thread - not `_run_ingestion_job` reused directly, since that function's
# job-log format assumes exactly one document. Sequential rather than
# parallel deliberately: each stage already shells out to a subprocess
# (conversion/extraction) and hits the same Neo4j/Postgres/LLM-provider
# rate limits a single job would, so running N of them concurrently would
# mostly just contend with itself rather than actually go faster - and for
# a batch that could be thousands of files, keeping this simple and
# resource-bounded matters more than shaving wall-clock time. A batch is
# just a big pile of independent single-document runs, so one slow/bad file
# never blocks the rest: caught and recorded as a per-file failure, then the
# loop moves on to the next file.
#
# The per-file job log itself CAN get large for a genuinely big batch
# (thousands of files x dozens of conversion/extraction log lines each), so
# unlike a single-document job's full `log` list, a batch job only keeps a
# capped rolling tail (_BATCH_LOG_TAIL_MAX) of the most recent lines, plus a
# per-file one-line result in `files` that's never trimmed - that's the
# part a user actually needs to review after a 10,000-file run finishes.
_BATCH_LOG_TAIL_MAX = 500


def _batch_log(batch_id: str, line: str):
    with _JOBS_LOCK:
        job = _JOBS.get(batch_id)
        if job is None:
            return
        job["log"].append(line)
        if len(job["log"]) > _BATCH_LOG_TAIL_MAX:
            job["log"] = job["log"][-_BATCH_LOG_TAIL_MAX:]


def start_batch_ingestion_job(upload_paths: list[Path], file_type: str, doc_meta_template: dict, mode: str) -> str:
    """Same shape as start_ingestion_job() but for many files at once - e.g.
    a whole folder of case-law PDFs uploaded together. doc_meta_template is
    applied to every file EXCEPT document_id/document_name, which are always
    derived per-file from that file's own filename (a single shared
    document_id for thousands of files would silently collide/overwrite in
    the graph)."""
    batch_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[batch_id] = {
            "type": "batch",
            "status": "running",
            "log": [],
            "total": len(upload_paths),
            "completed": 0,
            "failed": 0,
            "current_file": None,
            "files": [],   # [{"filename", "status": "ok"|"error", "error": str|None, "result": dict|None}]
            "result": None,
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
        }
    thread = threading.Thread(
        target=_run_batch_ingestion_job, args=(batch_id, upload_paths, file_type, doc_meta_template, mode), daemon=True
    )
    thread.start()
    return batch_id


def get_batch_job(batch_id: str) -> dict | None:
    with _JOBS_LOCK:
        return dict(_JOBS[batch_id]) if batch_id in _JOBS else None


def _run_batch_ingestion_job(batch_id: str, upload_paths: list[Path], file_type: str, doc_meta_template: dict, mode: str):
    from domain_packs.tax.config import DOMAIN_CONFIG

    total = len(upload_paths)
    for i, upload_path in enumerate(upload_paths, 1):
        with _JOBS_LOCK:
            if batch_id not in _JOBS:
                return  # job registry entry gone (shouldn't normally happen) - stop rather than log into nothing
            _JOBS[batch_id]["current_file"] = upload_path.name
        _batch_log(batch_id, f"[{i}/{total}] {upload_path.name}: converting...")

        # doc_id derived per file so 10,000 uploads never collide on one
        # shared id - document_name likewise, unless the template explicitly
        # set one (rare for a batch, but respected if present).
        doc_meta = dict(doc_meta_template)
        doc_meta["document_id"] = f"{upload_path.stem}"
        doc_meta.setdefault("document_name", upload_path.stem)

        try:
            md_path, convert_stats, convert_log = convert_document(upload_path, file_type)
            graph_path, chunks_path, extract_stats, extract_log = run_extraction(md_path, file_type, doc_meta)
            ingest_stats = pipeline_ops.ingest_graph_json(
                DOMAIN_CONFIG, str(graph_path), str(chunks_path) if chunks_path else None, mode
            )
            embed_stats = {"chunks_embedded": 0}
            if chunks_path:
                embed_stats = pipeline_ops.embed_chunks_json(DOMAIN_CONFIG, str(chunks_path))

            file_result = {
                "filename": upload_path.name, "status": "ok", "error": None,
                "result": {
                    "convert": convert_stats, "extract": extract_stats,
                    "ingest": ingest_stats, "embed": embed_stats,
                },
            }
            _batch_log(batch_id, f"[{i}/{total}] {upload_path.name}: OK - "
                                  f"{ingest_stats.get('entities_in_file', '?')} entities, "
                                  f"{ingest_stats.get('relationships_in_file', '?')} relationships")
            with _JOBS_LOCK:
                _JOBS[batch_id]["completed"] += 1
                _JOBS[batch_id]["files"].append(file_result)
        except Exception as e:
            _batch_log(batch_id, f"[{i}/{total}] {upload_path.name}: FAILED - {e}")
            with _JOBS_LOCK:
                _JOBS[batch_id]["failed"] += 1
                _JOBS[batch_id]["files"].append(
                    {"filename": upload_path.name, "status": "error", "error": str(e), "result": None}
                )
            # Deliberately no `raise`/`break` here - one bad file must never
            # abort the rest of a large batch.

    # 2026-10-13: NEW - refresh the semantic_graph entry-point store ONCE per
    # batch (one file_type per batch), not once per file - it re-embeds the
    # whole label each time, so per-file would redo that work N times. Only
    # if at least one file actually loaded.
    with _JOBS_LOCK:
        any_completed = _JOBS.get(batch_id, {}).get("completed", 0) > 0
    if any_completed:
        _batch_log(batch_id, refresh_entry_point_index(file_type))

    # Same reasoning as the single-document job above - a batch can easily
    # be the first time a whole new label (e.g. CaseLaw) shows up.
    graphrag_module.invalidate_schema_cache()

    with _JOBS_LOCK:
        if batch_id in _JOBS:
            _JOBS[batch_id]["status"] = "done"
            _JOBS[batch_id]["current_file"] = None
            _JOBS[batch_id]["finished_at"] = time.time()
    _batch_log(batch_id, f"Batch complete: {total} file(s), "
                          f"{_JOBS.get(batch_id, {}).get('completed', '?')} succeeded, "
                          f"{_JOBS.get(batch_id, {}).get('failed', '?')} failed.")
