"""
APPLICATION LAYER (Flask) - GENERALIZED from tax_kg_poc/webapp/app.py.

Loads ONE domain pack at process start (FRAMEWORK_DOMAIN env var, or --domain
via `python -m framework.webapp.app --domain tax`, default "tax") and wires
the framework's generic engines together exactly as before:

    /api/ask pipeline:
        0. cache.get_cached_answer()      -> return early on cache hit
        1. intent.detect_intent()         -> classify the question (domain's PromptPack)
        2. GraphRAGRetriever.retrieve()   -> graph + vector candidates (domain's DomainConfig)
        3. reranker.rerank()              -> BGE reranker narrows to top-N
        4. grounding.ground_answer()      -> Citation & Grounding metadata
        5. LLM answer generation          -> domain's answer_generation prompt
        6. cache.set_cached_answer()

/api/graph/seed, /api/graph/search, /api/node/<id>/neighbors (graph
visualization) talk to Neo4j directly and run through the domain's
PiiMasker before returning, same as before - not part of the QA pipeline,
but just as much a "system output".

Nothing in this file names "tax", "Section", or any other domain concept -
every domain-specific behavior comes from whichever DomainConfig/PromptPack
pack_loader.load_domain_pack() resolves.
"""
import argparse
import json
import os
import threading  # 2026-08-26: NEW - quality-checked-ask background job (see _ASK_JOBS below)
import time        # 2026-08-26: NEW - job started_at/finished_at timestamps
import uuid         # 2026-08-26: NEW - job_id generation, same pattern as domain_packs/tax/pipeline.py's _JOBS
from concurrent.futures import ThreadPoolExecutor  # 2026-09-07: NEW - runs the two evaluators concurrently in fast mode (see _run_fast_quality_checked_ask())
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

from framework.cli.pack_loader import load_pack_module
from framework.core.infra import llm_client, neo4j_client, settings_store
from framework.core.infra import audit_log
from framework.core.infra import chat_history
from framework.core.infra import evaluation_log  # 2026-08-26: NEW - see module docstring
from framework.core.ingest import reset as reset_module
from framework.core.ingest.embedder import EMBEDDING_MODEL_NAME, VECTOR_INDEX_NAME
from framework.core.ingest.validate import run_sanity_checks
from framework.core.retrieve import answer_evaluation, answer_quality, cache, category_retrieval, comparative_response, data_explorer, grounding, observability, prompt_config, reranker, reranker_config, response_config, retrieval_router
from framework.core.retrieve import structured_data_workspace  # 2026-09-08: NEW - agentic retriever's attached-file storage (see below)
from framework.core.retrieve import sandbox  # 2026-09-08: NEW - isolated-subprocess execution of the structured data query agent's generated pandas code (see sandbox.py's module docstring)
from framework.core.retrieve import retrieval_orchestrator  # 2026-09-10: NEW - agentic retrieval_mode's bounded plan/retrieve/sufficiency-check loop (see that module's docstring)
from framework.core.retrieve import hierarchical_orchestrator  # 2026-09-11: NEW - structured_research retrieval_mode's fixed Fact Sheet/Law/Case Law/Tribunal/Opinion hierarchy (see that module's docstring)
from framework.core.retrieve.graphrag import GraphRAGRetriever
from framework.core.retrieve.intent import detect_intent
from framework.core.retrieve.pii import PiiMasker
from framework.core.retrieve import graphrag as graphrag_module
from framework.core.infra.dotenv_path import env_path as _env_path

# Anchored to the repo root by file location, NOT process cwd - see
# framework.core.infra.dotenv_path's docstring for why the previous plain
# load_dotenv() (cwd-relative search) could silently miss .env entirely
# depending on how/where this app was launched from.
load_dotenv(_env_path())

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--domain", default=None)
_known_args, _ = _ap.parse_known_args()

PACK = load_pack_module(_known_args.domain)
DOMAIN_CONFIG = PACK.DOMAIN_CONFIG
PROMPT_PACK = PACK.PROMPT_PACK
# Not every domain pack necessarily ships an ingestion-UI orchestrator (only
# tax does, so far) - the upload/convert/ingest routes below check for this
# and return 501 if it's missing, rather than assuming every pack has one.
PIPELINE = getattr(PACK, "PIPELINE", None)
# 2026-08-26: NEW. Same optional-extra pattern as PIPELINE above - not every
# domain pack necessarily defines a Knowledge & Analysis Framework category
# list (see category_retrieval.py / domain_packs/tax/config.py's
# TAX_CATEGORY_SPECS). /api/ask below just skips the category-coverage pass
# entirely when this is None, rather than assuming every pack has one.
CATEGORY_SPECS = getattr(PACK, "CATEGORY_SPECS", None)

retriever = GraphRAGRetriever(DOMAIN_CONFIG, PROMPT_PACK)
pii = PiiMasker(DOMAIN_CONFIG)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
# 2026-08-27: RAISED from 200MB. Real Act PDFs (especially scanned ones like
# the full Income-tax Act, 1961 or the Income-tax Rules, 2026) run 70-180MB
# EACH - a batch upload of even 2-3 of them together (one multipart POST,
# all files combined) blew straight through the old 200MB ceiling. Flask
# rejects an over-limit request with a bare HTML 413 page BEFORE any route
# handler runs, which the frontend's `await res.json()` then fails to parse
# ("Unexpected token '<'... is not valid JSON") - a confusing symptom for
# what is really just "the batch was too big". Raised to a generous 1GB to
# comfortably cover a handful of large scanned Acts in one batch; see the
# global error handlers below for making ANY future over-the-line rejection
# (413 or otherwise) return a readable JSON error instead of an HTML page.
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB

try:
    # Best-effort at import time - if Neo4j isn't reachable yet (e.g. not
    # started), don't crash the whole webapp over a chat-history index;
    # the routes below will just surface their own error on first use.
    chat_history.ensure_indexes()
except Exception as e:
    app.logger.warning(f"chat_history.ensure_indexes() failed at startup (will retry lazily): {e}")

try:
    # 2026-08-26: NEW - same best-effort startup pattern as chat_history above.
    evaluation_log.ensure_indexes()
except Exception as e:
    app.logger.warning(f"evaluation_log.ensure_indexes() failed at startup (will retry lazily): {e}")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.after_request
def _disable_static_caching(response):
    """This app's static files (index.html/app.css/industry.css) change
    constantly during active development, but Flask's default static file
    handler sends Cache-Control/ETag headers that let browsers reuse a
    stale cached copy after a plain refresh - which looks exactly like a
    "regression" (old broken JS/CSS silently still running) even though the
    file on disk is already fixed. Force revalidation on every request so a
    normal refresh always picks up the latest saved file. Narrow and
    reversible - only touches the response, not the served content."""
    if request.path == "/" or request.path.startswith(("/static", "/vis-network", "/index.html", "/app.css", "/industry.css")) or request.path.endswith((".html", ".css", ".js")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.route("/api/schema")
def api_schema():
    return jsonify(graphrag_module.get_schema_summary())


# 2026-08-27: NEW - "View Ontology" screen. Distinct from /api/schema above:
# that one reports the LIVE Neo4j schema (what's actually been ingested);
# this reports the STATIC ontology DEFINITION (domain_packs/<domain>/
# ontology/schema.json) - what the ontology declares, whether or not
# anything of that type has been ingested yet. Not every domain pack's
# PIPELINE module necessarily exposes SCHEMA_PATH (only tax does, so far,
# same optional-extra pattern as PIPELINE itself) - 501s cleanly if not.
@app.route("/api/ontology")
def api_ontology():
    schema_path = getattr(PIPELINE, "SCHEMA_PATH", None) if PIPELINE else None
    if not schema_path:
        return jsonify({
            "error": f"Domain pack '{DOMAIN_CONFIG.domain_name}' doesn't expose a SCHEMA_PATH - "
                     f"the ontology view isn't available for this pack."
        }), 501
    from framework.core.retrieve import ontology_view
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except (OSError, ValueError) as e:
        return jsonify({"error": f"Could not read/parse {schema_path}: {e}"}), 500
    return jsonify(ontology_view.build_ontology_view(schema))


# ============================================================================
# GENERATE OPINIONS - 2026-08-27: NEW. Upload documents (case laws,
# contracts, notices) into an EPHEMERAL, per-session workspace (see
# framework/core/retrieve/opinion_workspace.py's module docstring for why
# this is deliberately NOT persisted into the main Neo4j graph - client-
# confidential documents shouldn't land in the shared firm-wide KG), ask
# questions grounded in just those documents, and generate a structured
# BIG4-style opinion / file-suit response / clarification-questions output.
# Uses PROMPT_PACK's "document_qa"/"opinion_generation" prompts, so this is
# only available for domain packs that define them (tax does) - same
# optional-extra pattern as PIPELINE.
# ============================================================================

def _opinions_available():
    try:
        PROMPT_PACK.get_prompt("opinion_generation")
        return True
    except KeyError:
        return False


def _require_opinions():
    if not _opinions_available():
        return jsonify({
            "error": f"Domain pack '{DOMAIN_CONFIG.domain_name}' doesn't define the opinion_generation/"
                     f"document_qa prompts - the Generate Opinions screen isn't available for this pack."
        }), 501
    return None


def _opinion_kg_context(question: str) -> str | None:
    """Best-effort supplementary grounding from the MAIN knowledge graph
    (statutes/case law already ingested) alongside the uploaded documents -
    optional, and never allowed to break opinion generation if it fails."""
    try:
        from framework.core.retrieve import grounding
        retrieved = retriever.retrieve(question, retrieval_mode="hybrid")
        parts = []
        if retrieved.get("plain_rows"):
            parts.append("Graph query results:\n" + grounding.graph_rows_to_context_json(retrieved["plain_rows"], max_chars=4000))
        top_chunks = (retrieved.get("candidate_chunks") or [])[:8]
        if top_chunks:
            parts.append("Retrieved passages from the main graph:\n" + "\n\n".join(
                f"[{c['chunk_id']}] {c['text'][:1200]}" for c in top_chunks
            ))
        return "\n\n".join(parts) if parts else None
    except Exception as e:
        app.logger.warning(f"KG grounding for opinion workspace failed (continuing without it): {e}")
        return None


@app.route("/api/opinions/workspaces", methods=["GET", "POST"])
def api_opinions_workspaces():
    err = _require_opinions()
    if err:
        return err
    from framework.core.retrieve import opinion_workspace
    if request.method == "GET":
        return jsonify({"workspaces": opinion_workspace.list_workspaces()})
    body = request.json or {}
    workspace_id = opinion_workspace.create_workspace(body.get("matter_context") or {})
    return jsonify({"workspace_id": workspace_id})


@app.route("/api/opinions/workspaces/<workspace_id>", methods=["GET", "DELETE"])
def api_opinions_workspace_detail(workspace_id):
    err = _require_opinions()
    if err:
        return err
    from framework.core.retrieve import opinion_workspace
    if request.method == "DELETE":
        if not opinion_workspace.delete_workspace(workspace_id):
            return jsonify({"error": "No such workspace."}), 404
        return jsonify({"deleted": True})
    # GET - full detail (documents/qa_history/opinions), for rehydrating the
    # UI on page load or when switching workspaces - the list route only
    # returns summary counts, not the full history.
    workspace = opinion_workspace.get_workspace(workspace_id)
    if workspace is None:
        return jsonify({"error": "No such workspace."}), 404
    return jsonify({
        "workspace_id": workspace_id,
        "matter_context": workspace["matter_context"],
        "created_at": workspace["created_at"],
        "documents": [
            {"name": d["name"], "char_count": d["char_count"], "uploaded_at": d["uploaded_at"]}
            for d in workspace["documents"]
        ],
        "qa_history": workspace["qa_history"],
        "opinions": workspace["opinions"],
    })


@app.route("/api/opinions/workspaces/<workspace_id>/documents", methods=["POST"])
def api_opinions_upload(workspace_id):
    err = _require_opinions() or _require_pipeline()
    if err:
        return err
    from framework.core.retrieve import opinion_workspace
    if opinion_workspace.get_workspace(workspace_id) is None:
        return jsonify({"error": "No such workspace."}), 404
    files = request.files.getlist("files") or ([request.files["file"]] if "file" in request.files else [])
    if not files:
        return jsonify({"error": "No files (expected multipart field 'files' or 'file')."}), 400
    added, errors = [], []
    for f in files:
        if not f.filename:
            continue
        try:
            content = f.read()
            saved_path = PIPELINE.save_upload(f.filename, content)
            added.append(opinion_workspace.add_document(workspace_id, saved_path, f.filename))
        except Exception as e:
            app.logger.exception(f"Failed adding '{f.filename}' to opinion workspace {workspace_id}")
            errors.append({"filename": f.filename, "error": str(e)})
    return jsonify({"added": added, "errors": errors})


@app.route("/api/opinions/workspaces/<workspace_id>/ask", methods=["POST"])
def api_opinions_ask(workspace_id):
    err = _require_opinions()
    if err:
        return err
    from framework.core.retrieve import opinion_workspace
    body = request.json or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "'question' is required."}), 400
    try:
        kg_context = _opinion_kg_context(question) if body.get("use_kg") else None
        record = opinion_workspace.ask(workspace_id, question, PROMPT_PACK, kg_context=kg_context)
        return jsonify(record)
    except KeyError:
        return jsonify({"error": "No such workspace."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Failed in /api/opinions/.../ask")
        return jsonify({"error": str(e)}), 500


@app.route("/api/opinions/workspaces/<workspace_id>/generate", methods=["POST"])
def api_opinions_generate(workspace_id):
    err = _require_opinions()
    if err:
        return err
    from framework.core.retrieve import opinion_workspace
    body = request.json or {}
    opinion_type = body.get("opinion_type") or "opinion"
    question = (body.get("question") or "").strip() or None
    try:
        kg_context = _opinion_kg_context(question or opinion_type) if body.get("use_kg") else None
        record = opinion_workspace.generate_opinion(
            workspace_id, opinion_type, PROMPT_PACK, user_question=question, kg_context=kg_context
        )
        return jsonify(record)
    except KeyError:
        return jsonify({"error": "No such workspace."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Failed in /api/opinions/.../generate")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# INGESTION UI: file-type selection, upload, convert-to-Markdown, and the
# full extract -> ingest -> embed pipeline (requirements 1-2 of the
# ingestion UI: choose file type, upload, convert, then ingest respecting
# the selected LLM's max chunk size). Delegates everything domain-specific
# to the loaded pack's PIPELINE module (domain_packs/tax/pipeline.py for the
# tax pack) - this file stays domain-agnostic.
# ============================================================================

def _require_pipeline():
    if PIPELINE is None:
        return jsonify({
            "error": f"Domain pack '{DOMAIN_CONFIG.domain_name}' doesn't expose a PIPELINE "
                     f"orchestrator - the ingestion UI isn't available for this pack."
        }), 501
    return None


@app.route("/api/file-types")
def api_file_types():
    err = _require_pipeline()
    if err:
        return err
    return jsonify({
        "file_types": PIPELINE.FILE_TYPES,
        "llm": PIPELINE.llm_chunk_info(),
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    err = _require_pipeline()
    if err:
        return err
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request (expected multipart field 'file')."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400
    try:
        content = f.read()
        # 2026-10-14: duplicate reporting (see domain_packs/tax/ingest_dedup.py)
        # - identical bytes reuse the existing upload, and the UI is told when
        # this exact file is already in the graph so it can offer "re-ingest
        # anyway" instead of silently creating a duplicate document. getattr:
        # a pack without these helpers keeps the plain save_upload() path.
        dedup_info, already_ingested = {}, []
        if hasattr(PIPELINE, "save_upload_dedup"):
            saved_path, dedup_info = PIPELINE.save_upload_dedup(f.filename, content)
            already_ingested = PIPELINE.describe_upload_duplicates(saved_path)
        else:
            saved_path = PIPELINE.save_upload(f.filename, content)
        return jsonify({
            "upload_path": str(saved_path), "filename": f.filename, "size_bytes": len(content),
            **dedup_info, "already_ingested": already_ingested,
        })
    except Exception as e:
        app.logger.exception("Failed in /api/upload")
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload/batch", methods=["POST"])
def api_upload_batch():
    """Multi-file counterpart to /api/upload - accepts every file under the
    multipart field 'files' (repeated, as a browser's `<input multiple>`
    naturally sends) in ONE request, saves each, and returns their upload
    paths in the same order they were received - the order start_batch_
    ingestion_job() then processes them in."""
    err = _require_pipeline()
    if err:
        return err
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files (expected multipart field 'files', repeated)."}), 400
    try:
        uploaded = []
        for f in files:
            if not f.filename:
                continue
            content = f.read()
            saved_path = PIPELINE.save_upload(f.filename, content)
            uploaded.append({"upload_path": str(saved_path), "filename": f.filename, "size_bytes": len(content)})
        if not uploaded:
            return jsonify({"error": "No valid files in the request."}), 400
        return jsonify({"files": uploaded, "count": len(uploaded)})
    except Exception as e:
        app.logger.exception("Failed in /api/upload/batch")
        return jsonify({"error": str(e)}), 500


@app.route("/api/convert", methods=["POST"])
def api_convert():
    err = _require_pipeline()
    if err:
        return err
    body = request.json or {}
    upload_path = body.get("upload_path")
    file_type = body.get("file_type")
    if not upload_path or not file_type:
        return jsonify({"error": "upload_path and file_type are required."}), 400
    try:
        md_path, stats, log_lines = PIPELINE.convert_document(Path(upload_path), file_type)
        preview = md_path.read_text(encoding="utf-8", errors="replace")
        return jsonify({
            "md_path": str(md_path), "stats": stats, "log": log_lines,
            "preview": preview[:4000], "preview_truncated": len(preview) > 4000,
        })
    except Exception as e:
        app.logger.exception("Failed in /api/convert")
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest/start", methods=["POST"])
def api_ingest_start():
    err = _require_pipeline()
    if err:
        return err
    body = request.json or {}
    upload_path = body.get("upload_path")
    file_type = body.get("file_type")
    doc_meta = body.get("doc_meta") or {}
    mode = body.get("mode", "base")
    if not upload_path or not file_type:
        return jsonify({"error": "upload_path and file_type are required."}), 400
    try:
        # 2026-10-14: force_reingest - "re-ingest anyway" for a file whose
        # exact bytes are already in the graph (otherwise the job skips it).
        extra = {"force_reingest": True} if body.get("force_reingest") else {}
        job_id = PIPELINE.start_ingestion_job(Path(upload_path), file_type, doc_meta, mode, **extra)
        return jsonify({"job_id": job_id})
    except Exception as e:
        app.logger.exception("Failed in /api/ingest/start")
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest/status/<job_id>")
def api_ingest_status(job_id):
    err = _require_pipeline()
    if err:
        return err
    job = PIPELINE.get_job(job_id)
    if job is None:
        return jsonify({"error": f"No such job: {job_id}"}), 404
    return jsonify(job)


@app.route("/api/ingest/batch/start", methods=["POST"])
def api_ingest_batch_start():
    """Same as /api/ingest/start, but for many uploads at once (e.g. 10,000
    case-law PDFs uploaded via /api/upload/batch). document_id/document_name
    are always derived per-file from that file's filename - doc_meta here
    only supplies the shared fields (source_type/authority_level/
    jurisdiction/etc.), same as the file type's `defaults`."""
    err = _require_pipeline()
    if err:
        return err
    body = request.json or {}
    upload_paths = body.get("upload_paths")
    file_type = body.get("file_type")
    doc_meta = body.get("doc_meta") or {}
    mode = body.get("mode", "base")
    if not upload_paths or not isinstance(upload_paths, list) or not file_type:
        return jsonify({"error": "upload_paths (non-empty list) and file_type are required."}), 400
    try:
        extra = {"force_reingest": True} if body.get("force_reingest") else {}  # 2026-10-14: see /api/ingest/start
        batch_id = PIPELINE.start_batch_ingestion_job([Path(p) for p in upload_paths], file_type, doc_meta, mode,
                                                      **extra)
        return jsonify({"batch_id": batch_id, "total": len(upload_paths)})
    except Exception as e:
        app.logger.exception("Failed in /api/ingest/batch/start")
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest/batch/status/<batch_id>")
def api_ingest_batch_status(batch_id):
    err = _require_pipeline()
    if err:
        return err
    job = PIPELINE.get_batch_job(batch_id)
    if job is None:
        return jsonify({"error": f"No such batch job: {batch_id}"}), 404
    return jsonify(job)


# ============================================================================
# VALIDATE GRAPH (requirement 3) and CLEAR ALL DATA (requirement 5)
# ============================================================================

@app.route("/api/validate")
def api_validate():
    try:
        with neo4j_client.get_session() as session:
            report = run_sanity_checks(session, DOMAIN_CONFIG)
        return jsonify(report)
    except Exception as e:
        app.logger.exception("Failed in /api/validate")
        return jsonify({"error": str(e)}), 500


@app.route("/api/response-config", methods=["GET"])
def api_response_config_get():
    return jsonify(response_config.load_response_config())


@app.route("/api/response-config", methods=["POST"])
def api_response_config_post():
    """Saves the user's response-structure config. Hot-reloadable - the
    very next /api/ask call re-reads this file, no restart needed."""
    body = request.json or {}
    try:
        saved = response_config.save_response_config(body)
        return jsonify(saved)
    except Exception as e:
        app.logger.exception("Failed in /api/response-config POST")
        return jsonify({"error": str(e)}), 500


@app.route("/api/comparative-response-config", methods=["GET"])
def api_comparative_response_config_get():
    return jsonify(comparative_response.load_config())


@app.route("/api/comparative-response-config", methods=["POST"])
def api_comparative_response_config_post():
    """Saves the user's favourable/unfavourable-, for/against-, comparison-,
    hold/withhold-type response structure config (see
    comparative_response.py). Hot-reloadable - the very next /api/ask call
    whose question matches the trigger keywords re-reads this file, no
    restart needed."""
    body = request.json or {}
    try:
        saved = comparative_response.save_config(body)
        return jsonify(saved)
    except Exception as e:
        app.logger.exception("Failed in /api/comparative-response-config POST")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reranker-config", methods=["GET"])
def api_reranker_config_get():
    return jsonify(reranker_config.load_config())


@app.route("/api/reranker-config", methods=["POST"])
def api_reranker_config_post():
    """Saves the reranker's relevance-filtering config (see
    reranker_config.py - min_relevance/fallback_relevance/
    fallback_max_chunks/top_n). Hot-reloadable - the very next /api/ask call
    re-reads this file, no restart needed."""
    body = request.json or {}
    try:
        saved = reranker_config.save_config(body)
        return jsonify(saved)
    except Exception as e:
        app.logger.exception("Failed in /api/reranker-config POST")
        return jsonify({"error": str(e)}), 500


@app.route("/api/retrieval-router-config", methods=["GET"])
def api_retrieval_router_config_get():
    return jsonify(retrieval_router.load_config())


@app.route("/api/retrieval-router-config", methods=["POST"])
def api_retrieval_router_config_post():
    """Saves the graph-expansion routing config (see retrieval_router.py -
    lineage_keywords/expansion_seed_top_k/expansion_max_neighbors). Hot-
    reloadable - the very next /api/ask call whose question matches the
    trigger keywords re-reads this file, no restart needed."""
    body = request.json or {}
    try:
        saved = retrieval_router.save_config(body)
        return jsonify(saved)
    except Exception as e:
        app.logger.exception("Failed in /api/retrieval-router-config POST")
        return jsonify({"error": str(e)}), 500


@app.route("/api/prompt-config", methods=["GET"])
def api_prompt_config_get():
    """Returns the user's saved prompt override (if any) AND the domain
    pack's current built-in default template/version, so the Settings page
    can show the override editor pre-filled with the override when one is
    enabled, or pre-filled with the built-in default as a starting point
    when the user has never saved one - either way giving them the exact
    text that's actually in effect right now to edit."""
    override = prompt_config.load_prompt_override()
    default_prompt = PROMPT_PACK.get_prompt("answer_generation")
    # 2026-10-13: NEW - surfaces why a saved, enabled override is being
    # ignored (see prompt_config.resolve_answer_system_prompt()'s matching
    # BUGFIX comment), so the Settings page can show it instead of the
    # toggle claiming "in effect now" for an override that isn't.
    problems = prompt_config.validate_template(override["template"]) if override["template"] else []
    return jsonify({
        "enabled": override["enabled"],
        "in_effect": override["enabled"] and not problems,
        "problems": problems,
        "template": override["template"],
        "default_template": default_prompt.template,
        "default_version": default_prompt.version,
    })


@app.route("/api/prompt-config", methods=["POST"])
def api_prompt_config_post():
    """Saves the user's answer_generation prompt override. Hot-reloadable -
    the very next /api/ask call re-reads this file via
    prompt_config.resolve_answer_system_prompt(), no restart needed.
    Send {"enabled": false} (or omit template) to revert to the domain
    pack's built-in prompt without losing the saved draft text."""
    body = request.json or {}
    try:
        saved = prompt_config.save_prompt_override(body)
        return jsonify(saved)
    except Exception as e:
        app.logger.exception("Failed in /api/prompt-config POST")
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/clear", methods=["POST"])
def api_graph_clear():
    """Erases every node/relationship in the currently connected Neo4j
    database and flushes the response cache, so a subsequent ingestion run
    starts from a genuinely empty graph. Requires an explicit confirmation
    string in the request body - not just any POST - since this is
    irreversible."""
    body = request.json or {}
    if body.get("confirm") != "DELETE ALL DATA":
        return jsonify({
            "error": "Refusing to clear the graph without confirmation. "
                     "POST {\"confirm\": \"DELETE ALL DATA\"} to proceed."
        }), 400
    try:
        with neo4j_client.get_session() as session:
            stats = reset_module.clear_all_data(session)
        cache_cleared = cache.clear_all_cache()
        graphrag_module.invalidate_schema_cache()  # force schema re-fetch (labels/rel types are now empty)
        return jsonify({"graph": stats, "cache_keys_cleared": cache_cleared})
    except Exception as e:
        app.logger.exception("Failed in /api/graph/clear")
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/seed")
def api_graph_seed():
    label = request.args.get("label", "Section")
    limit = int(request.args.get("limit", 40))

    try:
        label_safe = neo4j_client.safe_identifier(label)
        with neo4j_client.get_session() as session:
            node_rows = list(session.run(f"MATCH (n:{label_safe}) RETURN n LIMIT $limit", limit=limit))
            nodes = {}
            for row in node_rows:
                n = graphrag_module.vertex_to_node(row["n"])
                nodes[n["db_id"]] = n

            edges = []
            if nodes:
                id_list = list(nodes.keys())
                edge_rows = list(session.run(
                    "MATCH (a)-[r]->(b) WHERE elementId(a) IN $ids AND elementId(b) IN $ids RETURN r",
                    ids=id_list,
                ))
                for row in edge_rows:
                    edges.append(graphrag_module.edge_to_link(row["r"]))

        masked_nodes = pii.mask_nodes(list(nodes.values()), context=f"/api/graph/seed label={label}")
        return jsonify({"nodes": masked_nodes, "edges": edges})
    except Exception as e:
        app.logger.exception("Failed in /api/graph/seed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/search")
def api_graph_search():
    """Explorer's "Find" box. Two-tier lookup:
      1. Precise statutory citation match (unchanged) - q parsed via the
         domain's citation_rules (e.g. "12A" / "80CCD(1B)") and matched
         exactly against full_citation. Kept first/fast since it's still
         the common case for statute lookups and avoids substring
         false-positives (e.g. "12A" would CONTAINS-match "112A" too).
      2. Free-text fallback - if q doesn't parse as a citation, or the
         citation didn't match anything in the graph, fall back to a
         case-insensitive substring search across every text field a user
         might plausibly type: a case's name/citation, an Issue/Argument/
         Conclusion's own text, a Section's short title, a holding, or the
         node's id. This is what makes "reckitt" find a CaseLaw whose
         case_name is "Reckitt Benckiser Healthcare India... v. Dy.
         Commissioner...", not just exact statutory citations - added
         after a user report that case names, issues, and citations typed
         into this box all came back "does not exist" even though the
         data was ingested.
    Both tiers feed the SAME seed_ids into the shared neighborhood-building
    logic below (PART_OF descendants + one-hop neighbors), so a free-text
    hit renders exactly like a citation hit once a seed is found - possibly
    several seeds at once if more than one node matches.
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "q is required"}), 400

    limit = int(request.args.get("limit", 150))
    seed_limit = int(request.args.get("seed_limit", 20))

    try:
        with neo4j_client.get_session() as session:
            citation = retriever.normalize_citation_query(q)
            matched_citation = None
            seed_rows = []
            if citation:
                seed_rows = list(session.run(
                    "MATCH (n) WHERE toLower(n.full_citation) = toLower($citation) "
                    "AND labels(n)[0] <> 'Chunk' RETURN n",
                    citation=citation,
                ))
                if seed_rows:
                    matched_citation = citation

            if not seed_rows:
                seed_rows = list(session.run(
                    """
                    MATCH (n)
                    WHERE labels(n)[0] <> 'Chunk' AND (
                        toLower(coalesce(n.full_citation, '')) CONTAINS toLower($q)
                        OR toLower(coalesce(n.case_name, '')) CONTAINS toLower($q)
                        OR toLower(coalesce(n.hasText, '')) CONTAINS toLower($q)
                        OR toLower(coalesce(n.short_title, '')) CONTAINS toLower($q)
                        OR toLower(coalesce(n.holding, '')) CONTAINS toLower($q)
                        OR toLower(coalesce(n.id, '')) CONTAINS toLower($q)
                    )
                    RETURN n
                    LIMIT $seed_limit
                    """,
                    q=q, seed_limit=seed_limit,
                ))

            if not seed_rows:
                return jsonify({
                    "nodes": [], "edges": [], "seed_ids": [],
                    "error": f"No node found matching {q!r}"
                             + (f" (read as citation {citation!r}, and " if citation else " (")
                             + "no free-text match across case names, issue/argument text, "
                               "section titles, holdings, or ids).",
                })

            nodes = {}
            seed_ids = []
            for row in seed_rows:
                n = graphrag_module.vertex_to_node(row["n"])
                nodes[n["db_id"]] = n
                seed_ids.append(n["db_id"])

            desc_rows = list(session.run(
                """
                MATCH (seed)<-[:PART_OF*1..6]-(d)
                WHERE elementId(seed) IN $seed_ids
                RETURN DISTINCT d
                LIMIT $limit
                """,
                seed_ids=seed_ids, limit=limit,
            ))
            for row in desc_rows:
                n = graphrag_module.vertex_to_node(row["d"])
                nodes[n["db_id"]] = n

            edges = {}
            all_ids = list(nodes.keys())
            edge_rows = list(session.run(
                "MATCH (a)-[r]->(b) WHERE elementId(a) IN $ids AND elementId(b) IN $ids RETURN r",
                ids=all_ids,
            ))
            for row in edge_rows:
                e = graphrag_module.edge_to_link(row["r"])
                edges[(e["start_id"], e["type"], e["end_id"])] = e

            neighbor_rows = list(session.run(
                "MATCH (a)-[r]-(b) WHERE elementId(a) IN $seed_ids RETURN r, b LIMIT $limit",
                seed_ids=seed_ids, limit=limit,
            ))
            for row in neighbor_rows:
                b = graphrag_module.vertex_to_node(row["b"])
                e = graphrag_module.edge_to_link(row["r"])
                nodes[b["db_id"]] = b
                edges[(e["start_id"], e["type"], e["end_id"])] = e

        masked_nodes = pii.mask_nodes(list(nodes.values()), context=f"/api/graph/search q={q!r}")
        return jsonify({
            "nodes": masked_nodes, "edges": list(edges.values()),
            "seed_ids": seed_ids, "citation": matched_citation,
        })
    except Exception as e:
        app.logger.exception("Failed in /api/graph/search")
        return jsonify({"error": str(e)}), 500


@app.route("/api/node/<node_db_id>/neighbors")
def api_node_neighbors(node_db_id):
    limit = int(request.args.get("limit", 25))
    try:
        with neo4j_client.get_session() as session:
            rows = list(session.run(
                "MATCH (a)-[r]-(b) WHERE elementId(a) = $node_db_id RETURN a, r, b LIMIT $limit",
                node_db_id=node_db_id, limit=limit,
            ))

            nodes, edges = {}, []
            for row in rows:
                a = graphrag_module.vertex_to_node(row["a"])
                b = graphrag_module.vertex_to_node(row["b"])
                r = graphrag_module.edge_to_link(row["r"])
                nodes[a["db_id"]] = a
                nodes[b["db_id"]] = b
                edges.append(r)

        masked_nodes = pii.mask_nodes(list(nodes.values()), context=f"/api/node/{node_db_id}/neighbors")
        return jsonify({"nodes": masked_nodes, "edges": edges})
    except Exception as e:
        app.logger.exception("Failed in /api/node/<id>/neighbors")
        return jsonify({"error": str(e)}), 500


@app.route("/api/node/<node_db_id>/hierarchy")
def api_node_hierarchy(node_db_id):
    """Backs double-click expansion in both graph panes: instead of just the
    node's immediate neighbors (see /api/node/<id>/neighbors above), this
    walks the PART_OF structural-hierarchy relationship both up (ancestors,
    e.g. SubClause -> Clause -> SubSection -> Section) and down (descendants)
    from the node, PLUS one hop of every OTHER relationship type touching the
    node, so double-click reveals the node's entire place in the document
    hierarchy AND everything directly connected to it by non-structural
    relationships (e.g. a CaseLaw's INTERPRETS-linked Issue nodes, cited
    Sections, precedent CaseLaw nodes, LegalConcepts) in one action - not
    just the PART_OF chain, which a case-law-style reasoning graph (Issue/
    Conclusion/Argument/precedent citations - all framework-agnostic, not
    PART_OF) never participates in at all. Three separate queries rather
    than one combined query with several variable-length MATCHes off the
    same seed node, to avoid a cartesian-product blowup between the
    expansions. Node types that never participate in a PART_OF chain (e.g.
    CaseLaw/Issue/Conclusion) simply come back with no extra ancestor/
    descendant rows from the first two queries - same "nothing extra to
    add" behavior already relied on elsewhere (see GraphRAGRetriever.
    get_parent_section_context) - and rely on the one-hop query instead.
    The one-hop query is capped (`neighbor_limit`, default 60) since some
    case-law relationship types (HAS_ARGUMENT) can fan out heavily - a
    bounded one-hop step here still lets further double-clicks on THOSE
    newly-added nodes drill in further, rather than trying to pull a whole
    reasoning graph in one shot."""
    limit = int(request.args.get("limit", 300))
    neighbor_limit = int(request.args.get("neighbor_limit", 60))
    try:
        with neo4j_client.get_session() as session:
            nodes, edges = {}, {}

            seed_rows = list(session.run(
                "MATCH (a) WHERE elementId(a) = $node_db_id RETURN a LIMIT 1",
                node_db_id=node_db_id,
            ))
            for row in seed_rows:
                a = graphrag_module.vertex_to_node(row["a"])
                nodes[a["db_id"]] = a

            # Ancestors: PART_OF chains up to a single root, so the longest
            # upward path already contains every shorter one as a prefix.
            anc_rows = list(session.run(
                "MATCH p = (a)-[:PART_OF*1..15]->(anc) WHERE elementId(a) = $node_db_id "
                "WITH p ORDER BY length(p) DESC LIMIT 1 "
                "UNWIND relationships(p) AS r "
                "RETURN startNode(r) AS x, r AS r, endNode(r) AS y",
                node_db_id=node_db_id,
            ))
            # Descendants: the PART_OF tree can branch (a Section can have
            # several SubSections), so collect every distinct edge across all
            # downward paths instead of just the longest one.
            desc_rows = list(session.run(
                "MATCH p = (a)<-[:PART_OF*1..15]-(desc) WHERE elementId(a) = $node_db_id "
                "UNWIND relationships(p) AS r "
                "RETURN DISTINCT startNode(r) AS x, r AS r, endNode(r) AS y LIMIT $limit",
                node_db_id=node_db_id, limit=limit,
            ))
            # One hop of every OTHER relationship type (any direction) - this
            # is what actually reaches a CaseLaw's Issue/precedent/citation/
            # LegalConcept nodes, none of which sit in a PART_OF chain.
            neighbor_rows = list(session.run(
                "MATCH (a)-[r]-(b) WHERE elementId(a) = $node_db_id AND type(r) <> 'PART_OF' "
                "RETURN a AS x, r AS r, b AS y LIMIT $neighbor_limit",
                node_db_id=node_db_id, neighbor_limit=neighbor_limit,
            ))

            for row in anc_rows + desc_rows + neighbor_rows:
                x = graphrag_module.vertex_to_node(row["x"])
                y = graphrag_module.vertex_to_node(row["y"])
                r = graphrag_module.edge_to_link(row["r"])
                nodes[x["db_id"]] = x
                nodes[y["db_id"]] = y
                edges[(r["start_id"], r["type"], r["end_id"])] = r

        masked_nodes = pii.mask_nodes(list(nodes.values()), context=f"/api/node/{node_db_id}/hierarchy")
        return jsonify({"nodes": masked_nodes, "edges": list(edges.values())})
    except Exception as e:
        app.logger.exception("Failed in /api/node/<id>/hierarchy")
        return jsonify({"error": str(e)}), 500


@app.route("/api/entity/<entity_id>/chunk")
def api_entity_chunk(entity_id):
    """Backs the graph view's node-hover popup: the entity's own node
    properties are already in the hoverNode payload the frontend already
    has (no round-trip needed for those) - this endpoint returns the
    Chunk-node-only metadata (chunk_id/chunk_type/element_type/short_title/
    page_number/file_type/etc.) that's stored on a SEPARATE Chunk node and
    was previously invisible on hover entirely."""
    try:
        chunks = retriever.get_chunk_properties_for_entity(entity_id)
        masked = pii.mask_nodes(
            [{"label": "Chunk", "properties": c} for c in chunks],
            context=f"/api/entity/{entity_id}/chunk",
        )
        return jsonify({"chunks": [m["properties"] for m in masked]})
    except Exception as e:
        app.logger.exception("Failed in /api/entity/<id>/chunk")
        return jsonify({"error": str(e)}), 500



# ============================================================================
# DATA EXPLORER - 2026-08-29: NEW. "View Neo4j data" panel, per explicit
# request: browse the Knowledge Index (the 4 local semantic_graph
# entry-point vector stores), Metadata (any Neo4j entity label), and Chunks,
# each searchable/filterable and paginated. See
# framework/core/retrieve/data_explorer.py for the read-only query engine -
# this section is just the thin Flask/PII-masking boundary around it, same
# division of responsibility as /api/graph/* above.
# ============================================================================
@app.route("/api/data-explorer/summary")
def api_data_explorer_summary():
    try:
        with neo4j_client.get_session() as session:
            return jsonify(data_explorer.get_summary(session))
    except Exception as e:
        app.logger.exception("Failed in /api/data-explorer/summary")
        return jsonify({"error": str(e)}), 500


@app.route("/api/data-explorer/nodes")
def api_data_explorer_nodes():
    label = request.args.get("label", "").strip()
    if not label:
        return jsonify({"error": "label is required"}), 400
    search = request.args.get("search", "")
    page = request.args.get("page", 1)
    page_size = request.args.get("page_size", data_explorer.DEFAULT_PAGE_SIZE)
    try:
        with neo4j_client.get_session() as session:
            result = data_explorer.list_nodes(session, label, search=search, page=page, page_size=page_size)
        result["rows"] = pii.mask_nodes(
            [{"label": label, "db_id": r["db_id"], "properties": r["properties"]} for r in result["rows"]],
            context=f"/api/data-explorer/nodes label={label!r} search={search!r}",
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Failed in /api/data-explorer/nodes")
        return jsonify({"error": str(e)}), 500


@app.route("/api/data-explorer/chunks")
def api_data_explorer_chunks():
    search = request.args.get("search", "")
    page = request.args.get("page", 1)
    page_size = request.args.get("page_size", data_explorer.DEFAULT_PAGE_SIZE)
    try:
        with neo4j_client.get_session() as session:
            result = data_explorer.list_chunks(session, search=search, page=page, page_size=page_size)
        result["rows"] = pii.mask_nodes(
            [{"label": "Chunk", "db_id": r["db_id"], "properties": r["properties"]} for r in result["rows"]],
            context=f"/api/data-explorer/chunks search={search!r}",
        )
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Failed in /api/data-explorer/chunks")
        return jsonify({"error": str(e)}), 500


@app.route("/api/data-explorer/index/<index_name>")
def api_data_explorer_index(index_name):
    search = request.args.get("search", "")
    page = request.args.get("page", 1)
    page_size = request.args.get("page_size", data_explorer.DEFAULT_PAGE_SIZE)
    try:
        with neo4j_client.get_session() as session:
            result = data_explorer.list_knowledge_index(session, index_name, search=search,
                                                          page=page, page_size=page_size)
        result["rows"] = pii.mask_nodes(
            [{"label": (r.get("labels") or ["?"])[0], "db_id": r["db_id"], "properties": r["properties"]}
             for r in result["rows"]],
            context=f"/api/data-explorer/index/{index_name} search={search!r}",
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Failed in /api/data-explorer/index/<index_name>")
        return jsonify({"error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
def api_ask():
    body = request.json or {}
    question = body.get("question", "").strip()
    metadata_filters = body.get("filters") or None
    # 2026-08-26: NEW. "hybrid" (default, unchanged behavior) vs
    # "vector_only" ("Simple RAG" toggle on the Ask screen - see
    # graphrag.GraphRAGRetriever.retrieve()'s retrieval_mode docstring).
    # Validated here (not trusted blindly) since it flows into Cypher-vs-not
    # branching downstream.
    # 2026-09-10: NEW - "agentic" added. Routes through
    # retrieval_orchestrator.run_agentic_retrieval() (bounded plan/retrieve/
    # sufficiency-check loop) instead of GraphRAGRetriever.retrieve()'s fixed
    # single pass - see _run_unstructured_agent()'s retrieval_mode=="agentic"
    # branch below. Not offered on /api/ask/fast, which is explicitly a
    # low-latency path (no NL2Cypher, optional rerank skip) - agentic mode's
    # multiple planning/sufficiency LLM calls run directly counter to that.
    retrieval_mode = body.get("retrieval_mode") or "hybrid"
    if retrieval_mode not in ("hybrid", "vector_only", "semantic_graph", "agentic", "structured_research"):
        return jsonify({"error": f"Invalid retrieval_mode {retrieval_mode!r} - must be 'hybrid', "
                                  f"'vector_only', 'semantic_graph', 'agentic', or 'structured_research'."}), 400
    # 2026-09-03: NEW - answer_mode axis (Lookup/Research/Opinion), separate
    # from retrieval_mode above. Defaults to "research" so a caller that
    # omits it (any existing UI/script/replay) behaves exactly as before.
    answer_mode = body.get("answer_mode") or "research"
    if answer_mode not in ("lookup", "research", "opinion"):
        return jsonify({"error": f"Invalid answer_mode {answer_mode!r} - must be 'lookup', "
                                  f"'research', or 'opinion'."}), 400
    # 2026-09-03: NEW - opt-in "auto-evaluate immediately after generation"
    # toggle. Accepts a bool loosely (any truthy JSON value) rather than
    # strictly validating type, same tolerance the rest of this route gives
    # optional flags - a malformed value just falls back to False (off).
    auto_evaluate = bool(body.get("auto_evaluate"))
    # 2026-09-05: NEW - opt-in "download the chunks this answer used" flag -
    # see _api_ask_impl()'s include_full_chunks docstring.
    include_full_chunks = bool(body.get("include_full_chunks"))
    # 2026-09-08: NEW - opaque token returned by POST /api/ask/attach_file,
    # NOT the Neo4j chat_history session id (see structured_data_workspace.
    # attach_file()'s docstring for why they're different tokens). Optional -
    # omitted/unknown -> get_attachment() returns None -> identical behavior
    # to before file attachment existed.
    data_session_id = body.get("data_session_id") or None
    if not question:
        return jsonify({"error": "question is required"}), 400
    try:
        attached_file_context = (
            structured_data_workspace.get_attachment(data_session_id) if data_session_id else None
        )
        # 2026-09-08: NEW - routes through the agentic-retriever orchestrator
        # (see _run_agentic_ask()'s module comment above) instead of calling
        # _api_ask_impl() directly. attached_file_context resolves to a real
        # dict once a file has been attached via POST /api/ask/attach_file
        # for this data_session_id; otherwise None, in which case
        # _classify_agent_routing() always resolves to UNSTRUCTURED_ONLY with
        # NO extra LLM call, and _run_unstructured_agent() calls
        # _api_ask_impl() exactly as this route already did - a zero-
        # behavior-change path for every caller that never attaches a file.
        response, agent_trace = _run_agentic_ask(
            question, metadata_filters, retrieval_mode, answer_mode,
            attached_file_context=attached_file_context,
            auto_evaluate=auto_evaluate, include_full_chunks=include_full_chunks,
        )
        # 2026-09-08: NEW - surfaced for the evidence panel's "Agents
        # invoked" step (index.html) - see this route's docstring above.
        response["agent_trace"] = agent_trace
        return jsonify(response)
    except Exception as e:
        # Every other route in this file fails into a JSON {"error": ...}
        # response - this one previously didn't, so a transient failure
        # anywhere in the pipeline (LLM provider unreachable, Neo4j down
        # mid-request, etc.) surfaced as a raw, bodiless 500 instead of
        # something the UI could actually show the user. Caught here,
        # narrowly, around just the pipeline body - not the request
        # validation above, which should keep failing fast with its own
        # specific 400s.
        app.logger.exception("Failed in /api/ask")
        return jsonify({"error": f"Something went wrong answering this question: {e}"}), 500


@app.route("/api/ask/attach_file", methods=["POST"])
def api_ask_attach_file():
    """Attaches a structured data file (.xlsx/.xls/.csv/.json) to be queried
    alongside the tax knowledge graph on this Ask screen - see
    structured_data_workspace.py's module docstring for the ephemeral,
    in-memory, never-persisted-to-Neo4j storage model this uses (same
    rationale as opinion_workspace.py: attached data may be real,
    confidential client data).

    Multipart upload, single field 'file' (one attachment per chat, not a
    list - see attach_file()'s docstring). Body also accepts an optional
    'data_session_id' form field: pass the token returned by a PRIOR call
    to this same route to REPLACE that session's attachment (e.g. the user
    swaps in a different file mid-chat); omit it on the first attach in a
    new chat and this route mints a fresh token and returns it - the
    frontend then holds that token in memory and sends it back on every
    subsequent attach/detach/ask call for this chat.

    Deliberately NOT keyed to the Neo4j chat_history session id: that id
    only exists once the first question/answer has been saved (see
    api_chats()'s docstring - create_session() requires a question), so a
    file attached before any question is asked would have no chat session
    id yet. This route's token exists independently of chat history."""
    if "file" not in request.files:
        return jsonify({"error": "No file (expected multipart field 'file')."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400
    data_session_id = (request.form.get("data_session_id") or "").strip() or None
    try:
        content = f.read()
        result = structured_data_workspace.attach_file(data_session_id, f.filename, content)
        return jsonify(result)
    except ValueError as e:
        # Bad extension or unparseable file - user-input problem, not a
        # server bug (see _parse_file()'s docstring).
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Failed in /api/ask/attach_file")
        return jsonify({"error": f"Could not attach file: {e}"}), 500


@app.route("/api/ask/detach_file", methods=["POST"])
def api_ask_detach_file():
    """Removes a chat's attached structured data file (e.g. the user clicks
    the 'x' next to the attached filename on the Ask screen). Idempotent -
    detaching a session with nothing attached is not an error."""
    body = request.json or {}
    data_session_id = (body.get("data_session_id") or "").strip()
    if not data_session_id:
        return jsonify({"error": "data_session_id is required"}), 400
    try:
        removed = structured_data_workspace.detach_file(data_session_id)
        return jsonify({"detached": removed})
    except Exception as e:
        app.logger.exception("Failed in /api/ask/detach_file")
        return jsonify({"error": str(e)}), 500


def _dedupe_semantic_graph_trace(trace: dict | None) -> dict | None:
    """2026-08-29: NEW. Collapses the raw semantic_graph_trace dict (which
    can have the SAME file appear multiple times - once from the main
    retrieve() pass, again from one or more category_retrieval.py fallback
    calls, e.g. the same Act .md file backing several different Section
    hits) into one row per distinct (name, path) pair, keeping the
    highest-similarity occurrence, sorted for a stable, readable UI list.
    Returns None unchanged if `trace` itself is None (non-semantic_graph
    requests)."""
    if trace is None:
        return None

    used_by_key: dict[tuple, dict] = {}
    for row in trace.get("files_used", []):
        key = (row.get("name", ""), row.get("path", ""))
        existing = used_by_key.get(key)
        if existing is None or row.get("similarity", 0) > existing.get("similarity", 0):
            used_by_key[key] = row
    files_used = sorted(used_by_key.values(), key=lambda r: r.get("similarity", 0), reverse=True)

    not_found_by_key: dict[tuple, dict] = {}
    for row in trace.get("files_not_found", []):
        key = (row.get("name", ""), row.get("path", ""), row.get("reason", ""))
        not_found_by_key[key] = row
    files_not_found = sorted(not_found_by_key.values(), key=lambda r: r.get("name", ""))

    return {"files_used": files_used, "files_not_found": files_not_found}


def _resolve_include_parent_context(retrieval_mode: str, include_parent_context: bool | None) -> bool:
    """2026-10-14: NEW. Whether _api_ask_impl() hydrates PART_OF parent-
    section context (GraphRAGRetriever.get_parent_section_context()) for its
    top chunks. An explicit True/False from the caller wins; None keeps the
    rule that existed before this parameter: every mode EXCEPT "vector_only",
    the graph-free Simple RAG ablation baseline (parent context comes from a
    PART_OF graph walk, so it must not leak into that baseline)."""
    if include_parent_context is not None:
        return bool(include_parent_context)
    return retrieval_mode != "vector_only"


def _api_ask_impl(question: str, metadata_filters: dict | None, retrieval_mode: str = "hybrid",
                   extra_gen_instructions: str = "", skip_internal_quality_gate: bool = False,
                   bypass_cache: bool = False, answer_mode: str = "research",
                   auto_evaluate: bool = False,
                   supplementary_chunks: list[dict] | None = None,
                   include_full_chunks: bool = False,
                   fast_mode: bool = False, skip_rerank: bool = False,
                   include_parent_context: bool | None = None,
                   mode_label: str | None = None) -> tuple[dict, dict]:
    """2026-08-26: EXTENDED with 4 new params and a new return shape.
    Returns (response, internal) - `response` is the exact JSON-able dict
    the /api/ask route sends to the client (unchanged shape); `internal` is
    {"top_chunks": [...], "plain_rows": [...]} - the full chunk TEXT and raw
    graph rows this answer was generated from, which the public `response`
    deliberately does NOT carry in full (it's already capped/summarized for
    the client). Added so the new quality-checked-ask background job
    (_run_quality_checked_ask_job() below) can feed the SAME material into
    answer_evaluation.evaluate_answer() without re-deriving it.

    New params, all no-ops at their defaults (so a plain call with just
    question/metadata_filters behaves byte-for-byte as before):
      - retrieval_mode: threaded into retriever.retrieve()/category_retrieval
        (see graphrag.py's docstring) - "hybrid" (default) or "vector_only".
      - extra_gen_instructions: appended into the FIRST generation call's
        context, same mechanism the internal wider-context retry below
        already uses for its own gap_text - lets an external caller (the
        quality-checked-ask retry loop) hand this call independent-evaluator
        feedback from a PRIOR attempt on the same question.
      - skip_internal_quality_gate: when True, skips the existing
        judge_answer()-driven wider-context retry below entirely (still
        computes `quality` for grounding_info, just doesn't act on it). The
        quality-checked-ask loop sets this on its 2nd+ attempt, so a single
        outer retry doesn't ALSO trigger an inner retry - each outer
        iteration should cost exactly one generation call once genuinely in
        feedback-driven-regeneration territory.
      - bypass_cache: when True, skips both the cache read at the top and
        the cache write at the bottom. The quality-checked-ask loop sets
        this for every attempt (mid-loop attempts must never read a stale
        cached answer for the same question/filters, and must never
        overwrite the shared cache with a not-yet-finalized draft) - it
        caches the FINAL chosen result itself, once, after the loop ends.
      - answer_mode: 2026-09-03 NEW - "research" (default, unchanged
        behavior - the customer-approved answer_generation a/b/c/d
        skeleton), "lookup" (answer_generation_lookup - a fast, direct
        single-answer mode), or "opinion" (answer_generation_opinion - a
        client-facing advisory mode). Selects which PROMPT_PACK entry drives
        the system prompt; see tax_prompts.py for all three templates. Part
        of the cache key so switching modes on the same question doesn't
        return a stale answer written for a different mode.
      - supplementary_chunks: 2026-09-05 NEW - optional extra candidate
        chunks (same {"chunk_id", "text", ...} shape retriever.retrieve()
        produces) to merge into this call's own retrieved candidate pool
        BEFORE reranking, deduped by chunk_id. Used by the quality-checked-
        ask retry loop (_run_quality_checked_ask_job below): when an
        attempt scores low, the loop runs a SEPARATE supplementary
        retrieval pass seeded by the evaluator's own feedback (see
        _fetch_feedback_supplementary_chunks()) and passes the result in
        here for the NEXT attempt, so that attempt's generation sees a
        WIDER, feedback-informed pool - not just the same evidence again
        with a "try harder" instruction, which is all the retry loop did
        before this parameter existed. Reranking still scores every
        candidate (original + supplementary) against the ORIGINAL question
        text, never the feedback text itself, so relevance ranking isn't
        biased toward chunks that merely happen to repeat the evaluator's
        wording. None (the default) is a complete no-op - every existing
        caller that doesn't pass this behaves byte-for-byte as before.
      - auto_evaluate: 2026-09-03 NEW - opt-in. When True, runs
        answer_evaluation v2 immediately after generation (reusing this
        call's own top_chunks/plain_rows, no second retrieval) and attaches
        the result as response["auto_evaluation"], for the UI to pop up as
        a scorecard right after the answer renders. Off by default - NOT
        part of the cache key (an evaluation is judged against generation
        material, not a property of the question/filters/mode combination
        the cache key already covers; a cache hit simply won't carry an
        auto_evaluation field, which the UI treats as "not requested for
        this cached turn" rather than an error).
      - include_full_chunks: 2026-09-05 NEW - opt-in. When True, attaches
        the FULL, untruncated text of every chunk in the final top_chunks
        (post any quality-gate retry above) as response["chunks_used"] -
        [{"chunk_id", "document_name", "page", "text", "rerank_score"}, ...]
        - so the UI can offer a "download the chunks this answer was built
        from" export. Off by default: response WITHOUT this flag deliberately
        does NOT carry full chunk text (see this function's main docstring -
        "the public response deliberately does NOT carry it in full"), only
        a truncated preview via grounding_info["source_references"] - this
        flag is the one opt-in exception to that rule, only when a caller
        (the Ask screen's "Download chunks" button flow) actually wants the
        full text sent back down.
      - fast_mode: 2026-09-07 NEW - opt-in low-latency path, per explicit
        request to bring response time down to a few seconds. Skips
        detect_intent() entirely (removes one LLM round-trip; OUT_OF_SCOPE
        short-circuiting is skipped too - a fast_mode question always goes
        through full retrieval rather than being classified out of scope
        first) and passes fast_mode=True into retriever.retrieve(), which
        uses a FIXED comprehensive Cypher traversal seeded from vector-search
        hits instead of NL2Cypher (removes a second LLM round-trip) - see
        graphrag.py's retrieve()/comprehensive_graph_retrieval() docstrings.
        Does NOT by itself skip reranking - see skip_rerank below, a
        separate flag, since a caller may want fast retrieval with or
        without the cross-encoder pass independently.
      - skip_rerank: 2026-09-07 NEW - opt-in. Passed straight through to
        reranker.rerank() (see that function's own skip_rerank docstring for
        exactly what changes) - bypasses the BGE cross-encoder pass, the
        single largest per-request latency cost measured in this project
        (90-200+ seconds for 25-30 pairs via Langfuse). Independent of
        fast_mode: a caller can request fast_mode retrieval with reranking
        still on, or normal retrieval with reranking off, though the two are
        expected to be used together in practice (see /api/ask/fast).
      - include_parent_context / mode_label: 2026-10-14 NEW, INTERNAL ONLY
        (not exposed on any route). Added for _run_unstructured_agent()'s
        retrieval_mode="agentic" branch, which reuses retrieval_mode=
        "vector_only" purely as the cheapest retrieval plumbing for an
        evidence pool the agentic orchestrator ALREADY gathered (graph
        included) - see _resolve_include_parent_context().
        include_parent_context=None (default) keeps the historical rule
        (parent-section context for every mode except the graph-free
        "vector_only" Simple RAG baseline); True/False forces it.
        mode_label (default: retrieval_mode) is the mode name this answer is
        cached under and reported as in response["retrieval_mode"]/evaluation
        logs - without it an agentic answer was cached under the SAME key as
        a real Simple RAG answer to the same question, so either mode could
        be served the other's cached answer.
    """

    # 2026-09-07: NEW - fast_mode skips detect_intent() (one LLM round-trip)
    # entirely, per this function's fast_mode docstring above. There is no
    # classified intent to check against OUT_OF_SCOPE in this path - a
    # fast_mode question always proceeds straight to retrieval rather than
    # being screened out first, trading that screening for speed.
    # 2026-10-13: BUGFIX - the cache READ key below included a
    # ':fast'/':normal' suffix but the cache WRITE at the end of this
    # function used only f"{retrieval_mode}:{answer_mode}", so the two keys
    # never matched and every /api/ask was a guaranteed cache MISS (the
    # cache was being written but never read back). Built ONCE here and
    # reused for both the read and the write so they can't drift apart again.
    # 2026-10-14: keyed on reported_mode (mode_label, else retrieval_mode) -
    # see this function's mode_label docstring for the agentic/vector_only
    # cache collision this prevents.
    reported_mode = mode_label or retrieval_mode
    use_parent_context = _resolve_include_parent_context(retrieval_mode, include_parent_context)
    cache_extra_key = f"{reported_mode}:{answer_mode}:{'fast' if fast_mode else 'normal'}"
    if not bypass_cache:
        cached = cache.get_cached_answer(question, metadata_filters, extra_key=cache_extra_key)
        if cached:
            cached["from_cache"] = True
            return cached, {"top_chunks": [], "plain_rows": []}

    run_id = cache.start_agent_run()
    trace = observability.start_trace("ask_pipeline", {"question": question}, run_id=run_id)

    detected_intent = None if fast_mode else detect_intent(question, DOMAIN_CONFIG, PROMPT_PACK, trace=trace)

    if not fast_mode:
        cache.update_agent_state(run_id, "intent_detection", {"intent": detected_intent})
        observability.log_span(trace, "intent_detection", question, detected_intent)

        if detected_intent == "OUT_OF_SCOPE":
            response = {
                "question": question,
                "intent": detected_intent,
                "cypher": None,
                "answer": f"That doesn't look related to the {DOMAIN_CONFIG.domain_name} knowledge graph. "
                          f"Try asking about entities, relationships, or specific citations.",
                "nodes": [], "edges": [], "source_chunks": [], "grounding": None,
            }
            cache.finish_agent_run(run_id)
            observability.end_trace(trace, response)
            observability.flush_langfuse()
            return response, {"top_chunks": [], "plain_rows": []}

    # 2026-08-29: NEW - one dict shared across the main retrieve() call AND
    # every category_retrieval.py fallback call below, so both accumulate
    # into the SAME "files_used"/"files_not_found" lists - see
    # semantic_entry_point_search()'s docstring in graphrag.py. Only
    # populated under retrieval_mode="semantic_graph"; stays None otherwise
    # so hybrid/vector_only requests don't pay for bookkeeping they don't use.
    semantic_graph_trace = {"files_used": [], "files_not_found": []} if retrieval_mode == "semantic_graph" else None

    retrieved = retriever.retrieve(question, intent=detected_intent, metadata_filters=metadata_filters,
                                    trace=trace, retrieval_mode=retrieval_mode,
                                    semantic_graph_trace=semantic_graph_trace, fast_mode=fast_mode)
    cache.update_agent_state(run_id, "graphrag_retrieve", {
        "cypher": retrieved["cypher"],
        "n_graph_rows": len(retrieved["plain_rows"]),
        "n_candidate_chunks": len(retrieved["candidate_chunks"]),
        "retrieval_scope": retrieved["retrieval_scope"],
    })
    observability.log_span(trace, "graphrag_retrieve", question, {
        "cypher": retrieved["cypher"], "n_candidates": len(retrieved["candidate_chunks"]),
        "retrieval_scope": retrieved["retrieval_scope"],
    })

    # 2026-09-05: NEW - merge in any supplementary_chunks (see this
    # function's docstring), deduped by chunk_id, BEFORE reranking - so the
    # existing rerank call below transparently considers the wider pool with
    # no changes of its own, and still ranks everything against the
    # ORIGINAL `question` (never the feedback text that produced the
    # supplementary chunks). A supplementary chunk that duplicates one
    # already in candidate_chunks is skipped (first occurrence wins) rather
    # than scored twice.
    candidate_chunks = retrieved["candidate_chunks"]
    if supplementary_chunks:
        seen_chunk_ids = {str(c.get("chunk_id")) for c in candidate_chunks}
        merged_extra = [c for c in supplementary_chunks if str(c.get("chunk_id")) not in seen_chunk_ids]
        if merged_extra:
            candidate_chunks = candidate_chunks + merged_extra
            app.logger.info(f"[_api_ask_impl] merged {len(merged_extra)} feedback-driven supplementary "
                             f"chunk(s) into the candidate pool ({len(retrieved['candidate_chunks'])} -> "
                             f"{len(candidate_chunks)}) before reranking.")

    # 2026-09-03: was a hardcoded top_n=25 - now sourced from
    # reranker_config.load_config() (a Settings-page control) when not
    # explicitly overridden, so a Settings edit changes what reaches the LLM
    # on the very next question. See reranker_config.py's docstring.
    # 2026-09-07: NEW - skip_rerank threaded straight through, per this
    # function's own skip_rerank docstring / reranker.rerank()'s.
    top_chunks = reranker.rerank(question, candidate_chunks, skip_rerank=skip_rerank)
    cache.update_agent_state(run_id, "bge_rerank", {"top_chunk_ids": [c["chunk_id"] for c in top_chunks]})
    # 2026-08-29: routed through summarize_chunks_for_trace() - candidate_chunks
    # can carry full whole-file-hydrated "text" under semantic_graph mode
    # (up to 4MB each), and 30-39 of them blew past Langfuse's per-item size
    # limit, silently dropping this span entirely. See that function's
    # docstring in observability.py.
    observability.log_span(trace, "bge_rerank",
                            observability.summarize_chunks_for_trace(retrieved["candidate_chunks"]),
                            observability.summarize_chunks_for_trace(top_chunks))

    # 2026-08-26: NEW. Tax Knowledge & Analysis Framework category coverage -
    # runs graph-first, vector-fallback retrieval for EACH of the six
    # framework categories (Fact Sheet, Law & Regulatory Framework, Case Law
    # Hierarchy, Opinion & Expert Views, Client Specific Data, Final Output &
    # Impact Analysis), independent of whatever the general hybrid retrieval
    # above already found - see category_retrieval.py's module docstring for
    # why this exists as a SEPARATE pass. Skipped entirely if the loaded
    # domain pack doesn't define CATEGORY_SPECS (see PACK loading above).
    category_results = []
    if CATEGORY_SPECS:
        try:
            category_results = category_retrieval.run_category_retrieval(
                question, retriever, CATEGORY_SPECS, reranker_module=reranker, retrieval_mode=retrieval_mode,
                semantic_graph_trace=semantic_graph_trace,
            )
        except Exception as e:
            # Best-effort - a failure here should degrade to "no category
            # coverage this time," not break the whole /api/ask call.
            app.logger.warning(f"[category_retrieval] run_category_retrieval failed: {e}")
            category_results = []
        cache.update_agent_state(run_id, "category_retrieval",
                                  {"categories": category_retrieval.summarize_category_coverage(category_results)})
        observability.log_span(trace, "category_retrieval", question,
                                category_retrieval.summarize_category_coverage(category_results))

    grounding_info = grounding.ground_answer(question, top_chunks, retrieved["plain_rows"])

    # Attach document-level provenance (document_id/name/source_type/
    # authority_level/jurisdiction + the source entity's own citation) to
    # every cited chunk, so the UI can show "which document did this come
    # from" alongside each citation, not just the chunk text - requirement:
    # answers should surface chunk details, document details, citations,
    # relationships, and the retrieval path used.
    doc_details = retriever.get_document_details([c["chunk_id"] for c in top_chunks])
    # Parent-child chunking, retrieval half: top_chunks are the small,
    # precise units vector search/reranking matched on - this pulls each
    # one's full parent Section (or GenericDocument) text too, so answer
    # generation below sees the complete surrounding provision, not just an
    # isolated clause. See GraphRAGRetriever.get_parent_section_context().
    # 2026-08-26: skipped in retrieval_mode="vector_only" - this expansion
    # walks the graph's PART_OF relationships, so including it would let
    # graph-derived content leak into the "Simple RAG" baseline's context,
    # defeating the point of the comparison (see graphrag.py's docstring).
    # 2026-10-14: decided by _resolve_include_parent_context() (same default
    # rule, plus the internal include_parent_context override agentic uses).
    parent_context = retriever.get_parent_section_context([c["chunk_id"] for c in top_chunks]) \
        if use_parent_context else {}
    for ref in grounding_info["source_references"]:
        ref["document"] = doc_details.get(str(ref["chunk_id"]), {})
        ref["parent_context"] = parent_context.get(str(ref["chunk_id"]), {})

    cache.update_agent_state(run_id, "citation_grounding", grounding_info)
    # 2026-08-29: same oversized-trace reasoning as the "bge_rerank" span above.
    observability.log_span(trace, "citation_grounding",
                            observability.summarize_chunks_for_trace(top_chunks), grounding_info)

    # "Nodes actually used to answer this question" for the UI's graph
    # view - retrieved["nodes"] alone only covers the NL2Cypher-matched
    # subgraph, which is EMPTY whenever a question was answered purely via
    # vector search (the common case for a broad/semantic question with no
    # exact citation to match on - graphrag.py's retrieve() still ran, but
    # `graph_result["nodes"]` never got populated because there was no
    # Cypher query to run). Merging in each top (post-rerank) chunk's
    # linked entity closes that gap - the graph view can then show exactly
    # the same nodes cited in the answer, whichever retrieval path found
    # them.
    vector_matched_entities = pii.mask_nodes(
        retriever.get_entities_for_chunks([c["chunk_id"] for c in top_chunks]),
        context=f"/api/ask (vector-matched) question={question[:80]!r}",
    )
    retrieval_nodes_by_id = {n["db_id"]: n for n in retrieved["nodes"]}
    for n in vector_matched_entities:
        retrieval_nodes_by_id.setdefault(n["db_id"], n)
    retrieval_nodes = list(retrieval_nodes_by_id.values())

    # Relationships DIRECTLY connecting the final node set above (e.g. PART_OF
    # between a Section and its SubSections) - retrieved["edges"] on its own
    # only ever contains edges the NL2Cypher query itself matched, which is
    # empty/sparse for a vector-search-driven answer even when the real
    # hierarchy between the displayed nodes exists in the graph. Without this,
    # the graph view shows correctly-labeled but fully disconnected nodes
    # (reported directly by a user for "what deductions are allowed in
    # section 80C" - Section_80C and a dozen SubSections with no lines
    # between them).
    retrieval_edges_by_id = {e["db_id"]: e for e in retrieved["edges"]}
    for e in retriever.get_edges_among_nodes([n["db_id"] for n in retrieval_nodes]):
        retrieval_edges_by_id.setdefault(e["db_id"], e)
    retrieval_edges = list(retrieval_edges_by_id.values())

    # 2026-08-26: EXTENDED. Previously this only checked the general hybrid
    # retrieval (plain_rows/top_chunks) - now also checks whether the
    # category-coverage pass above found ANYTHING for any category (e.g. a
    # question that the general retrieval's single Cypher/vector pass missed
    # but a category's own targeted graph-keyword or vector-fallback search
    # caught). Only bails out to the "nothing found" message when BOTH have
    # come up empty.
    category_found_something = any(r.status in ("graph", "vector_fallback") for r in category_results)
    if not retrieved["plain_rows"] and not top_chunks and not category_found_something:
        response = {
            "question": question,
            "intent": detected_intent,
            "cypher": retrieved["cypher"],
            "answer": "I couldn't find anything in the knowledge graph or source documents that "
                      "answers this question, so I'm not going to guess. Try rephrasing, or ask "
                      "about a specific entity/citation.",
            "nodes": retrieved["nodes"],
            "edges": retrieved["edges"],
            "source_chunks": [],
            "grounding": grounding_info,
            "retrieval_scope": retrieved["retrieval_scope"],
            "category_coverage": category_retrieval.summarize_category_coverage(category_results),
            "from_cache": False,
        }
        cache.finish_agent_run(run_id)
        observability.end_trace(trace, response)
        observability.flush_langfuse()
        return response, {"top_chunks": [], "plain_rows": retrieved["plain_rows"]}

    # 2026-09-03 NEW - answer_mode selects which PROMPT_PACK entry drives
    # this call. "research" (default) keeps the exact prior behavior
    # (answer_generation, Settings-override-eligible). "lookup"/"opinion" are
    # real, separate prompt names (not versions of answer_generation - see
    # tax_prompts.py's module comment on why PromptPack.get_prompt() only
    # returning the LATEST version of a name rules out doing this as
    # versions) with their own fully-specified structure - the Settings-page
    # system-prompt override (prompt_config.py) intentionally stays scoped
    # to "research" mode only; lookup/opinion have no override UI yet.
    _ANSWER_MODE_PROMPT_NAMES = {
        "research": "answer_generation",
        "lookup": "answer_generation_lookup",
        "opinion": "answer_generation_opinion",
    }
    answer_prompt_name = _ANSWER_MODE_PROMPT_NAMES.get(answer_mode, "answer_generation")
    answer_prompt = PROMPT_PACK.get_prompt(answer_prompt_name)
    if answer_mode == "research":
        # A user-saved Settings-page override (see prompt_config.py) takes
        # precedence over the domain pack's built-in template when enabled -
        # resolved fresh on every call, same hot-reload pattern as
        # response_config.py/llm_client's refresh_model_config(), so an edit in
        # the UI takes effect on the very next question, no restart needed.
        answer_template, prompt_is_override = prompt_config.resolve_answer_system_prompt(answer_prompt.template)
        # A custom override (or the built-in v4 master research prompt) already
        # fully specifies its own mandatory response structure - appending the
        # generic user-configurable "Intent/Direct Answer/Citations" 3-section
        # directive on top would fight it with conflicting structure
        # instructions. Only append the directive for prompts that actually rely
        # on it (v1-v3, which were deliberately left structure-light).
        prompt_defines_own_structure = prompt_is_override or answer_prompt.version not in ("v1", "v2", "v3")
    else:
        # lookup/opinion prompts always fully specify their own structure -
        # no Settings override, no generic structure directive.
        answer_template, prompt_is_override = answer_prompt.template, False
        prompt_defines_own_structure = True
    resp_cfg = response_config.load_response_config()
    answer_system_prompt = answer_template
    if not prompt_defines_own_structure:
        answer_system_prompt += response_config.build_structure_directive(resp_cfg)

    # 2026-08-29: NEW. Favourable/unfavourable-, for/against-, comparison-,
    # hold/withhold-type questions get a DEDICATED structure (see
    # comparative_response.py) - unlike response_config's directive above,
    # this is appended REGARDLESS of prompt_defines_own_structure, because a
    # comparative question's structure need is a property of the QUESTION,
    # not of which prompt version happens to be active (the master v4/v5
    # prompt's own default structure isn't built for a side-by-side
    # Revenue-vs-taxpayer breakdown). `comparative_structure_applied` is
    # surfaced in the response so the UI/evidence trail can show it fired.
    comparative_cfg = comparative_response.load_config()
    comparative_structure_applied = comparative_response.is_comparative_question(question, comparative_cfg)
    if comparative_structure_applied:
        answer_system_prompt += comparative_response.build_structure_directive(comparative_cfg)

    def _chunk_source_label(chunk_id, doc_details_map: dict) -> str:
        """Builds the bracketed label shown above each chunk's text in the
        LLM's own context - the ONLY place the model can learn a chunk's
        real document name/page. Previously this label was just
        "chunk_id=N", so no answer_generation prompt version could ever
        instruct the model to cite "<document name> - Page N" (the format
        the customer's expected-output samples use throughout) - the model
        simply didn't have that information available, and fell back to
        citing the internal chunk_id instead. Falls back gracefully through
        document_name+page -> document_name alone -> entity/chunk citation
        -> chunk_id alone, mirroring the answer_generation prompt's own
        citation-hierarchy fallback instructions (see tax_prompts.py v5,
        Section 8)."""
        detail = doc_details_map.get(str(chunk_id), {}) or {}
        doc_name = detail.get("document_name")
        page = detail.get("page")
        if doc_name and page:
            return f"chunk_id={chunk_id} | {doc_name}, p. {page}"
        if doc_name:
            return f"chunk_id={chunk_id} | {doc_name}"
        fallback_citation = detail.get("entity_citation") or detail.get("chunk_citation")
        if fallback_citation:
            return f"chunk_id={chunk_id} | {fallback_citation}"
        return f"chunk_id={chunk_id}"

    def _build_context_parts(chunks: list[dict], extra_instructions: str = "",
                              doc_details_map: dict | None = None, parent_context_map: dict | None = None) -> list[str]:
        # Default to the outer (first-pass) doc_details/parent_context when
        # no override is given - a caller regenerating over a WIDER chunk
        # pool (see the quality-gate retry below) passes its own maps in,
        # since the outer ones only cover the original top_chunks and would
        # otherwise silently degrade every new chunk's citation label back
        # to a bare chunk_id.
        doc_details_map = doc_details_map if doc_details_map is not None else doc_details
        parent_context_map = parent_context_map if parent_context_map is not None else parent_context

        parts = [f"Question: {question}", f"Detected intent: {detected_intent}"]
        if extra_instructions:
            parts.append(extra_instructions)
        # 2026-08-26: NEW. Tax Knowledge & Analysis Framework category
        # coverage block - see category_retrieval.py's format_category_context()
        # and the category_results computed above. Included in every
        # generation call (first pass AND the quality-gate retry below, since
        # both share this function) so the model can structure its response
        # under the six category headings with real evidence-or-gap
        # information for each one, not just guess.
        if category_results:
            parts.append(category_retrieval.format_category_context(category_results))
        if retrieved["plain_rows"]:
            rows_json = grounding.graph_rows_to_context_json(retrieved["plain_rows"])
            parts.append(f"Graph query results (JSON):\n{rows_json}")
        if chunks:
            chunk_block = "\n\n---\n\n".join(
                f"[{_chunk_source_label(c['chunk_id'], doc_details_map)}]\n{c['text'][:1500]}" for c in chunks
            )
            parts.append(f"Retrieved source text chunks (reranked by relevance):\n{chunk_block}")

            # Parent-child chunking: also give the LLM each matched chunk's
            # full parent Section/GenericDocument text, not just the
            # isolated small chunk - deduplicated (several chunks often
            # share one parent) and capped per-parent so one huge Section
            # can't blow the prompt budget the way the raw chunk text above
            # is already capped.
            seen_parents = set()
            parent_blocks = []
            for c in chunks:
                pc = parent_context_map.get(str(c["chunk_id"]))
                if not pc or not pc.get("parent_text") or pc["parent_id"] in seen_parents:
                    continue
                seen_parents.add(pc["parent_id"])
                parent_blocks.append(f"[{pc.get('parent_label', 'parent')}={pc['parent_id']} | "
                                      f"{pc.get('parent_citation', '')}]\n{pc['parent_text'][:3000]}")
            if parent_blocks:
                parts.append(
                    "Full parent-section context for the chunks above (for comprehensive understanding - "
                    "still cite the precise document/page citation above when answering, not just the parent):\n"
                    + "\n\n---\n\n".join(parent_blocks)
                )
        return parts

    def _generate(chunks: list[dict], call_name: str, extra_instructions: str = "",
                   doc_details_map: dict | None = None, parent_context_map: dict | None = None) -> str:
        context_parts = _build_context_parts(chunks, extra_instructions, doc_details_map, parent_context_map)
        text = llm_client.complete(
            system=answer_system_prompt, user_content="\n\n".join(context_parts), max_tokens=5000,
            trace=trace, name=call_name,
        ).strip()
        observability.log_span(trace, call_name, context_parts, text)
        return text

    answer_text = _generate(top_chunks, "answer_generation", extra_gen_instructions)

    # Quality gate: an independent LLM pass grades THIS answer (not the
    # retrieval candidates) on groundedness/completeness/specificity - see
    # framework.core.retrieve.answer_quality. If it scores below the bar
    # and there's actually more candidate material available than we used
    # (top_chunks was already capped at 25 by reranker.rerank() above but
    # retrieved["candidate_chunks"] may hold more), retry ONCE with a wider
    # pool and the judge's own gap list fed back in as instructions - a
    # bounded improvement pass, not an open retry loop. If the retry still
    # scores low (or there was nothing more to draw on), the better of the
    # two attempts is returned anyway, honestly, with its score attached.
    # 2026-09-07: NEW - fast_mode skips this judge call ENTIRELY (not just
    # its retry action below) - judge_answer() is itself a full LLM
    # round-trip that ran unconditionally even when skip_internal_quality_gate
    # only suppressed the retry it could trigger. fast_mode's own outer
    # dual-evaluator retry loop (see _run_fast_quality_checked_ask() below)
    # is what actually drives quality in this path, making this inner gate
    # redundant latency for fast_mode specifically - it stays fully active
    # for every non-fast caller, unchanged.
    if fast_mode:
        quality = {"available": False, "overall": None, "gaps": [],
                   "reason": "Skipped under fast_mode - see _api_ask_impl's fast_mode docstring."}
    else:
        quality = answer_quality.judge_answer(question, answer_text, top_chunks, retrieved["plain_rows"], trace=trace)
        observability.log_span(trace, "answer_quality_judge", answer_text, quality)

    # 2026-08-26: NEW guard - skip_internal_quality_gate. The quality-checked-
    # ask background job (see _run_quality_checked_ask_job() below) already
    # runs its OWN outer retry loop driven by the richer answer_evaluation
    # scorecard; without this guard, its 2nd+ attempts would each ALSO
    # trigger this inner wider-context retry, making retry cost/latency
    # unpredictable (1 or 2 generation calls per outer attempt depending on
    # this gate). `quality` is still computed above either way (grounding_info
    # below expects it) - only the RETRY ACTION is skipped. fast_mode implies
    # this guard too (quality["available"] is already False above, so the
    # condition below short-circuits regardless, but skip_internal_quality_gate
    # is also treated as True for fast_mode for clarity/defense-in-depth).
    more_candidates_available = len(retrieved["candidate_chunks"]) > len(top_chunks)
    if not (skip_internal_quality_gate or fast_mode) and quality["available"] and quality["overall"] is not None \
            and quality["overall"] < answer_quality.QUALITY_MIN_SCORE and more_candidates_available:
        wider_chunks = reranker.rerank(question, retrieved["candidate_chunks"], top_n=40,
                                        min_relevance=0.3)  # wider net + lower bar than the primary pass
        # Look these up BEFORE generating the retry, not after (as this used
        # to) - _build_context_parts()/_chunk_source_label() need a
        # doc_details/parent_context map that actually covers the WIDER
        # chunk set at generation time so the retry's own citations can name
        # real documents/pages, not just fall back to bare chunk_id for
        # every chunk introduced by widening. Kept in separate wider_*
        # variables (not assigned into the outer doc_details/parent_context)
        # until we know the retry actually wins, so a losing retry can't
        # leave those names pointing at the wrong chunk set for the
        # first-pass answer_text/top_chunks being kept below.
        wider_ids = [c["chunk_id"] for c in wider_chunks]
        wider_doc_details = retriever.get_document_details(wider_ids)
        # 2026-08-26: same vector_only guard as the first-pass parent_context above.
        wider_parent_context = retriever.get_parent_section_context(wider_ids) if use_parent_context else {}

        gap_text = "; ".join(quality["gaps"]) if quality["gaps"] else "the answer was judged too thin or under-cited"
        retry_instructions = (
            f"IMPORTANT: A prior draft of this answer was reviewed and found lacking ({gap_text}). "
            f"You now have a WIDER set of source chunks below than before - use them to give a more "
            f"complete, detailed, precisely-cited answer. Do not repeat the same gaps."
        )
        retry_text = _generate(wider_chunks, "answer_generation_retry", retry_instructions,
                                doc_details_map=wider_doc_details, parent_context_map=wider_parent_context)
        retry_quality = answer_quality.judge_answer(question, retry_text, wider_chunks, retrieved["plain_rows"], trace=trace)
        observability.log_span(trace, "answer_quality_judge_retry", retry_text, retry_quality)

        # Keep whichever attempt actually scored better - the retry is a
        # genuine attempt to improve, not an assumed improvement.
        retry_is_better = (
            retry_quality["available"] and retry_quality["overall"] is not None
            and (not quality["available"] or quality["overall"] is None or retry_quality["overall"] >= quality["overall"])
        )
        if retry_is_better:
            answer_text, top_chunks, quality = retry_text, wider_chunks, retry_quality
            doc_details, parent_context = wider_doc_details, wider_parent_context
            grounding_info = grounding.ground_answer(question, top_chunks, retrieved["plain_rows"])
            for ref in grounding_info["source_references"]:
                ref["document"] = doc_details.get(str(ref["chunk_id"]), {})
                ref["parent_context"] = parent_context.get(str(ref["chunk_id"]), {})

    grounding_info["answer_quality"] = quality

    citation_check = grounding.verify_citations(answer_text, top_chunks, retrieved["plain_rows"],
                                                 doc_details=doc_details)
    grounding_info["citation_check"] = citation_check
    if not citation_check["citation_ok"]:
        answer_text += ("\n\n(Note: this answer should have cited specific sources but didn't - "
                         "treat it with extra caution and check the retrieved context yourself.)")
    observability.log_span(trace, "citation_verification", answer_text, citation_check)

    # 2026-08-29: NEW - best-effort, non-blocking check for whether the
    # model actually followed comparative_response.py's structure (see that
    # module's docstring) - per explicit report that a comparative question's
    # answer wasn't following the configured structure. Only runs when the
    # directive was actually appended (comparative_structure_applied), and
    # checks the FINAL answer_text (post quality-gate retry, if any) - the
    # same "flag, don't silently guess" pattern as citation_check above.
    comparative_structure_check = (
        comparative_response.check_structure_compliance(answer_text, comparative_cfg)
        if comparative_structure_applied else {"checked": False}
    )
    grounding_info["comparative_structure_check"] = comparative_structure_check

    # Best-effort, non-blocking sanity check on the "d. Referred Sources"
    # bibliography an answer_generation v5-style prompt asks the model to
    # compile - a compiled reference list is the part of this response
    # format most prone to an LLM inventing a plausible-looking source it
    # never actually cited in the body. Never alters answer_text or blocks
    # the response - just logs a warning for review if something doesn't
    # match an actually-retrieved document.
    referred_sources_check = grounding.check_referred_sources(answer_text, doc_details)
    grounding_info["referred_sources_check"] = referred_sources_check

    # 2026-09-03: NEW - optional, opt-in "auto-evaluate immediately after
    # generation" - a SEPARATE thing from both answer_quality.judge_answer()
    # above (a thin internal gate) and the manual "Evaluate this answer"
    # button (answer_evaluation v1, re-retrieves fresh). This reuses the
    # SAME top_chunks/plain_rows this call already retrieved/generated from
    # (no second retrieval pass - the whole point is "evaluate what was
    # actually just used", and it's already sitting right here), and runs
    # answer_evaluation v2 (Legal accuracy / Reasoning quality / Statutory
    # precision / Law/fact distinction / Authority handling / Completeness +
    # Overall - see tax_prompts.py). Off by default (auto_evaluate=False) -
    # opt-in toggle per explicit request, since it's one extra LLM call and
    # a few seconds of latency on top of every question it's enabled for.
    # Best-effort: a failure here must never break the answer itself.
    auto_evaluation = None
    if auto_evaluate:
        try:
            auto_eval_prompt = PROMPT_PACK.get_prompt("answer_evaluation", version="v2").template
            auto_evaluation = answer_evaluation.evaluate_answer(
                question, answer_text, top_chunks, retrieved["plain_rows"], auto_eval_prompt, trace=trace,
            )
            observability.log_span(trace, "answer_evaluation_auto", answer_text, auto_evaluation)
            try:
                evaluation_log.record_evaluation(question, answer_text, auto_evaluation,
                                                  source="auto_post_generation", retrieval_mode=reported_mode)
            except Exception as log_err:
                app.logger.warning(f"evaluation_log.record_evaluation() failed (non-fatal): {log_err}")
        except Exception as e:
            app.logger.warning(f"[auto_evaluate] answer_evaluation failed (non-fatal): {e}")
            auto_evaluation = {"available": False, "overall": None, "dimensions": [], "verdict": "",
                                "strengths": [], "issues": [],
                                "reason": f"Auto-evaluation failed: {e}"}

    # 2026-09-05: NEW - a SECOND, independent scorecard, run alongside
    # auto_evaluation above under the SAME auto_evaluate toggle (no new UI
    # control - per explicit request "nothing else changes"). Context: a
    # customer was pasting answers into ChatGPT and asking it to grade them
    # cold (no retrieved context supplied, since the customer doesn't have
    # it) - producing a different score than auto_evaluation above, which
    # deliberately grades ONLY against the retrieved context. Both questions
    # are legitimate but different ("is this traceable to what we retrieved"
    # vs "is this correct as a matter of actual tax law"), so this is a
    # SEPARATE field, not a replacement - see general_knowledge_evaluation's
    # prompt-registration comment in tax_prompts.py for the full reasoning.
    # Calls the exact same evaluate_answer() with top_chunks=[]/plain_rows=[]
    # and no_context_mode=True so the model has nothing to ground against and
    # can only draw on its own knowledge - closer to replicating the
    # customer's blind ChatGPT-paste comparison. Best-effort, same pattern as
    # auto_evaluation above: a failure here must never break the answer.
    general_knowledge_evaluation = None
    if auto_evaluate:
        try:
            gk_eval_prompt = PROMPT_PACK.get_prompt("general_knowledge_evaluation", version="v1").template
            general_knowledge_evaluation = answer_evaluation.evaluate_answer(
                question, answer_text, [], [], gk_eval_prompt, trace=trace, no_context_mode=True,
            )
            observability.log_span(trace, "answer_evaluation_general_knowledge",
                                    answer_text, general_knowledge_evaluation)
            try:
                evaluation_log.record_evaluation(question, answer_text, general_knowledge_evaluation,
                                                  source="auto_general_knowledge", retrieval_mode=reported_mode)
            except Exception as log_err:
                app.logger.warning(f"evaluation_log.record_evaluation() failed (non-fatal): {log_err}")
        except Exception as e:
            app.logger.warning(f"[auto_evaluate] general_knowledge_evaluation failed (non-fatal): {e}")
            general_knowledge_evaluation = {"available": False, "overall": None, "dimensions": [], "verdict": "",
                                             "strengths": [], "issues": [],
                                             "reason": f"General-knowledge evaluation failed: {e}"}

    # 2026-09-05: NEW - optional, opt-in full-chunk-text export (see this
    # function's docstring). Built from the FINAL top_chunks/doc_details
    # (post any quality-gate retry above, so a downloaded chunk set always
    # matches what the on-screen answer was actually generated from) - same
    # document-name/page label lookup _chunk_source_label() already uses for
    # the LLM's own context, reused here (not duplicated) so the exported
    # label always matches what the model actually saw. Best-effort: a
    # failure here must never break the answer itself.
    chunks_used = None
    if include_full_chunks:
        try:
            chunks_used = [
                {
                    "chunk_id": c["chunk_id"],
                    "document_name": (doc_details.get(str(c["chunk_id"]), {}) or {}).get("document_name"),
                    "page": (doc_details.get(str(c["chunk_id"]), {}) or {}).get("page"),
                    "rerank_score": c.get("rerank_score"),
                    "text": c.get("text", ""),
                }
                for c in top_chunks
            ]
        except Exception as e:
            app.logger.warning(f"[include_full_chunks] building chunks_used failed (non-fatal): {e}")
            chunks_used = None

    # Hardcoded disclaimer, appended server-side rather than trusted to the
    # LLM to reproduce verbatim on every call - the customer's expected
    # output samples all use the exact same disclaimer paragraph
    # word-for-word, and the active answer_generation prompt (v5) explicitly
    # instructs the model NOT to write one itself (see tax_prompts.py).
    # Optional per domain pack - a domain pack with no "answer_disclaimer"
    # prompt entry just doesn't get one, keeping this file domain-agnostic.
    try:
        disclaimer_text = PROMPT_PACK.get_prompt("answer_disclaimer").template.strip()
    except KeyError:
        disclaimer_text = None
    if disclaimer_text:
        answer_text += f"\n\nDISCLAIMER\n{disclaimer_text}"

    response = {
        "question": question,
        "intent": detected_intent,
        "cypher": retrieved["cypher"],
        "answer": answer_text,
        "nodes": retrieval_nodes,
        "edges": retrieval_edges,
        "source_chunks": grounding_info["source_references"],
        "grounding": grounding_info,
        "retrieval_scope": retrieved["retrieval_scope"],
        # 2026-08-26: NEW. Lightweight (status + chunk counts only, no chunk
        # text - see summarize_category_coverage()) so the UI can eventually
        # show which framework categories had evidence, without bloating the
        # response payload with the full context text already sent to the LLM.
        "category_coverage": category_retrieval.summarize_category_coverage(category_results),
        "retrieval_mode": reported_mode,  # 2026-08-26: NEW - surfaced so the UI can label which mode produced this answer (2026-10-14: mode_label-aware)
        "answer_mode": answer_mode,  # 2026-09-03: NEW - surfaced so the UI can badge lookup/research/opinion
        # 2026-09-05: NEW - count of feedback-driven supplementary chunks
        # actually merged in (0 if supplementary_chunks was None/empty, or
        # every one of them happened to already be in the original pool) -
        # lets the quality-checked-ask progress log/iteration history show
        # whether a retry attempt's wider evidence pool actually added
        # anything new, not just that it was attempted.
        "supplementary_chunks_merged": len(candidate_chunks) - len(retrieved["candidate_chunks"]),
        # 2026-09-03: NEW - only present (non-null) when auto_evaluate=True was
        # requested - see this function's docstring. Same shape as the manual
        # "Evaluate this answer" button's response (answer_evaluation.evaluate_answer()'s
        # return dict), but scored on v2's dimension set and computed unconditionally
        # server-side rather than requiring a second round-trip from the UI.
        "auto_evaluation": auto_evaluation,
        # 2026-09-05: NEW - only present (non-null) when auto_evaluate=True was
        # requested (same toggle as auto_evaluation above - see this
        # function's docstring and the general_knowledge_evaluation block
        # above for why this is a SEPARATE field rather than folded into
        # auto_evaluation). Same {available, overall, dimensions, verdict,
        # strengths, issues} shape, but graded with no retrieved context at
        # all - the model's own general tax-law knowledge only.
        "general_knowledge_evaluation": general_knowledge_evaluation,
        # 2026-09-05: NEW - only present (non-null) when include_full_chunks=True
        # was requested - see this function's docstring. Powers the Ask
        # screen's "Download chunks used" button.
        "chunks_used": chunks_used,
        # 2026-08-29: NEW - True when this question matched the comparative/
        # for-against trigger keywords and the dedicated structure directive
        # (see comparative_response.py) was appended to the system prompt.
        "comparative_structure_applied": comparative_structure_applied,
        # 2026-08-29: NEW - only present (non-null) under retrieval_mode=
        # "semantic_graph". {"files_used": [...], "files_not_found": [...]} -
        # every .md file the answer's evidence was hydrated from (or tried
        # to be, and couldn't) across the main retrieve() pass AND every
        # category's fallback - see _dedupe_semantic_graph_trace() above and
        # Semantic_Graph_Retrieval_Architecture.md.
        "semantic_graph_files": _dedupe_semantic_graph_trace(semantic_graph_trace),
        "from_cache": False,
    }

    if not bypass_cache:
        cache.set_cached_answer(question, response, metadata_filters, extra_key=cache_extra_key)  # 2026-10-13: same key as the read above
    cache.finish_agent_run(run_id)
    observability.end_trace(trace, response)
    observability.flush_langfuse()

    # 2026-08-26: return shape changed from jsonify(response) to a plain
    # (response, internal) tuple - see this function's docstring above.
    return response, {"top_chunks": top_chunks, "plain_rows": retrieved["plain_rows"]}


# ============================================================================
# ANSWER EVALUATION - the Ask screen's on-demand "Evaluate this answer"
# button. Deliberately a SEPARATE, user-triggered call rather than reusing
# the internal answer_quality.judge_answer() quality gate that already runs
# during /api/ask (see answer_evaluation.py's module docstring for why) -
# richer scorecard, explicit strengths/issues, and only runs when the user
# actually wants it (an extra LLM call on every question would be wasteful).
# Re-runs retrieval+rerank fresh for the SAME question (identical code path
# to _api_ask_impl above) rather than trying to reuse cached state from the
# original /api/ask call - this keeps the evaluation genuinely independent
# (judged against freshly-retrieved material, not whatever the app claims it
# used) and avoids adding a new caching/storage dependency.
# ============================================================================

@app.route("/api/ask/evaluate", methods=["POST"])
def api_ask_evaluate():
    body = request.json or {}
    question = (body.get("question") or "").strip()
    answer = (body.get("answer") or "").strip()
    metadata_filters = body.get("filters") or None
    # 2026-08-26: BUG FIX. This route previously always re-retrieved in the
    # DEFAULT "hybrid" mode regardless of what mode actually produced the
    # answer being evaluated - so an answer generated under "vector_only"
    # (Simple RAG) got judged against a hybrid retrieval's different (often
    # richer) context, dinging groundedness/completeness for material the
    # answer never had access to. Root-caused from a real report: the SAME
    # answer scored 8/10 in the quality-checked loop (which correctly reuses
    # the generating call's own retrieval_mode) and only 5/10 from this
    # button. The caller must now pass the retrieval_mode the answer was
    # actually generated under (the frontend reads it off the answer's own
    # `retrieval_mode` field, not whatever mode is currently selected in the
    # UI - see index.html's currentAskRetrievalMode) - defaults to "hybrid"
    # only for backward compatibility with a caller that omits it.
    # 2026-09-10: NEW - "agentic" accepted here too, so an answer generated
    # under retrieval_mode="agentic" can still be independently re-evaluated
    # (this route scores an ALREADY-GENERATED answer against whatever mode
    # it was produced under - see the comment above about reusing the
    # generating call's own retrieval_mode).
    retrieval_mode = body.get("retrieval_mode") or "hybrid"
    if retrieval_mode not in ("hybrid", "vector_only", "semantic_graph", "agentic", "structured_research"):
        return jsonify({"error": f"Invalid retrieval_mode {retrieval_mode!r} - must be 'hybrid', "
                                  f"'vector_only', 'semantic_graph', 'agentic', or 'structured_research'."}), 400
    if not question or not answer:
        return jsonify({"error": "Both 'question' and 'answer' are required."}), 400

    try:
        trace = observability.start_trace("ask_evaluate", {"question": question})

        detected_intent = detect_intent(question, DOMAIN_CONFIG, PROMPT_PACK, trace=trace)
        if detected_intent == "OUT_OF_SCOPE":
            evaluation = {
                "available": False, "overall": None, "dimensions": [], "verdict": "",
                "strengths": [], "issues": [],
                "reason": "Question was classified as out of scope for this knowledge graph, "
                          "so there's no retrieved context to evaluate the answer against.",
            }
            observability.end_trace(trace, evaluation)
            observability.flush_langfuse()
            return jsonify({"evaluation": evaluation})

        retrieved = retriever.retrieve(question, intent=detected_intent, metadata_filters=metadata_filters,
                                        trace=trace, retrieval_mode=retrieval_mode)
        # 2026-09-03: was a hardcoded top_n=25 - now sourced from
        # reranker_config.load_config() (a Settings-page control) when not
        # explicitly overridden, so a Settings edit changes what reaches the
        # LLM on the very next question. See reranker_config.py's docstring.
        top_chunks = reranker.rerank(question, retrieved["candidate_chunks"])

        # 2026-09-03: pinned to version="v1" explicitly - a new "v2" was
        # registered for the separate auto-evaluate-after-generation popup
        # (see api_ask()'s AUTO_EVALUATE handling below), and get_prompt()
        # with no version returns the LATEST version, which would otherwise
        # silently switch this pre-existing manual "Evaluate" button over to
        # v2's different dimension set the moment v2 was added.
        eval_prompt = PROMPT_PACK.get_prompt("answer_evaluation", version="v1").template
        evaluation = answer_evaluation.evaluate_answer(
            question, answer, top_chunks, retrieved["plain_rows"], eval_prompt, trace=trace,
        )
        observability.log_span(trace, "answer_evaluation", answer, evaluation)
        observability.end_trace(trace, evaluation)
        observability.flush_langfuse()
        # 2026-08-26: NEW - log this manual evaluation to the feedback history
        # (see evaluation_log.py) so it's reviewable later from the
        # "Evaluation Feedback" nav page, not just visible in this response.
        # Best-effort/non-fatal - a logging failure must never break the
        # evaluation the user is actually looking at.
        try:
            evaluation_log.record_evaluation(question, answer, evaluation,
                                              source="manual", retrieval_mode=retrieval_mode)
        except Exception as log_err:
            app.logger.warning(f"evaluation_log.record_evaluation() failed (non-fatal): {log_err}")

        # 2026-09-05: NEW - a SECOND, independent scorecard, run alongside the
        # grounded evaluation above on every manual "Evaluate this answer"
        # click - see general_knowledge_evaluation's prompt-registration
        # comment in tax_prompts.py and the matching block in _api_ask_impl
        # above for the full reasoning (a customer comparing our score
        # against a blind ChatGPT-paste grade needs a score computed the same
        # context-free way to compare against). Best-effort: a failure here
        # must never break the grounded evaluation the button already
        # produced successfully above.
        try:
            gk_eval_prompt = PROMPT_PACK.get_prompt("general_knowledge_evaluation", version="v1").template
            general_knowledge_evaluation = answer_evaluation.evaluate_answer(
                question, answer, [], [], gk_eval_prompt, trace=trace, no_context_mode=True,
            )
            observability.log_span(trace, "answer_evaluation_general_knowledge", answer, general_knowledge_evaluation)
            try:
                evaluation_log.record_evaluation(question, answer, general_knowledge_evaluation,
                                                  source="manual_general_knowledge", retrieval_mode=retrieval_mode)
            except Exception as log_err:
                app.logger.warning(f"evaluation_log.record_evaluation() failed (non-fatal): {log_err}")
        except Exception as e:
            app.logger.warning(f"[api_ask_evaluate] general_knowledge_evaluation failed (non-fatal): {e}")
            general_knowledge_evaluation = {"available": False, "overall": None, "dimensions": [], "verdict": "",
                                             "strengths": [], "issues": [],
                                             "reason": f"General-knowledge evaluation failed: {e}"}

        return jsonify({"evaluation": evaluation, "general_knowledge_evaluation": general_knowledge_evaluation})
    except Exception as e:
        app.logger.exception("Failed in /api/ask/evaluate")
        return jsonify({"error": f"Something went wrong evaluating this answer: {e}"}), 500


# ============================================================================
# QUALITY-CHECKED ASK - 2026-08-26: NEW. A background-job version of
# /api/ask that automatically runs the rich answer_evaluation scorecard
# after generating an answer, and retries - regenerating with the
# evaluator's own feedback folded in as extra instructions - until the score
# reaches TARGET_SCORE or MAX_RETRIES is exhausted, whichever comes first.
# Added per explicit request: "run the evaluation post generating the
# response automatically and retry basis the feedback unless you get a
# score of 9+ out of 10." Progress is exposed via the SAME background-job +
# status-polling pattern this app already uses for batch ingestion (see
# domain_packs/tax/pipeline.py's _JOBS/_run_ingestion_job) rather than a
# plain synchronous request, since a full retry loop (up to
# DEFAULT_QUALITY_MAX_RETRIES+1 generate+evaluate LLM round trips) can
# easily take minutes - a synchronous call that long would look hung.
#
# Retrieval (Cypher/vector/category-coverage) runs ONCE per attempt inside
# _api_ask_impl() - there is no separate "retrieval-only-once" shortcut here,
# because _api_ask_impl doesn't expose retrieval and generation as separable
# steps. Each retry therefore re-runs the full pipeline, but with
# skip_internal_quality_gate=True from attempt 2 onward so the (redundant,
# once already in feedback-driven-regeneration territory) inner
# judge_answer-based wider-context retry never ALSO fires - each outer
# attempt costs exactly one generation + one evaluation call.
# ============================================================================

# 2026-08-26: CHANGED (was 9.0) - per explicit request "if the answer score
# reaches 8 and above, stop the attempt". The loop already stops as soon as
# score >= target_score (see the `if score is not None and score >=
# target_score` check below) - this only lowers the bar it stops at.
DEFAULT_QUALITY_TARGET_SCORE = 8.0
DEFAULT_QUALITY_MAX_RETRIES = 3   # -> 4 attempts total, per user's explicit choice
MAX_QUALITY_MAX_RETRIES = 6       # hard ceiling regardless of what a caller requests - bounds worst-case cost/latency

_ASK_JOBS: dict[str, dict] = {}
_ASK_JOBS_LOCK = threading.Lock()


def _ask_job_update(job_id: str, **fields):
    with _ASK_JOBS_LOCK:
        if job_id in _ASK_JOBS:
            _ASK_JOBS[job_id].update(fields)


def _ask_job_log(job_id: str, line: str):
    with _ASK_JOBS_LOCK:
        if job_id in _ASK_JOBS:
            _ASK_JOBS[job_id]["log"].append(line)


def _merge_evaluation_issues(evaluation: dict, secondary_evaluation: dict | None = None) -> list[str]:
    """2026-09-05: NEW - collects 'issues' from up to two evaluations (the
    grounded evaluator, plus optionally the separate general-knowledge
    evaluator - see general_knowledge_evaluation in tax_prompts.py), deduped
    while preserving order, for callers that want a single combined feedback
    list rather than picking one evaluator's issues arbitrarily. Per explicit
    request: "when auto-evaluate is checked, then it should use the feedback
    from both the evaluators and then retry." secondary_evaluation may be
    None (not run, or unavailable) - in which case this is identical to
    reading evaluation['issues'] alone, so existing single-evaluator callers
    are unaffected."""
    merged, seen = [], set()
    for ev in (evaluation, secondary_evaluation):
        if not ev:
            continue
        for issue in (ev.get("issues") or []):
            key = issue.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(issue)
    return merged


def _build_eval_retry_instructions(score: float | None, target_score: float, evaluation: dict,
                                    secondary_evaluation: dict | None = None) -> str:
    """2026-08-26: NEW - extracted from the retry loop below so the SAME
    feedback-phrasing is used whether the feedback comes from a mid-loop
    evaluation or from a `seed_evaluation` a caller hands in up front (see
    seed_evaluation param below) - one place decides what "here's what to
    fix" looks like to the model.

    2026-09-05: CHANGED - accepts an optional `secondary_evaluation` (the
    general-knowledge evaluator's result, run alongside the grounded one in
    the retry loop below) - its issues are merged in via
    _merge_evaluation_issues() so the model is told to address feedback from
    BOTH evaluators, not just the grounded one. `score`/`target_score` still
    refer to the grounded evaluator's score only, since that's the number the
    retry loop is actually driving toward (see the loop's comments on why
    general-knowledge feedback informs generation but not the stop
    condition)."""
    issue_text = "; ".join(_merge_evaluation_issues(evaluation, secondary_evaluation)) or \
        "the reviewer found the sourcing, completeness, or precision lacking"
    score_text = f"{score:.1f}/10" if score is not None else "an unclear/low score"
    return (
        f"IMPORTANT: An independent reviewer scored your previous attempt at this exact question "
        f"{score_text} (target: {target_score:.1f}/10). Specific issues raised: {issue_text}. "
        f"Address these directly in this revised answer - do not repeat the same gaps."
    )


# 2026-09-05: NEW - feedback-aware re-retrieval for the quality-checked-ask
# retry loop, per explicit request to confirm/validate whether a retry
# actually fetches NEW context based on the evaluator's feedback rather than
# just regenerating over the same evidence with a "try harder" instruction
# (it didn't, before this - see _build_eval_retry_instructions()'s docstring
# and the retry loop's prior behavior, which only passed feedback text to
# GENERATION, never back into retrieval).
FEEDBACK_SUPPLEMENTARY_TOP_K = 15  # kept modest - this is a SUPPLEMENT to the original pool, not a replacement


def _fetch_feedback_supplementary_chunks(question: str, evaluation: dict, retrieval_mode: str,
                                          secondary_evaluation: dict | None = None) -> list[dict]:
    """Runs ONE extra vector-search pass seeded by the evaluator's own
    'issues' feedback, so the NEXT retry attempt can see chunks the
    original retrieval pass missed - not just be told to try harder over
    the identical evidence. Deliberately simple: concatenates the issues
    text onto the ORIGINAL question to form one augmented query string, and
    reuses vector_search_chunks() as-is (no new LLM call to extract
    keywords - cheap, adds no extra latency/cost beyond one more ANN
    lookup, matching the approved plan's "no extra LLM call" choice).

    2026-09-05: CHANGED - accepts an optional `secondary_evaluation` (the
    general-knowledge evaluator's result), merged in via
    _merge_evaluation_issues() alongside `evaluation`'s own issues, so a
    "missing statutory exception" flagged only by the general-knowledge
    evaluator (which has no context to check against, but may still catch a
    gap the grounded evaluator - limited to what was actually retrieved -
    can't see) also gets a chance to surface better-matching chunks on the
    next attempt, not just issues the grounded evaluator itself raised.

    Only runs for retrieval_mode != "vector_only" callers in the sense that
    it works identically for all three modes (it's a pure vector-search
    supplement, independent of the graph-traversal path) - it does NOT
    re-run NL2Cypher/graph traversal, only ever adds vector-matched chunks,
    since the evaluator's feedback is text-shaped ("missing X exception"),
    which vector search is a much more direct match for than trying to
    synthesize a new Cypher query from prose feedback.

    Returns [] (never raises) on any failure - a failed supplementary
    fetch must degrade to "retry proceeds with just the original pool",
    never break the retry loop itself."""
    issues = _merge_evaluation_issues(evaluation, secondary_evaluation)
    if not issues:
        return []
    augmented_query = f"{question}\n\nAlso specifically address: {'; '.join(issues)}"
    try:
        return retriever.vector_search_chunks(augmented_query, top_k=FEEDBACK_SUPPLEMENTARY_TOP_K)
    except Exception as e:
        app.logger.warning(f"[feedback_supplementary_retrieval] failed (non-fatal, retry proceeds "
                            f"without it): {e}")
        return []


def _run_quality_checked_ask_job(job_id: str, question: str, metadata_filters: dict | None,
                                  retrieval_mode: str, max_retries: int, target_score: float,
                                  seed_evaluation: dict | None = None, answer_mode: str = "research",
                                  include_full_chunks: bool = False, data_session_id: str | None = None):
    """Runs in a background thread (started by /api/ask/quality_checked/start
    below) - never touches the Flask request/response objects directly, only
    the _ASK_JOBS registry, which the status route polls.

    `seed_evaluation` (2026-08-26: NEW): an already-computed evaluation dict
    (same shape answer_evaluation.evaluate_answer() returns) to seed attempt
    1's feedback with, instead of attempt 1 generating blind - used by the
    Ask screen's "Improve this answer" button, which starts this same job
    off the evaluation the user just ran manually via /api/ask/evaluate, so
    the first regeneration already tries to address that feedback rather
    than reproducing the same answer and wasting an attempt rediscovering
    problems the user already told it about.

    `data_session_id` (2026-09-08: NEW) - see this function's early
    STRUCTURED/BOTH short-circuit just below: the evaluate-and-retry loop
    that makes up the rest of this function only makes sense for an answer
    GENERATED from retrieved evidence (a legal answer can be reworded,
    re-grounded, improved against reviewer feedback). A structured-data
    agent's answer is a deterministic computation over a DataFrame - there
    is nothing to "regenerate for higher quality" about `df['amount'].sum()`
    beyond what _run_structured_agent()'s own one-retry-on-execution-error
    already does. So when routing resolves to STRUCTURED_ONLY or BOTH, this
    function runs the agentic path ONCE (no evaluation/retry loop at all)
    and reports it as an already-"done" job - see the approved plan: "route
    once, skip retry for structured.\""""
    file_attached_context = structured_data_workspace.get_attachment(data_session_id) if data_session_id else None
    if file_attached_context is not None:
        routing_decision, raw_routing_output = _classify_agent_routing(question, file_attached_context)
        if routing_decision in ("STRUCTURED_ONLY", "BOTH"):
            _ask_job_update(job_id, phase="generating", attempt=1, max_attempts=1)
            _ask_job_log(job_id, f"Routed to {routing_decision} (attached file) - the quality-checked "
                                   f"evaluate/retry loop does not apply to a structured-data computation, "
                                   f"running once (see the structured agent's own one-retry-on-error logic).")
            try:
                response, agent_trace = _run_agentic_ask(
                    question, metadata_filters, retrieval_mode, answer_mode,
                    attached_file_context=file_attached_context,
                    include_full_chunks=include_full_chunks,
                )
                response["agent_trace"] = agent_trace
                response["quality_check"] = {
                    "enabled": False,
                    "skipped_reason": f"Routed to {routing_decision} (attached file) - evaluate/retry "
                                       f"does not apply to a structured-data computation.",
                }
                _ask_job_update(job_id, status="done", phase="done", result=response, finished_at=time.time())
            except Exception as e:
                app.logger.exception(f"[quality_checked_ask] job {job_id} failed (agentic/structured path)")
                _ask_job_update(job_id, status="error", error=str(e), finished_at=time.time())
            return

    max_attempts = max_retries + 1
    best_score = None       # None until at least one attempt produces a usable score
    best_response = None
    best_evaluation = None
    # 2026-09-05: NEW - mirrors best_evaluation, for the separate general-
    # knowledge evaluator now run alongside it each attempt (see the loop
    # below) - tracked so the FINAL response's quality_check can carry both,
    # even though only best_evaluation's score decides which attempt "wins".
    best_general_knowledge_evaluation = None
    # 2026-08-26: NEW - full per-attempt record (answer text + evaluation),
    # separate from the lightweight {attempt,score,verdict} list written into
    # _ASK_JOBS for progress polling below. Attached to the FINAL response's
    # quality_check.attempts so it flows into chat history automatically
    # (the frontend already saves the whole response object as the answer -
    # see index.html's _finishAskUI) - added per explicit request to record
    # what changed across iterations for later review, not just the final
    # score.
    attempt_history: list[dict] = []
    extra_instructions = (
        _build_eval_retry_instructions(seed_evaluation.get("overall"), target_score, seed_evaluation)
        if seed_evaluation else ""
    )
    # 2026-09-05: NEW - see _fetch_feedback_supplementary_chunks()'s docstring
    # and _api_ask_impl()'s supplementary_chunks param. Seeded from
    # seed_evaluation too (not just mid-loop evaluations below), so the
    # "Improve this answer" button's very first regeneration ALSO gets a
    # feedback-informed wider pool, not just feedback TEXT with the same old
    # evidence - consistent with how extra_instructions is already seeded
    # from seed_evaluation above.
    supplementary_chunks = (
        _fetch_feedback_supplementary_chunks(question, seed_evaluation, retrieval_mode)
        if seed_evaluation else []
    )
    if seed_evaluation:
        _ask_job_log(job_id, "Seeded with your independent evaluation's feedback - "
                              "attempt 1 will try to address it directly."
                              + (f" Fetched {len(supplementary_chunks)} supplementary chunk(s) based on that "
                                 f"feedback." if supplementary_chunks else ""))

    try:
        for attempt in range(1, max_attempts + 1):
            _ask_job_update(job_id, phase="generating", attempt=attempt, max_attempts=max_attempts)
            _ask_job_log(job_id, f"Attempt {attempt}/{max_attempts}: generating answer"
                                  + (" (revising per reviewer feedback"
                                     + (f", with {len(supplementary_chunks)} supplementary chunk(s)"
                                        if supplementary_chunks else "") + ")..."
                                     if extra_instructions else "..."))

            # 2026-09-11: BUGFIX - was calling _api_ask_impl() directly here,
            # which sends retrieval_mode="agentic" straight into
            # GraphRAGRetriever.retrieve() (no "agentic" branch there, never
            # will be - see retrieval_orchestrator.py's docstring), silently
            # falling through to plain hybrid-style retrieval instead of
            # running the orchestrator's plan/retrieve/sufficiency-check
            # loop. Reported bug: selecting "Agentic" with "Quality-checked
            # answer" also checked produced an answer that looked like it
            # never ran the agentic loop at all. Routing through
            # _run_unstructured_agent() here (the SAME function /api/ask
            # already uses) means this loop picks up its retrieval_mode=
            # "agentic" branch for free, and stays behaviorally identical to
            # before for every other retrieval_mode (that function is a thin
            # wrapper - see its own docstring).
            response, internal, per_attempt_trace = _run_unstructured_agent(
                question, metadata_filters, retrieval_mode, answer_mode,
                extra_gen_instructions=extra_instructions,
                skip_internal_quality_gate=(attempt > 1),
                bypass_cache=True,
                supplementary_chunks=supplementary_chunks or None,
                # 2026-09-05: NEW - requested on EVERY attempt (not just the
                # one that ends up winning) - cheap (chunk text is already in
                # memory either way, this just includes it in the response
                # dict) and simplest: whichever attempt's `response` dict
                # ends up assigned to best_response below already carries its
                # own correct chunks_used, no extra bookkeeping needed.
                include_full_chunks=include_full_chunks,
            )
            # 2026-09-11: NEW - surfaced so a winning agentic attempt's
            # evidence panel still shows its retrieval_iterations breakdown
            # (see index.html's "Agentic retrieval loop" rendering) even
            # though it arrived via the quality-checked-ask job path rather
            # than a plain /api/ask call - same field _run_agentic_ask()
            # already attaches as response["agent_trace"]["agents_invoked"].
            response["agent_trace"] = {"routing_decision": "UNSTRUCTURED_ONLY", "routing_reason": "",
                                        "file_attached": False, "agents_invoked": [per_attempt_trace]}

            if response.get("intent") == "OUT_OF_SCOPE" or not internal.get("top_chunks"):
                _ask_job_log(job_id, "No groundable answer was produced - nothing to evaluate or retry.")
                best_response = response
                break

            _ask_job_update(job_id, phase="evaluating", attempt=attempt)
            _ask_job_log(job_id, f"Attempt {attempt}/{max_attempts}: running independent evaluation...")

            # 2026-09-03: pinned to version="v1" - see api_ask_evaluate()'s
            # matching comment above for why this must stay explicit now
            # that a "v2" exists for the separate auto-evaluate popup.
            eval_prompt = PROMPT_PACK.get_prompt("answer_evaluation", version="v1").template
            evaluation = answer_evaluation.evaluate_answer(
                question, response["answer"], internal["top_chunks"], internal["plain_rows"], eval_prompt,
            )
            score = evaluation["overall"] if evaluation.get("available") and evaluation.get("overall") is not None else None

            # 2026-09-05: NEW - run the SECOND, independent general-knowledge
            # evaluator on every attempt too, per explicit request: "when
            # auto-evaluate is checked, then it should use the feedback from
            # both the evaluators and then retry." Its score does NOT drive
            # best_score/the target-score stop condition below (that stays
            # tied to the grounded evaluator, which is what "8+/10" has
            # always meant here) - only its `issues` feed into the next
            # attempt's generation instructions and supplementary retrieval,
            # alongside the grounded evaluator's own issues. Best-effort:
            # a failure here must never break the retry loop itself.
            try:
                gk_eval_prompt = PROMPT_PACK.get_prompt("general_knowledge_evaluation", version="v1").template
                gk_evaluation = answer_evaluation.evaluate_answer(
                    question, response["answer"], [], [], gk_eval_prompt, no_context_mode=True,
                )
            except Exception as e:
                _ask_job_log(job_id, f"(general-knowledge evaluation failed, non-fatal: {e})")
                gk_evaluation = {"available": False, "overall": None, "dimensions": [], "verdict": "",
                                  "strengths": [], "issues": [], "reason": f"General-knowledge evaluation failed: {e}"}

            attempt_history.append({
                "attempt": attempt, "answer": response["answer"], "evaluation": evaluation,
                # 2026-09-05: NEW - kept alongside "evaluation" (not merged
                # into it) so the iteration-history UI can still show each
                # evaluator's own scores/assessments distinctly.
                "general_knowledge_evaluation": gk_evaluation,
            })
            # 2026-08-26: NEW - log every attempt (not just the final/best one) to
            # the feedback history, per explicit request to have all evaluation
            # feedback reviewable later in one place. Best-effort/non-fatal, same
            # as the /api/ask/evaluate route's call - a logging failure must never
            # interrupt the retry loop itself.
            try:
                evaluation_log.record_evaluation(
                    question, response["answer"], evaluation,
                    source="quality_checked_attempt", retrieval_mode=retrieval_mode,
                    attempt_number=attempt,
                )
                evaluation_log.record_evaluation(
                    question, response["answer"], gk_evaluation,
                    source="quality_checked_attempt_general_knowledge", retrieval_mode=retrieval_mode,
                    attempt_number=attempt,
                )
            except Exception as log_err:
                _ask_job_log(job_id, f"(feedback history logging failed, non-fatal: {log_err})")

            _ask_job_log(job_id, f"Attempt {attempt}/{max_attempts}: scored "
                                  + (f"{score:.1f}/10 (grounded)" if score is not None else "unavailable (grounded evaluator call failed)")
                                  + (f", {gk_evaluation['overall']:.1f}/10 (general knowledge)."
                                     if gk_evaluation.get("available") and gk_evaluation.get("overall") is not None
                                     else ", general-knowledge score unavailable."))
            with _ASK_JOBS_LOCK:
                if job_id in _ASK_JOBS:
                    _ASK_JOBS[job_id]["latest_score"] = score
                    _ASK_JOBS[job_id]["attempts"].append(
                        {"attempt": attempt, "score": score, "verdict": evaluation.get("verdict", ""),
                         "general_knowledge_score": gk_evaluation.get("overall")})

            if best_score is None or (score is not None and score > best_score):
                best_score, best_response, best_evaluation = score, response, evaluation
                best_general_knowledge_evaluation = gk_evaluation

            if score is not None and score >= target_score:
                _ask_job_log(job_id, f"Reached target score ({score:.1f} >= {target_score:.1f}) - done.")
                break

            if attempt >= max_attempts:
                _ask_job_log(job_id, f"Reached the retry limit ({max_retries} retries) without hitting the "
                                      f"target score - returning the best-scoring attempt "
                                      f"({best_score:.1f}/10)." if best_score is not None else
                                      f"Reached the retry limit ({max_retries} retries); no attempt produced a "
                                      f"usable score - returning the last attempt.")
                break

            extra_instructions = _build_eval_retry_instructions(score, target_score, evaluation, gk_evaluation)
            # 2026-09-05: NEW - re-fetch supplementary chunks for the NEXT
            # attempt, seeded by THIS attempt's own evaluation feedback FROM
            # BOTH EVALUATORS (not last iteration's, and not just carried
            # over) - each retry's supplementary pool is freshly targeted at
            # whatever THAT attempt was specifically dinged for by either
            # reviewer. See _fetch_feedback_supplementary_chunks()'s docstring.
            supplementary_chunks = _fetch_feedback_supplementary_chunks(
                question, evaluation, retrieval_mode, secondary_evaluation=gk_evaluation)
            if supplementary_chunks:
                _ask_job_log(job_id, f"Fetched {len(supplementary_chunks)} supplementary chunk(s) targeting "
                                      f"the issues raised above (from both evaluators), for the next attempt.")

        if best_response is None:
            raise RuntimeError("quality-checked ask loop produced no response at all")

        best_response["quality_check"] = {
            "enabled": True,
            "target_score": target_score,
            "max_retries": max_retries,
            "reached_target": best_score is not None and best_score >= target_score,
            "final_score": best_score,
            "final_evaluation": best_evaluation,
            # 2026-09-05: NEW - the winning attempt's general-knowledge score,
            # alongside final_evaluation above - see best_general_knowledge_evaluation's
            # declaration comment. Also surfaced as the response's top-level
            # "general_knowledge_evaluation" field below (same field name the
            # separate auto-evaluate popup uses) so the frontend's existing
            # dual-scorecard rendering (_evalResultHtmlDual) works unchanged
            # for quality-checked answers too.
            "final_general_knowledge_evaluation": best_general_knowledge_evaluation,
            "seeded_from_manual_evaluation": seed_evaluation is not None,
            # 2026-08-26: NEW - full iteration trail (every attempt's answer
            # + evaluation, in order) for later review - see this function's
            # docstring above.
            "attempts": attempt_history,
        }
        # 2026-09-05: NEW - mirror final_general_knowledge_evaluation onto the
        # same top-level field name the auto-evaluate popup already uses, so
        # renderAskAnswer()/_evalResultHtmlDual() in index.html picks it up
        # for quality-checked answers with no separate frontend branch needed.
        best_response["general_knowledge_evaluation"] = best_general_knowledge_evaluation
        _ask_job_update(job_id, status="done", phase="done", result=best_response, finished_at=time.time())
    except Exception as e:
        app.logger.exception(f"[quality_checked_ask] job {job_id} failed")
        _ask_job_update(job_id, status="error", error=str(e), finished_at=time.time())


# ============================================================================
# FAST QUALITY-CHECKED ASK - 2026-09-07: NEW. A low-latency sibling of
# _run_quality_checked_ask_job() above, per explicit request: "the current
# retrieval is taking long... how can i get the response within 3-8 seconds
# with multiple iterations (2 iterations)... ensure comprehensive
# cypher... accuracy and context adherence score to be 85% and above in
# first shot." Three changes from the normal quality-checked loop, used
# TOGETHER here:
#   1. Runs SYNCHRONOUSLY (returns the final result directly) rather than as
#      a background job polled over HTTP - a fast path has no need for
#      progress polling, and polling itself adds a round-trip.
#   2. Calls _api_ask_impl(fast_mode=True, skip_rerank=...) - skips
#      detect_intent()/NL2Cypher's LLM round-trips and (optionally) the BGE
#      cross-encoder pass - see that function's fast_mode/skip_rerank
#      docstrings.
#   3. Runs the grounded and general-knowledge evaluators CONCURRENTLY
#      (ThreadPoolExecutor) instead of sequentially - llm_client.complete()
#      is explicitly thread-safe for exactly this (see ModelConfig's
#      docstring: "Immutable snapshot... resolved fresh on every call") -
#      roughly halving the two-evaluator latency cost versus the normal
#      loop's sequential grounded-then-general-knowledge calls.
# Still reuses _merge_evaluation_issues()/_build_eval_retry_instructions()/
# _fetch_feedback_supplementary_chunks() unchanged - the FEEDBACK mechanics
# are identical to the normal loop, only what runs sequentially vs
# concurrently and which retrieval path is used are different.
FAST_MAX_ITERATIONS = int(os.environ.get("FAST_ASK_MAX_ITERATIONS", "2"))
FAST_TARGET_SCORE = float(os.environ.get("FAST_ASK_TARGET_SCORE", "8.5"))  # 85% on this project's 0-10 scale


def _run_both_evaluators_concurrently(question: str, answer_text: str, top_chunks: list[dict],
                                       plain_rows: list[dict]) -> tuple[dict, dict]:
    """Runs the grounded (answer_evaluation v1) and general-knowledge
    (general_knowledge_evaluation v1, no_context_mode=True) evaluators in
    parallel threads instead of one after the other, and waits for both.
    Returns (grounded_evaluation, general_knowledge_evaluation) - same two
    dicts _run_quality_checked_ask_job()'s sequential calls produce, just
    computed concurrently. Each call is independently best-effort: if either
    raises, that ONE evaluation degrades to an unavailable placeholder
    (matching the sequential loop's existing per-evaluator try/except
    behavior) without waiting on or cancelling the other."""
    def _grounded():
        eval_prompt = PROMPT_PACK.get_prompt("answer_evaluation", version="v1").template
        return answer_evaluation.evaluate_answer(question, answer_text, top_chunks, plain_rows, eval_prompt)

    def _general_knowledge():
        gk_eval_prompt = PROMPT_PACK.get_prompt("general_knowledge_evaluation", version="v1").template
        return answer_evaluation.evaluate_answer(question, answer_text, [], [], gk_eval_prompt, no_context_mode=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        grounded_future = pool.submit(_grounded)
        gk_future = pool.submit(_general_knowledge)

        try:
            evaluation = grounded_future.result()
        except Exception as e:
            app.logger.warning(f"[fast_ask] grounded evaluation failed (non-fatal): {e}")
            evaluation = {"available": False, "overall": None, "dimensions": [], "verdict": "",
                          "strengths": [], "issues": [], "reason": f"Grounded evaluation failed: {e}"}

        try:
            gk_evaluation = gk_future.result()
        except Exception as e:
            app.logger.warning(f"[fast_ask] general-knowledge evaluation failed (non-fatal): {e}")
            gk_evaluation = {"available": False, "overall": None, "dimensions": [], "verdict": "",
                              "strengths": [], "issues": [], "reason": f"General-knowledge evaluation failed: {e}"}

    return evaluation, gk_evaluation


def _run_fast_quality_checked_ask(question: str, metadata_filters: dict | None, retrieval_mode: str,
                                   answer_mode: str, skip_rerank: bool,
                                   data_session_id: str | None = None) -> dict:
    """Synchronous fast-path equivalent of _run_quality_checked_ask_job() -
    see the module comment above this function for what's different and why.
    Capped at FAST_MAX_ITERATIONS attempts (default 2, per explicit
    request), targeting FAST_TARGET_SCORE (default 8.5/10 = 85%) on the
    GROUNDED evaluator's score only - identical stop-condition semantics to
    the normal loop (general-knowledge feedback informs the next attempt's
    generation/retrieval, never the stop condition itself).

    `data_session_id` (2026-09-08: NEW) - same STRUCTURED_ONLY/BOTH
    short-circuit as _run_quality_checked_ask_job() (see that function's
    docstring for why the evaluate/retry loop doesn't apply to a
    deterministic structured-data computation) - runs the agentic path once
    and returns immediately, skipping the fast retry loop entirely."""
    file_attached_context = structured_data_workspace.get_attachment(data_session_id) if data_session_id else None
    if file_attached_context is not None:
        routing_decision, raw_routing_output = _classify_agent_routing(question, file_attached_context)
        if routing_decision in ("STRUCTURED_ONLY", "BOTH"):
            response, agent_trace = _run_agentic_ask(
                question, metadata_filters, retrieval_mode, answer_mode,
                attached_file_context=file_attached_context,
            )
            response["agent_trace"] = agent_trace
            response["quality_check"] = {
                "enabled": False,
                "skipped_reason": f"Routed to {routing_decision} (attached file) - fast-mode evaluate/"
                                   f"retry does not apply to a structured-data computation.",
            }
            return response

    best_score = None
    best_response = None
    best_evaluation = None
    best_general_knowledge_evaluation = None
    attempt_history: list[dict] = []
    extra_instructions = ""
    supplementary_chunks: list[dict] = []

    for attempt in range(1, FAST_MAX_ITERATIONS + 1):
        response, internal = _api_ask_impl(
            question, metadata_filters, retrieval_mode=retrieval_mode,
            extra_gen_instructions=extra_instructions,
            skip_internal_quality_gate=True,
            bypass_cache=True,
            answer_mode=answer_mode,
            supplementary_chunks=supplementary_chunks or None,
            fast_mode=True,
            skip_rerank=skip_rerank,
        )

        if response.get("intent") == "OUT_OF_SCOPE" or not internal.get("top_chunks"):
            best_response = response
            break

        evaluation, gk_evaluation = _run_both_evaluators_concurrently(
            question, response["answer"], internal["top_chunks"], internal["plain_rows"])
        score = evaluation["overall"] if evaluation.get("available") and evaluation.get("overall") is not None else None

        attempt_history.append({
            "attempt": attempt, "answer": response["answer"], "evaluation": evaluation,
            "general_knowledge_evaluation": gk_evaluation,
        })

        if best_score is None or (score is not None and score > best_score):
            best_score, best_response, best_evaluation = score, response, evaluation
            best_general_knowledge_evaluation = gk_evaluation

        if score is not None and score >= FAST_TARGET_SCORE:
            break
        if attempt >= FAST_MAX_ITERATIONS:
            break

        extra_instructions = _build_eval_retry_instructions(score, FAST_TARGET_SCORE, evaluation, gk_evaluation)
        # Same feedback-aware supplementary retrieval the normal loop uses -
        # not skipped for speed, since it's a single extra vector-search
        # call (cheap relative to an LLM round-trip) and materially improves
        # what the 2nd (final) attempt can draw on.
        supplementary_chunks = _fetch_feedback_supplementary_chunks(
            question, evaluation, retrieval_mode, secondary_evaluation=gk_evaluation)

    if best_response is None:
        raise RuntimeError("fast quality-checked ask loop produced no response at all")

    best_response["quality_check"] = {
        "enabled": True,
        "fast_mode": True,
        "target_score": FAST_TARGET_SCORE,
        "max_retries": FAST_MAX_ITERATIONS - 1,
        "reached_target": best_score is not None and best_score >= FAST_TARGET_SCORE,
        "final_score": best_score,
        "final_evaluation": best_evaluation,
        "final_general_knowledge_evaluation": best_general_knowledge_evaluation,
        "seeded_from_manual_evaluation": False,
        "attempts": attempt_history,
    }
    best_response["general_knowledge_evaluation"] = best_general_knowledge_evaluation
    return best_response


@app.route("/api/ask/fast", methods=["POST"])
def api_ask_fast():
    """Synchronous fast-response endpoint - see _run_fast_quality_checked_ask()'s
    module comment for the three changes (fast_mode retrieval, optional
    reranker skip, concurrent dual-evaluation) that make this meaningfully
    faster than /api/ask/quality_checked/start's background-job path, at the
    cost of narrower per-attempt retrieval polish (no NL2Cypher, no BGE
    reranking unless skip_rerank=false is explicitly requested) - see this
    project's own accompanying developer notes on the accuracy/latency
    trade-off this represents."""
    body = request.json or {}
    question = (body.get("question") or "").strip()
    metadata_filters = body.get("filters") or None
    retrieval_mode = body.get("retrieval_mode") or "hybrid"
    if retrieval_mode not in ("hybrid", "vector_only", "semantic_graph"):
        return jsonify({"error": f"Invalid retrieval_mode {retrieval_mode!r} - must be 'hybrid', "
                                  f"'vector_only', or 'semantic_graph'."}), 400
    answer_mode = body.get("answer_mode") or "research"
    if answer_mode not in ("lookup", "research", "opinion"):
        return jsonify({"error": f"Invalid answer_mode {answer_mode!r} - must be 'lookup', "
                                  f"'research', or 'opinion'."}), 400
    # 2026-09-07: NEW - defaults to skipping the reranker (the single largest
    # measured per-request latency cost - see reranker.rerank()'s skip_rerank
    # docstring) since that's the whole point of this route; an explicit
    # skip_rerank=false lets a caller opt back into cross-encoder scoring
    # while keeping every other fast_mode saving.
    skip_rerank = bool(body.get("skip_rerank", True))
    # 2026-09-08: NEW - see _run_fast_quality_checked_ask()'s docstring for
    # the STRUCTURED_ONLY/BOTH short-circuit this enables.
    data_session_id = body.get("data_session_id") or None
    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        started_at = time.time()
        result = _run_fast_quality_checked_ask(question, metadata_filters, retrieval_mode, answer_mode,
                                                 skip_rerank, data_session_id=data_session_id)
        result["fast_ask_elapsed_seconds"] = round(time.time() - started_at, 2)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Failed in /api/ask/fast")
        return jsonify({"error": f"Something went wrong generating a fast response: {e}"}), 500


# ============================================================================
# AGENTIC RETRIEVER - 2026-09-08: NEW. An orchestrator that routes a question
# to one or both of two agents based on intent, per explicit request:
# "An agent orchestrator, based on intention and requirement directs the task
# to other agents (tax unstructured data query agent... structured data
# query agent...). If data not attached, only uses tax unstructured data
# query agent only. If the intent is to only respond from the attached
# sources, then only structured query agent is invoked."
#
# THREE PIECES, kept deliberately separate:
#   1. _classify_agent_routing() - a NEW, cheap classification (agent_routing
#      prompt) deciding UNSTRUCTURED_ONLY / STRUCTURED_ONLY / BOTH. Skipped
#      entirely (hardcoded to UNSTRUCTURED_ONLY, no LLM call) when no file is
#      attached - there is nothing to route between in that case.
#   2. _run_unstructured_agent() - a THIN WRAPPER around the existing,
#      unchanged _api_ask_impl() pipeline (KG traversal + chunk retrieval +
#      reranking + generation - everything already built). Not new retrieval
#      logic - just packages the existing call with an agent-trace record for
#      the evidence panel.
#   3. _run_structured_agent() - 2026-09-08: IMPLEMENTED. Generates pandas
#      code from the attached file's schema (structured_data_query prompt),
#      executes it via sandbox.run_sandboxed() (isolated subprocess - see
#      that module's docstring for the full threat-model reasoning), and
#      retries once with the failure fed back if the first attempt's code
#      doesn't execute cleanly (STRUCTURED_AGENT_MAX_ATTEMPTS).
#
# _run_agentic_ask() is the single entry point app.py's /api/ask route calls -
# attached_file_context is resolved there from the request's data_session_id
# via structured_data_workspace.get_attachment() (see that route's code).
# ============================================================================

def _classify_agent_routing(question: str, attached_file_context: dict | None, trace=None) -> tuple[str, str]:
    """Returns (routing_decision, raw_model_output). routing_decision is one
    of "UNSTRUCTURED_ONLY"/"STRUCTURED_ONLY"/"BOTH" - always
    "UNSTRUCTURED_ONLY" with NO LLM call when attached_file_context is None
    (see this section's module comment for why). Falls back to
    "UNSTRUCTURED_ONLY" (the safest default - it's the path that already
    exists and is well-tested) if the model's output doesn't cleanly match a
    known label, same "don't crash the pipeline over a classification
    hiccup" reasoning intent.py's detect_intent() already uses.

    2026-09-08: BUGFIX - takes the full attached_file_context dict (not just
    a bool) and threads its filename/schema_summary into the agent_routing
    v2 prompt (see tax_prompts.py's v2 entry for the reported bug this
    fixes: routing was previously blind to the file's actual content, so a
    question clearly answerable from the file - "What deliverables are
    done?" against a status-report spreadsheet - was misrouted to
    UNSTRUCTURED_ONLY purely because nothing in the question text itself
    mentioned "file" or "spreadsheet")."""
    if attached_file_context is None:
        return "UNSTRUCTURED_ONLY", ""

    prompt = PROMPT_PACK.get_prompt("agent_routing", version="v2")
    system = prompt.template.format(
        filename=attached_file_context.get("filename", "attached file"),
        schema_summary=attached_file_context.get("schema_summary", ""),
    )
    raw = llm_client.complete(system=system, user_content=question, max_tokens=20,
                               trace=trace, name="agent_routing").strip().upper()
    if raw in ("UNSTRUCTURED_ONLY", "STRUCTURED_ONLY", "BOTH"):
        return raw, raw
    app.logger.warning(f"[agent_routing] Unrecognized model output {raw!r}, defaulting to UNSTRUCTURED_ONLY")
    return "UNSTRUCTURED_ONLY", raw


def _run_unstructured_agent(question: str, metadata_filters: dict | None, retrieval_mode: str,
                             answer_mode: str, attached_file_context: dict | None = None,
                             **api_ask_impl_kwargs) -> tuple[dict, dict, dict]:
    """Thin wrapper around the EXISTING, UNCHANGED _api_ask_impl() pipeline -
    see this section's module comment. Returns (response, internal,
    agent_trace_record) - the same (response, internal) tuple _api_ask_impl()
    itself returns, plus a third element describing what this agent did, in
    the shape the evidence panel's "Agents invoked" step (index.html) reads.
    **api_ask_impl_kwargs passes through any of _api_ask_impl()'s other
    params (auto_evaluate, fast_mode, skip_rerank, etc.) unchanged - this
    wrapper does not restrict or alter which of _api_ask_impl()'s existing
    features remain available when called through the orchestrator.

    2026-09-10: NEW - retrieval_mode="agentic" branches BEFORE reaching
    _api_ask_impl() at all. GraphRAGRetriever.retrieve() (which
    _api_ask_impl() calls internally) has no "agentic" branch and never will
    - see retrieval_orchestrator.py's module docstring for why that loop
    lives outside GraphRAGRetriever entirely, calling its existing tool
    methods directly instead. So this wrapper: (1) runs the orchestrator's
    bounded plan/retrieve/sufficiency-check loop itself to get an evidence
    pool, (2) calls _api_ask_impl() with retrieval_mode="vector_only" (the
    cheapest real retrieval path - see that mode's own docstring) purely so
    generation/reranking/the quality gate/caching all run completely
    unchanged, feeding the orchestrator's chunks in via the EXISTING
    supplementary_chunks mechanism (see _api_ask_impl()'s docstring) so nothing
    downstream of retrieval needs to know agentic mode exists, then (3)
    patches the returned response's "cypher"/"plain_rows" fields back to what
    the orchestrator actually found - vector_only's own retrieve() call
    always reports cypher=None/plain_rows=[] (it deliberately skips graph
    retrieval - see that mode's docstring), which would otherwise make the
    evidence panel claim agentic mode never touched the graph even when its
    planner chose CYPHER and found rows.

    2026-10-14: FIXED the former KNOWN GAP here. Because generation is
    reached via retrieval_mode="vector_only", it used to inherit that mode's
    skip of get_parent_section_context() ("vector_only" deliberately excludes
    PART_OF-graph-derived surrounding-provision text to stay a clean
    graph-free baseline) - an artifact of reusing vector_only's plumbing, not
    a design choice: agentic answers cited chunks WITHOUT the surrounding
    section a hybrid answer citing the identical chunk carried. The call now
    passes include_parent_context=True explicitly (real Simple RAG calls keep
    the default None -> skipped), and mode_label="agentic" so the answer is
    cached/reported as agentic rather than colliding with a genuine
    vector_only answer's cache entry (and being badged "Simple RAG")."""
    started_at = time.time()

    if retrieval_mode == "agentic":
        orchestrator_result = retrieval_orchestrator.run_agentic_retrieval(retriever, question, trace=None)
        supplementary = api_ask_impl_kwargs.pop("supplementary_chunks", None) or []
        response, internal = _api_ask_impl(
            question, metadata_filters, retrieval_mode="vector_only", answer_mode=answer_mode,
            supplementary_chunks=orchestrator_result["chunks"] + supplementary,
            include_parent_context=True, mode_label="agentic",  # 2026-10-14: see docstring
            **api_ask_impl_kwargs)
        # Patch in what the orchestrator actually found - see this branch's
        # comment above for why vector_only's own retrieve() call can't
        # report these fields itself.
        response["cypher"] = "; ".join(orchestrator_result["cypher_queries"]) or None
        response["plain_rows"] = orchestrator_result["plain_rows"]
        internal["plain_rows"] = orchestrator_result["plain_rows"]
        agent_trace_record = {
            "agent": "tax_unstructured_query_agent",
            "action": "Ran the agentic retrieval loop (planner + tool calls + sufficiency check, "
                       f"{len(orchestrator_result['iterations'])} iteration(s)), then generated an "
                       "answer from the accumulated evidence.",
            "cypher_ran": bool(orchestrator_result["cypher_queries"]),
            "chunks_used": len(internal.get("top_chunks") or []),
            "retrieval_mode": retrieval_mode,
            "retrieval_iterations": orchestrator_result["iterations"],
            "final_sufficiency_verdict": orchestrator_result["final_verdict"],
            "elapsed_seconds": round(time.time() - started_at, 2),
        }
        return response, internal, agent_trace_record

    if retrieval_mode == "structured_research":
        # 2026-09-11: NEW - the customer-specified fixed hierarchy (Fact
        # Sheet -> Law -> Case Law/Tribunal/Opinion -> Synthesis), see
        # hierarchical_orchestrator.py's module docstring. Unlike "agentic"
        # above, this does NOT route through _api_ask_impl() at all - the
        # final answer comes from the hierarchical_synthesis prompt's
        # mandated 4-point structure (change in law / opinion-changed-by-
        # law / opinion-changed-by-case-law / position-changed-by-case-law),
        # which is a fundamentally different generation shape than
        # _api_ask_impl()'s normal single-pass answer_generation templates -
        # forcing it through that path would mean fighting its cache key,
        # quality-gate retry, and reranking logic for a generation style
        # none of that machinery was built for. api_ask_impl_kwargs
        # (auto_evaluate, include_full_chunks, etc.) are accepted for
        # interface consistency with every other retrieval_mode but mostly
        # do not apply here yet - see the response fields below for what is
        # actually populated.
        hierarchy_result = hierarchical_orchestrator.run_hierarchical_research(
            retriever, question, attached_file_context=attached_file_context, trace=None)

        law_findings_text = "\n".join(
            f"- {sq['label']}: {'found evidence' if sq['found_evidence'] else 'NOT FOUND'}"
            for sq in hierarchy_result["law"].get("sub_queries", [])
        ) or "(no sub-queries recorded)"
        case_law_text = (
            f"Favourable: {[c['citation'] for c in hierarchy_result['case_law']['favourable']] or 'none'}\n"
            f"Unfavourable: {[c['citation'] for c in hierarchy_result['case_law']['unfavourable']] or 'none'}"
        )
        tribunal_text = (
            f"HC: {[c['citation'] for c in hierarchy_result['tribunal']['hc']] or 'none'}\n"
            f"SC: {[c['citation'] for c in hierarchy_result['tribunal']['sc']] or 'none'}\n"
            f"Jurisdictional: {[c['citation'] for c in hierarchy_result['tribunal']['jurisdictional']] or 'none'}\n"
            f"Pre-change-in-law: {[c['citation'] for c in hierarchy_result['tribunal']['pre_change_in_law']] or 'none'}"
        )
        opinion_text = (
            f"Latest: {[o['citation'] for o in hierarchy_result['opinion']['latest']] or 'none'}\n"
            f"Superseded: {[o['citation'] for o in hierarchy_result['opinion']['superseded']] or 'none'}"
            if hierarchy_result["opinion"]["candidates_found"]
            else "NOT AVAILABLE - no KPMG opinions found (not yet ingested into the graph, or none "
                 "relevant to this matter)."
        )

        synthesis_prompt = PROMPT_PACK.get_prompt("hierarchical_synthesis", version="v1").template.format(
            facts=hierarchy_result["fact_sheet"]["facts"],
            tax_position=hierarchy_result["fact_sheet"]["tax_position"],
            law_findings=law_findings_text,
            case_law_findings=case_law_text,
            tribunal_findings=tribunal_text,
            opinion_findings=opinion_text,
            question=question,
        )
        answer_text = llm_client.complete(system=synthesis_prompt, user_content=question, max_tokens=3000,
                                           name="hierarchical_synthesis").strip()

        all_cypher = hierarchy_result["law"].get("cypher_queries", [])
        response = {
            "question": question, "intent": None,
            "cypher": "; ".join(all_cypher) or None,
            "answer": answer_text,
            "nodes": [], "edges": [],
            "source_chunks": [], "grounding": None,
            "retrieval_scope": {"retrieval_mode": "structured_research"},
            # 2026-09-11: BUGFIX - every other retrieval_mode branch sets
            # this top-level key (see hybrid's response["retrieval_mode"] =
            # retrieval_mode a few lines below in the non-structured_research
            # path) - index.html's renderAskAnswer() reads result.retrieval_mode
            # (NOT retrieval_scope.retrieval_mode) to sync the mode badge, so
            # without this key it fell back to "hybrid" and every Structured
            # Research answer was mislabeled "KG + RAG" regardless of what
            # actually ran.
            "retrieval_mode": "structured_research",
            # 2026-09-11: BUGFIX - same missing-key pattern as retrieval_mode
            # above, for the SEPARATE answer_mode axis (Lookup/Research/
            # Opinion). This branch always generates via the fixed
            # hierarchical_synthesis template regardless of which answer_mode
            # was selected (see this route's docstring above for why - the
            # 4-point change-in-law structure is a customer-mandated
            # methodology, not a swappable output style) - reporting the
            # answer_mode the user actually selected here would be
            # misleading. Always reports "research" (the closest existing
            # label to what this format actually is) so the badge is honest
            # about what was generated, regardless of the toggle's position;
            # the frontend additionally disables the toggle for this
            # retrieval_mode so the user isn't offered a choice this branch
            # can't honor in the first place - see index.html's
            # FAST_INCOMPATIBLE_RETRIEVAL_MODES-style handling for
            # "structured_research".
            "answer_mode": "research",
        }
        internal = {"top_chunks": hierarchy_result["law"].get("chunks", []),
                    "plain_rows": hierarchy_result["law"].get("plain_rows", [])}
        agent_trace_record = {
            "agent": "tax_unstructured_query_agent",
            "action": "Ran the structured-research hierarchy (Fact Sheet, Law, Case Law, Tribunal Case "
                       "Law, Opinion - Case Law/Tribunal/Opinion in parallel), then synthesized a final "
                       "answer addressing change-in-law and change-in-interpretation.",
            "cypher_ran": bool(all_cypher),
            "chunks_used": len(internal["top_chunks"]),
            "retrieval_mode": retrieval_mode,
            "hierarchy_stage_trace": hierarchy_result["stage_trace"],
            "elapsed_seconds": round(time.time() - started_at, 2),
        }
        return response, internal, agent_trace_record

    response, internal = _api_ask_impl(question, metadata_filters, retrieval_mode=retrieval_mode,
                                        answer_mode=answer_mode, **api_ask_impl_kwargs)
    agent_trace_record = {
        "agent": "tax_unstructured_query_agent",
        "action": "Queried the tax knowledge graph (Cypher traversal) and document chunks (vector "
                   "search + reranking), then generated an answer from the retrieved evidence.",
        "cypher_ran": bool(response.get("cypher")),
        "chunks_used": len(internal.get("top_chunks") or []),
        "retrieval_mode": retrieval_mode,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    return response, internal, agent_trace_record


# 2026-09-08: NEW - bounds how many times _run_structured_agent() will
# re-prompt the LLM after a sandboxed execution failure (a bad column name,
# a KeyError, a syntax slip). One retry, not an open-ended loop - a second
# consecutive failure is surfaced to the user as an honest "couldn't answer
# from the attached file" rather than burning further LLM calls on a
# question the model has already failed twice to translate into code.
STRUCTURED_AGENT_MAX_ATTEMPTS = 2


def _generate_structured_query_code(question: str, schema_summary: str, retry_context: str = "",
                                     trace=None) -> str:
    """One LLM call: question + schema -> pandas code text. Strips markdown
    code fences defensively (```python ... ``` or bare ``` ... ```) since
    models asked for "code only" still sometimes wrap it in fences anyway -
    exec()'ing a string that still has ``` in it would fail with a
    SyntaxError before the sandbox's own error handling even gets a chance
    to describe the REAL problem."""
    prompt = PROMPT_PACK.get_prompt("structured_data_query", version="v1")
    system = prompt.template.format(schema_summary=schema_summary, question=question,
                                     retry_context=retry_context)
    raw = llm_client.complete(system=system, user_content=question, max_tokens=800,
                               trace=trace, name="structured_data_query")
    code = raw.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines).strip()
    return code


def _stringify_structured_result(result) -> str:
    """Turns whatever run_sandboxed() returned in result["result"] (a
    scalar, a pandas Series, or a pandas DataFrame - see sandbox.py's
    _child_worker() docstring) into a plain string suitable for dropping
    into an answer or the agent_merge prompt. DataFrames/Series use
    to_string() (bounded, readable) rather than repr() (which can add
    truncation ellipses/index noise less suited to a final answer)."""
    import pandas as pd
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.to_string()
    return str(result)


def _run_structured_agent(question: str, attached_file_context: dict, trace=None) -> dict:
    """Generates pandas code from the attached file's schema + the question,
    executes it in sandbox.py's isolated subprocess, and retries ONCE
    (re-prompting with the failure appended as context) if execution fails -
    see STRUCTURED_AGENT_MAX_ATTEMPTS. Returns {"answer": <str>,
    "agent_trace_record": {...}} - the shape _run_agentic_ask() already
    expects from this function (see that function's body above).

    TRANSPARENCY DEFAULT (flagged, not silently assumed - see this
    project's own notes on the open question of whether generated code
    should be fully visible to the whole team or restricted to an admin/
    debug view): agent_trace_record includes the ACTUAL generated code and
    row/column counts, matching the existing evidence-panel convention of
    showing chunk text/Cypher/file trails to everyone. If this needs to be
    restricted later, that's a UI-layer change (hide the field for
    non-admins) rather than a change to what this function computes."""
    schema_summary = attached_file_context["schema_summary"]
    dataframes = attached_file_context["sheets"]
    filename = attached_file_context["filename"]

    retry_context = ""
    last_error = None
    generated_code = None
    exec_result = None

    for attempt in range(1, STRUCTURED_AGENT_MAX_ATTEMPTS + 1):
        generated_code = _generate_structured_query_code(question, schema_summary, retry_context, trace=trace)
        exec_result = sandbox.run_sandboxed(generated_code, dataframes)
        if exec_result.get("ok"):
            break
        last_error = exec_result.get("error", "Unknown execution error")
        app.logger.warning(f"[structured_agent] attempt {attempt} failed: {last_error}")
        retry_context = (
            f"\nYour PREVIOUS attempt at this code failed with this error - fix it:\n"
            f"--- previous code ---\n{generated_code}\n--- error ---\n{last_error}\n"
        )

    agent_trace_record = {
        "agent": "structured_data_query_agent",
        "action": f"Generated pandas code and executed it (sandboxed subprocess) against the attached "
                    f"file {filename!r} to compute an answer directly from its data.",
        "filename": filename,
        "generated_code": generated_code,
        "attempts": attempt if exec_result.get("ok") else STRUCTURED_AGENT_MAX_ATTEMPTS,
        "succeeded": bool(exec_result.get("ok")),
        "elapsed_seconds": exec_result.get("elapsed_seconds"),
    }

    if not exec_result.get("ok"):
        agent_trace_record["error"] = last_error
        return {
            "answer": f"I couldn't compute an answer from the attached file {filename!r} - the "
                       f"generated query failed after {STRUCTURED_AGENT_MAX_ATTEMPTS} attempt(s): {last_error}",
            "agent_trace_record": agent_trace_record,
        }

    result_text = _stringify_structured_result(exec_result["result"])
    agent_trace_record["result_preview"] = result_text[:500]
    return {
        "answer": f"From the attached file {filename!r}:\n\n{result_text}",
        "agent_trace_record": agent_trace_record,
    }


def _run_agentic_ask(question: str, metadata_filters: dict | None, retrieval_mode: str, answer_mode: str,
                      attached_file_context: dict | None = None, trace=None,
                      **api_ask_impl_kwargs) -> tuple[dict, dict]:
    """Orchestrator entry point. `attached_file_context` is None (the common
    case today - see this section's module comment) or a dict describing a
    session-attached structured data file (shape TBD by the file-attachment
    slice - not yet implemented, see _run_structured_agent()).

    Returns (response, agent_trace) - `response` is the SAME public response
    shape _api_ask_impl() already produces (so every existing frontend
    rendering path keeps working unchanged) with its "answer" field
    overwritten by the merged answer when both agents ran; `agent_trace` is a
    NEW dict - {"routing_decision", "routing_reason", "agents_invoked": [...]}
    - attached by the caller (the /api/ask route, once wired) as
    response["agent_trace"] for the evidence panel to render."""
    file_attached = attached_file_context is not None
    routing_decision, raw_routing_output = _classify_agent_routing(question, attached_file_context, trace=trace)

    agents_invoked = []
    unstructured_response = None
    structured_result = None

    if routing_decision in ("UNSTRUCTURED_ONLY", "BOTH"):
        # 2026-09-11: NEW - attached_file_context threaded through so
        # retrieval_mode="structured_research"'s Fact Sheet stage can use it
        # (see fact_sheet_agent.build_fact_sheet()'s attached_file_context
        # param) - every other retrieval_mode ignores this kwarg entirely,
        # so this is a no-op for hybrid/vector_only/semantic_graph/agentic.
        unstructured_response, internal, trace_record = _run_unstructured_agent(
            question, metadata_filters, retrieval_mode, answer_mode,
            attached_file_context=attached_file_context, **api_ask_impl_kwargs)
        agents_invoked.append(trace_record)

    if routing_decision in ("STRUCTURED_ONLY", "BOTH"):
        structured_result = _run_structured_agent(question, attached_file_context, trace=trace)
        agents_invoked.append(structured_result.get("agent_trace_record", {}))

    if routing_decision == "STRUCTURED_ONLY":
        # Structured-only: the structured agent's own result IS the answer -
        # no unstructured pipeline ran at all, so there's no separate
        # "response" scaffold (grounding/source_chunks/cypher/etc.) to slot
        # it into. Builds a minimal response shape consistent with what
        # _api_ask_impl() returns for its own OUT_OF_SCOPE short-circuit
        # (same pattern: a plain answer string, empty evidence fields).
        response = {
            "question": question, "intent": None, "cypher": None,
            "answer": structured_result["answer"],
            "nodes": [], "edges": [], "source_chunks": [], "grounding": None,
        }
    elif routing_decision == "BOTH":
        merge_prompt = PROMPT_PACK.get_prompt("agent_merge", version="v1").template.format(
            unstructured_agent_answer=unstructured_response["answer"],
            structured_agent_answer=structured_result["answer"],
            question=question,
        )
        merged_answer = llm_client.complete(system=merge_prompt, user_content=question, max_tokens=2000,
                                             trace=trace, name="agent_merge").strip()
        response = dict(unstructured_response)
        response["answer"] = merged_answer
    else:  # UNSTRUCTURED_ONLY
        response = unstructured_response

    agent_trace = {
        "routing_decision": routing_decision,
        "routing_reason": raw_routing_output,
        "file_attached": file_attached,
        "agents_invoked": agents_invoked,
    }
    return response, agent_trace


@app.route("/api/ask/quality_checked/start", methods=["POST"])
def api_ask_quality_checked_start():
    body = request.json or {}
    question = (body.get("question") or "").strip()
    metadata_filters = body.get("filters") or None
    # 2026-09-10: NEW - "agentic" accepted here too - see /api/ask's
    # retrieval_mode validation above for what it does.
    retrieval_mode = body.get("retrieval_mode") or "hybrid"
    if retrieval_mode not in ("hybrid", "vector_only", "semantic_graph", "agentic", "structured_research"):
        return jsonify({"error": f"Invalid retrieval_mode {retrieval_mode!r} - must be 'hybrid', "
                                  f"'vector_only', 'semantic_graph', 'agentic', or 'structured_research'."}), 400
    # 2026-09-03: NEW - see /api/ask's answer_mode validation above; same axis.
    answer_mode = body.get("answer_mode") or "research"
    if answer_mode not in ("lookup", "research", "opinion"):
        return jsonify({"error": f"Invalid answer_mode {answer_mode!r} - must be 'lookup', "
                                  f"'research', or 'opinion'."}), 400
    # 2026-09-05: NEW - see /api/ask's include_full_chunks handling above.
    include_full_chunks = bool(body.get("include_full_chunks"))
    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        max_retries = int(body.get("max_retries", DEFAULT_QUALITY_MAX_RETRIES))
    except (TypeError, ValueError):
        max_retries = DEFAULT_QUALITY_MAX_RETRIES
    max_retries = max(0, min(max_retries, MAX_QUALITY_MAX_RETRIES))
    try:
        target_score = float(body.get("target_score", DEFAULT_QUALITY_TARGET_SCORE))
    except (TypeError, ValueError):
        target_score = DEFAULT_QUALITY_TARGET_SCORE
    target_score = max(0.0, min(target_score, 10.0))
    # 2026-08-26: NEW - optional. When the "Improve this answer" button on
    # the Ask screen starts this job off an evaluation the user already ran
    # manually (see api_ask_evaluate() above), it sends that evaluation dict
    # here so attempt 1 is seeded with its feedback rather than starting
    # blind. Loosely validated - only used if it looks like a real
    # evaluation dict (has an "issues" or "overall" key); anything else is
    # ignored rather than erroring the whole request over an optional extra.
    seed_evaluation = body.get("seed_evaluation")
    if not (isinstance(seed_evaluation, dict) and ("issues" in seed_evaluation or "overall" in seed_evaluation)):
        seed_evaluation = None
    # 2026-09-08: NEW - see _run_quality_checked_ask_job()'s docstring for
    # the STRUCTURED_ONLY/BOTH short-circuit this enables.
    data_session_id = body.get("data_session_id") or None

    job_id = uuid.uuid4().hex[:12]
    with _ASK_JOBS_LOCK:
        _ASK_JOBS[job_id] = {
            "status": "running", "phase": "starting", "attempt": 0, "max_attempts": max_retries + 1,
            "latest_score": None, "attempts": [], "log": [], "result": None, "error": None,
            "started_at": time.time(), "finished_at": None,
        }
    thread = threading.Thread(
        target=_run_quality_checked_ask_job,
        args=(job_id, question, metadata_filters, retrieval_mode, max_retries, target_score, seed_evaluation,
              answer_mode, include_full_chunks, data_session_id),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id, "max_attempts": max_retries + 1, "target_score": target_score})


@app.route("/api/ask/quality_checked/status/<job_id>")
def api_ask_quality_checked_status(job_id):
    with _ASK_JOBS_LOCK:
        job = dict(_ASK_JOBS[job_id]) if job_id in _ASK_JOBS else None
    if job is None:
        return jsonify({"error": f"No such job: {job_id}"}), 404
    return jsonify(job)


# ============================================================================
# EVALUATION FEEDBACK HISTORY - 2026-08-26: NEW. Read-only listing over the
# evaluation_log.py Neo4j records (see that module's docstring) - powers the
# "Evaluation Feedback" left-nav page, which shows every question/answer
# evaluation this app has ever run (both manual "Evaluate" clicks and every
# attempt inside a quality-checked auto-retry run) so a user can review the
# full history of scores + reviewer feedback in one place, not just whatever
# happens to still be on screen. Added per explicit request.
# ============================================================================

@app.route("/api/evaluations")
def api_evaluations():
    try:
        limit = int(request.args.get("limit", 300))
    except (TypeError, ValueError):
        limit = 300
    limit = max(1, min(limit, 1000))
    try:
        records = evaluation_log.list_evaluations(limit=limit)
        return jsonify({"evaluations": records})
    except Exception as e:
        app.logger.exception("Failed in /api/evaluations")
        return jsonify({"error": f"Could not load evaluation history: {e}"}), 500


# ============================================================================
# DASHBOARD - real KPIs/health/activity, sourced from Neo4j (graph size,
# AuditRun history) + Langfuse (24h stats) + embedder config. No fabricated
# numbers anywhere: any figure that can't be honestly computed comes back as
# null/"unavailable" and the frontend renders an explicit empty state for it
# rather than a placeholder number.
# ============================================================================

@app.route("/api/dashboard/summary")
def api_dashboard_summary():
    try:
        with neo4j_client.get_session() as session:
            counts = reset_module.count_by_label(session)

            # Documents ingested: distinct document_id values recorded on
            # AuditRun nodes across every ingestion-shaped script this app
            # runs (single + batch, both file-type paths) - the same
            # AuditRun trail domain_packs/tax/pipeline.py's engines already
            # write to via framework.core.infra.audit_log at ingest time.
            doc_rows = list(session.run(
                "MATCH (r:AuditRun) WHERE r.document_id IS NOT NULL "
                "RETURN count(DISTINCT r.document_id) AS c"
            ))
            documents_ingested = doc_rows[0]["c"] if doc_rows else 0

            # Recent activity feed: the last 12 AuditRun events, most recent
            # first - real pipeline history, not a mock feed.
            activity_rows = list(session.run(
                "MATCH (r:AuditRun) WHERE r.started_at IS NOT NULL "
                "RETURN r.run_id AS run_id, r.script AS script, r.document_id AS document_id, "
                "       r.status AS status, r.started_at AS started_at, r.finished_at AS finished_at "
                "ORDER BY r.started_at DESC LIMIT 12"
            ))
            activity = [dict(row) for row in activity_rows]

            # Ingestion pipeline health: status breakdown of the last 20 runs
            # per distinct script name actually seen in AuditRun - "5 stages"
            # per the README is aspirational for a prototype with a fixed
            # mock list; here we report whatever real script names show up
            # (a real, honest stand-in), each with real success/running/error
            # counts from the last 20 runs of that script.
            stage_rows = list(session.run(
                "MATCH (r:AuditRun) WHERE r.script IS NOT NULL "
                "RETURN r.script AS script, r.status AS status, r.started_at AS started_at "
                "ORDER BY r.started_at DESC LIMIT 500"
            ))
            by_script: dict[str, list[dict]] = {}
            for row in stage_rows:
                by_script.setdefault(row["script"], []).append(row)
            pipeline_health = []
            for script, rows in list(by_script.items())[:5]:
                recent = rows[:20]
                errors = sum(1 for r in recent if r["status"] == "error")
                running = sum(1 for r in recent if r["status"] == "running")
                total = len(recent)
                pct_healthy = round(100 * (total - errors) / total) if total else 100
                # `script` is the raw AuditRun.script value, which is often a
                # full dotted module/function path (e.g.
                # "framework.core.ingest.pipeline_ops.ingest_graph_json") -
                # too long to sit next to a progress bar without overlapping
                # it. `stage_short` is just the last dotted segment (still
                # real data, not invented) for compact display; the UI can
                # still show the full `stage` value in a title/tooltip.
                stage_short = script.rsplit(".", 1)[-1] if script else script
                pipeline_health.append({
                    "stage": script, "stage_short": stage_short,
                    "sample_size": total, "errors": errors, "running": running,
                    "pct_healthy": pct_healthy,
                    "status_text": "Flagged" if errors else ("Running" if running else "Healthy"),
                })

        embedding_coverage = None
        if counts["total_nodes"]:
            chunk_row = None
            with neo4j_client.get_session() as session:
                chunk_row = session.run("MATCH (c:Chunk) RETURN count(c) AS c").single()
            total_chunks = chunk_row["c"] if chunk_row else 0
            if total_chunks:
                embedding_coverage = round(100 * counts["chunks_with_embedding"] / total_chunks, 1)

        langfuse_24h = observability.get_recent_traces(hours=24, limit=200)
        gen_stats = observability.get_generation_stats(hours=24)

        return jsonify({
            "kpis": {
                "documents_ingested": documents_ingested,
                "questions_asked_24h": langfuse_24h["stats"]["count"] if langfuse_24h["available"] else None,
                "questions_asked_available": langfuse_24h["available"],
                "kg_nodes": counts["total_nodes"],
                "avg_answer_latency_seconds": gen_stats["avg_latency_seconds"] if gen_stats["available"] else None,
                "avg_answer_latency_available": gen_stats["available"],
            },
            "pipeline_health": pipeline_health,
            "activity": activity,
            "graph_size": {
                "nodes": counts["total_nodes"],
                "relationships": counts["total_relationships"],
                "embeddings": counts["chunks_with_embedding"],
            },
            "langfuse_24h": {
                "available": langfuse_24h["available"],
                "reason": langfuse_24h.get("reason"),
                "traces": langfuse_24h["stats"]["count"],
                "avg_latency_seconds": langfuse_24h["stats"]["avg_latency_seconds"],
                "total_cost": langfuse_24h["stats"]["total_cost"],
            },
            "embedding_coverage": {
                "pct": embedding_coverage,
                "model": EMBEDDING_MODEL_NAME,
            },
        })
    except Exception as e:
        app.logger.exception("Failed in /api/dashboard/summary")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# MONITORING - real Langfuse read-side data (observability.py's
# get_recent_traces/get_generation_stats), degrading to an honest empty
# state (not an error page) when Langfuse isn't configured.
# ============================================================================

@app.route("/api/monitoring/summary")
def api_monitoring_summary():
    try:
        hours = int(request.args.get("hours", 24))
        limit = int(request.args.get("limit", 100))
        data = observability.get_recent_traces(hours=hours, limit=limit)
        return jsonify(data)
    except Exception as e:
        app.logger.exception("Failed in /api/monitoring/summary")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# SETTINGS - real Neo4j connectivity + config, real Langfuse config, real
# data-source list derived from the loaded pack's FILE_TYPES (see
# settings_store.py's module docstring for why these specific 3 rows and not
# the prototype's 5 hardcoded ones), toggle state persisted to a small JSON
# file (settings_store.py) - not wired to actually pause anything, documented
# as such.
# ============================================================================

@app.route("/api/settings/summary")
def api_settings_summary():
    try:
        neo4j_connected = False
        neo4j_error = None
        try:
            neo4j_client.verify_connectivity()
            neo4j_connected = True
        except Exception as e:
            neo4j_error = str(e)

        file_types = PIPELINE.FILE_TYPES if PIPELINE is not None else {}
        stored = settings_store.load_settings()

        # Last-sync per file type: most recent finished AuditRun with that
        # file_type recorded in its details/args, if any - "Never" (None) if
        # nothing's been ingested yet for that type. Best-effort: AuditRun
        # doesn't always carry file_type explicitly in older records, so this
        # is reported as "unavailable" rather than guessed when absent.
        last_sync_by_type = {}
        if neo4j_connected:
            try:
                with neo4j_client.get_session() as session:
                    rows = list(session.run(
                        "MATCH (r:AuditRun) WHERE r.finished_at IS NOT NULL "
                        "RETURN r.args AS args, r.finished_at AS finished_at "
                        "ORDER BY r.finished_at DESC LIMIT 200"
                    ))
                import json as _json
                for row in rows:
                    try:
                        args = _json.loads(row["args"] or "{}")
                    except Exception:
                        args = {}
                    ft = args.get("file_type")
                    if ft and ft not in last_sync_by_type:
                        last_sync_by_type[ft] = row["finished_at"]
            except Exception:
                pass

        data_sources = []
        for key, ft in file_types.items():
            data_sources.append({
                "id": key,
                "name": ft.get("label", key),
                "description": ft.get("description", ""),
                "enabled": stored["data_sources"].get(key, {}).get("enabled", True),
                "last_sync": last_sync_by_type.get(key),
            })

        embedding_dim = None
        try:
            from framework.core.ingest import embedder as embedder_module
            embedding_dim = embedder_module.embedding_dimension()
        except Exception:
            pass  # loading the model just to report its dim isn't worth doing eagerly; report null if unavailable

        # refresh_langfuse_config() re-reads .env and is a cheap no-op unless
        # the key values actually changed since the last check (see its
        # docstring) - called here so a live .env edit (keys added/changed
        # while this process was already running) shows up on the Settings
        # page immediately, instead of only after a full app restart or an
        # explicit toggle click (the two things that used to be the ONLY
        # way LANGFUSE_CONFIGURED/LANGFUSE_ENABLED ever got re-evaluated -
        # this route used to just read those globals as of process-start).
        observability.refresh_langfuse_config()

        return jsonify({
            "neo4j": {
                "connected": neo4j_connected, "error": neo4j_error,
                "uri": neo4j_client.NEO4J_URI, "database": neo4j_client.NEO4J_DATABASE,
            },
            "langfuse": {
                "enabled": observability.LANGFUSE_ENABLED, "host": observability.LANGFUSE_HOST,
                "configured": observability.LANGFUSE_CONFIGURED,
                "toggle_on": stored.get("langfuse_tracing_enabled", True),
                "reason": observability.LANGFUSE_DISABLED_REASON,
            },
            "data_sources": data_sources,
            "embedding_model": {
                "name": EMBEDDING_MODEL_NAME, "dimensions": embedding_dim,
                "index": f"Neo4j vector index (cosine) — '{VECTOR_INDEX_NAME}'",
            },
            # 2026-08-27: NEW - current LLM sampling temperature (settings_store-
            # backed, see that module's docstring). null means "no override -
            # using whatever the active provider's own API default is", which
            # is also what the settings file defaults to.
            "temperature": {
                "value": stored.get("temperature"),
                "min": settings_store.TEMPERATURE_MIN, "max": settings_store.TEMPERATURE_MAX,
                "provider": llm_client.USE_MODEL,
            },
        })
    except Exception as e:
        app.logger.exception("Failed in /api/settings/summary")
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings/data-sources/<source_id>", methods=["POST"])
def api_settings_toggle_data_source(source_id):
    body = request.json or {}
    if "enabled" not in body:
        return jsonify({"error": "'enabled' (bool) is required"}), 400
    try:
        saved = settings_store.set_data_source_enabled(source_id, bool(body["enabled"]))
        return jsonify(saved)
    except Exception as e:
        app.logger.exception("Failed in /api/settings/data-sources/<id>")
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings/langfuse", methods=["POST"])
def api_settings_langfuse():
    """Flips the Settings screen's Langfuse on/off preference. The actual
    public/secret key/host are NEVER accepted here or written to .env by
    this app - they only ever come from .env itself (set them by hand, the
    normal way any other secret in this project is configured); this route
    just persists the toggle (settings_store.set_langfuse_tracing_enabled)
    and rebuilds observability's client against .env + the new toggle
    state. Returns real enabled/reason status - turning the toggle on when
    .env has no (or bad) keys reports back honestly (enabled: false + why),
    it never claims success it can't back up."""
    body = request.json or {}
    if "enabled" not in body:
        return jsonify({"error": "'enabled' (bool) is required"}), 400
    try:
        settings_store.set_langfuse_tracing_enabled(bool(body["enabled"]))
        status = observability.refresh_langfuse_config(force=True)
        return jsonify({
            "enabled": status["enabled"], "reason": status["reason"],
            "configured": observability.LANGFUSE_CONFIGURED, "host": observability.LANGFUSE_HOST,
        })
    except Exception as e:
        app.logger.exception("Failed in /api/settings/langfuse")
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings/temperature", methods=["POST"])
def api_settings_temperature():
    """2026-08-27: NEW - sets the LLM sampling temperature from the Settings
    page, per explicit request ("add an option to change the temperature in
    settings page and it should be hot pickup everytime i ask a question").
    Just persists to settings_store.py (the same JSON-file, re-read-on-every-
    call store the data-source/langfuse toggles already use) - no restart
    needed, since llm_client.refresh_model_config() re-reads it fresh on
    every single LLM call (see that function's docstring). `temperature:
    null` resets to "no override - use the provider's own default"."""
    body = request.json or {}
    if "temperature" not in body:
        return jsonify({"error": "'temperature' (float or null) is required"}), 400
    try:
        saved = settings_store.set_temperature(body["temperature"])
        return jsonify({"temperature": saved.get("temperature"),
                         "min": settings_store.TEMPERATURE_MIN, "max": settings_store.TEMPERATURE_MAX})
    except Exception as e:
        app.logger.exception("Failed in /api/settings/temperature")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# INGESTION METHODS WITHOUT A REAL BACKEND (Scrape URL / Paste text / Watch
# folder) - the task requires these be UI-complete but honestly 501, not
# faked as successful. Upload and Bulk-batch already have real endpoints
# above (/api/upload[/batch] + /api/ingest[/batch]/start).
# ============================================================================

@app.route("/api/ingest/scrape-url", methods=["POST"])
def api_ingest_scrape_url():
    return jsonify({
        "error": "Scrape URL ingestion is not implemented yet in this build - there is no web-scraping "
                 "engine wired up behind this button. Use Upload or Bulk batch instead.",
    }), 501


@app.route("/api/ingest/paste-text", methods=["POST"])
def api_ingest_paste_text():
    return jsonify({
        "error": "Paste-text ingestion is not implemented yet in this build - pasted text has no "
                 "document-conversion path wired up. Save it as a .txt/.md file and use Upload instead.",
    }), 501


@app.route("/api/ingest/watch-folder", methods=["POST"])
def api_ingest_watch_folder():
    return jsonify({
        "error": "Watch-folder ingestion is not implemented yet in this build - there is no folder-"
                 "watcher/scheduler process in this codebase. Use Upload or Bulk batch instead.",
    }), 501


# ============================================================================
# DELETE NEO4J DATA - guarded destructive wipe. Step 2's itemized counts come
# from reset_module.count_by_label() (real per-label counts); the delete
# itself reuses the existing /api/graph/clear engine and additionally logs
# the operation via audit_log.start_run/finish_run (script="delete_all_data"),
# matching how every other pipeline operation in this app is audited.
# ============================================================================

@app.route("/api/delete/counts")
def api_delete_counts():
    try:
        with neo4j_client.get_session() as session:
            counts = reset_module.count_by_label(session)
        return jsonify(counts)
    except Exception as e:
        app.logger.exception("Failed in /api/delete/counts")
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete/confirm", methods=["POST"])
def api_delete_confirm():
    body = request.json or {}
    if body.get("confirm") != "DELETE ALL DATA":
        return jsonify({
            "error": "Refusing to delete without confirmation. "
                     "POST {\"confirm\": \"DELETE ALL DATA\"} to proceed."
        }), 400

    run_id = audit_log.start_run("delete_all_data", document_id=None, args={"source": "delete_screen"})
    try:
        with neo4j_client.get_session() as session:
            before_counts = reset_module.count_by_label(session)
            stats = reset_module.clear_all_data(session)
        cache_cleared = cache.clear_all_cache()
        graphrag_module.invalidate_schema_cache()
        audit_log.finish_run(run_id, "success", details={
            "nodes_deleted": stats["nodes_deleted"], "batches": stats["batches"],
            "before_counts": before_counts, "cache_keys_cleared": cache_cleared,
        })
        return jsonify({
            "graph": stats, "cache_keys_cleared": cache_cleared,
            "before_counts": before_counts, "run_id": run_id, "logged": True,
        })
    except Exception as e:
        audit_log.finish_run(run_id, "error", details={"error": str(e)})
        app.logger.exception("Failed in /api/delete/confirm")
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete/documents/backfill", methods=["POST"])
def api_delete_documents_backfill():
    """Non-destructive - only ADDS document_id/document_name/source_type/
    authority_level/jurisdiction to CaseLaw graphs ingested before
    extract_case_law.py started stamping them, so they become visible/
    selectable in the Delete screen's per-document picker without needing a
    re-ingest. See reset_module.backfill_case_law_document_metadata()."""
    try:
        with neo4j_client.get_session() as session:
            stats = reset_module.backfill_case_law_document_metadata(session)
        graphrag_module.invalidate_schema_cache()
        return jsonify(stats)
    except Exception as e:
        app.logger.exception("Failed in /api/delete/documents/backfill")
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete/documents")
def api_delete_documents_list():
    """Lists every distinct ingested document (by document_id) for the
    Delete screen's per-document picker - see reset_module.list_documents()."""
    try:
        with neo4j_client.get_session() as session:
            documents = reset_module.list_documents(session)
        return jsonify({"documents": documents})
    except Exception as e:
        app.logger.exception("Failed in /api/delete/documents")
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete/documents/confirm", methods=["POST"])
def api_delete_documents_confirm():
    """Selective delete - only the entities/chunks belonging to the given
    document_id(s), not a full wipe. Uses a DIFFERENT confirmation phrase
    than /api/delete/confirm ("DELETE SELECTED" vs "DELETE ALL DATA") so the
    two destructive actions can never be triggered by copy-pasting the wrong
    confirmation string."""
    body = request.json or {}
    document_ids = body.get("document_ids") or []
    if not isinstance(document_ids, list) or not document_ids:
        return jsonify({"error": "document_ids must be a non-empty list."}), 400
    if body.get("confirm") != "DELETE SELECTED":
        return jsonify({
            "error": "Refusing to delete without confirmation. "
                     "POST {\"confirm\": \"DELETE SELECTED\", \"document_ids\": [...]} to proceed."
        }), 400

    run_id = audit_log.start_run("delete_documents", document_id=None,
                                  args={"source": "delete_screen", "document_ids": document_ids})
    try:
        with neo4j_client.get_session() as session:
            before = {d["document_id"]: d for d in reset_module.list_documents(session)
                      if d["document_id"] in document_ids}
            stats = reset_module.delete_documents(session, document_ids)
        cache_cleared = cache.clear_all_cache()
        graphrag_module.invalidate_schema_cache()
        audit_log.finish_run(run_id, "success", details={
            "entities_deleted": stats["entities_deleted"], "chunks_deleted": stats["chunks_deleted"],
            "batches": stats["batches"], "before": before, "cache_keys_cleared": cache_cleared,
        })
        return jsonify({
            "graph": stats, "cache_keys_cleared": cache_cleared,
            "before": before, "run_id": run_id, "logged": True,
        })
    except Exception as e:
        audit_log.finish_run(run_id, "error", details={"error": str(e)})
        app.logger.exception("Failed in /api/delete/documents/confirm")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chats", methods=["GET", "POST"])
def api_chats():
    """GET lists all saved chat sessions (sidebar); POST creates a new one,
    auto-named from `question`, and appends `question`/`answer` as its first
    message in one call - see chat_history.create_session()'s docstring for
    why chat creation and first-message append happen together rather than
    as two separate round trips."""
    if request.method == "GET":
        try:
            return jsonify(chat_history.list_sessions())
        except Exception as e:
            app.logger.exception("Failed in GET /api/chats")
            return jsonify({"error": str(e)}), 500

    body = request.json or {}
    question = (body.get("question") or "").strip()
    answer = body.get("answer")
    if not question:
        return jsonify({"error": "question is required"}), 400
    try:
        result = chat_history.create_session(question, answer, name=body.get("name"))
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Failed in POST /api/chats")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chats/<chat_id>", methods=["GET", "PATCH", "DELETE"])
def api_chat_detail(chat_id):
    if request.method == "GET":
        try:
            session_obj = chat_history.get_session(chat_id)
            if session_obj is None:
                return jsonify({"error": "Chat not found"}), 404
            return jsonify(session_obj)
        except Exception as e:
            app.logger.exception("Failed in GET /api/chats/<id>")
            return jsonify({"error": str(e)}), 500

    if request.method == "PATCH":
        body = request.json or {}
        new_name = (body.get("name") or "").strip()
        if not new_name:
            return jsonify({"error": "name is required"}), 400
        try:
            updated = chat_history.rename_session(chat_id, new_name)
            if updated is None:
                return jsonify({"error": "Chat not found"}), 404
            return jsonify(updated)
        except Exception as e:
            app.logger.exception("Failed in PATCH /api/chats/<id>")
            return jsonify({"error": str(e)}), 500

    # DELETE
    try:
        deleted = chat_history.delete_session(chat_id)
        if not deleted:
            return jsonify({"error": "Chat not found"}), 404
        return jsonify({"deleted": True})
    except Exception as e:
        app.logger.exception("Failed in DELETE /api/chats/<id>")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chats/<chat_id>/messages", methods=["POST"])
def api_chat_append_message(chat_id):
    body = request.json or {}
    question = (body.get("question") or "").strip()
    answer = body.get("answer")
    if not question:
        return jsonify({"error": "question is required"}), 400
    try:
        if chat_history.get_session(chat_id) is None:
            return jsonify({"error": "Chat not found"}), 404
        message = chat_history.append_message(chat_id, question, answer)
        return jsonify({"message": message})
    except Exception as e:
        app.logger.exception("Failed in POST /api/chats/<id>/messages")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# GLOBAL JSON ERROR HANDLERS - 2026-08-27: NEW. Every route above returns
# JSON on its OWN handled errors (try/except -> jsonify({"error": ...})),
# but a request can also fail BEFORE any route body runs at all - oversized
# upload (413, once MAX_CONTENT_LENGTH is exceeded), a bad/mistyped URL
# (404), or a genuinely unhandled exception Flask itself catches (500). Those
# were previously returned as Flask's default HTML error pages, which broke
# every frontend fetch call that blindly does `await res.json()` on the
# response - the browser then throws a confusing "Unexpected token '<'...
# is not valid JSON" with NO indication of the real cause (e.g. "the upload
# was larger than the server allows"). These three handlers make every
# error response from this app JSON, matching every route's own error shape,
# so the frontend's existing `err.message` display actually shows something
# useful instead of a JSON-parse crash.
# ============================================================================
@app.errorhandler(413)
def handle_payload_too_large(e):
    max_mb = app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)
    return jsonify({
        "error": f"Upload too large - this server accepts at most {max_mb:.0f}MB per request. "
                 f"If uploading multiple files at once, try fewer/smaller files per batch."
    }), 413


@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({"error": f"No such endpoint: {request.path}"}), 404


@app.errorhandler(500)
def handle_server_error(e):
    app.logger.exception("Unhandled server error")
    return jsonify({"error": "Internal server error - see server logs for details."}), 500


if __name__ == "__main__":
    # use_reloader=False is deliberate: with it on (Flask's default when
    # debug=True), any file change anywhere under this process's watched
    # tree - including this app's own static/*.py edits, or even
    # response_config.json/webapp_settings.json being written by a request -
    # respawns a brand-new worker process. A fresh process means every
    # module-level "load once" cache (the embedding model in graphrag.py,
    # the reranker in reranker.py, the SentenceTransformer in embedder.py)
    # is gone and reloads from disk on the very next request - this is what
    # produced the "it reloads model weights on every question" symptom.
    # debug=True is kept (still get tracebacks/auto server errors in the
    # browser during development) but the reloader itself is off, so model
    # weights genuinely load once per run, as intended.
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
