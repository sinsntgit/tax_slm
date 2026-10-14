"""
Ephemeral, per-session "Generate Opinions" workspace: upload one or more
documents (contracts, case laws, notices) for a specific client matter, ask
questions grounded ONLY in those documents (optionally supplemented by the
main KG), and generate a structured BIG4-style opinion / file-suit response
/ clarification-questions output - all scoped to an in-memory workspace_id,
never written to the permanent Neo4j graph.

WHY EPHEMERAL, NOT PERSISTED TO THE MAIN KG (2026-08-27 design note): the
documents uploaded here are typically client-specific and often
confidential (a specific client's contract, a notice they received) -
merging them into the shared firm-wide knowledge graph would put
client-confidential material in a graph every user of this app can query.
Kept in-process memory instead (same pattern as framework/webapp/app.py's
_JOBS/_ASK_JOBS job registries) - workspace state does NOT survive an app
restart, which is the deliberate tradeoff for keeping this data out of the
permanent, shared store. A later version could add explicit, opt-in
persistence (e.g. only when a user explicitly saves a finished opinion),
but that is NOT what this module does today - flagged as an open design
question in the accompanying delivery notes, not decided unilaterally here.

Text extraction reuses the existing generic PDF/DOCX-to-text conversion
already built for the ingestion pipeline (domain_packs/tax/converters/
pdf_to_md_generic.py) rather than writing a new converter, since "convert a
document to plain text" is the identical problem already solved there.
"""
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

from framework.core.infra import llm_client, step_logger

_WORKSPACES: dict[str, dict] = {}

# Per-document character cap when building a prompt - keeps a workspace with
# a handful of uploaded contracts/judgments within a sane prompt budget
# without needing a full chunking/embedding pipeline for what's usually a
# small, ephemeral set of documents (unlike the permanent KG's Act/case-law
# corpus, which genuinely needs chunking - see chunking.py).
MAX_DOC_CHARS_IN_PROMPT = 40_000

OPINION_TYPE_INSTRUCTIONS = {
    "opinion": (
        "MODE: FULL LEGAL/TAX OPINION. Produce a complete, formal opinion memo addressing the "
        "matter fully, as a Big Four tax partner would sign off on."
    ),
    "file_suit_response": (
        "MODE: RESPONSE TO A FILED NOTICE/SUIT. The uploaded document(s) include a notice, show-"
        "cause letter, or suit the client has received. Draft the substance of a formal response: "
        "the facts as the client's side sees them, the legal grounds for the response/defense, and "
        "the relief or outcome being sought - framed as a firm's advice on how to respond, not as "
        "the filed submission itself."
    ),
    "clarification_questions": (
        "MODE: CLARIFICATION QUESTIONS ONLY. Do NOT attempt a full opinion. Your primary job is to "
        "identify every material question that must be answered before a full opinion could "
        "responsibly be given - gaps in facts, ambiguous clauses, missing documents, unclear "
        "timelines, jurisdictional uncertainty, etc."
    ),
}


def create_workspace(matter_context: dict | None = None) -> str:
    workspace_id = uuid.uuid4().hex[:12]
    _WORKSPACES[workspace_id] = {
        "documents": [],  # [{"name", "text", "char_count", "uploaded_at"}]
        "matter_context": matter_context or {},
        "created_at": time.time(),
        "qa_history": [],       # [{"question", "answer", "asked_at"}]
        "opinions": [],         # [{"opinion_type", "result": {...}, "generated_at"}]
    }
    step_logger.log(f"[opinion_workspace] created workspace {workspace_id}")
    return workspace_id


def get_workspace(workspace_id: str) -> dict | None:
    return _WORKSPACES.get(workspace_id)


def list_workspaces() -> list[dict]:
    return [
        {
            "workspace_id": wid,
            "document_count": len(w["documents"]),
            "document_names": [d["name"] for d in w["documents"]],
            "matter_context": w["matter_context"],
            "created_at": w["created_at"],
            "opinion_count": len(w["opinions"]),
        }
        for wid, w in sorted(_WORKSPACES.items(), key=lambda kv: -kv[1]["created_at"])
    ]


def delete_workspace(workspace_id: str) -> bool:
    removed = _WORKSPACES.pop(workspace_id, None) is not None
    if removed:
        step_logger.log(f"[opinion_workspace] deleted workspace {workspace_id}")
    return removed


def _extract_text(upload_path: Path) -> str:
    """Converts an uploaded file to plain text. Runs the existing generic
    converter as a subprocess (same pattern domain_packs/tax/pipeline.py
    already uses for every converter) so a malformed upload can't take
    down the webapp process itself."""
    suffix = upload_path.suffix.lower()
    if suffix in (".txt", ".md"):
        return upload_path.read_text(encoding="utf-8", errors="replace")

    converters_dir = Path(__file__).resolve().parents[3] / "domain_packs" / "tax" / "converters"
    out_path = upload_path.with_suffix(".extracted.md")
    cmd = [sys.executable, str(converters_dir / "pdf_to_md_generic.py"),
           "--in", str(upload_path), "--out", str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    if proc.returncode != 0 or not out_path.exists():
        tail = (proc.stderr or proc.stdout or "")[-500:]
        raise RuntimeError(f"Could not extract text from {upload_path.name}: {tail}")
    return out_path.read_text(encoding="utf-8", errors="replace")


def add_document(workspace_id: str, upload_path: Path, display_name: str) -> dict:
    workspace = _WORKSPACES.get(workspace_id)
    if workspace is None:
        raise KeyError(f"No such workspace: {workspace_id}")
    text = _extract_text(upload_path)
    doc = {"name": display_name, "text": text, "char_count": len(text), "uploaded_at": time.time()}
    workspace["documents"].append(doc)
    step_logger.log(f"[opinion_workspace] {workspace_id}: added '{display_name}' ({len(text)} chars)")
    return {"name": display_name, "char_count": len(text)}


def _document_context_block(workspace: dict) -> str:
    if not workspace["documents"]:
        return "(no documents uploaded)"
    parts = []
    for doc in workspace["documents"]:
        text = doc["text"]
        truncated = text if len(text) <= MAX_DOC_CHARS_IN_PROMPT else \
            text[:MAX_DOC_CHARS_IN_PROMPT] + "\n...[truncated - document exceeds per-document prompt budget]"
        parts.append(f"=== {doc['name']} ===\n{truncated}")
    return "\n\n".join(parts)


def _matter_context_block(workspace: dict) -> str:
    mc = workspace.get("matter_context") or {}
    if not mc:
        return "(none provided)"
    return "\n".join(f"- {k}: {v}" for k, v in mc.items() if v)


def ask(workspace_id: str, question: str, prompt_pack, kg_context: str | None = None) -> dict:
    workspace = _WORKSPACES.get(workspace_id)
    if workspace is None:
        raise KeyError(f"No such workspace: {workspace_id}")
    if not workspace["documents"]:
        raise ValueError("No documents uploaded to this workspace yet - upload at least one before asking a question.")

    prompt = prompt_pack.get_prompt("document_qa")
    kg_block = (f"SUPPLEMENTARY KNOWLEDGE-GRAPH CONTEXT (statutes/case law from the main graph, if relevant):\n"
                f"{kg_context}") if kg_context else ""
    system_prompt = prompt.template.format(
        document_context=_document_context_block(workspace), kg_context_block=kg_block, question=question,
    )
    answer = llm_client.complete(
        system=system_prompt, user_content="Answer the question above based only on the material provided.",
        max_tokens=2000, name="document_qa", purpose="retrieval",
    ).strip()

    record = {"question": question, "answer": answer, "asked_at": time.time()}
    workspace["qa_history"].append(record)
    return record


_OPINION_JSON_KEYS = (
    "facts", "assumptions", "questions", "opinion_applicable_law", "analysis",
    "risk_classification", "opinion_recommendation", "opinion_conclusion",
    "limitations", "reliance_clauses",
)


def _parse_opinion_json(raw: str) -> dict:
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    parsed = json.loads(cleaned)
    # Normalize - guarantee every expected key exists so the frontend never
    # has to defensively check for a missing field.
    result = {k: parsed.get(k, "" if k not in ("questions", "opinion_applicable_law") else []) for k in _OPINION_JSON_KEYS}
    return result


def generate_opinion(workspace_id: str, opinion_type: str, prompt_pack,
                      user_question: str | None = None, kg_context: str | None = None) -> dict:
    workspace = _WORKSPACES.get(workspace_id)
    if workspace is None:
        raise KeyError(f"No such workspace: {workspace_id}")
    if not workspace["documents"]:
        raise ValueError("No documents uploaded to this workspace yet - upload at least one before generating an opinion.")
    if opinion_type not in OPINION_TYPE_INSTRUCTIONS:
        raise ValueError(f"Unknown opinion_type {opinion_type!r} - expected one of {sorted(OPINION_TYPE_INSTRUCTIONS)}")

    prompt = prompt_pack.get_prompt("opinion_generation")
    matter_block = _matter_context_block(workspace)
    if user_question:
        matter_block += f"\n\nSPECIFIC QUESTION FROM THE USER TO ADDRESS: {user_question}"
    kg_block = (f"SUPPLEMENTARY KNOWLEDGE-GRAPH CONTEXT (statutes/case law from the main graph, if relevant):\n"
                f"{kg_context}") if kg_context else ""
    system_prompt = prompt.template.format(
        opinion_type_instructions=OPINION_TYPE_INSTRUCTIONS[opinion_type],
        matter_context=matter_block,
        document_context=_document_context_block(workspace),
        kg_context_block=kg_block,
    )
    raw = llm_client.complete(
        system=system_prompt, user_content="Produce the JSON object now.",
        max_tokens=4000, name="opinion_generation", purpose="retrieval",
    )
    try:
        result = _parse_opinion_json(raw)
        parse_ok = True
    except (json.JSONDecodeError, ValueError) as e:
        step_logger.log(f"[opinion_workspace] {workspace_id}: opinion JSON parse failed ({e}) - returning raw text")
        result = {"raw_text": raw}
        parse_ok = False

    record = {
        "opinion_type": opinion_type, "result": result, "parse_ok": parse_ok,
        "generated_at": time.time(), "user_question": user_question,
    }
    workspace["opinions"].append(record)
    return record
