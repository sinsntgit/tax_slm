"""
Duplicate-document prevention for the tax pack's ingestion pipeline
(domain_packs/tax/pipeline.py calls into this; kept in its own module so
pipeline.py's own changes stay small). 2026-10-14: NEW.

BUGFIX for: data/converted_md on the owner's machine held the SAME judgment
converted 2-4 times (Samsung x4, Vodafone x3, Engineering Analysis x2) under
different 8-hex prefixes. Root cause: pipeline.save_upload() prefixed every
upload with a fresh uuid4, and everything downstream keys off that name -
convert_document()'s .md path is `<upload stem>.md`, and the batch path's
document_id is the upload stem - so each re-upload of an identical PDF
became a brand-new document: re-converted, re-extracted (a fresh LLM call)
and re-loaded. CaseLaw ids are slugs of the LLM-extracted citation
(extract_case_law.py), which differs run to run ("Dy. Commissioner" vs
"Deputy Commissioner"), so the graph ended up with duplicate CaseLaw nodes
and duplicate Issue/Conclusion/Argument/Chunk sets as well.

Fix, in three deterministic layers:
  1. save_upload_dedup(): the upload prefix is the first 8 hex of the file's
     SHA-256, and byte-identical content returns the EXISTING upload path
     (pre-fix uuid-prefixed uploads are matched by name+size+hash) - so the
     same bytes always map to the same upload / .md / batch document_id.
  2. Ingest time: every entity loaded from a file is stamped with
     `source_file_sha256` (stamp_source_file_hash()); a job whose file is
     already in the graph is SKIPPED with a clear log line unless the caller
     passes force_reingest=True ("re-ingest anyway") - prepare_duplicate_check().
  3. A forced re-ingest of a judgment converges on the SAME CaseLaw node
     (converge_case_law_ids()) even when the LLM words the citation
     differently this time.
Duplicates already in the graph are cleaned with
framework/cli/merge_duplicate_caselaw.py (dry run first).
"""
import glob
import hashlib
import json
import re
from pathlib import Path

from framework.core.ingest import provenance  # 2026-10-14 (phase0/review-integrity): _is_stub()

SOURCE_FILE_HASH_PROP = "source_file_sha256"
# extract_case_law.py / extract_circular.py precedent/reference stub nodes
# all carry extraction_notes starting with this.
_STUB_NOTE_PREFIX = "Stub created from"


def sha256_of_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_existing_upload(upload_dir: Path, digest: str, name: str, size: int) -> Path | None:
    """An already-saved upload with exactly these bytes, or None. Checks the
    content-hash-prefixed name first, then pre-2026-10-14 uuid-prefixed
    uploads of the same original filename and size (always hash-verified -
    a name+size match alone is never trusted)."""
    candidates = sorted(upload_dir.glob(f"{digest[:8]}_*")) + sorted(upload_dir.glob(f"*_{glob.escape(name)}"))
    seen = set()
    for p in candidates:
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        try:
            if p.stat().st_size == size and sha256_of_file(p) == digest:
                return p
        except OSError:
            continue
    return None


def save_upload_dedup(upload_dir: Path, filename: str, content: bytes) -> tuple[Path, dict]:
    """Saves an upload as `<sha256[:8]>_<filename>` and returns (path, info).
    If byte-identical content was uploaded before (under any name), nothing
    is written and the EXISTING path is returned with
    info["duplicate_upload"]=True. info always carries content_sha256."""
    name = Path(filename).name
    digest = sha256_of_bytes(content)
    existing = find_existing_upload(upload_dir, digest, name, len(content))
    if existing is not None:
        return existing, {"content_sha256": digest, "duplicate_upload": True, "existing_upload": existing.name}
    path = upload_dir / f"{digest[:8]}_{name}"
    path.write_bytes(content)
    return path, {"content_sha256": digest, "duplicate_upload": False, "existing_upload": None}


def find_ingested_copies(content_sha256: str, md_path: Path | None = None) -> list[dict]:
    """Documents already in the graph that were ingested from these exact
    bytes: entities stamped with source_file_sha256 (ingested after this
    fix), or - for older ingestions, which carry no hash - entities whose
    sourceMdPath is the .md this upload converts to (same upload path =>
    same bytes, now that upload names are content-addressed). Returns
    [{"document_id", "document_name", "file_type", "nodes"}]. Label-less
    scan (no cross-label index exists in Neo4j 5 Community) - fine at this
    corpus size. Raises if Neo4j is unreachable; the caller decides."""
    from framework.core.infra import neo4j_client
    md = str(Path(md_path).resolve()) if md_path is not None else None
    with neo4j_client.get_session() as session:
        rows = session.run(
            f"""
            MATCH (n)
            WHERE n.{SOURCE_FILE_HASH_PROP} = $sha OR ($md IS NOT NULL AND n.sourceMdPath = $md)
            RETURN n.document_id AS document_id, n.document_name AS document_name,
                   n.file_type AS file_type, count(n) AS nodes
            ORDER BY nodes DESC
            LIMIT 10
            """,
            sha=content_sha256, md=md,
        )
        return [dict(r) for r in rows]


def find_case_law_ids_for_file(content_sha256: str, md_path: Path | None = None) -> list[str]:
    """Ids of the judgment's own (non-stub - stubs have no node_id) CaseLaw
    node(s) previously ingested from these exact bytes; same matching rules
    as find_ingested_copies()."""
    from framework.core.infra import neo4j_client
    md = str(Path(md_path).resolve()) if md_path is not None else None
    with neo4j_client.get_session() as session:
        rows = session.run(
            f"""
            MATCH (n:CaseLaw)
            WHERE n.node_id IS NOT NULL
              AND (n.{SOURCE_FILE_HASH_PROP} = $sha OR ($md IS NOT NULL AND n.sourceMdPath = $md))
            RETURN DISTINCT n.id AS id
            ORDER BY id
            """,
            sha=content_sha256, md=md,
        )
        return [r["id"] for r in rows if r["id"]]


def _is_stub(entity: dict) -> bool:
    """2026-10-14 (phase0/review-integrity): keyed on provenance first. Was
    extraction_notes-only, which missed every stub that doesn't use that
    wording - chunking._ensure_amending_act()'s AmendingAct stubs
    (provenance=cited_reference, extraction_notes="") were stamped with the
    citing Act PDF's source_file_sha256, i.e. recorded as produced by a file
    that only cited them. A present-but-unranked/lower provenance counts as a
    stub (provenance.rank() treats unknown values as untrusted). The legacy
    notes check stays as a fallback for graph JSON written before provenance
    existed. A MISSING provenance is NOT a stub (load-time default is
    ingested_document)."""
    props = entity.get("properties") or {}
    prov = props.get("provenance")
    if prov and provenance.rank(prov) < provenance.rank(provenance.INGESTED_DOCUMENT):
        return True
    return str(props.get("extraction_notes") or "").startswith(_STUB_NOTE_PREFIX)


def stamp_source_file_hash(graph: dict, content_sha256: str) -> int:
    """Stamps source_file_sha256 on every entity this file actually produced.
    Precedent/reference STUBS (another case/circular merely cited here) are
    skipped: stamping them would mark a DIFFERENT document's node as coming
    from this file, and base-mode `SET n += props` would overwrite the real
    document's own hash when the stub MERGEs onto it. Returns the count."""
    n = 0
    for ent in graph.get("entities", []):
        if _is_stub(ent):
            continue
        ent.setdefault("properties", {})[SOURCE_FILE_HASH_PROP] = content_sha256
        n += 1
    return n


def main_case_law_id(graph: dict) -> str | None:
    """The judgment's own CaseLaw id in an extract_case_law.py graph (the
    only CaseLaw entity with a node_id - precedent stubs get none)."""
    mains = [e["id"] for e in graph.get("entities", [])
             if e.get("type") == "CaseLaw" and (e.get("properties") or {}).get("node_id")
             and not _is_stub(e)]  # 2026-10-14 (phase0/review-integrity): opinion stubs DO carry node_id
    return mains[0] if len(mains) == 1 else None


def converge_case_law_ids(graph: dict, chunks: list[dict], new_case_id: str, target_case_id: str) -> int:
    """Rewrites a fresh extraction's ids from `new_case_id` (slug of THIS
    run's LLM-worded citation) to `target_case_id` (the CaseLaw node already
    in the graph for the same file), so a forced re-ingest MERGEs onto the
    existing node and its Issue/Conclusion/Argument/LegalEvent/Chunk ids
    instead of minting a parallel set. Only extract_case_law.py's exact id
    shapes are touched: the case id itself, `..._to_<case_id>` children
    (Issue/Argument, and Conclusion via its Issue), `LegalEvent_<eventType>_
    <case_id>_<subject>` and `caselaw_<case_id>_<n>` chunk ids - a precedent
    stub that merely shares a prefix is never rewritten. Mutates in place;
    returns the number of id strings rewritten."""
    if not new_case_id or not target_case_id or new_case_id == target_case_id:
        return 0
    chunk_re = re.compile(rf"^caselaw_{re.escape(new_case_id)}_(\d+)$")
    event_types = {(e.get("properties") or {}).get("eventType") for e in graph.get("entities", [])
                   if e.get("type") == "LegalEvent"} - {None, ""}
    changed = 0

    def remap(x):
        nonlocal changed
        if not isinstance(x, str):
            return x
        out = x
        m = chunk_re.match(x)
        if x == new_case_id:
            out = target_case_id
        elif x.endswith(f"_to_{new_case_id}"):
            out = x[: -len(new_case_id)] + target_case_id
        elif m:
            out = f"caselaw_{target_case_id}_{m.group(1)}"
        else:
            for et in event_types:
                prefix = f"LegalEvent_{et}_{new_case_id}_"
                if x.startswith(prefix):
                    out = f"LegalEvent_{et}_{target_case_id}_" + x[len(prefix):]
                    break
        if out != x:
            changed += 1
        return out

    for ent in graph.get("entities", []):
        ent["id"] = remap(ent["id"])
        props = ent.get("properties") or {}
        if "node_id" in props:
            props["node_id"] = remap(props["node_id"])
        if props.get("source_chunk_ids"):
            props["source_chunk_ids"] = ",".join(remap(c) for c in str(props["source_chunk_ids"]).split(","))
    for rel in graph.get("relationships", []):
        rel["source"] = remap(rel["source"])
        rel["target"] = remap(rel["target"])
    for c in chunks or []:
        c["chunk_id"] = remap(c.get("chunk_id"))
        c["entity_id"] = remap(c.get("entity_id"))
    return changed


def duplicate_message(copies: list[dict], content_sha256: str) -> str:
    docs = "; ".join(f"document_id={c.get('document_id')!r} ({c.get('document_name') or '-'}, "
                     f"{c.get('file_type') or '-'}, {c.get('nodes')} node(s))" for c in copies)
    return (f"SKIPPED - this exact file (sha256 {content_sha256[:12]}...) is already in the graph: {docs}. "
            f"Nothing was converted, extracted or loaded. To re-ingest it anyway, tick "
            f"'Re-ingest even if already ingested' (API: \"force_reingest\": true) - a re-ingested "
            f"judgment converges on its existing CaseLaw node.")


def prepare_duplicate_check(upload_path: Path, md_path: Path, force_reingest: bool, log) -> tuple[str, list[dict] | None]:
    """Returns (content_sha256, copies). copies is None when the job should
    PROCEED, a non-empty list when it must be skipped as a duplicate. Logs
    every decision via `log(line)`. A Neo4j failure here never blocks
    ingestion (logged; the job proceeds - its load stage needs Neo4j anyway
    and will surface the real error)."""
    content_sha256 = sha256_of_file(upload_path)
    try:
        copies = find_ingested_copies(content_sha256, md_path)
    except Exception as e:
        log(f"WARNING: duplicate check could not query Neo4j ({e}) - proceeding without it.")
        return content_sha256, None
    if copies and not force_reingest:
        log(duplicate_message(copies, content_sha256))
        return content_sha256, copies
    if copies:
        log(f"Re-ingesting anyway (force_reingest) - this exact file is already in the graph as "
            f"{', '.join(repr(c.get('document_id')) for c in copies)}.")
    return content_sha256, None


def finalize_graph_for_load(graph_path: Path, chunks_path: Path | None, file_type: str, content_sha256: str,
                            md_path: Path, log) -> None:
    """Post-extraction, pre-load: (case law) converges on an existing CaseLaw
    id for the same file, then stamps source_file_sha256. Rewrites the
    graph/chunks JSON in place."""
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    chunks = (json.loads(Path(chunks_path).read_text(encoding="utf-8"))
              if chunks_path and Path(chunks_path).exists() else None)
    if file_type == "caselaw_pdf":
        new_id = main_case_law_id(graph)
        try:
            existing = find_case_law_ids_for_file(content_sha256, md_path)
        except Exception as e:
            existing = []
            log(f"WARNING: could not look up an existing CaseLaw node for this file ({e}).")
        if new_id and len(existing) == 1 and existing[0] != new_id:
            n = converge_case_law_ids(graph, chunks or [], new_id, existing[0])
            log(f"Same file as existing CaseLaw {existing[0]!r} - re-pointed this run's {new_id!r} and its "
                f"derived ids ({n} id string(s)) onto it, so the re-ingest updates that node instead of "
                f"duplicating it.")
        elif len(existing) > 1:
            # Leave ids alone rather than guess which copy is canonical.
            log(f"WARNING: {len(existing)} CaseLaw nodes already exist for this same file ({existing}) - not "
                f"converging; run `python -m framework.cli.merge_duplicate_caselaw --domain tax` (dry run "
                f"first) to clean up.")
    stamped = stamp_source_file_hash(graph, content_sha256)
    Path(graph_path).write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    if chunks is not None:
        Path(chunks_path).write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Stamped {SOURCE_FILE_HASH_PROP}={content_sha256[:12]}... on {stamped} entity(ies).")
