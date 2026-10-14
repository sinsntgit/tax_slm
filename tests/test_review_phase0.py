"""
2026-10-14: review/phase0 - independent post-merge review of the five Phase 0
branches. Every test here DEMONSTRATES A BUG found in the merged main and is
marked xfail(strict=True) so the suite stays green until it is fixed; when a
fix lands the test XPASSes, strict mode fails the run, and the marker must be
removed (turning it into a regression test). Each reason names the file/line.

No product code is changed on this branch.
"""
import re
import sys
from pathlib import Path

import pytest

from domain_packs.tax.config import DOMAIN_CONFIG
from framework.core.ingest import provenance
from framework.core.ingest.graph_loader import GraphIngestionEngine

CONVERTERS = Path(__file__).resolve().parents[1] / "domain_packs" / "tax" / "converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

# Reuse the opinions branch's interpreting fake (it already understands the
# loader's Cypher, including `MATCH (n:Label) RETURN n` used by fuzzy matching).
from test_provenance_loader import FakeGraph  # noqa: E402


@pytest.fixture
def graph():
    return FakeGraph()


@pytest.fixture
def engine():
    return GraphIngestionEngine(DOMAIN_CONFIG)


# --- 1. cross-run fuzzy resolution merges DIFFERENT opinions / authorities ------------

@pytest.mark.xfail(strict=True, reason=(
    "BUG graph_loader.load_entities() + entity_resolution: FirmOpinion has no resolution key "
    "(config.RESOLUTION_KEY_PROPERTIES) and is not a structural label, so the id-fallback key "
    "'firmopinion firmopinion_op_02' scores 96.6 vs '..._op_01' (>= HIGH_SIMILARITY 95) and the second "
    "opinion ingested is auto-merged onto the FIRST opinion's node, overwriting it (base mode SET n +=)."))
def test_second_firm_opinion_is_not_fuzzy_merged_onto_first(graph, engine):
    def opinion(op_id, title):
        return {"id": f"FirmOpinion_{op_id}", "type": "FirmOpinion", "properties": {
            "opinion_id": op_id.replace("_", "-"), "title": title, "provenance": provenance.INGESTED_DOCUMENT}}

    engine.load_entities(graph, [opinion("OP_01", "Withholding on FTS")], mode="base")
    _labels, resolved = engine.load_entities(graph, [opinion("OP_02", "Capital gains on demerger")], mode="base")
    assert resolved == {}
    assert "FirmOpinion_OP_02" in graph.nodes
    assert graph.props("FirmOpinion_OP_01")["title"] == "Withholding on FTS"


@pytest.mark.xfail(strict=True, reason=(
    "BUG graph_loader.find_cross_graph_match() on opinion stubs: an opinion's cited_reference stub "
    "CBDTCircular_8_of_2026 (circular 8 not ingested) fuzzy-resolves (97.1) onto the REAL "
    "CBDTCircular_3_of_2026, the stub is skipped as lower-provenance and resolved_id_map re-points the "
    "opinion's RELIES_ON_AUTHORITY edge (provenance ingested_document) at the WRONG circular. Same for "
    "Article_12 vs Article_13 and ITR1962__Rule_10 vs Rule_11."))
def test_opinion_stub_is_not_rewired_to_a_different_authority(graph, engine):
    engine.load_entities(graph, [{"id": "CBDTCircular_3_of_2026", "type": "CBDTCircular",
                                  "properties": {"circular_number": "3 of 2026", "hasText": "real circular 3"}}],
                         mode="base")
    stub = {"id": "CBDTCircular_8_of_2026", "type": "CBDTCircular", "properties": provenance.mark_stub(
        {"circular_number": "8 of 2026", "full_citation": "CBDT Circular No. 8/2026"}, cited_in="opinion_OP_01")}
    _labels, resolved = engine.load_entities(graph, [stub], mode="base")
    assert resolved.get("CBDTCircular_8_of_2026") in (None, "CBDTCircular_8_of_2026")


# --- 2. merge_duplicate_caselaw --drop-duplicate-subtree deletes OTHER cases' data ------

def _cypher_where_to_python(where: str) -> str:
    """Translates the exact predicate shapes _drop_duplicate_subtree() uses into
    a Python expression over (labels, props, dup), so the test evaluates the
    script's REAL WHERE clause rather than a re-implementation of it."""
    py = where.replace("$dup", "dup")
    py = re.sub(r"\bn:(\w+)", r"('\1' in labels)", py)
    py = re.sub(r"\bn\.(\w+)", r"str(props.get('\1') or '')", py)
    operand = r"(\([^()]*\)|'[^']*')"
    py = re.sub(r"(str\(props\.get\('\w+'\) or ''\)) ENDS WITH " + operand, r"\1.endswith(\2)", py)
    py = re.sub(r"(str\(props\.get\('\w+'\) or ''\)) STARTS WITH " + operand, r"\1.startswith(\2)", py)
    py = re.sub(r"(str\(props\.get\('\w+'\) or ''\)) CONTAINS " + operand, r"(\2 in \1)", py)
    return py.replace(" OR ", " or ").replace(" AND ", " and ")


class _CaptureSession:
    def __init__(self):
        self.queries = []

    def run(self, q, **p):
        self.queries.append((q, p))

        class _R:
            def single(self_inner):
                return {"c": 0}
        return _R()


@pytest.mark.xfail(strict=True, reason=(
    "BUG framework/cli/merge_duplicate_caselaw.py:_drop_duplicate_subtree() - the Chunk/LegalEvent "
    "predicates are unanchored prefix/substring matches ('caselaw_' + dup + '_' / CONTAINS '_' + dup + '_'). "
    "Case ids are 60-char-truncated citation slugs, so a duplicate id that is a PREFIX of another case's "
    "id (CaseLaw_Samsung_Electronics_Co_Ltd vs CaseLaw_Samsung_Electronics_Co_Ltd_v_ITO_2010 - typical of "
    "the same-file re-ingests this flag targets) DETACH DELETEs the other case's chunks and LegalEvents."))
def test_drop_duplicate_subtree_never_matches_another_cases_ids():
    from framework.cli import merge_duplicate_caselaw as merge
    s = _CaptureSession()
    merge._drop_duplicate_subtree(s, "CaseLaw_Samsung_Electronics_Co_Ltd", dry_run=True)
    where = re.search(r"WHERE (.*) RETURN count", s.queries[0][0], re.S).group(1)
    expr = _cypher_where_to_python(where)
    dup = "CaseLaw_Samsung_Electronics_Co_Ltd"

    def matches(labels, props):
        return eval(expr, {}, {"labels": set(labels), "props": props, "dup": dup, "str": str})

    # sanity: the duplicate's own subtree IS matched (translation is faithful)
    assert matches({"Chunk"}, {"chunk_id": f"caselaw_{dup}_3"})
    assert matches({"Issue"}, {"id": f"Issue_1_to_{dup}"})
    other = "CaseLaw_Samsung_Electronics_Co_Ltd_v_ITO_2010"
    assert not matches({"Chunk"}, {"chunk_id": f"caselaw_{other}_3"}), "deletes the canonical/other case's chunk"
    assert not matches({"LegalEvent"}, {"id": f"LegalEvent_overruled_{other}_CaseLaw_X"})


class _DegreeSession:
    def __init__(self, degrees):
        self.degrees = degrees

    def run(self, q, **p):
        class _R:
            def __init__(self, d):
                self.d = d

            def single(self):
                return {"deg": self.d}
        return _R(self.degrees[p["id"]])


@pytest.mark.xfail(strict=True, reason=(
    "BUG merge_duplicate_caselaw._pick_canonical() ignores provenance (opinions branch): a heavily-cited "
    "cited_reference precedent STUB out-degrees the real judgment, becomes canonical, keeps "
    "provenance=cited_reference/confidence=0.3/'Stub created from...' (fill never overwrites), and the "
    "ingested judgment node is DETACH DELETEd - the verified judgment is relabelled as unverified."))
def test_pick_canonical_prefers_ingested_judgment_over_stub():
    from framework.cli import merge_duplicate_caselaw as merge
    stub = {"id": "CaseLaw_Vodafone_International_Holdings_BV_v_Union_of_India", "node_id": None,
            "provenance": provenance.CITED_REFERENCE, "ingestion_date": "2026-01-01"}
    real = {"id": "CaseLaw_Vodafone_International_Holdings_B_V_v_UOI_2012_341_ITR_1_SC", "node_id": "x",
            "provenance": provenance.INGESTED_DOCUMENT, "ingestion_date": "2026-02-01"}
    s = _DegreeSession({stub["id"]: 25, real["id"]: 9})
    assert merge._pick_canonical(s, [stub, real])["id"] == real["id"]


# --- 3. PII: opinion pseudonymisation + plain-row masking --------------------------------

# 2026-10-14: fixed on phase0/review-pii (looks_like_party_name accepts a
# single capitalised word; redact() is case-insensitive) - now a regression test.
def test_opinion_redaction_catches_single_word_and_uppercase_client():
    import extract_opinion
    red = extract_opinion.build_redactions("Infosys", "", "")
    assert "Infosys" not in extract_opinion.redact("Infosys paid fees to the non-resident.", red)
    red2 = extract_opinion.build_redactions("Zomato Limited", "", "")
    assert "ZOMATO" not in extract_opinion.redact("ZOMATO LIMITED (the Company) paid...", red2)


@pytest.mark.xfail(strict=True, reason=(
    "BUG pii.PiiMasker.mask_plain_rows(): ALWAYS_MASKED_FIELDS (config.py now lists FirmOpinion "
    "client_name/counterparty_name, the opinions branch's ONLY protection for real client identity) are "
    "applied only when a row value is a NODE dict. An NL2Cypher row like {'o.client_name': 'Zomato "
    "Limited'} (RETURN o.client_name) is a scalar, gets regex-only masking, and the raw client name reaches "
    "the answer LLM, the LLM call log and the UI."))
def test_plain_row_scalar_client_name_is_masked(monkeypatch):
    from framework.core.retrieve import pii
    monkeypatch.setattr(pii, "_log_pii_access", lambda *a, **k: None)
    masker = pii.PiiMasker(DOMAIN_CONFIG)
    rows = masker.mask_plain_rows([{"o.client_name": "Zomato Limited", "client_name": "Zomato Limited"}])
    assert "Zomato" not in str(rows)


@pytest.mark.xfail(strict=True, reason=(
    "BUG app._run_unstructured_agent() agentic branch: response['plain_rows'] is overwritten AFTER "
    "_api_ask_impl() with orchestrator_result['plain_rows'] straight from GraphRAGRetriever."
    "run_graph_retrieval(), which never masks (retrieve() is the 'single enforcement point', bypassed "
    "here) - FirmOpinion node properties (client_name) go to the browser unmasked."))
def test_agentic_plain_rows_are_pii_masked(monkeypatch):
    from framework.webapp import app as webapp
    from framework.core.retrieve import pii
    from test_agentic_parent_context import FakeRetriever

    monkeypatch.setattr(pii, "_log_pii_access", lambda *a, **k: None)
    monkeypatch.setattr(webapp, "retriever", FakeRetriever())
    monkeypatch.setattr(webapp, "CATEGORY_SPECS", None)
    monkeypatch.setattr(webapp.reranker, "rerank", lambda q, chunks, **kw: list(chunks))
    monkeypatch.setattr(webapp.llm_client, "complete", lambda *a, **k: "ANSWER")
    monkeypatch.setattr(webapp.cache, "get_cached_answer", lambda *a, **k: None)
    monkeypatch.setattr(webapp.cache, "set_cached_answer", lambda *a, **k: None)
    node = {"id": "FirmOpinion_OP_01", "label": "FirmOpinion",
            "properties": {"client_name": "Zomato Limited", "opinion_id": "OP-01"}}
    orchestrator_result = {"chunks": [{"chunk_id": "opinion_FirmOpinion_OP_01_2", "text": "x", "source": "graph"}],
                           "cypher_queries": ["MATCH (o:FirmOpinion) RETURN o"], "plain_rows": [{"o": node}],
                           "iterations": [{}], "final_verdict": "sufficient"}
    monkeypatch.setattr(webapp.retrieval_orchestrator, "run_agentic_retrieval",
                        lambda retriever, question, trace=None: orchestrator_result)
    response, _internal, _trace = webapp._run_unstructured_agent(
        "Which client got the FTS opinion?", None, "agentic", "research", fast_mode=True, skip_rerank=True)
    assert "Zomato" not in str(response["plain_rows"])


# --- 4. dedup: stub detection ignores provenance ------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "BUG ingest_dedup._is_stub() keys on extraction_notes 'Stub created from', not on provenance: "
    "chunking._ensure_amending_act()'s AmendingAct stubs (provenance=cited_reference, extraction_notes='') "
    "are stamped with the citing Act PDF's source_file_sha256, i.e. marked as produced by a file that "
    "only cited them."))
def test_stamp_source_hash_skips_every_cited_reference_stub():
    from domain_packs.tax import ingest_dedup
    graph = {"entities": [
        {"id": "AmendingAct_15_2024", "type": "AmendingAct",
         "properties": {"provenance": "cited_reference", "sourceMdPath": "", "extraction_notes": ""}},
    ]}
    ingest_dedup.stamp_source_file_hash(graph, "f" * 64)
    assert ingest_dedup.SOURCE_FILE_HASH_PROP not in graph["entities"][0]["properties"]
