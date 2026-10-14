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

# FIXED 2026-10-14 (phase0/review-integrity): was xfail - "FirmOpinion has no resolution key and is not
# a structural label, so the id-fallback key 'firmopinion firmopinion_op_02' scores 96.6 vs '..._op_01'
# and the second opinion is auto-merged onto the first". FirmOpinion is now an exact-id label
# (config.STRUCTURAL_LABELS) and id-fallback keys can no longer auto-merge (graph_loader).
def test_second_firm_opinion_is_not_fuzzy_merged_onto_first(graph, engine):
    def opinion(op_id, title):
        return {"id": f"FirmOpinion_{op_id}", "type": "FirmOpinion", "properties": {
            "opinion_id": op_id.replace("_", "-"), "title": title, "provenance": provenance.INGESTED_DOCUMENT}}

    engine.load_entities(graph, [opinion("OP_01", "Withholding on FTS")], mode="base")
    _labels, resolved = engine.load_entities(graph, [opinion("OP_02", "Capital gains on demerger")], mode="base")
    assert resolved == {}
    assert "FirmOpinion_OP_02" in graph.nodes
    assert graph.props("FirmOpinion_OP_01")["title"] == "Withholding on FTS"


# FIXED 2026-10-14 (phase0/review-integrity): was xfail - "an opinion's cited_reference stub
# CBDTCircular_8_of_2026 fuzzy-resolves (97.1) onto the REAL CBDTCircular_3_of_2026 and resolved_id_map
# re-points the opinion's RELIES_ON_AUTHORITY edge at the WRONG circular; same for Article_12/13 and
# Rule_10/11". Circulars now key on a canonical circular_number, every fuzzy path vetoes differing number
# tokens (entity_resolution.key_guard), and Article/Rule are exact-id. See tests/test_resolution_identity.py.
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
    # 2026-10-14 (phase0/review-integrity): Cypher `=~` is a FULL-string regex match.
    py = re.sub(r"(str\(props\.get\('\w+'\) or ''\)) =~ " + operand, r"bool(re.fullmatch(\2, \1))", py)
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


# FIXED 2026-10-14 (phase0/review-integrity): was xfail - "the Chunk/LegalEvent predicates are
# unanchored prefix/substring matches, so a duplicate id that is a PREFIX of another case's id DETACH
# DELETEs the other case's chunks and LegalEvents". Id-shaped nodes are now matched by end-anchored ids /
# a full-match chunk-id regex (query 1, evaluated below); LegalEvents by their EVENT_INSTRUMENT edge to the
# duplicate (query 2, never by id text). _cypher_where_to_python() gained `=~` (Cypher full match) and
# $dup_re to evaluate the new chunk predicate faithfully.
def test_drop_duplicate_subtree_never_matches_another_cases_ids():
    from framework.cli import merge_duplicate_caselaw as merge
    s = _CaptureSession()
    merge._drop_duplicate_subtree(s, "CaseLaw_Samsung_Electronics_Co_Ltd", dry_run=True)
    where = re.search(r"WHERE (.*) RETURN count", s.queries[0][0], re.S).group(1)
    expr = _cypher_where_to_python(where)
    dup = "CaseLaw_Samsung_Electronics_Co_Ltd"
    assert s.queries[0][1]["dup_re"] == re.escape(dup)

    def matches(labels, props):
        return eval(expr, {}, {"labels": set(labels), "props": props, "dup": dup, "str": str,
                               "dup_re": re.escape(dup), "re": re})

    # sanity: the duplicate's own subtree IS matched (translation is faithful)
    assert matches({"Chunk"}, {"chunk_id": f"caselaw_{dup}_3"})
    assert matches({"Issue"}, {"id": f"Issue_1_to_{dup}"})
    other = "CaseLaw_Samsung_Electronics_Co_Ltd_v_ITO_2010"
    assert not matches({"Chunk"}, {"chunk_id": f"caselaw_{other}_3"}), "deletes the canonical/other case's chunk"
    assert not matches({"LegalEvent"}, {"id": f"LegalEvent_overruled_{other}_CaseLaw_X"})
    # 2026-10-14: the dup's OWN chunks of any number still match; a chunk of a case whose id is the dup
    # plus a numeric-looking suffix does not.
    assert matches({"Chunk"}, {"chunk_id": f"caselaw_{dup}_12"})
    assert not matches({"Chunk"}, {"chunk_id": f"caselaw_{dup}_2010_3"})
    assert not matches({"Issue"}, {"id": f"Issue_1_to_{other}"})
    # LegalEvents: a second, relationship-anchored query - never id-text matching.
    events_q = s.queries[1][0]
    assert "LegalEvent" in events_q and "EVENT_INSTRUMENT" in events_q and "{id: $dup}" in events_q
    assert "CONTAINS" not in events_q and "STARTS WITH" not in events_q


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


# FIXED 2026-10-14 (phase0/review-integrity): was xfail - "_pick_canonical() ignores provenance: a
# heavily-cited cited_reference STUB out-degrees the real judgment, becomes canonical, and the ingested
# judgment node is DETACH DELETEd". Provenance now sorts before degree.
def test_pick_canonical_prefers_ingested_judgment_over_stub():
    from framework.cli import merge_duplicate_caselaw as merge
    stub = {"id": "CaseLaw_Vodafone_International_Holdings_BV_v_Union_of_India", "node_id": None,
            "provenance": provenance.CITED_REFERENCE, "ingestion_date": "2026-01-01"}
    real = {"id": "CaseLaw_Vodafone_International_Holdings_B_V_v_UOI_2012_341_ITR_1_SC", "node_id": "x",
            "provenance": provenance.INGESTED_DOCUMENT, "ingestion_date": "2026-02-01"}
    s = _DegreeSession({stub["id"]: 25, real["id"]: 9})
    assert merge._pick_canonical(s, [stub, real])["id"] == real["id"]


# --- 3. PII: opinion pseudonymisation + plain-row masking --------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "BUG extract_opinion.party_variants()/redact(): a single-word client name ('Infosys') fails "
    "looks_like_party_name() (no corporate suffix, < 2 capitalised words) so it is never pseudonymised, "
    "and redact() is case-sensitive so 'INFOSYS'/'ZOMATO LIMITED' survive in chunk text - which is shown "
    "to the answer LLM and to users researching other clients' matters."))
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

# FIXED 2026-10-14 (phase0/review-integrity): was xfail - "ingest_dedup._is_stub() keys on extraction_notes,
# not on provenance: AmendingAct stubs (provenance=cited_reference, extraction_notes='') are stamped with the
# citing Act PDF's source_file_sha256". _is_stub() now keys on provenance, notes kept as legacy fallback.
def test_stamp_source_hash_skips_every_cited_reference_stub():
    from domain_packs.tax import ingest_dedup
    graph = {"entities": [
        {"id": "AmendingAct_15_2024", "type": "AmendingAct",
         "properties": {"provenance": "cited_reference", "sourceMdPath": "", "extraction_notes": ""}},
    ]}
    ingest_dedup.stamp_source_file_hash(graph, "f" * 64)
    assert ingest_dedup.SOURCE_FILE_HASH_PROP not in graph["entities"][0]["properties"]
