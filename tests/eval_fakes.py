"""
2026-10-14: NEW - offline stand-ins for the eval harness tests: canned
/api/ask responses (shaped exactly like framework/webapp/app.py::_api_ask_impl's
response) and a tiny threaded Flask app serving them on a free localhost port,
so eval.run is exercised over real HTTP (timeouts, retries, malformed bodies)
with no Neo4j / LLM / .env.
"""
import json
import threading
import time

from flask import Flask, jsonify, request
from werkzeug.serving import make_server

EA_DOC = "Engineering Analysis Centre of Excellence v CIT (2021) 432 ITR 471.txt"
CIRC_DOC = "CBDT Order Section 119 TDS certificates March 2026.pdf"
DISCLAIMER = "\n\nDISCLAIMER\nThis is general information, not advice. Royalty is taxable as fees for included services."

GOOD_RESEARCH_ANSWER = f"""## Analysis
## a. Executive Summary - Core Position of Taxability
* **Holding:** Payments for software licences are not royalty and no TDS under Section 195 is required. {EA_DOC} - Page 8
* **Litigation risk assessment:** Low - binding Supreme Court authority.

## b. Detailed Analysis
### i. Applicable Law & Framework
Section 9(1)(vi) and Article 12 of the DTAA; the DTAA governs where more beneficial under s.90(2). {EA_DOC} - Page 8
### ii. Jurisprudence & Case Law Review
Engineering Analysis Centre of Excellence v. CIT held the amounts are not royalty. {EA_DOC} - Page 9

## c. Conclusion
The payments do not constitute royalty; there is no obligation to deduct tax under section 195.

## d. Referred Sources
* **Acts & Rules** Income-tax Act, 1961
* **DTAA** Article 12
* **Case Law** {EA_DOC} - Page 8
* **Circulars / Instructions** None cited in the provided context.
* **FEMA / RBI materials** None used.
* **Web Research sources** None used.
"""

FABRICATED_ANSWER = GOOD_RESEARCH_ANSWER.replace(
    "## c. Conclusion",
    "See also Imaginary Software Pvt Ltd v. DCIT (2019) 999 ITR 1.txt - Page 4 and chunk_id=999.\n\n## c. Conclusion")

MALFORMED_FORMAT_ANSWER = ("The payments are not royalty under Section 195. " + EA_DOC + " - Page 8\n"
                           "## c. Conclusion\nNo TDS.\n")


def source_chunks(doc=EA_DOC, pages=(8, 9)):
    return [{"chunk_id": f"c{p}", "via": "vector", "preview": "payments for computer software are not royalty; section 195",
             "rerank_score": 0.9,
             "document": {"document_name": doc, "page": p, "chunk_citation": None, "entity_citation": None},
             "parent_context": {}} for p in pages]


def ask_response(answer, *, answer_mode="research", retrieval_mode="hybrid", chunks=None, quality=0.8,
                 from_cache=False, citation_ok=True):
    chunks = source_chunks() if chunks is None else chunks
    return {
        "question": "q", "intent": "SEMANTIC", "cypher": "", "answer": answer + DISCLAIMER,
        "nodes": [], "edges": [], "source_chunks": chunks,
        "grounding": {
            "source_references": chunks, "confidence": 0.8,
            "citation_check": {"citations_found": [EA_DOC], "citation_ok": citation_ok},
            "referred_sources_check": {"checked": True, "unverified_sources": []},
            "answer_quality": {"available": True, "overall": quality, "groundedness": quality,
                               "completeness": quality, "specificity": quality, "statutory_fidelity": quality,
                               "holding_discipline": quality, "authority_hierarchy": quality, "gaps": []},
        },
        "retrieval_mode": retrieval_mode, "answer_mode": answer_mode,
        "chunks_used": [{"chunk_id": c["chunk_id"], "document_name": c["document"]["document_name"],
                         "page": c["document"]["page"], "rerank_score": 0.9,
                         "text": "The Supreme Court in Engineering Analysis Centre of Excellence held that payments "
                                 "are not royalty under section 9(1)(vi) and no TDS under section 195."}
                        for c in chunks],
        "auto_evaluation": None, "general_knowledge_evaluation": None, "from_cache": from_cache,
    }


def evaluation_response(overall=8.0):
    return {"evaluation": {"available": True, "overall": overall, "verdict": "ok", "strengths": [], "issues": [],
                           "dimensions": [{"name": "Groundedness / accuracy", "score": overall, "assessment": ""}]},
            "general_knowledge_evaluation": {"available": True, "overall": overall - 1, "dimensions": []}}


class FakeTarget:
    """behaviors: {question_substring: callable(count) -> ("json", status, obj) | ("raw", status, text)
    | ("sleep", seconds, obj)}; unmatched questions get a good answer."""

    def __init__(self, behaviors=None, judge_overall=8.0, quality=0.8):
        self.behaviors = behaviors or {}
        self.calls = {}
        self.judge_calls = 0
        self.judge_overall = judge_overall
        self.quality = quality
        self.app = Flask("fake_target")
        self._lock = threading.Lock()

        @self.app.route("/api/settings/summary")
        def settings():
            return jsonify({"temperature": {"value": 0.0, "provider": "FAKE"},
                            "embedding_model": {"name": "fake-embedder"}, "neo4j": {"connected": False}})

        @self.app.route("/api/ask", methods=["POST"])
        @self.app.route("/api/ask/fast", methods=["POST"])
        def ask():
            body = request.json or {}
            q = body.get("question", "")
            with self._lock:
                self.calls[q] = self.calls.get(q, 0) + 1
                n = self.calls[q]
            for key, fn in self.behaviors.items():
                if key in q:
                    kind, a, b = fn(n)
                    if kind == "json":
                        return jsonify(b), a
                    if kind == "raw":
                        return b, a, {"Content-Type": "text/html"}
                    if kind == "sleep":
                        time.sleep(a)
                        return jsonify(b), 200
            return jsonify(ask_response(GOOD_RESEARCH_ANSWER, answer_mode=body.get("answer_mode", "research"),
                                        quality=self.quality))

        @self.app.route("/api/ask/evaluate", methods=["POST"])
        def evaluate():
            with self._lock:
                self.judge_calls += 1
            return jsonify(evaluation_response(self.judge_overall))

    def __enter__(self):
        self.server = make_server("127.0.0.1", 0, self.app, threaded=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.thread.join(timeout=5)


def write_gold(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")
    return path


EA_AUTH = {"id": "engineering-analysis", "kind": "case", "label": "Engineering Analysis",
           "aliases": ["Engineering Analysis Centre of Excellence", "Engineering Analysis"]}
S195_AUTH = {"id": "s195", "kind": "section", "label": "Section 195", "aliases": ["Section 195"], "section": "195"}


def gold_item(id_, question, *, category="research", answer_mode="research", required=None, mnc=None, kps=None):
    return {
        "id": id_, "question": question, "category": category, "answer_mode": answer_mode,
        "required_authorities": required if required is not None else [EA_AUTH, S195_AUTH],
        "expected_key_points": kps if kps is not None else [
            {"point": "Not royalty", "source": "test fixture", "match_any": [r"not\s+royalty"]}],
        "must_not_claim": mnc if mnc is not None else [
            {"claim": "It is royalty", "patterns": [r"(?:constitutes?|is|are)\s+royalt(?:y|ies)\b"]}],
        "verified_by": "claude-draft", "verified": False, "notes": "test item",
    }
