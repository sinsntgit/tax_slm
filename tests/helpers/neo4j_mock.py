"""
Scripted fake Neo4j driver/session for offline unit tests - 2026-10-14: NEW.

There is no Neo4j server in the cloud/CI environment, so any code path that
opens a session (`neo4j_client.get_session()`, `session.run(...)`) needs a
stand-in. This one is deliberately tiny: you script WHICH records come back
for WHICH query by a Cypher substring match, and every query + its params is
recorded so the test can assert on what was actually sent.

Quick start
-----------
    from tests.helpers.neo4j_mock import FakeNeo4j

    def test_something(monkeypatch):
        neo = FakeNeo4j()
        neo.on("MATCH (n:CaseLaw)", [{"db_id": "4:abc:1", "case_name": "X v. Y"}])
        neo.on("elementId(n) IN $db_ids", lambda query, params: [
            {"db_id": i, "full_citation": f"cite-{i}"} for i in params["db_ids"]
        ])
        neo.install(monkeypatch)        # patches framework.core.infra.neo4j_client

        ...call the code under test...

        assert neo.ran("MATCH (n:CaseLaw)")                 # was it called?
        q = neo.last("MATCH (n:CaseLaw)")                   # the RecordedQuery
        assert q.params["stance"] == "revenue_favorable"

Rules
-----
- `on(substring, records)`: the FIRST rule (in registration order) whose
  substring occurs in the query text wins. `records` may be
    * a list of dicts (returned on every matching call),
    * a callable `(query_text, params) -> list[dict]` (computed per call),
    * an Exception instance/class (raised from `.run()` - to test the
      code's error handling / fallback tier).
  Pass `once=True` to consume the rule after its first match (so a later
  rule or the default applies to subsequent calls - handy for "first call
  returns X, second returns Y").
- Unmatched queries return an empty result by default; construct with
  `FakeNeo4j(strict=True)` to raise `AssertionError` instead (catches a query
  you didn't expect).
- `neo4j.Query(text, timeout=...)` objects are unwrapped to their text, and
  `session.run(q, {"a": 1}, b=2)` merges both param styles, same as the real
  driver.
- `execute_read(fn, *a)` / `execute_write(fn, *a)` / `begin_transaction()`
  call through to the same scripted `.run()`.

Records behave like `neo4j.Record` for the ways this codebase uses them:
`r["key"]`, `r.get("key")`, `dict(r)`, `r.keys()`, `r.values()`, `r.data()`.
Result objects support iteration, `list()`, `.single()`, `.data()`,
`.value()`, `.values()`, `.peek()` and `.consume()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class FakeRecord(dict):
    """dict subclass, so dict(r) / r["k"] / r.get("k") / r.keys() all work."""

    def data(self, *keys):
        return {k: self[k] for k in keys} if keys else dict(self)

    def value(self, key=0, default=None):
        if isinstance(key, int):
            vals = list(self.values())
            return vals[key] if key < len(vals) else default
        return self.get(key, default)


class FakeResult:
    def __init__(self, records: list[dict]):
        self._records = [r if isinstance(r, FakeRecord) else FakeRecord(r) for r in records]

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    def single(self, strict: bool = False):
        if not self._records:
            if strict:
                raise ValueError("No records")
            return None
        return self._records[0]

    def peek(self):
        return self._records[0] if self._records else None

    def data(self, *keys):
        return [r.data(*keys) for r in self._records]

    def value(self, key=0, default=None):
        return [r.value(key, default) for r in self._records]

    def values(self, *keys):
        return [list(r.data(*keys).values()) for r in self._records]

    def consume(self):
        return None


@dataclass
class RecordedQuery:
    text: str
    params: dict
    timeout: Any = None


@dataclass
class _Rule:
    substring: str
    records: Any
    once: bool = False
    used: bool = False


@dataclass
class FakeNeo4j:
    """The scripted backend shared by every FakeDriver/FakeSession it hands out."""
    strict: bool = False
    rules: list = field(default_factory=list)
    queries: list = field(default_factory=list)
    sessions_opened: int = 0

    # --- scripting ------------------------------------------------------
    def on(self, substring: str, records: Any = None, once: bool = False) -> "FakeNeo4j":
        self.rules.append(_Rule(substring, [] if records is None else records, once))
        return self

    # --- assertions -----------------------------------------------------
    def matching(self, substring: str) -> list[RecordedQuery]:
        return [q for q in self.queries if substring in q.text]

    def ran(self, substring: str) -> bool:
        return bool(self.matching(substring))

    def last(self, substring: str = "") -> RecordedQuery:
        hits = self.matching(substring)
        if not hits:
            raise AssertionError(f"No query containing {substring!r} was run. Ran: "
                                 f"{[q.text[:80] for q in self.queries]}")
        return hits[-1]

    # --- execution ------------------------------------------------------
    def _execute(self, query, parameters=None, **kwargs) -> FakeResult:
        text = getattr(query, "text", query)
        timeout = getattr(query, "timeout", None)
        params = dict(parameters or {})
        params.update(kwargs)
        self.queries.append(RecordedQuery(str(text), params, timeout))
        for rule in self.rules:
            if rule.used or rule.substring not in str(text):
                continue
            if rule.once:
                rule.used = True
            payload = rule.records
            if isinstance(payload, type) and issubclass(payload, BaseException):
                raise payload(f"scripted failure for {rule.substring!r}")
            if isinstance(payload, BaseException):
                raise payload
            if callable(payload):
                payload = payload(str(text), params)
            return FakeResult(list(payload or []))
        if self.strict:
            raise AssertionError(f"Unscripted Cypher query: {str(text)[:200]}")
        return FakeResult([])

    # --- driver/session factories -------------------------------------
    def session(self, **_kwargs) -> "FakeSession":
        self.sessions_opened += 1
        return FakeSession(self)

    def driver(self) -> "FakeDriver":
        return FakeDriver(self)

    def install(self, monkeypatch) -> "FakeNeo4j":
        """Patches framework.core.infra.neo4j_client so get_session()/
        get_driver()/verify_connectivity() all hit this fake. Modules that do
        `from framework.core.infra import neo4j_client` and call
        `neo4j_client.get_session()` at CALL time are covered automatically;
        a module that did `from ...neo4j_client import get_session` at import
        time must be patched separately (monkeypatch.setattr(mod, "get_session", neo.session))."""
        from framework.core.infra import neo4j_client
        monkeypatch.setattr(neo4j_client, "get_session", self.session)
        monkeypatch.setattr(neo4j_client, "get_driver", self.driver)
        monkeypatch.setattr(neo4j_client, "verify_connectivity", lambda: None)
        return self


class FakeTransaction:
    def __init__(self, backend: FakeNeo4j):
        self._backend = backend

    def run(self, query, parameters=None, **kwargs):
        return self._backend._execute(query, parameters, **kwargs)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession(FakeTransaction):
    def execute_read(self, fn: Callable, *args, **kwargs):
        return fn(FakeTransaction(self._backend), *args, **kwargs)

    execute_write = execute_read
    read_transaction = execute_read   # neo4j 4.x names, in case older code uses them
    write_transaction = execute_read

    def begin_transaction(self, *a, **k):
        return FakeTransaction(self._backend)


class FakeDriver:
    def __init__(self, backend: FakeNeo4j):
        self._backend = backend

    def session(self, **kwargs):
        return self._backend.session(**kwargs)

    def verify_connectivity(self):
        return None

    def close(self):
        pass
