"""
Gold-set loader + validator.

2026-10-14: NEW. Gold items live in eval/gold/*.jsonl - ONE JSON object per
line. JSONL was chosen over YAML because (a) the repo has no YAML dependency
(PyYAML is not in requirements.txt) and the harness must stay stdlib-only,
(b) one item per line gives clean git diffs / blame per question when tax
professionals sign items off, and (c) appending a new item can never
re-indent or break its neighbours. Blank lines and lines starting with '//'
are ignored so a file can carry a short header comment.

The schema (eval/gold_item.schema.json) is the single source of truth; this
module implements the small subset of JSON Schema that file uses (type,
required, properties, additionalProperties, enum, pattern, minLength,
minItems, items, $ref into $defs, format=regex) rather than adding a
jsonschema dependency. Cross-item checks the schema can't express (unique
ids across files, verified=true must name a real reviewer) are done here too.

Every problem in every file is collected and reported together, as
"<file>:<line> [<id>] <path>: <message>", so fixing a gold file is one pass,
not one error per run.

CLI:  python -m eval.goldset eval/gold/      (validate + print counts)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "gold_item.schema.json"

# The placeholder reviewer every machine-drafted item carries. verified=true
# with this reviewer is rejected - a draft can't sign itself off.
DRAFT_REVIEWER = "claude-draft"


class GoldValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"{len(errors)} gold-set validation error(s):\n  " + "\n  ".join(errors))


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


_JSON_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "integer": int, "number": (int, float),
}


def _resolve(schema: dict, root: dict) -> dict:
    ref = schema.get("$ref")
    if not ref:
        return schema
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported $ref {ref!r}")
    node = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def validate_against_schema(value, schema: dict, root: dict | None = None, path: str = "$") -> list[str]:
    """Returns a list of "<json path>: <message>" errors (empty = valid)."""
    root = root if root is not None else schema
    schema = _resolve(schema, root)
    errors: list[str] = []

    expected = schema.get("type")
    if expected:
        py_type = _JSON_TYPES[expected]
        # bool is a subclass of int in Python - don't let True pass as an integer/number.
        if not isinstance(value, py_type) or (expected in ("integer", "number") and isinstance(value, bool)):
            return [f"{path}: expected {expected}, got {type(value).__name__}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")

    if isinstance(value, str):
        if len(value.strip()) < schema.get("minLength", 0):
            errors.append(f"{path}: must be at least {schema['minLength']} non-blank characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: {value!r} does not match pattern {schema['pattern']}")
        if schema.get("format") == "regex":
            try:
                re.compile(value)
            except re.error as e:
                errors.append(f"{path}: invalid regular expression {value!r} ({e})")

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: needs at least {schema['minItems']} item(s)")
        if "items" in schema:
            for i, item in enumerate(value):
                errors.extend(validate_against_schema(item, schema["items"], root, f"{path}[{i}]"))

    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required field '{key}'")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}: unknown field '{key}' (allowed: {', '.join(props)})")
        for key, sub in props.items():
            if key in value:
                errors.extend(validate_against_schema(value[key], sub, root, f"{path}.{key}"))
    return errors


def _gold_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.glob("*.jsonl") if p.is_file())
    raise FileNotFoundError(f"gold path not found: {path}")


def load_gold(path: str | Path, *, ids: list[str] | None = None,
              categories: list[str] | None = None) -> list[dict]:
    """Loads + validates every gold item under `path` (a .jsonl file or a
    directory of them). Raises GoldValidationError listing ALL problems.
    `ids`/`categories` optionally filter the (already fully validated) set -
    filtering happens AFTER validation so a typo elsewhere in the gold set is
    never silently hidden by a filter."""
    path = Path(path)
    schema = load_schema()
    files = _gold_files(path)
    if not files:
        raise GoldValidationError([f"{path}: no *.jsonl gold files found"])

    errors: list[str] = []
    items: list[dict] = []
    seen: dict[str, str] = {}
    for f in files:
        for lineno, raw in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            where = f"{f.name}:{lineno}"
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{where}: not valid JSON ({e.msg} at column {e.colno})")
                continue
            item_id = item.get("id", "?") if isinstance(item, dict) else "?"
            item_errors = validate_against_schema(item, schema)
            if isinstance(item, dict):
                if item_id in seen:
                    item_errors.append(f"$.id: duplicate id (first defined at {seen[item_id]})")
                else:
                    seen[item_id] = where
                if item.get("verified") is True and item.get("verified_by") == DRAFT_REVIEWER:
                    item_errors.append(f"$.verified: an item drafted by '{DRAFT_REVIEWER}' cannot be "
                                       f"verified=true - set verified_by to the signing tax professional")
                auth_ids = [a.get("id") for a in (item.get("required_authorities") or []) +
                            (item.get("optional_authorities") or []) if isinstance(a, dict)]
                dupes = [a for a, n in Counter(auth_ids).items() if n > 1]
                if dupes:
                    item_errors.append(f"$.required_authorities: duplicate authority id(s) {dupes}")
            errors.extend(f"{where} [{item_id}] {e}" for e in item_errors)
            if not item_errors:
                item = dict(item)
                item["_source_file"] = f.name
                items.append(item)
    if errors:
        raise GoldValidationError(errors)

    if ids:
        wanted = set(ids)
        unknown = wanted - {i["id"] for i in items}
        if unknown:
            raise GoldValidationError([f"--ids: unknown gold id(s) {sorted(unknown)}"])
        items = [i for i in items if i["id"] in wanted]
    if categories:
        items = [i for i in items if i["category"] in set(categories)]
    return items


def gold_fingerprint(items: list[dict]) -> str:
    """Stable hash of the gold content actually used in a run (ignoring
    load-time bookkeeping keys) - compare.py uses it to warn when two runs
    were scored against different gold sets."""
    canon = [
        {k: v for k, v in sorted(i.items()) if not k.startswith("_")}
        for i in sorted(items, key=lambda x: x["id"])
    ]
    return hashlib.sha256(json.dumps(canon, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.goldset", description="Validate gold-set files.")
    ap.add_argument("path", nargs="?", default=str(Path(__file__).resolve().parent / "gold"))
    args = ap.parse_args(argv)
    try:
        items = load_gold(args.path)
    except GoldValidationError as e:
        print(str(e), file=sys.stderr)
        return 1
    by_cat = Counter(i["category"] for i in items)
    verified = sum(1 for i in items if i["verified"])
    print(f"OK: {len(items)} gold items ({verified} verified, {len(items) - verified} unverified drafts)")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:<14} {n}")
    print(f"fingerprint: {gold_fingerprint(items)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
