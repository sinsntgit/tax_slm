"""
Step 1b: Fix a real conversion bug found in income_tax_act_1961.md while
verifying it against the source PDF.

THE BUG: the PDF-to-Markdown converter's heading-pattern detector treats any
line starting "N.(text)" / "N. Text" as a new "Section N" heading. That's
correct in the main Act body, but wrong in two situations that both produce
duplicate "#### Section N." headings colliding with the real Section N:

  1. Inside the Act's 14 Schedules, internal Rules/paragraphs restart their
     own numbering from 1 (e.g. the Third Schedule "Procedure for Recovery of
     Tax" has its own Rules 1-14) - the converter labels these "Section 1",
     "Section 2", ... "Section 14" again, identical to the real sections of
     the same number.
  2. Inside numbered tables/lists embedded in ordinary Section text (e.g. a
     district list in an Explanation, or a schedule-like industry
     classification table), individual row numbers occasionally get promoted
     to a heading the same way.

Confirmed by inspection: 72 section numbers have duplicate "#### Section N."
headings (124 duplicate occurrences total: 2 before the first Schedule
boundary, 122 after it). Every single one checked by hand was a false
positive - never a legitimate second use of a section number.

THE FIX:
  - Every "#### Section N." heading that falls AFTER a "## THE <NAME>
    SCHEDULE" boundary is relabeled to "#### THE <NAME> SCHEDULE -- Rule N.
    <title>" - not just the ones that happen to collide with an earlier
    Section number. Real Act sections never appear inside a Schedule, so
    ANY "Section" heading there is mislabeled Schedule content, whether or
    not its number happens to coincide with a real section elsewhere (many
    low numbers like 18, 19, 21... were long ago omitted/renumbered out of
    the Act body, so a Schedule's internal Rule 18 never collides with a
    duplicate check based on "have we seen this number before" - it has to
    be a pure schedule-boundary check instead). Renaming these out of the
    "Section" namespace is what stops them from colliding with real Section
    N ids during KG ingestion (see 02_extract_entities.py's matching update).
  - Occurrences BEFORE any Schedule boundary that duplicate an earlier
    Section number (only 2 found): these are pure noise - a numbered table
    row or a cross-reference number that happened to land at the start of a
    line after a page break, not a real provision. The "Section N." framing
    is dropped entirely and the line's real text is kept as plain body text.

A small number of relabeled headings have title text that itself looks
merged/garbled (e.g. two rules' text glued together at a page break) - this
script can't safely guess the correct wording, so it flags those separately
in the console output and in the "flagged_for_review" section of the JSON
report, WITHOUT silently rewriting their words. The heading is still moved
out of the Section namespace either way, so the KG-collision problem is fixed
regardless of whether the title text itself is later cleaned up by hand.

Usage:
    python 01b_fix_md_schedule_headings.py --md ../data/income_tax_act_1961.md \\
        --out ../data/income_tax_act_1961_v2.md --report fix_report.json

Writes to a NEW file (--out) rather than overwriting the input, so the
original converter output is preserved for diffing/audit.

PORTING NOTE (tax_kg_poc_revised): identical logic to
tax_kg_poc/scripts/01b_fix_md_schedule_headings.py. The only change is the
`step_logger` import below - the original bare `import step_logger` relied
on being run from a flat scripts/ folder where step_logger.py was a sibling;
here it lives at framework/core/infra/step_logger.py, so the import is
resolved the same way every other converter in this folder resolves it (a
sys.path bootstrap up to the tax_kg_poc_revised root, then an absolute
import). `fix_markdown()` itself - the function pdf_to_md_act.py actually
calls via dynamic module loading (see its run_01b_safety_net()) - is
untouched.
"""
import argparse
import json
import re

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
from framework.core.infra import step_logger

SCHEDULE_RE = re.compile(r"^## THE (.+) SCHEDULE\s*$")
HEADING_RE = re.compile(r"^#### Section ([0-9A-Za-z-]+)\.\s*(.*)$")

# Heuristics for "does this title look like it might be merged/garbled text
# rather than a clean provision title" - used only to FLAG for human review,
# never to silently rewrite wording.
GARBLED_PATTERNS = [
    re.compile(r"\)"),                    # stray closing paren with no opener in the title
    re.compile(r"\d\.\d+[A-Za-z]"),       # tariff/classification-code-like "84.71Foo"
    re.compile(r"[A-Za-z]\d{2,}"),        # word glued directly to a multi-digit number, e.g. "Centres84"
]


def looks_garbled(title: str) -> bool:
    if ")" in title and "(" not in title:
        return True
    for pat in GARBLED_PATTERNS[1:]:
        if pat.search(title):
            return True
    return False


def fix_markdown(text: str) -> tuple[str, dict]:
    lines = text.split("\n")
    seen_numbers = set()
    current_schedule = None

    rewritten = []
    flagged = []

    for i, line in enumerate(lines):
        sm = SCHEDULE_RE.match(line)
        if sm:
            current_schedule = sm.group(1).strip()
            continue

        hm = HEADING_RE.match(line)
        if not hm:
            continue

        num, title = hm.group(1), hm.group(2)
        is_dup = num in seen_numbers
        seen_numbers.add(num)

        if current_schedule:
            # Real Act sections never occur inside a Schedule - EVERY
            # "Section" heading here is mislabeled Schedule content,
            # regardless of whether `num` happens to duplicate an earlier
            # Section number (many low numbers were long ago omitted from
            # the Act body, so a Schedule's own Rule 18 etc. would never
            # trip a duplicate check).
            pass
        elif not is_dup:
            continue  # first occurrence outside any Schedule - the real Section, leave alone

        garbled = looks_garbled(title)
        if garbled:
            flagged.append({
                "line": i + 1, "section_no": num, "title": title,
                "context": "schedule" if current_schedule else "body",
                "schedule": current_schedule,
            })

        if current_schedule:
            new_line = f"#### THE {current_schedule} SCHEDULE — Rule {num}. {title}"
        else:
            new_line = title  # demote: drop the fabricated "Section N." framing entirely

        rewritten.append({
            "line": i + 1, "old": line, "new": new_line,
            "context": "schedule" if current_schedule else "body",
        })
        lines[i] = new_line

    report = {
        "total_headings_rewritten": len(rewritten),
        "rewritten_in_schedule": sum(1 for r in rewritten if r["context"] == "schedule"),
        "rewritten_in_body": sum(1 for r in rewritten if r["context"] == "body"),
        "flagged_for_review": flagged,
        "changes": rewritten,
    }
    return "\n".join(lines), report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, help="Path to the source income_tax_act_1961.md")
    parser.add_argument("--out", required=True, help="Path to write the corrected markdown")
    parser.add_argument("--report", default="fix_md_schedule_headings_report.json")
    args = parser.parse_args()

    with step_logger.Step(f"Reading {args.md}"):
        with open(args.md, "r", encoding="utf-8") as f:
            text = f.read()

    with step_logger.Step("Rewriting false Section headings"):
        fixed_text, report = fix_markdown(text)
        step_logger.log(f"  {report['total_headings_rewritten']} heading(s) rewritten "
                         f"({report['rewritten_in_schedule']} relabeled to Schedule/Rule, "
                         f"{report['rewritten_in_body']} demoted to plain text)")
        if report["flagged_for_review"]:
            step_logger.log(f"  {len(report['flagged_for_review'])} heading(s) flagged as possibly "
                             f"garbled/merged text - review these against the source PDF by hand:")
            for f in report["flagged_for_review"]:
                step_logger.log(f"    line {f['line']} (was Section {f['section_no']}, "
                                 f"{f['context']}): {f['title']!r}")

    with step_logger.Step(f"Writing corrected markdown to {args.out}"):
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(fixed_text)

    with step_logger.Step(f"Writing report to {args.report}"):
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
