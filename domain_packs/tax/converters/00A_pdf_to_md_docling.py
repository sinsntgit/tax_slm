"""
Step 0A: Convert income_tax_act_1961.pdf to Markdown using Docling
(https://github.com/docling-project/docling), as a layout-model-based
alternative to the custom "font decode + heading-regex" converter that
produced income_tax_act_1961.md (see 01b_fix_md_schedule_headings.py for the
141 mislabeled headings that converter's regex-based heading detection
produced - Schedule-internal Rules getting promoted to "#### Section N."
because the regex only looks at text patterns, not visual layout).

Docling instead runs a trained layout model over each page to classify text
blocks (heading vs body vs table vs list, etc.) from their actual visual
presentation - font size/weight/position - which should avoid the specific
"a numbered table row looks like 'N. text' so it must be a heading" failure
mode entirely, since a table row and a real heading don't look alike on the
page even when their text does. It CANNOT, however, know that a "Rule 11 of
the Second Schedule" is semantically different from "Section 11" of the Act
body just from layout - that distinction is domain knowledge, not a layout
feature - so it likely will still label both as headings of the same visual
style. Verify the output against the checks this script runs (and against
income_tax_act_1961_v2.md) before trusting it as an ingestion source.

IMPORTANT - RESOURCE REQUIREMENTS (learned the hard way while building this):
Docling's standard PDF pipeline pulls in torch + transformers + its own
layout/table-structure models (docling-ibm-models). On Linux, plain
`pip install docling` resolves the CUDA build of torch by default - that
drags in the full nvidia-cublas/cudnn/cusparse/nccl/etc. stack (several GB)
even on a CPU-only machine, because PyPI only hosts the CUDA wheel for Linux;
the small CPU-only wheel lives on a separate index (download.pytorch.org)
that many locked-down networks (including the sandbox this script was
authored in) don't allow. Before running this for real:
  - Make sure you have several GB of free disk for the model weights + torch.
  - If your network allows it, install the CPU-only torch build FIRST so
    docling's install doesn't pull the CUDA stack:
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        pip install docling
    Otherwise just `pip install docling` and accept the larger download.
  - First run downloads the layout + table-structure model weights from
    HuggingFace (one-time, cached under ~/.cache/docling/models after that -
    see --artifacts-path below to pre-fetch/pin them instead).
  - This was NOT executed end-to-end in the sandbox this script was written
    in (2 CPUs, ~3.4GB free disk, and download.pytorch.org blocked by the
    network policy) - only verified against Docling's official docs/source
    for API correctness. Run the --max-pages smoke test below FIRST on your
    own machine before doing the full 1098-page conversion.

Usage:
    # Smoke test on just the first 15 pages before committing to the full run:
    python 00A_pdf_to_md_docling.py --pdf ../data/income_tax_act_1961.pdf \\
        --out ../data/income_tax_act_1961_docling_sample.md --max-pages 15

    # Full conversion:
    python 00A_pdf_to_md_docling.py --pdf ../data/income_tax_act_1961.pdf \\
        --out ../data/income_tax_act_1961_docling.md

    # Faster/rougher tables (default is the more accurate, slower TableFormer mode):
    python 00A_pdf_to_md_docling.py --pdf ../data/income_tax_act_1961.pdf \\
        --out ../data/income_tax_act_1961_docling.md --fast-tables

    # Pre-fetched/pinned model weights (see docling-tools models download):
    python 00A_pdf_to_md_docling.py --pdf ../data/income_tax_act_1961.pdf \\
        --out ../data/income_tax_act_1961_docling.md --artifacts-path /local/path/to/models

PORTING NOTE (tax_kg_poc_revised): identical logic to
tax_kg_poc/scripts/00A_pdf_to_md_docling.py. The only change is the
`poppler_utils`/`step_logger` imports below - the originals were bare
same-directory imports that relied on being run from a flat scripts/
folder; here those two live at framework/core/convert/poppler_utils.py and
framework/core/infra/step_logger.py, so they're resolved the same way every
other converter in this folder resolves them (a sys.path bootstrap up to
the tax_kg_poc_revised root, then an absolute import). This file's own
convert()/check_heading_duplicates()/check_word_count_vs_pdftotext() -
the functions pdf_to_md_act.py actually calls via dynamic module loading,
see its load_00a_sanity_checks() - are untouched. Docling itself remains an
optional, lazily-imported dependency (only pulled in inside convert(), never
at module import time), so pdf_to_md_act.py's --validate safety-net checks
work without docling being installed at all.
"""
import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
from framework.core.convert import poppler_utils
from framework.core.infra import step_logger


def convert(args):
    # Imported lazily so `--help` and argument errors don't require docling
    # (and its heavy dependency chain) to be importable first.
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    # Native-text PDF (PageMaker/Distiller output, not a scan) - OCR would
    # only slow things down and risks introducing recognition errors on text
    # that's already perfectly extractable, so it stays off.
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True,
        mode=TableFormerMode.FAST if args.fast_tables else TableFormerMode.ACCURATE,
    )
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=args.threads, device=AcceleratorDevice.CPU,
    )
    if args.artifacts_path:
        pipeline_options.artifacts_path = args.artifacts_path

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    convert_kwargs = {}
    if args.max_pages:
        convert_kwargs["max_num_pages"] = args.max_pages
    if args.max_file_size_mb:
        convert_kwargs["max_file_size"] = args.max_file_size_mb * 1024 * 1024

    with step_logger.Step(f"Converting {args.pdf} with Docling "
                           f"(threads={args.threads}, tables={'FAST' if args.fast_tables else 'ACCURATE'}"
                           f"{f', max_pages={args.max_pages}' if args.max_pages else ''})"):
        t0 = time.time()
        result = converter.convert(str(args.pdf), **convert_kwargs)
        elapsed = time.time() - t0
        step_logger.log(f"  Converted in {elapsed:.1f}s "
                         f"({elapsed / max(len(result.document.pages), 1):.2f}s/page, "
                         f"{len(result.document.pages)} page(s) processed)")

    with step_logger.Step("Exporting to Markdown"):
        markdown_text = result.document.export_to_markdown()

    with step_logger.Step(f"Writing {args.out}"):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown_text, encoding="utf-8")

    if args.json_out:
        with step_logger.Step(f"Writing DoclingDocument JSON to {args.json_out}"):
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result.document.export_to_dict()), encoding="utf-8")

    return markdown_text, result


# --- POST-CONVERSION SANITY CHECKS ------------------------------------------
# The same class of checks that caught 141 bad headings in the old converter's
# output (see the conversation this script came out of) - run automatically
# here so a Docling conversion doesn't get trusted blind either. Docling's
# headings won't look like "#### Section N." (it doesn't know Income-tax Act
# conventions), so this checks what IS knowable generically: heading-text
# duplication (a real red flag regardless of source) and overall text
# completeness against a raw pdftotext extraction.

def check_heading_duplicates(markdown_text: str) -> dict:
    headings = re.findall(r"^#{1,6}\s+(.+)$", markdown_text, flags=re.MULTILINE)
    counts = Counter(h.strip() for h in headings if h.strip())
    dupes = {h: c for h, c in counts.items() if c > 1}
    return {"total_headings": len(headings), "unique_headings": len(counts), "duplicate_headings": dupes}


def check_word_count_vs_pdftotext(markdown_text: str, pdf_path: Path) -> dict:
    md_words = len(markdown_text.split())
    try:
        # encoding="utf-8"/errors="replace" - NOT optional, see pipeline.py's
        # _run_subprocess() for the full explanation: `text=True` alone
        # decodes via the platform locale encoding (cp1252 on Windows),
        # which real Act/circular/case-law text routinely contains
        # characters outside of (smart quotes, etc.) - that previously
        # crashed the subprocess reader thread with UnicodeDecodeError,
        # left `.stdout` as None, and turned THIS non-critical sanity check
        # into a hard AttributeError that aborted an otherwise-successful
        # conversion (the .md file was already written by this point).
        raw = subprocess.run([poppler_utils.poppler_bin("pdftotext"), "-layout", str(pdf_path), "-"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=300, check=True).stdout
        pdf_words = len(raw.split())
        pct_diff = 100 * (md_words - pdf_words) / pdf_words if pdf_words else None
        return {"pdftotext_available": True, "pdf_word_count": pdf_words,
                "md_word_count": md_words, "pct_diff": pct_diff}
    except Exception as e:
        # Broadened from a narrow (FileNotFoundError, CalledProcessError,
        # TimeoutExpired) tuple to bare Exception - this is a nice-to-have
        # cross-check, not something that should ever be able to abort a
        # conversion whose real output (the .md file) has already been
        # written successfully. Any failure here degrades to "couldn't run
        # the check," never a crash.
        return {"pdftotext_available": False, "md_word_count": md_words, "error": str(e)}


def run_sanity_checks(markdown_text: str, pdf_path: Path) -> dict:
    report = {
        "headings": check_heading_duplicates(markdown_text),
        "word_count": check_word_count_vs_pdftotext(markdown_text, pdf_path),
    }
    with step_logger.Step("Post-conversion sanity checks"):
        h = report["headings"]
        step_logger.log(f"  {h['total_headings']} heading(s), {len(h['duplicate_headings'])} "
                         f"distinct heading TEXT repeated more than once")
        if h["duplicate_headings"]:
            step_logger.log("  Top repeated headings (inspect these - repetition is exactly "
                             "the pattern that flagged the old converter's Schedule-heading bug):")
            for text, count in sorted(h["duplicate_headings"].items(), key=lambda kv: -kv[1])[:10]:
                step_logger.log(f"    x{count}: {text[:80]!r}")
        w = report["word_count"]
        if w["pdftotext_available"]:
            step_logger.log(f"  word count: PDF={w['pdf_word_count']}, MD={w['md_word_count']} "
                             f"({w['pct_diff']:+.2f}% diff)")
            if w["pct_diff"] is not None and abs(w["pct_diff"]) > 5:
                step_logger.log("  WARNING: word count diverges by more than 5% - check for "
                                 "dropped pages/sections (common cause: max_num_pages/max_file_size "
                                 "silently truncating, or a page the layout model failed on).")
        else:
            step_logger.log(f"  pdftotext not available for cross-check ({w.get('error')}) - "
                             f"MD word count: {w['md_word_count']}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True, type=Path, help="Path to the source PDF")
    parser.add_argument("--out", required=True, type=Path, help="Path to write the converted Markdown")
    parser.add_argument("--json-out", type=Path, default=None,
                         help="Optional: also write the full DoclingDocument as JSON here "
                              "(has per-item page numbers/bounding boxes for provenance, "
                              "which the flat Markdown export does not preserve)")
    parser.add_argument("--report", type=Path, default=None,
                         help="Optional: write the sanity-check report as JSON here")
    parser.add_argument("--max-pages", type=int, default=None,
                         help="Only convert the first N pages - use this for a smoke test "
                              "before running the full 1098-page document")
    parser.add_argument("--max-file-size-mb", type=float, default=None,
                         help="Abort if the PDF exceeds this size in MB (safety limit)")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4,
                         help="CPU threads for the layout/table models (default: os.cpu_count())")
    parser.add_argument("--fast-tables", action="store_true",
                         help="Use TableFormerMode.FAST instead of the default ACCURATE "
                              "(faster, less precise table structure - worth trying first "
                              "on constrained hardware)")
    parser.add_argument("--artifacts-path", type=str, default=None,
                         help="Path to pre-fetched Docling model weights "
                              "(see: docling-tools models download) - skips the first-run "
                              "HuggingFace download, useful for offline/air-gapped runs")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    markdown_text, result = convert(args)
    report = run_sanity_checks(markdown_text, args.pdf)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        step_logger.log(f"Sanity-check report written to {args.report}")

    step_logger.log(f"Done - {args.out} ({len(markdown_text.split())} words, "
                     f"{len(result.document.pages)} page(s))")


if __name__ == "__main__":
    main()
