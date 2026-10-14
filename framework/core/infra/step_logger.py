"""
Shared, minimal progress-logging helper - PROMOTED unchanged from
tax_kg_poc/scripts/step_logger.py. 100% domain-agnostic already (no tax
concepts anywhere in it), so this is a straight copy: every script/module in
every domain pack imports this same instance.

Usage:
    from framework.core.infra import step_logger
    step_logger.set_verbose(args.verbose)
    step_logger.log("Reading file...")
    with step_logger.Step("Extracting entities"):
        ...
"""
import time
from datetime import datetime

VERBOSE = False


def set_verbose(value: bool):
    global VERBOSE
    VERBOSE = bool(value)


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def fmt(msg: str) -> str:
    return f"[{_timestamp()}] {msg}"


def log(msg: str):
    print(fmt(msg), flush=True)


def vlog(msg: str):
    if VERBOSE:
        log(msg)


class Step:
    def __init__(self, name: str):
        self.name = name
        self._t0 = None

    def __enter__(self):
        self._t0 = time.time()
        log(f"START  - {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self._t0
        if exc_type:
            log(f"FAILED - {self.name} ({elapsed:.1f}s) - {exc_type.__name__}: {exc}")
        else:
            log(f"DONE   - {self.name} ({elapsed:.1f}s)")
        return False
