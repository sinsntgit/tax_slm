"""
Standalone worker script for sandbox.py's run_sandboxed(). Launched as its
OWN Python process via subprocess.Popen (see run_sandboxed()'s docstring for
why subprocess, not multiprocessing.Process, is used here) - NEVER imports
anything from the `framework` or `app` packages beyond the one thing it
absolutely needs (pickle protocol compatibility is stdlib-only, so nothing
from this project is required at all). This isolation is deliberate: this
file must be safely runnable as `python _sandbox_worker.py <in> <out>` with
NOTHING else already loaded, so that launching it never re-imports Flask,
Neo4j drivers, sentence-transformers, or any other of app.py's heavy
module-level setup.

WHY THIS FILE EXISTS SEPARATELY FROM sandbox.py (2026-09-08: BUGFIX):
the original design used multiprocessing.Process, which works by FORKING on
Linux (the child is a cheap copy-on-write duplicate of the already-running
parent - fast, no re-import) but by SPAWNING on Windows (the only start
method Windows supports) - and spawn works by starting a brand-new Python
interpreter that re-imports the __main__ module before it can even unpickle
the worker function to run. Since this app is started as
`python -m framework.webapp.app`, __main__ on Windows IS app.py - so every
single sandboxed execution was silently re-running app.py's ENTIRE
module-level setup (Flask app construction, Neo4j connections, prompt pack
loading, embedding/reranker model loading) inside the child before it ever
reached the actual one-line pandas computation. That easily blew past the
10-second timeout even for a trivial query on a 16-row spreadsheet - this
was reported as "Execution timed out after 10 seconds" on Windows for a
query that runs in under a tenth of a second on Linux (where fork hides the
problem entirely). Switching to subprocess.Popen with THIS standalone
script sidesteps the whole issue: there is no "child re-imports __main__"
step for an ordinary subprocess - it just runs a plain .py file cold, same
as running any script from a terminal.

PROTOCOL: argv[1] is a path to a pickle file containing
{"code": str, "dataframes": {sheet_name: DataFrame}} written by the parent
(run_sandboxed()). argv[2] is a path this worker should pickle its own
{"ok": bool, "result"/"error": ...} response dict into. Using files (not
stdout/stdin) avoids any risk of pandas/numpy startup banner text or a
generated `print(...)` statement in the LLM's code corrupting a
binary-over-stdout protocol.
"""
import sys


def _build_restricted_globals(dataframes):
    """See sandbox.py's _build_restricted_globals() - kept as an exact,
    independent copy here (not imported from sandbox.py) so this worker
    file has ZERO dependency on the rest of this project, per this module's
    docstring above. Any change to the restricted-builtins allowlist must be
    made in BOTH places - see sandbox.py's own copy for the full docstring
    explaining the allowlist's reasoning."""
    import builtins as _builtins_mod
    safe_names = (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
        "float", "format", "frozenset", "getattr", "hasattr", "hash", "int",
        "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
        "next", "object", "pow", "print", "range", "repr", "reversed", "round",
        "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
        "True", "False", "None", "Exception", "ValueError", "TypeError",
        "KeyError", "IndexError", "StopIteration", "ZeroDivisionError",
        "AttributeError", "RuntimeError", "ArithmeticError",
    )
    safe_builtins = {name: getattr(_builtins_mod, name) for name in safe_names
                     if hasattr(_builtins_mod, name)}

    import pandas as pd
    import numpy as np

    g = {"__builtins__": safe_builtins, "pd": pd, "np": np, "dfs": dataframes}
    if len(dataframes) == 1:
        g["df"] = next(iter(dataframes.values()))
    return g


def main() -> None:
    import pickle
    import time
    import traceback

    # 2026-09-08: NEW - diagnostic timing, temporary, to pin down a reported
    # Windows timeout that could not be reproduced on Linux (see sandbox.py's
    # matching log lines on the parent side). Written to stderr (not
    # step_logger - this process has no logging setup of its own and
    # shouldn't import anything beyond stdlib/pandas/numpy, per this file's
    # module docstring) so run_sandboxed() can capture and log it.
    t_start = time.monotonic()
    print(f"[_sandbox_worker] process started at {t_start:.3f} (module-level)", file=sys.stderr, flush=True)

    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "rb") as f:
        payload = pickle.load(f)
    code = payload["code"]
    dataframes = payload["dataframes"]
    t_after_load = time.monotonic()
    print(f"[_sandbox_worker] loaded input pickle in {t_after_load - t_start:.3f}s", file=sys.stderr, flush=True)

    # Resource limits (memory/CPU) - POSIX-only, same as before. A no-op on
    # Windows; the parent's wall-clock subprocess timeout is the enforced
    # backstop on every platform regardless (see run_sandboxed()'s docstring).
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    except Exception:
        pass

    try:
        t_before_build = time.monotonic()
        exec_globals = _build_restricted_globals(dataframes)  # imports pandas/numpy - the likely slow step
        t_after_build = time.monotonic()
        print(f"[_sandbox_worker] built globals (pandas/numpy import) in "
              f"{t_after_build - t_before_build:.3f}s", file=sys.stderr, flush=True)

        exec(code, exec_globals)
        t_after_exec = time.monotonic()
        print(f"[_sandbox_worker] exec() of generated code took "
              f"{t_after_exec - t_after_build:.3f}s", file=sys.stderr, flush=True)

        result = exec_globals.get("result")
        if result is None:
            response = {"ok": False, "error": "Generated code did not assign a `result` variable."}
        else:
            response = {"ok": True, "result": result}
    except Exception as e:
        response = {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=4)}

    with open(out_path, "wb") as f:
        pickle.dump(response, f)
    print(f"[_sandbox_worker] total worker time: {time.monotonic() - t_start:.3f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
