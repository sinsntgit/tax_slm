"""
Sandboxed execution of LLM-generated pandas code, for the structured data
query agent (see structured_data_workspace.py and structured_data_query
agent in app.py that generates the code this module runs).

THREAT MODEL (explicitly discussed and approved by the user before this was
built - see the "how do i weigh in on step 3" / "okay go ahead with option B"
exchange): this deployment is a trusted-team-only tool inside a VPC with no
outside access - the user's own words: "it is just me and my team. the data
may be real client data as well - but is deployed within the VPC, so no
outside access is available." That means the realistic risk here is NOT a
malicious external attacker trying to escape a sandbox to pivot further into
the network - it's an LLM generating pandas code that is buggy, inefficient,
or occasionally nonsensical (an infinite loop from a bad groupby, a typo'd
column access that hangs instead of raising, an accidental O(n^3) join that
eats all available memory). The design below is sized for THAT blast radius:
a hard wall-clock timeout, memory/CPU limits, and restricted builtins as
defense-in-depth - not a full container-per-execution model (Docker, gVisor,
Firecracker), which would be proportionate to an adversarial-code threat
model but is overkill (and a much heavier ops burden - image builds, orchestration,
per-call container startup latency) for "one team's LLM-generated data-analysis
snippets running against their own already-authenticated tool."

WHY subprocess.Popen running a STANDALONE script (_sandbox_worker.py), NOT
multiprocessing.Process (2026-09-08: CHANGED - see the bugfix note below for
why the original design broke on Windows):
- A thread shares the parent's memory and GIL - a runaway thread can still be
  hard to kill cleanly and can corrupt/leak parent process state. A separate
  OS process gives real memory isolation and can be killed unconditionally if
  it won't stop.
- resource.setrlimit() applies to a process (and its threads), so hard
  memory/CPU ceilings (POSIX-only - see run_sandboxed()'s docstring) are only
  meaningful at the process boundary anyway.
- subprocess.Popen launches an ordinary, independent Python process running
  _sandbox_worker.py "cold" - there is no dependency on how the PARENT
  process itself was started, unlike multiprocessing.Process (see below).

BUGFIX (2026-09-08) - why this was multiprocessing.Process originally, and
why that broke on Windows: multiprocessing.Process forks on Linux (the child
is a cheap copy-on-write duplicate of the already-running parent - fast, no
re-import) but can only SPAWN on Windows - and spawn starts a brand-new
interpreter that re-imports the __main__ module before it can even unpickle
the worker function to run. Since this app is started as
`python -m framework.webapp.app`, __main__ on Windows IS app.py - so every
sandboxed execution was silently re-running app.py's entire module-level
setup (Flask app construction, Neo4j connections, prompt pack loading,
embedding/reranker model loading) inside the child before it ever reached
the actual pandas computation, reliably blowing the 10-second timeout even
for a trivial query on a tiny spreadsheet (reported: "Execution timed out
after 10 seconds" on Windows for a query that completes in well under a
second on Linux, where fork hides the problem entirely - this project's own
testing during development used the Linux sandbox and never caught it).
subprocess.Popen has no such "child re-imports __main__" step - it just runs
_sandbox_worker.py as a plain, independent script, identically on Windows
and Linux.

WHY THE DATAFRAME(S) ARE PASSED VIA A TEMP PICKLE FILE, NOT A SHARED
FILESYSTEM PATH TO THE ORIGINAL UPLOAD: the attached data may be real,
confidential client data (same reasoning as structured_data_workspace.py's
module docstring) - the DataFrame(s) already live in memory only (never
written to disk by structured_data_workspace.py); this module writes them to
a TEMP file solely as the handoff mechanism to the subprocess (Popen has no
in-memory equivalent of multiprocessing.Pipe), and deletes that temp file
immediately after the subprocess exits (see run_sandboxed()'s finally block) -
so the data touches disk only transiently, for the duration of one query, in
the OS temp directory, not the working directory or any persisted location.

WHAT THIS DOES NOT DEFEND AGAINST (documented honestly, not silently):
restricted builtins are a best-effort deterrent against a generated snippet
CASUALLY calling os.system() or opening a socket - they are not a hard
security boundary against a determined adversary deliberately trying to
escape (e.g. via ctypes, or byte-code manipulation, or a builtins-restriction
bypass gadget). That level of hardening (seccomp-bpf, gVisor, a real
container) is NOT implemented here because it does not match this
deployment's threat model above. If this tool's deployment model changes
(e.g. external/less-trusted users get access, or the "no outside access"
VPC assumption changes), this module's isolation approach should be
revisited.
"""
import os
import pickle
import subprocess
import sys
import tempfile
import time

from framework.core.infra import step_logger

# ---------------------------------------------------------------------------
# Tunables. Deliberately generous enough for legitimate pandas analysis over
# the row caps structured_data_workspace.py already enforces
# (MAX_ROWS_PER_SHEET = 50_000), but tight enough that a runaway/infinite-loop
# snippet doesn't hang a user-facing request for more than a few seconds past
# what they'd tolerate anyway.
# ---------------------------------------------------------------------------
EXECUTION_TIMEOUT_SECONDS = 10
MEMORY_LIMIT_BYTES = 1024 * 1024 * 1024  # 1 GB - generous for a <=50k-row DataFrame - enforced
                                          # inside _sandbox_worker.py (POSIX-only, see its own note).

_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sandbox_worker.py")


def run_sandboxed(code: str, dataframes: dict, timeout: float = EXECUTION_TIMEOUT_SECONDS) -> dict:
    """Runs `code` (a string of LLM-generated pandas code expected to set a
    `result` variable) against `dataframes` ({sheet_name: DataFrame}) in an
    isolated child PROCESS (subprocess.Popen running _sandbox_worker.py -
    see this module's docstring for why that file is standalone and why
    subprocess replaced multiprocessing.Process here), per this module's
    documented threat model.

    Returns one of:
      {"ok": True, "result": <value>, "elapsed_seconds": float}
      {"ok": False, "error": "...", "elapsed_seconds": float}
    - NEVER raises for a code/timeout/resource-limit failure (those are all
      expected, routine outcomes for LLM-generated code and are reported as
      data, not exceptions) - only a genuine harness bug (e.g. failing to
      even launch the subprocess) would raise here.

    PLATFORM NOTE: resource.setrlimit (inside _sandbox_worker.py) is
    POSIX-only. On Windows the memory/CPU rlimits silently no-op there and
    only the wall-clock `timeout` below still applies - that's an accepted,
    documented asymmetry, not a bug to fix here (this deployment's
    production target is Linux, inside a VPC - see this module's docstring).
    The wall-clock timeout itself, and the subprocess kill on timeout, work
    identically on both platforms.
    """
    started = time.monotonic()
    tmp_dir = tempfile.mkdtemp(prefix="sandbox_")
    in_path = os.path.join(tmp_dir, "in.pkl")
    out_path = os.path.join(tmp_dir, "out.pkl")
    # 2026-09-08: NEW - diagnostic timing, temporary, to pin down a reported
    # Windows timeout that this module's own testing (on Linux) could not
    # reproduce. Logs exactly when the subprocess is launched and how long
    # Popen() itself takes to return (a slow antivirus scan of a freshly
    # written python.exe launch would show up here as launch_seconds being
    # unexpectedly large, BEFORE the worker has done anything) - see
    # _sandbox_worker.py's own timing log for the pandas-import/exec split
    # on the other side of the same call.
    step_logger.log(f"[sandbox] launching worker: sys.executable={sys.executable!r}, "
                     f"script={_WORKER_SCRIPT!r}, timeout={timeout}s")

    try:
        with open(in_path, "wb") as f:
            pickle.dump({"code": code, "dataframes": dataframes}, f)

        popen_start = time.monotonic()
        proc = subprocess.Popen(
            # "-u" = unbuffered stdout/stderr - without it, a subprocess
            # writing to a PIPE (not a terminal) fully buffers its output,
            # so the worker's own diagnostic timing prints (see
            # _sandbox_worker.py) would never actually reach this parent
            # process's stderr capture if the worker gets killed on timeout
            # before its buffer flushes - defeating the point of adding them.
            [sys.executable, "-u", _WORKER_SCRIPT, in_path, out_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        step_logger.log(f"[sandbox] Popen() returned in {time.monotonic() - popen_start:.2f}s "
                         f"(pid={proc.pid}) - waiting up to {timeout}s for it to finish...")
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Timed out - the worker is still running (probably an infinite
            # loop or a genuinely huge computation past what this use case
            # should need). kill() sends SIGKILL on POSIX / TerminateProcess
            # on Windows - both unconditional, both work regardless of what
            # the generated code inside the worker is doing. The SECOND
            # communicate() call (after kill) still returns whatever the
            # worker had already written to stdout/stderr before it died -
            # 2026-09-08: NEW, captured and logged here so the diagnostic
            # timing prints from _sandbox_worker.py (module started / pickle
            # loaded / pandas imported / exec finished) are visible even on
            # the TIMEOUT path, not just the success path - this is exactly
            # what's needed to tell "process never even started running the
            # generated code" apart from "the generated code itself hung".
            proc.kill()
            timeout_stdout, timeout_stderr = proc.communicate()
            elapsed = time.monotonic() - started
            partial_stderr = (timeout_stderr or b"").decode("utf-8", errors="replace").strip()
            step_logger.log(f"[sandbox] execution timed out after {timeout}s - process killed. "
                             f"Popen() launch itself took {time.monotonic() - popen_start:.2f}s "
                             f"(see 'Popen() returned' line above). Worker's own progress before being "
                             f"killed (stderr): {partial_stderr[-2000:]!r}")
            return {"ok": False, "error": f"Execution timed out after {timeout} seconds "
                                            f"(likely an inefficient or infinite-looping query).",
                    "elapsed_seconds": elapsed}

        elapsed = time.monotonic() - started
        step_logger.log(f"[sandbox] worker finished in {elapsed:.2f}s total (returncode={proc.returncode}). "
                         f"stderr: {(stderr or b'')[-2000:].decode('utf-8', errors='replace')!r}")

        if proc.returncode != 0 or not os.path.exists(out_path):
            # The worker process crashed before writing a result (e.g. killed
            # by the OS OOM killer for exceeding the memory limit, or an
            # exception during pickle.load/dump itself rather than inside the
            # generated code's own try/except). stderr often has the real
            # reason (e.g. a Python traceback from a crash) - included for
            # diagnosability, truncated to keep the user-facing error short.
            stderr_text = (stderr or b"").decode("utf-8", errors="replace").strip()
            reason = stderr_text[-500:] if stderr_text else f"exit code {proc.returncode}, no output produced"
            step_logger.log(f"[sandbox] worker process failed without a result: {reason}")
            return {"ok": False, "error": f"Execution failed without a result ({reason}) - "
                                            f"likely exceeded available memory or crashed.",
                    "elapsed_seconds": elapsed}

        with open(out_path, "rb") as f:
            response = pickle.load(f)
        response["elapsed_seconds"] = elapsed
        if not response.get("ok"):
            step_logger.log(f"[sandbox] execution failed: {response.get('error')}")
        return response
    finally:
        # Best-effort cleanup - the pickled DataFrame(s) may contain real
        # client data (see this module's docstring), so this temp directory
        # should not outlive a single query. Not fatal if a file is locked/
        # already gone (e.g. the killed worker still had a handle open for a
        # moment on Windows) - the OS temp directory is periodically cleaned
        # regardless, this is a best-effort not a hard guarantee.
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
