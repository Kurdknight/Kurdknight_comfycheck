"""Executes a remedy and streams its output back to the panel.

Design constraints, in order of importance:

  1. The browser cannot specify a command. It sends a finding id; we run the
     argv WE generated during the last scan. Nothing else is executable.
  2. No shell. Ever. argv lists go straight to Popen with shell=False, so there
     is no quoting, no injection surface, no PATH surprises.
  3. One job at a time. Two concurrent pip processes writing the same
     site-packages is how you turn a broken environment into an unrecoverable
     one.
  4. Output streams live. A pip install of torch takes minutes; a spinner with
     no output is indistinguishable from a hang, and people kill it halfway -
     which is precisely the state you never want to leave a package in.
"""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field

from .models import Remedy


@dataclass
class Job:
    id: str
    finding_id: str
    title: str
    commands: list[list[str]]
    lines: list[str] = field(default_factory=list)
    status: str = "pending"     # pending | running | success | failed | cancelled
    exit_code: int | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    _proc: subprocess.Popen | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)
            # Keep a hard cap; pip on a slow connection emits thousands of
            # progress lines and we don't want to hold them all forever.
            if len(self.lines) > 4000:
                del self.lines[:1000]

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "finding_id": self.finding_id,
                "title": self.title,
                "status": self.status,
                "exit_code": self.exit_code,
                "lines": self.lines[since:],
                "total_lines": len(self.lines),
                "elapsed": round((self.finished_at or time.time()) - self.started_at, 1)
                if self.started_at else 0,
            }


_JOBS: dict[str, Job] = {}
_ACTIVE: str | None = None
_GLOBAL_LOCK = threading.Lock()


def active_job() -> Job | None:
    with _GLOBAL_LOCK:
        return _JOBS.get(_ACTIVE) if _ACTIVE else None


def get(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


def start(finding_id: str, remedy: Remedy) -> tuple[Job | None, str | None]:
    """Returns (job, error). Refuses to start if something is already running."""
    global _ACTIVE

    with _GLOBAL_LOCK:
        current = _JOBS.get(_ACTIVE) if _ACTIVE else None
        if current and current.status == "running":
            return None, (
                f"'{current.title}' is still running. Only one repair may run at a time - two pip "
                f"processes writing the same site-packages can corrupt it beyond repair."
            )

        job = Job(
            id=uuid.uuid4().hex[:12],
            finding_id=finding_id,
            title=remedy.title,
            commands=[list(c) for c in remedy.commands],
        )
        _JOBS[job.id] = job
        _ACTIVE = job.id

    threading.Thread(target=_run, args=(job,), daemon=True, name=f"comfydoctor-fix-{job.id}").start()
    return job, None


def cancel(job_id: str) -> bool:
    job = _JOBS.get(job_id)
    if not job or job.status != "running" or not job._proc:
        return False
    job.emit("")
    job.emit("[cancelled by user - the package being installed may be left half-written;")
    job.emit(" re-run the fix to finish the job cleanly]")
    try:
        job._proc.terminate()
    except Exception:
        return False
    job.status = "cancelled"
    return True


def _run(job: Job) -> None:
    job.status = "running"
    job.started_at = time.time()

    try:
        for i, argv in enumerate(job.commands, 1):
            if job.status == "cancelled":
                break
            if len(job.commands) > 1:
                job.emit(f"[step {i} of {len(job.commands)}]")
            job.emit("$ " + " ".join(argv))
            job.emit("")

            code = _stream(job, argv)
            if job.status == "cancelled":
                break
            if code != 0:
                job.status = "failed"
                job.exit_code = code
                job.emit("")
                job.emit(f"[failed with exit code {code}]")
                job.emit(_diagnose_failure(job))
                return
        else:
            job.status = "success"
            job.exit_code = 0
            job.emit("")
            job.emit("[done - restart ComfyUI, then run the scan again to confirm]")
    except Exception as e:
        job.status = "failed"
        job.emit(f"[ComfyDoctor could not run this command: {type(e).__name__}: {e}]")
    finally:
        job.finished_at = time.time()


def _stream(job: Job, argv: list[str]) -> int:
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,   # pip must never be able to block on a prompt
        text=True,
        bufsize=1,
        shell=False,                # non-negotiable
        creationflags=_no_window(),
    )
    job._proc = proc
    assert proc.stdout is not None
    for line in proc.stdout:
        job.emit(line.rstrip("\n"))
    proc.wait()
    return proc.returncode


def _no_window() -> int:
    """Stop Windows flashing a console window in the user's face."""
    import os

    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _diagnose_failure(job: Job) -> str:
    """Turn pip's failure into something actionable.

    A raw pip traceback is where most users give up. These four cases cover the
    overwhelming majority of what actually goes wrong.
    """
    text = "\n".join(job.lines[-60:]).lower()

    if "access is denied" in text or "permission denied" in text or "winerror 5" in text:
        return (
            "[why: a file was locked. ComfyUI is still holding the package you're replacing.\n"
            " Close ComfyUI completely, then run this command in a terminal:\n"
            "   " + " ".join(job.commands[0]) + "]"
        )
    if "no space left" in text or "not enough space" in text or "errno 28" in text:
        return "[why: the disk filled up. Free space and try again - a torch install needs ~5 GB of scratch.]"
    if "could not find a version" in text or "no matching distribution" in text:
        return (
            "[why: pip couldn't find that version for your Python/platform. Usually means your\n"
            " Python is too new for this package. See the System section of the report.]"
        )
    if "connection" in text or "timed out" in text or "ssl" in text:
        return "[why: the download failed. Check your connection or proxy and re-run - pip resumes from cache.]"
    return "[the pip output above has the reason. Nothing was left half-installed unless it says otherwise.]"
