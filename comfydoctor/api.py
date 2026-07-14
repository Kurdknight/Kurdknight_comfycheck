"""HTTP surface, mounted on ComfyUI's aiohttp server.

  GET  /comfydoctor/scan          -> ScanResult as JSON
  GET  /comfydoctor/report.html   -> self-contained HTML report (download)
  GET  /comfydoctor/report.md     -> markdown, anonymized, for pasting into an issue
  POST /comfydoctor/fix           -> {finding_id} -> {job_id}
  GET  /comfydoctor/fix/{job_id}  -> job status + new output lines (poll with ?since=N)
  POST /comfydoctor/fix/{job_id}/cancel

The old code registered routes with a Flask-style `@server.route` decorator that
ComfyUI's aiohttp server does not have - so none of its routes ever existed and
its Refresh/Save buttons had never worked. This is the actual API.
"""

from __future__ import annotations

import json

from . import report, runner
from .scan import last as last_scan
from .scan import remedy_for
from .scan import scan as run_scan

_registered = False


def register() -> bool:
    """Attach routes to the running ComfyUI server. Safe to call twice."""
    global _registered
    if _registered:
        return True

    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return False  # not inside ComfyUI (CLI mode) - nothing to register

    routes = PromptServer.instance.routes

    @routes.get("/comfydoctor/scan")
    async def _scan(request):
        result = await _in_thread(run_scan)
        return web.json_response(result.to_dict())

    @routes.get("/comfydoctor/report.html")
    async def _report_html(request):
        result = last_scan() or await _in_thread(run_scan)
        html = report.to_html(result)
        return web.Response(
            body=html.encode("utf-8"),
            content_type="text/html",
            headers={"Content-Disposition": 'attachment; filename="comfydoctor-report.html"'},
        )

    @routes.get("/comfydoctor/report.md")
    async def _report_md(request):
        result = last_scan() or await _in_thread(run_scan)
        return web.Response(text=report.to_markdown(result), content_type="text/plain")

    @routes.post("/comfydoctor/fix")
    async def _fix(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "expected JSON"}, status=400)

        finding_id = body.get("finding_id")
        if not isinstance(finding_id, str):
            return web.json_response({"error": "finding_id is required"}, status=400)

        # The client hands us an id, never a command. We execute only the argv we
        # generated ourselves during the last scan. If the id isn't in that scan
        # (stale panel, or someone poking the endpoint) there is simply nothing
        # to run.
        remedy = remedy_for(finding_id)
        if remedy is None:
            return web.json_response(
                {"error": "No runnable fix for that finding in the current scan. Re-scan and retry."},
                status=404,
            )

        job, err = runner.start(finding_id, remedy)
        if job is None:
            return web.json_response({"error": err}, status=409)
        return web.json_response({"job_id": job.id, "commands": remedy.as_shell()})

    @routes.get("/comfydoctor/fix/{job_id}")
    async def _fix_status(request):
        job = runner.get(request.match_info["job_id"])
        if not job:
            return web.json_response({"error": "unknown job"}, status=404)
        try:
            since = int(request.query.get("since", "0"))
        except ValueError:
            since = 0
        return web.json_response(job.snapshot(since))

    @routes.post("/comfydoctor/fix/{job_id}/cancel")
    async def _fix_cancel(request):
        ok = runner.cancel(request.match_info["job_id"])
        return web.json_response({"cancelled": ok})

    _registered = True
    return True


async def _in_thread(fn, *args):
    """A full scan takes ~1-3s (nvidia-smi + a few hundred dist-info reads).
    That is far too long to block ComfyUI's event loop, which is also serving
    the websocket that streams render previews."""
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)
