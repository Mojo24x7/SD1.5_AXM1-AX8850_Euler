import json, time, queue
from flask import Blueprint, request, jsonify, Response
from .job_manager import JOBS, JOBS_LOCK, now_iso

bp_jobs = Blueprint("jobs", __name__)

@bp_jobs.post("/api/jobs/cancel")
def api_jobs_cancel():
    job_id = (request.json or {}).get("job_id", "")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404

    job.kill()
    job.status = "cancelled"
    job.ended = now_iso()
    job.push({"type": "status", "status": job.status, "ended": job.ended, "returncode": -9})
    return jsonify({"ok": True})

@bp_jobs.get("/api/jobs/status")
def api_jobs_status():
    job_id = request.args.get("job_id", "")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404

    return jsonify({
        "ok": True,
        "job_id": job.job_id,
        "kind": job.kind,
        "status": job.status,
        "created": job.created,
        "started": job.started,
        "ended": job.ended,
        "returncode": job.returncode,
        "run_dir": job.run_dir,
        "progress": job.progress,
        "npu": job.npu,
    })

@bp_jobs.get("/api/jobs/events")
def api_jobs_events():
    job_id = request.args.get("job_id", "")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404

    def stream():
        # snapshot first (so UI updates immediately)
        snap = {
            "type": "snapshot",
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "npu": job.npu,
            "run_dir": job.run_dir,
        }
        yield f"data: {json.dumps(snap)}\n\n"
        # flush hint
        yield ": ok\n\n"

        while True:
            try:
                msg = job.pop(timeout=1.0)
                yield f"data: {json.dumps(msg)}\n\n"
                yield ":.\n\n"  # keep proxies flushing

                if msg.get("type") == "status" and msg.get("status") in ("done", "error", "cancelled"):
                    # drain remaining queue briefly
                    t_end = time.time() + 1.2
                    while time.time() < t_end:
                        try:
                            msg2 = job.pop(timeout=0.2)
                            yield f"data: {json.dumps(msg2)}\n\n"
                            yield ":.\n\n"
                        except queue.Empty:
                            break
                    break

            except queue.Empty:
                yield f"data: {json.dumps({'type':'ping','t':time.time()})}\n\n"
                yield ":.\n\n"

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # important if ever behind a proxy
    }
    return Response(stream(), mimetype="text/event-stream", headers=headers)
