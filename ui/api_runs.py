import os
from flask import Blueprint, request, jsonify, current_app
from .job_manager import find_rep_images

bp_runs = Blueprint("runs", __name__)

@bp_runs.get("/api/runs/list")
def api_runs_list():
    out_root = current_app.config["OUT_ROOT"]
    runs = []
    try:
        if os.path.isdir(out_root):
            for name in sorted(os.listdir(out_root)):
                p = os.path.join(out_root, name)
                if os.path.isdir(p):
                    imgs = find_rep_images(p)
                    runs.append({"name": name, "path": p, "mtime": os.path.getmtime(p), "images": imgs[:6]})
        runs.sort(key=lambda x: x["mtime"], reverse=True)
    except Exception:
        pass
    return jsonify({"ok": True, "out_root": out_root, "runs": runs})

@bp_runs.get("/api/runs/inspect")
def api_runs_inspect():
    run_path = request.args.get("path", "")
    if not run_path or not os.path.isdir(run_path):
        return jsonify({"ok": False, "error": "Invalid run path"}), 400
    return jsonify({
        "ok": True,
        "path": run_path,
        "images": find_rep_images(run_path),
        "files": [os.path.join(run_path, f) for f in sorted(os.listdir(run_path))][:200],
    })
