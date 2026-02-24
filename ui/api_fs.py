import os, io
import mimetypes
from flask import Blueprint, request, jsonify, send_file
from PIL import Image
from .job_manager import is_image

bp_fs = Blueprint("fs", __name__)

def list_dir(path: str):
    out = {"path": path, "entries": []}
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    full = os.path.join(path, e.name)
                    entry = {
                        "name": e.name,
                        "path": full,
                        "is_dir": e.is_dir(follow_symlinks=False),
                        "is_file": e.is_file(follow_symlinks=False),
                    }
                    if entry["is_file"] and is_image(full):
                        entry["is_image"] = True
                        entry["size"] = e.stat(follow_symlinks=False).st_size
                    else:
                        entry["is_image"] = False
                        entry["size"] = e.stat(follow_symlinks=False).st_size if entry["is_file"] else None
                    out["entries"].append(entry)
                except Exception:
                    continue
        out["entries"].sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        out["ok"] = True
    except Exception as ex:
        out["ok"] = False
        out["error"] = str(ex)
    return out

@bp_fs.get("/api/fs/list")
def api_fs_list():
    path = request.args.get("path", "/")
    return jsonify(list_dir(path))

@bp_fs.get("/api/fs/thumb")
def api_fs_thumb():
    path = request.args.get("path", "")
    if not path or not os.path.isfile(path) or not is_image(path):
        return jsonify({"ok": False, "error": "Not an image"}), 400

    max_w = int(request.args.get("w", "256"))
    max_h = int(request.args.get("h", "256"))

    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((max_w, max_h))
        bio = io.BytesIO()
        im.save(bio, format="JPEG", quality=85)
        bio.seek(0)
        # cache thumbs a bit to reduce load
        return send_file(bio, mimetype="image/jpeg", max_age=0)
    except Exception as ex:
        return jsonify({"ok": False, "error": str(ex)}), 500


# ✅ NEW: serve the original file (png/jpg/etc) so clicking works
@bp_fs.get("/api/fs/file")
def api_fs_file():
    path = request.args.get("path", "").strip()
    if not path:
        return jsonify({"ok": False, "error": "missing path"}), 400

    # Normalize
    path = os.path.abspath(path)

    if not os.path.isfile(path):
        return jsonify({"ok": False, "error": "not found"}), 404

    # Optional: limit to images only (safer)
    if not is_image(path):
        return jsonify({"ok": False, "error": "not an image"}), 400

    mime, _ = mimetypes.guess_type(path)
    return send_file(
        path,
        mimetype=mime or "application/octet-stream",
        as_attachment=False,
        conditional=True,
        #max_age=3600
    )
