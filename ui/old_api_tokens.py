import os
import json
import subprocess
from flask import Blueprint, request, jsonify, current_app

bp_tokens = Blueprint("tokens", __name__)

@bp_tokens.post("/api/tokens/count")
def api_tokens_count():
    cfg = request.json or {}
    prompt = cfg.get("prompt", "")
    negative = cfg.get("negative", "")

    token_py = current_app.config.get("TOKEN_PY") or "python3"
    token_helper = current_app.config.get("TOKEN_HELPER")
    tokenizer_dir = current_app.config.get("TOKENIZER_DIR")
    max_len = int(current_app.config.get("TOKEN_MAX_LEN", 77))

    if not token_helper or not os.path.isfile(token_helper):
        return jsonify({"ok": False, "error": "TOKEN_HELPER missing"}), 500
    if not tokenizer_dir or not os.path.isdir(tokenizer_dir):
        return jsonify({"ok": False, "error": "TOKENIZER_DIR missing"}), 500

    cmd = [
        token_py, token_helper,
        "--tokenizer_dir", tokenizer_dir,
        "--prompt", prompt,
        "--negative", negative,
        "--max_len", str(max_len),
        "--sum",
    ]

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if p.returncode != 0:
            return jsonify({"ok": False, "error": p.stderr.strip() or "token helper failed"}), 500
        out = (p.stdout or "").strip()
        return jsonify(json.loads(out))
    except Exception as ex:
        return jsonify({"ok": False, "error": str(ex)}), 500
