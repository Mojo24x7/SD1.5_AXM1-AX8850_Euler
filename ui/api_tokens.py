import os
import json
import subprocess
from flask import Blueprint, request, jsonify, current_app

bp_tokens = Blueprint("tokens", __name__)

def _cfg():
    token_py = current_app.config.get("TOKEN_PY") or "python3"
    token_helper = current_app.config.get("TOKEN_HELPER")
    tokenizer_dir = current_app.config.get("TOKENIZER_DIR")
    max_len = int(current_app.config.get("TOKEN_MAX_LEN", 77))

    if not token_helper or not os.path.isfile(token_helper):
        return None, ("TOKEN_HELPER missing", 500)
    if not tokenizer_dir or not os.path.isdir(tokenizer_dir):
        return None, ("TOKENIZER_DIR missing", 500)

    return (token_py, token_helper, tokenizer_dir, max_len), None

def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
    if p.returncode != 0:
        return None, (p.stderr.strip() or "token helper failed", 500)
    out = (p.stdout or "").strip()
    return json.loads(out), None

@bp_tokens.post("/api/tokens/count")
def api_tokens_count():
    cfg = request.json or {}
    prompt = cfg.get("prompt", "")
    negative = cfg.get("negative", "")

    c, err = _cfg()
    if err:
        msg, code = err
        return jsonify({"ok": False, "error": msg}), code
    token_py, token_helper, tokenizer_dir, max_len = c

    cmd = [
        token_py, token_helper,
        "--tokenizer_dir", tokenizer_dir,
        "--mode", "count",
        "--prompt", prompt,
        "--negative", negative,
        "--max_len", str(max_len),
        "--sum",
    ]
    out, err2 = _run(cmd)
    if err2:
        msg, code = err2
        return jsonify({"ok": False, "error": msg}), code
    return jsonify(out)

@bp_tokens.post("/api/tokens/analyze")
def api_tokens_analyze():
    cfg = request.json or {}
    text = cfg.get("text", "")

    c, err = _cfg()
    if err:
        msg, code = err
        return jsonify({"ok": False, "error": msg}), code
    token_py, token_helper, tokenizer_dir, max_len = c

    cmd = [
        token_py, token_helper,
        "--tokenizer_dir", tokenizer_dir,
        "--mode", "analyze",
        "--text", text,
        "--max_len", str(max_len),
    ]
    out, err2 = _run(cmd)
    if err2:
        msg, code = err2
        return jsonify({"ok": False, "error": msg}), code
    return jsonify(out)

@bp_tokens.post("/api/tokens/trim")
def api_tokens_trim():
    cfg = request.json or {}
    text = cfg.get("text", "")

    c, err = _cfg()
    if err:
        msg, code = err
        return jsonify({"ok": False, "error": msg}), code
    token_py, token_helper, tokenizer_dir, max_len = c

    cmd = [
        token_py, token_helper,
        "--tokenizer_dir", tokenizer_dir,
        "--mode", "trim",
        "--text", text,
        "--max_len", str(max_len),
    ]
    out, err2 = _run(cmd)
    if err2:
        msg, code = err2
        return jsonify({"ok": False, "error": msg}), code
    return jsonify(out)

