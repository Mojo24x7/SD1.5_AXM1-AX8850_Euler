#!/usr/bin/env python3
import os
import argparse

from .ui_app import create_app
from .api_fs import bp_fs
from .api_runs import bp_runs
from .api_jobs import bp_jobs
from .api_runners import bp_run
from .api_tokens import bp_tokens
from .ui_page import bp_ui


def _resolve_existing_tokenizer_dir(base_dir: str) -> str:
    """
    Prefer:
      base_dir/support/tokenizer/{vocab.json, merges.txt}
    Fallback:
      base_dir/support/tokenizer/tokenizer/{vocab.json, merges.txt}
    """
    cand1 = os.path.join(base_dir, "support", "tokenizer")
    cand2 = os.path.join(base_dir, "support", "tokenizer", "tokenizer")

    def ok(d: str) -> bool:
        return (os.path.isfile(os.path.join(d, "vocab.json")) and
                os.path.isfile(os.path.join(d, "merges.txt")))

    if ok(cand1):
        return cand1
    if ok(cand2):
        return cand2
    # return cand1 as default; api_tokens will error with a clear message
    return cand1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=os.path.expanduser("~/axm1_sd/sd15_euler512"))
    ap.add_argument("--out_root", default="", help="Default output root. If empty uses base_dir/out")
    ap.add_argument("--venv_py", default=os.path.expanduser("~/axm1_sd/.venv_sd15/bin/python"))
    ap.add_argument("--system_py", default="/usr/bin/python3")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)

    # Token counting helper (pure python, no transformers/torch)
    ap.add_argument("--token_py", default="", help="Python to run token helper (default: venv_py)")
    ap.add_argument("--token_helper", default="", help="Path to token_count_clip_bpe.py (default: alongside this server.py)")
    ap.add_argument("--tokenizer_dir", default="", help="Tokenizer dir with vocab.json+merges.txt (default: base_dir/support/tokenizer)")
    ap.add_argument("--token_max_len", type=int, default=77)

    args = ap.parse_args()

    base_dir = os.path.expanduser(args.base_dir)
    out_root = os.path.expanduser(args.out_root) if args.out_root else os.path.join(base_dir, "out")

    # Create app (your ui_app.py should set BASE_DIR/OUT_ROOT/VENV_PY/SYS_PY)
    app = create_app(base_dir=base_dir, out_root=out_root, venv_py=args.venv_py, system_py=args.system_py)

    # ---- Token config (IMPORTANT: no transformers) ----
    here = os.path.dirname(os.path.abspath(__file__))

    token_py = os.path.expanduser(args.token_py) if args.token_py else os.path.expanduser(args.venv_py)
    token_helper = os.path.expanduser(args.token_helper) if args.token_helper else os.path.join(here, "token_count_clip_bpe.py")
    tokenizer_dir = os.path.expanduser(args.tokenizer_dir) if args.tokenizer_dir else _resolve_existing_tokenizer_dir(base_dir)

    app.config["TOKEN_PY"] = token_py
    app.config["TOKEN_HELPER"] = token_helper
    app.config["TOKENIZER_DIR"] = tokenizer_dir
    app.config["TOKEN_MAX_LEN"] = int(args.token_max_len)

    # register modules
    app.register_blueprint(bp_ui)
    app.register_blueprint(bp_fs)
    app.register_blueprint(bp_runs)
    app.register_blueprint(bp_jobs)
    app.register_blueprint(bp_run)
    app.register_blueprint(bp_tokens)

    print(f"[UI] base_dir:      {base_dir}")
    print(f"[UI] out_root:      {out_root}")
    print(f"[UI] venv_py:       {app.config.get('VENV_PY')}")
    print(f"[UI] sys_py:        {app.config.get('SYS_PY')}")
    print(f"[UI] tokenizer_dir: {app.config.get('TOKENIZER_DIR')}")
    print(f"[UI] token_py:      {app.config.get('TOKEN_PY')}")
    print(f"[UI] token_helper:  {app.config.get('TOKEN_HELPER')}")
    print(f"[UI] listening:     http://{args.host}:{args.port}/")

    # LAN-only dev server (ok for your use)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
