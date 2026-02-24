import os, re
from flask import Flask
from typing import Dict

DEFAULT_VENV_PY = os.path.expanduser("~/axm1_sd/.venv_sd15/bin/python")
DEFAULT_SYS_PY = "/usr/bin/python3"

DEFAULT_SCRIPTS: Dict[str, str] = {
    "txt2img": "txt2img_axengine_euler.py",
    "img2img": "img2img_masked_axengine_euler.py",
    "mask":   "mask_gen.py",
}

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def default_run_name(prefix: str) -> str:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"

def build_run_dir(out_root: str, run_name: str) -> str:
    run_name = (run_name or "").strip() or default_run_name("run")
    run_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", run_name)
    return os.path.join(out_root, run_name)

def resolve_scripts(base_dir: str, scripts_dir: str) -> Dict[str, str]:
    scripts = {}
    for k, fn in DEFAULT_SCRIPTS.items():
        p1 = os.path.join(scripts_dir, fn)             # prefer scripts/
        p2 = os.path.join(base_dir, "scripts", fn)     # fallback
        if os.path.isfile(p1):
            scripts[k] = p1
        elif os.path.isfile(p2):
            scripts[k] = p2
        else:
            scripts[k] = p1
    return scripts

def create_app(
    base_dir: str,
    out_root: str,
    venv_py: str = DEFAULT_VENV_PY,
    system_py: str = DEFAULT_SYS_PY,
) -> Flask:
    app = Flask(__name__)
    base_dir = os.path.expanduser(base_dir)
    out_root = os.path.expanduser(out_root)

    app.config["BASE_DIR"] = base_dir
    app.config["OUT_ROOT"] = out_root
    app.config["VENV_PY"] = os.path.expanduser(venv_py)
    app.config["SYS_PY"] = system_py
    app.config["SCRIPTS_DIR"] = os.path.dirname(os.path.abspath(__file__))  # scripts/

    safe_mkdir(out_root)
    return app
