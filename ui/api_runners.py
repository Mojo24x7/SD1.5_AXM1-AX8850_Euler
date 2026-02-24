import os
from flask import Blueprint, request, jsonify, current_app
from .ui_app import build_run_dir, resolve_scripts, safe_mkdir, default_run_name
from .job_manager import spawn_job

bp_run = Blueprint("run", __name__)

def _abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def _join(*parts) -> str:
    return os.path.join(*parts)

def _file_exists(p: str) -> bool:
    return bool(p) and os.path.isfile(p)

def _dir_exists(p: str) -> bool:
    return bool(p) and os.path.isdir(p)

def _validate_support_paths(base_dir: str):
    """
    Validate local pack layout so we never hit HF online downloads.
    """
    sched_dir = _join(base_dir, "support", "scheduler")
    sched_cfg = _join(sched_dir, "scheduler_config.json")
    vae_cfg = _join(base_dir, "support", "vae", "config.json")
    tok_dir = _join(base_dir, "support", "tokenizer")

    missing = []
    if not _dir_exists(tok_dir):
        missing.append(f"missing tokenizer_dir: {tok_dir}")
    if not _file_exists(sched_cfg):
        missing.append(f"missing scheduler_config.json: {sched_cfg}")
    if not _file_exists(vae_cfg):
        missing.append(f"missing vae config.json: {vae_cfg}")

    return missing, tok_dir, sched_dir, vae_cfg

@bp_run.post("/api/run/txt2img")
def api_run_txt2img():
    cfg = request.json or {}

    base_dir = _abspath(cfg.get("base_dir") or current_app.config["BASE_DIR"])
    out_root = _abspath(cfg.get("output_root") or current_app.config["OUT_ROOT"])
    run_name = cfg.get("run_name") or default_run_name("txt2img")

    prompt = cfg.get("prompt", "")
    negative = cfg.get("negative", "")
    steps = int(cfg.get("steps", 30))
    guidance = float(cfg.get("cfg", 7.5))
    seed = int(cfg.get("seed", 1))

    npu_live = bool(cfg.get("npu_live", True))
    npu_poll = int(cfg.get("npu_poll_every", 5))
    npu_record = bool(cfg.get("npu_record", True))
    embed_meta = bool(cfg.get("embed_metadata", True))

    # Validate local support paths so we never hit HF
    missing, tok_dir, sched_dir, vae_cfg = _validate_support_paths(base_dir)
    if missing:
        return jsonify({"ok": False, "error": " / ".join(missing)}), 400

    run_dir = build_run_dir(out_root, run_name)
    safe_mkdir(run_dir)

    scripts = resolve_scripts(base_dir, current_app.config["SCRIPTS_DIR"])
    venv_py = current_app.config["VENV_PY"]

    out_png = _join(run_dir, "out.png")
    out_json = _join(run_dir, "out.json")

    # IMPORTANT:
    # - Use "-u" so stdout is unbuffered (continuous logs in UI)
    # - Also enforce PYTHONUNBUFFERED via job env
    cmd = [
        venv_py, "-u", scripts["txt2img"],
        "--weights_dir", _join(base_dir, "axmodels"),
        "--tokenizer_dir", tok_dir,
        "--scheduler_dir", sched_dir,
        "--vae_config", vae_cfg,
        "--te", "sd15_text_encoder_sim.axmodel",
        "--unet", "unet.axmodel",
        "--vae", "vae_decoder.axmodel",
        "--prompt", prompt,
        "--negative", negative,
        "--steps", str(steps),
        "--guidance", str(guidance),
        "--seed", str(seed),
        "--log_every", "1",       # smoother UI; you can change later
        "--eta_window", "8",
        "--warmup", "1",
    ]

    if npu_live:
        cmd += ["--npu_live", "--npu_poll_every", str(max(1, npu_poll))]
        if npu_record:
            cmd += ["--npu_record"]

    if embed_meta:
        cmd += ["--embed_metadata"]

    cmd += ["--metadata_json", out_json, "--out", out_png]

    job = spawn_job("txt2img", run_dir, cmd, env={"PYTHONUNBUFFERED": "1"})
    return jsonify({"ok": True, "job_id": job.job_id, "run_dir": run_dir})

@bp_run.post("/api/run/img2img")
def api_run_img2img():
    cfg = request.json or {}

    base_dir = _abspath(cfg.get("base_dir") or current_app.config["BASE_DIR"])
    out_root = _abspath(cfg.get("output_root") or current_app.config["OUT_ROOT"])
    run_name = cfg.get("run_name") or default_run_name("img2img")

    init_image = _abspath(cfg.get("init_image", ""))
    mask_path = _abspath(cfg.get("mask_path", ""))

    if not _file_exists(init_image):
        return jsonify({"ok": False, "error": f"init_image not found: {init_image}"}), 400
    if not _file_exists(mask_path):
        return jsonify({"ok": False, "error": f"mask_path not found: {mask_path}"}), 400

    prompt = cfg.get("prompt", "")
    negative = cfg.get("negative", "")
    steps = int(cfg.get("steps", 30))
    strength = float(cfg.get("strength", 0.85))
    guidance = float(cfg.get("cfg", 7.5))
    seed = int(cfg.get("seed", 93))
    mask_blur = int(cfg.get("mask_blur", 3))

    run_dir = build_run_dir(out_root, run_name)
    safe_mkdir(run_dir)

    scripts = resolve_scripts(base_dir, current_app.config["SCRIPTS_DIR"])
    venv_py = current_app.config["VENV_PY"]

    out_png = _join(run_dir, "edit_out.png")

    cmd = [
        venv_py, "-u", scripts["img2img"],
        "--base_dir", base_dir,
        "--init_image", init_image,
        "--mask", mask_path,
        "--prompt", prompt,
        "--negative", negative,
        "--steps", str(steps),
        "--strength", str(strength),
        "--guidance_scale", str(guidance),
        "--seed", str(seed),
        "--mask_blur", str(mask_blur),
        "--out", out_png,
        "--log_every", "1",
    ]

    job = spawn_job("img2img", run_dir, cmd, env={"PYTHONUNBUFFERED": "1"})
    return jsonify({"ok": True, "job_id": job.job_id, "run_dir": run_dir})

@bp_run.post("/api/run/mask")
def api_run_mask():
    cfg = request.json or {}

    base_dir = _abspath(cfg.get("base_dir") or current_app.config["BASE_DIR"])
    out_root = _abspath(cfg.get("output_root") or current_app.config["OUT_ROOT"])
    run_name = cfg.get("run_name") or default_run_name("mask")

    init_image = _abspath(cfg.get("init_image", ""))
    if not _file_exists(init_image):
        return jsonify({"ok": False, "error": f"init_image not found: {init_image}"}), 400

    mask_mode = cfg.get("mask_mode", "grabcut")
    rect = cfg.get("rect", "0,280,512,232")
    grabcut_iters = int(cfg.get("grabcut_iters", 5))
    hsv_low = cfg.get("hsv_low", "0,0,200")
    hsv_high = cfg.get("hsv_high", "179,80,255")
    roi = cfg.get("roi", "")
    preset = cfg.get("preset", "shirt")
    invert_mask = bool(cfg.get("invert_mask", False))
    erode = int(cfg.get("erode", 0))
    dilate = int(cfg.get("dilate", 3))
    blur = int(cfg.get("blur", 7))
    protect_face = bool(cfg.get("protect_face", False))
    face_pad = float(cfg.get("face_pad", 0.25))

    run_dir = build_run_dir(out_root, run_name)
    safe_mkdir(run_dir)

    scripts = resolve_scripts(base_dir, current_app.config["SCRIPTS_DIR"])
    sys_py = current_app.config["SYS_PY"]

    mask_out = _join(run_dir, "mask_debug.png")

    cmd = [
        sys_py, "-u", scripts["mask"],
        "--base_dir", base_dir,
        "--init_image", init_image,
        "--mask_mode", mask_mode,
        "--rect", rect,
        "--grabcut_iters", str(grabcut_iters),
        "--hsv_low", hsv_low,
        "--hsv_high", hsv_high,
        "--roi", roi,
        "--preset", preset,
        "--erode", str(erode),
        "--dilate", str(dilate),
        "--blur", str(blur),
        "--mask_out", mask_out,
    ]
    if invert_mask:
        cmd += ["--invert_mask"]
    if protect_face:
        cmd += ["--protect_face", "--face_pad", str(face_pad)]

    job = spawn_job("mask", run_dir, cmd, env={"PYTHONUNBUFFERED": "1"})
    return jsonify({"ok": True, "job_id": job.job_id, "run_dir": run_dir, "mask_path": mask_out})
