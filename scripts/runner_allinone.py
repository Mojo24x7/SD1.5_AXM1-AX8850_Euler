#!/usr/bin/env python3
"""
runner_allinone.py

Single entrypoint that orchestrates:
  - txt2img (venv .venv_sd15)
  - img2img masked (venv .venv_sd15)
  - auto-mask generation (SYSTEM python cv2) + img2img masked (venv)

Also:
  - Creates a run folder under output_root/run_name
  - Captures stdout/stderr into console.log
  - Writes run.json with args, paths, exit codes
  - (Placeholder) optional Real-ESRGAN upscaling hook for later

Assumptions (your current layout):
  base_dir = ~/axm1_sd/sd15_euler512
  txt2img script = base_dir/scripts/txt2img_axengine_euler.py
  img2img masked script = base_dir/scripts/img2img_masked_axengine_euler.py
  support paths exist under base_dir/support
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ------------------ utils ------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def iso_now():
    return datetime.now().isoformat(timespec="seconds")

def sanitize_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s or "run"

def default_run_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def which(p: str) -> Optional[str]:
    if os.path.isfile(p) and os.access(p, os.X_OK):
        return p
    for d in os.environ.get("PATH", "").split(":"):
        cand = os.path.join(d, p)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None

def run_subprocess(cmd: List[str], cwd: Optional[str], log_path: str) -> Tuple[int, str]:
    """
    Run command, tee stdout+stderr to console and to log file.
    Returns (returncode, combined_tail_for_summary).
    """
    ensure_dir(os.path.dirname(log_path) or ".")
    tail_lines: List[str] = []
    max_tail = 120

    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write("\n" + "=" * 80 + "\n")
        logf.write(f"[RUN] {iso_now()}\n")
        logf.write("[CMD] " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        logf.write("=" * 80 + "\n")

        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert p.stdout is not None
        for line in p.stdout:
            # Print to console
            sys.stdout.write(line)
            sys.stdout.flush()

            # Log to file
            logf.write(line)
            logf.flush()

            tail_lines.append(line.rstrip("\n"))
            if len(tail_lines) > max_tail:
                tail_lines = tail_lines[-max_tail:]

        rc = p.wait()
        logf.write(f"\n[EXIT] rc={rc}\n")
        return rc, "\n".join(tail_lines)

def build_txt2img_cmd(venv_py: str, base_dir: str, run_dir: str, args: argparse.Namespace) -> List[str]:
    scripts_dir = os.path.join(base_dir, "scripts")
    script = os.path.join(scripts_dir, "txt2img_axengine_euler.py")

    weights_dir = args.weights_dir or os.path.join(base_dir, "axmodels")
    tokenizer_dir = args.tokenizer_dir or os.path.join(base_dir, "support", "tokenizer")
    scheduler_dir = args.scheduler_dir or os.path.join(base_dir, "support", "scheduler")
    vae_config = args.vae_config or os.path.join(base_dir, "support", "vae", "config.json")

    out_png = os.path.join(run_dir, "out.png")
    out_json = os.path.join(run_dir, "run.json")

    cmd = [
        venv_py, script,
        "--weights_dir", weights_dir,
        "--tokenizer_dir", tokenizer_dir,
        "--te", args.te,
        "--unet", args.unet,
        "--vae", args.vae,
        "--prompt", args.prompt,
        "--negative", args.negative or "",
        "--steps", str(args.steps),
        "--guidance", str(args.cfg),
        "--seed", str(args.seed),
        "--log_every", str(args.log_every),
        "--eta_window", str(args.eta_window),
        "--warmup", str(args.warmup),
        "--scheduler_dir", scheduler_dir,
        "--vae_config", vae_config,
        "--metadata_json", out_json,
        "--embed_metadata",
        "--out", out_png,
    ]

    if args.npu_live:
        cmd += ["--npu_live"]
        if args.npu_poll_every > 0:
            cmd += ["--npu_poll_every", str(args.npu_poll_every)]
        if args.npu_record:
            cmd += ["--npu_record"]

    if args.no_md5:
        cmd += ["--no_md5"]

    return cmd

def build_img2img_masked_cmd(venv_py: str, base_dir: str, run_dir: str, mask_path: str, args: argparse.Namespace) -> List[str]:
    scripts_dir = os.path.join(base_dir, "scripts")
    script = os.path.join(scripts_dir, "img2img_masked_axengine_euler.py")

    out_png = os.path.join(run_dir, "out.png")

    cmd = [
        venv_py, script,
        "--base_dir", base_dir,
        "--init_image", args.init_image,
        "--mask", mask_path,
        "--prompt", args.prompt,
        "--negative", args.negative or "",
        "--steps", str(args.steps),
        "--strength", str(args.strength),
        "--guidance_scale", str(args.cfg),
        "--seed", str(args.seed),
        "--mask_blur", str(args.mask_blur),
        "--out", out_png,
        "--log_every", str(args.log_every),
    ]

    if args.quiet_axengine:
        cmd += ["--quiet_axengine"]

    return cmd

def build_mask_gen_cmd(sys_py: str, mask_gen_path: str, run_dir: str, args: argparse.Namespace) -> List[str]:
    cmd = [
        sys_py, mask_gen_path,
        "--init_image", args.init_image,
        "--out_dir", run_dir,
        "--task", args.task,
    ]

    # Raw overrides (optional)
    if args.mask_method:
        cmd += ["--method", args.mask_method]
    if args.mask_file:
        cmd += ["--mask_file", args.mask_file]
    if args.rect:
        cmd += ["--rect", args.rect]
    if args.roi:
        cmd += ["--roi", args.roi]
    if args.grabcut_iters >= 0:
        cmd += ["--grabcut_iters", str(args.grabcut_iters)]
    if args.hsv_low:
        cmd += ["--hsv_low", args.hsv_low]
    if args.hsv_high:
        cmd += ["--hsv_high", args.hsv_high]
    if args.preset:
        cmd += ["--preset", args.preset]

    if args.invert_mask:
        cmd += ["--invert"]

    # refine
    if args.erode >= 0:
        cmd += ["--erode", str(args.erode)]
    if args.dilate >= 0:
        cmd += ["--dilate", str(args.dilate)]
    if args.blur >= 0:
        cmd += ["--blur", str(args.blur)]

    # protect
    if args.protect_face:
        cmd += ["--protect_face"]
    if args.face_pad >= 0:
        cmd += ["--face_pad", str(args.face_pad)]

    return cmd

def maybe_realesrgan_placeholder(run_dir: str, args: argparse.Namespace, console_log: str) -> Dict:
    """
    Placeholder for later:
      - If args.upscale is set, we record intention and optionally run a binary if present.
      - For now, we do NOT fail the run if realesrgan isn't installed.
    """
    if not args.upscale:
        return {"enabled": False}

    out_png = os.path.join(run_dir, "out.png")
    out_up = os.path.join(run_dir, "out_upscaled.png")

    info = {
        "enabled": True,
        "factor": args.upscale_factor,
        "model": args.upscale_model,
        "bin": args.realesrgan_bin,
        "input": out_png,
        "output": out_up,
        "status": "planned",
    }

    # If user provided a realesrgan binary and it exists, we can try to run it later.
    # Keep it as a placeholder to implement properly when you decide which Real-ESRGAN runner you want.
    bin_path = args.realesrgan_bin.strip() if args.realesrgan_bin else ""
    if bin_path:
        resolved = which(bin_path) or (bin_path if os.path.exists(bin_path) else None)
        if resolved and os.path.isfile(resolved):
            info["status"] = "placeholder_not_executed"
            # We intentionally don't execute now to avoid guessing the command line you want.
            # Later we'll implement:
            #   realesrgan-ncnn-vulkan -i out.png -o out_upscaled.png -n <model> -s <factor>
        else:
            info["status"] = "bin_not_found"

    # write a note file
    with open(os.path.join(run_dir, "upscale_note.txt"), "w", encoding="utf-8") as f:
        f.write("Real-ESRGAN hook is enabled but not implemented yet.\n")
        f.write(json.dumps(info, indent=2))

    return info


# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(description="All-in-one runner for AX-M1 SD (txt2img / img2img masked / auto-mask).")

    ap.add_argument("--mode", required=True, choices=["txt2img", "img2img_masked", "img2img_auto_mask"],
                    help="Which pipeline to run")

    ap.add_argument("--base_dir", default=os.path.expanduser("~/axm1_sd/sd15_euler512"),
                    help="Project base dir containing scripts/, axmodels/, support/")

    # Interpreters
    ap.add_argument("--venv_python", default=os.path.expanduser("~/axm1_sd/.venv_sd15/bin/python"),
                    help="Python for SD inference (NO cv2 needed here)")
    ap.add_argument("--system_python", default="/usr/bin/python3",
                    help="System python containing cv2")
    ap.add_argument("--mask_gen", default="mask_gen.py",
                    help="Path to mask_gen.py (system-python tool). Can be relative or absolute.")

    # IO
    ap.add_argument("--output_root", default="",
                    help="Root output dir. If empty, uses <base_dir>/out")
    ap.add_argument("--run_name", default="",
                    help="Run folder name. If empty, uses timestamp. (UI can template this.)")

    # Common SD args
    ap.add_argument("--prompt", default="", help="Prompt (required for all modes)")
    ap.add_argument("--negative", default="", help="Negative prompt")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=7.5)

    # txt2img specific
    ap.add_argument("--weights_dir", default="")
    ap.add_argument("--tokenizer_dir", default="")
    ap.add_argument("--scheduler_dir", default="")
    ap.add_argument("--vae_config", default="")
    ap.add_argument("--te", default="sd15_text_encoder_sim.axmodel")
    ap.add_argument("--unet", default="unet.axmodel")
    ap.add_argument("--vae", default="vae_decoder.axmodel")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--eta_window", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--npu_live", action="store_true")
    ap.add_argument("--npu_poll_every", type=int, default=0)
    ap.add_argument("--npu_record", action="store_true")
    ap.add_argument("--no_md5", action="store_true")

    # img2img specific
    ap.add_argument("--init_image", default="", help="Init image path (required for img2img modes)")
    ap.add_argument("--strength", type=float, default=0.80)
    ap.add_argument("--mask_blur", type=int, default=3)
    ap.add_argument("--quiet_axengine", action="store_true")

    # img2img masked (import)
    ap.add_argument("--mask", default="", help="Mask path (required for img2img_masked)")

    # img2img auto-mask (human task + overrides)
    ap.add_argument("--task", default="keep_face_change_clothes",
                    help="Human goal for auto mask (e.g. keep_face_change_clothes, keep_person_change_background, remove_object)")
    ap.add_argument("--mask_method", default="", choices=["", "grabcut", "file", "rect", "color", "preset"],
                    help="Override mask method (advanced)")
    ap.add_argument("--mask_file", default="", help="Mask file to import (advanced)")
    ap.add_argument("--rect", default="", help="x,y,w,h (advanced)")
    ap.add_argument("--roi", default="", help="x,y,w,h for hsv (advanced)")
    ap.add_argument("--grabcut_iters", type=int, default=-1)
    ap.add_argument("--hsv_low", default="")
    ap.add_argument("--hsv_high", default="")
    ap.add_argument("--preset", default="")
    ap.add_argument("--invert_mask", action="store_true")
    ap.add_argument("--erode", type=int, default=-1)
    ap.add_argument("--dilate", type=int, default=-1)
    ap.add_argument("--blur", type=int, default=-1)
    ap.add_argument("--protect_face", action="store_true")
    ap.add_argument("--face_pad", type=float, default=-1.0)

    # Real-ESRGAN placeholder
    ap.add_argument("--upscale", action="store_true", help="Enable Real-ESRGAN hook (not implemented yet)")
    ap.add_argument("--upscale_factor", type=int, default=2)
    ap.add_argument("--upscale_model", default="realesrgan-x4plus")
    ap.add_argument("--realesrgan_bin", default="", help="Path to realesrgan binary (later)")

    args = ap.parse_args()

    if not args.prompt.strip():
        raise SystemExit("--prompt is required")

    # output root default
    output_root = args.output_root.strip() or os.path.join(args.base_dir, "out")
    ensure_dir(output_root)

    run_name = sanitize_name(args.run_name) if args.run_name.strip() else default_run_name()
    run_dir = os.path.join(output_root, run_name)
    ensure_dir(run_dir)

    console_log = os.path.join(run_dir, "console.log")

    # Resolve mask_gen path
    mask_gen_path = args.mask_gen
    if not os.path.isabs(mask_gen_path):
        # prefer same directory as runner if relative
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, mask_gen_path)
        if os.path.isfile(cand):
            mask_gen_path = cand

    # Basic run metadata
    run_meta: Dict = {
        "started": iso_now(),
        "mode": args.mode,
        "base_dir": args.base_dir,
        "run_dir": run_dir,
        "output_root": output_root,
        "run_name": run_name,
        "prompt": args.prompt,
        "negative": args.negative,
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "strength": args.strength,
        "init_image": args.init_image,
        "mask": args.mask,
        "task": args.task,
        "interpreters": {
            "venv_python": args.venv_python,
            "system_python": args.system_python,
            "mask_gen": mask_gen_path,
        },
        "artifacts": {},
        "exit": {},
    }
    write_json(os.path.join(run_dir, "run_meta_start.json"), run_meta)

    # Sanity checks
    if not os.path.isfile(args.venv_python):
        raise SystemExit(f"venv_python not found: {args.venv_python}")
    if not os.path.isfile(args.system_python):
        raise SystemExit(f"system_python not found: {args.system_python}")

    # Run pipeline
    rc_mask = None
    rc_sd = None
    tail = ""

    if args.mode == "txt2img":
        cmd = build_txt2img_cmd(args.venv_python, args.base_dir, run_dir, args)
        rc_sd, tail = run_subprocess(cmd, cwd=args.base_dir, log_path=console_log)
        run_meta["artifacts"]["out_png"] = os.path.join(run_dir, "out.png")
        run_meta["artifacts"]["out_json"] = os.path.join(run_dir, "run.json")

    elif args.mode == "img2img_masked":
        if not args.init_image:
            raise SystemExit("--init_image required for img2img_masked")
        if not args.mask:
            raise SystemExit("--mask required for img2img_masked")
        cmd = build_img2img_masked_cmd(args.venv_python, args.base_dir, run_dir, args.mask, args)
        rc_sd, tail = run_subprocess(cmd, cwd=args.base_dir, log_path=console_log)
        run_meta["artifacts"]["out_png"] = os.path.join(run_dir, "out.png")
        run_meta["artifacts"]["mask_used"] = args.mask

    elif args.mode == "img2img_auto_mask":
        if not args.init_image:
            raise SystemExit("--init_image required for img2img_auto_mask")

        # Step 1: mask generation (system python)
        cmd_mask = build_mask_gen_cmd(args.system_python, mask_gen_path, run_dir, args)
        rc_mask, _tail_mask = run_subprocess(cmd_mask, cwd=args.base_dir, log_path=console_log)

        mask_path = os.path.join(run_dir, "mask.png")
        if rc_mask != 0 or (not os.path.isfile(mask_path)):
            run_meta["exit"]["mask_rc"] = rc_mask
            run_meta["exit"]["sd_rc"] = None
            run_meta["finished"] = iso_now()
            write_json(os.path.join(run_dir, "run_meta_end.json"), run_meta)
            raise SystemExit(f"Mask generation failed rc={rc_mask}")

        # Step 2: masked img2img (venv)
        cmd_sd = build_img2img_masked_cmd(args.venv_python, args.base_dir, run_dir, mask_path, args)
        rc_sd, tail = run_subprocess(cmd_sd, cwd=args.base_dir, log_path=console_log)

        run_meta["artifacts"]["mask_used"] = mask_path
        run_meta["artifacts"]["mask_debug"] = os.path.join(run_dir, "mask_debug.png")
        run_meta["artifacts"]["overlay"] = os.path.join(run_dir, "overlay.png")
        run_meta["artifacts"]["mask_gen_json"] = os.path.join(run_dir, "mask_gen.json")
        run_meta["artifacts"]["out_png"] = os.path.join(run_dir, "out.png")

    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    # Real-ESRGAN hook (placeholder)
    run_meta["upscale"] = maybe_realesrgan_placeholder(run_dir, args, console_log)

    # Finalize metadata
    run_meta["exit"]["mask_rc"] = rc_mask
    run_meta["exit"]["sd_rc"] = rc_sd
    run_meta["finished"] = iso_now()

    write_json(os.path.join(run_dir, "run_meta_end.json"), run_meta)

    print("\n== runner_allinone ==")
    print(f"run_dir: {run_dir}")
    print(f"console_log: {console_log}")
    if rc_mask is not None:
        print(f"mask_rc: {rc_mask}")
    print(f"sd_rc: {rc_sd}")
    print("DONE.")


if __name__ == "__main__":
    main()
