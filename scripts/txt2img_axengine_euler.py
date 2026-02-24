#!/usr/bin/env python3
'''python3 scripts/txt2img_axengine_euler.py \
  --weights_dir ./axmodels \
  --tokenizer_dir ./support/tokenizer \
  --te sd15_text_encoder_sim.axmodel \
  --unet unet.axmodel \
  --vae vae_decoder.axmodel \
  --prompt "a realistic portrait photo, sharp focus, natural lighting, 85mm lens" \
  --negative "blurry, low quality, deformed, bad anatomy" \
  --steps 30 --guidance 7.5 --seed 1 \
  --log_every 2 --eta_window 8 --warmup 1 \
  --npu_live --npu_poll_every 2 --npu_record \
  --embed_metadata --metadata_json ./runs/run.json \
  --out ./runs/out.png
'''

import argparse
import os
import time
import hashlib
import json
import numpy as np
from collections import deque
from datetime import datetime
import subprocess
import re

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import torch
from transformers import CLIPTokenizer
from diffusers import EulerDiscreteScheduler

from axengine import InferenceSession


# -------------------- utils --------------------

def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def stat(name: str, x: np.ndarray) -> str:
    y = x.astype(np.float64, copy=False)
    return (f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
            f"min={y.min():.6f} max={y.max():.6f} mean={y.mean():.6f} std={y.std():.6f}")


def pick_name(io_list, keywords):
    for io in io_list:
        n = (getattr(io, "name", "") or "")
        nl = n.lower()
        for k in keywords:
            if k in nl:
                return n
    return None


def describe_session(tag: str, sess: InferenceSession):
    print(f"\n[DBG] === {tag} ===")
    ins = sess.get_inputs()
    outs = sess.get_outputs()
    print("[DBG] Inputs:")
    for i, x in enumerate(ins):
        print(f"  [{i}] name={getattr(x,'name',None)} shape={getattr(x,'shape',None)} type={getattr(x,'type',None)}")
    print("[DBG] Outputs:")
    for i, x in enumerate(outs):
        print(f"  [{i}] name={getattr(x,'name',None)} shape={getattr(x,'shape',None)} type={getattr(x,'type',None)}")


def load_scaling_factor(vae_config_path: str) -> float:
    # SD1.5 usually: 0.18215
    try:
        with open(vae_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        sf = cfg.get("scaling_factor", 0.18215)
        return float(sf)
    except Exception:
        return 0.18215


def fmt_hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    s = int(seconds + 0.5)
    h = s // 3600
    m = (s % 3600) // 60
    s2 = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s2:02d}"
    return f"{m:02d}:{s2:02d}"


# -------------------- NPU stats (axcl-smi) --------------------

def run_cmd_text(cmd_list):
    try:
        p = subprocess.run(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False
        )
        return p.returncode, (p.stdout or "").strip()
    except Exception as e:
        return 999, f"EXCEPTION: {e}"


def parse_axcl_smi_compact(text: str):
    """
    Parse axcl-smi ASCII table (best-effort) into a compact dict.
    Expected pattern (two lines per device):
      |    0  AX8850 ... | ... |  176 MiB /  945 MiB |
      |   --   46C ...   | 1%  100% | 1603 MiB / 7040 MiB |
    Returns None if not found.
    """
    lines = text.splitlines()
    idx = None
    for i, ln in enumerate(lines):
        if "AX8850" in ln and "MiB" in ln and "|" in ln:
            idx = i
            break
    if idx is None:
        return None

    mem_used = mem_total = None
    m = re.search(r"\|\s*([0-9]+)\s*MiB\s*/\s*([0-9]+)\s*MiB\s*\|", lines[idx])
    if m:
        mem_used = int(m.group(1))
        mem_total = int(m.group(2))

    temp_c = cpu_pct = npu_pct = None
    cmm_used = cmm_total = None

    if idx + 1 < len(lines):
        ln2 = lines[idx + 1]
        mt = re.search(r"\|\s*--\s+([0-9]+)C", ln2)
        if mt:
            temp_c = int(mt.group(1))

        mc = re.search(r"\|\s*([0-9]+)%\s+([0-9]+)%\s*\|", ln2)
        if mc:
            cpu_pct = int(mc.group(1))
            npu_pct = int(mc.group(2))

        mm = re.search(r"\|\s*([0-9]+)\s*MiB\s*/\s*([0-9]+)\s*MiB\s*\|", ln2)
        if mm:
            cmm_used = int(mm.group(1))
            cmm_total = int(mm.group(2))

    return {
        "temp_c": temp_c,
        "cpu_pct": cpu_pct,
        "npu_pct": npu_pct,
        "npu_mem_used_mib": mem_used,
        "npu_mem_total_mib": mem_total,
        "cmm_used_mib": cmm_used,
        "cmm_total_mib": cmm_total,
    }


def get_npu_compact(cmd_list):
    rc, out = run_cmd_text(cmd_list)
    if rc != 0:
        return {"rc": int(rc), "err": out}
    d = parse_axcl_smi_compact(out)
    if d is None:
        return {"rc": int(rc), "err": "Could not parse axcl-smi output"}
    d["rc"] = int(rc)
    return d


def fmt_npu_compact(d: dict) -> str:
    if not d or d.get("rc", 0) != 0:
        return "NPU: (smi err)"
    if "err" in d:
        return f"NPU: (smi parse err: {d.get('err')})"

    parts = []
    if d.get("temp_c") is not None:
        parts.append(f"T={d['temp_c']}C")
    if d.get("cpu_pct") is not None:
        parts.append(f"CPU={d['cpu_pct']}%")
    if d.get("npu_pct") is not None:
        parts.append(f"NPU={d['npu_pct']}%")
    if d.get("npu_mem_used_mib") is not None and d.get("npu_mem_total_mib") is not None:
        parts.append(f"Mem={d['npu_mem_used_mib']}/{d['npu_mem_total_mib']}MiB")
    if d.get("cmm_used_mib") is not None and d.get("cmm_total_mib") is not None:
        parts.append(f"CMM={d['cmm_used_mib']}/{d['cmm_total_mib']}MiB")
    return " | ".join(parts) if parts else "NPU: (no fields)"


# -------------------- SD helpers --------------------

def tokenize(tok: CLIPTokenizer, text: str, max_len: int = 77) -> np.ndarray:
    out = tok(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    # TE axmodel wants int32
    return out["input_ids"].astype(np.int32, copy=False)


def run_text_encoder(te: InferenceSession, input_ids_i32: np.ndarray, debug=False) -> np.ndarray:
    ins = te.get_inputs()
    outs = te.get_outputs()

    in_name = pick_name(ins, ["input_ids", "ids"]) or getattr(ins[0], "name", "input_ids")
    out_name = getattr(outs[0], "name", None)

    if debug:
        print("[DBG] TE in_name:", in_name, "out_name:", out_name)

    y = te.run(
        output_names=[out_name] if out_name else None,
        input_feed={in_name: input_ids_i32},
    )[0]

    if y.dtype != np.float32:
        y = y.astype(np.float32)
    return y


def build_unet_feed(unet: InferenceSession, sample: np.ndarray, timestep_val: int, ehs: np.ndarray, timestep_dtype: str):
    ins = unet.get_inputs()

    name_sample = pick_name(ins, ["sample", "latent"]) or getattr(ins[0], "name", "sample")
    name_t = pick_name(ins, ["timestep", "time", "t"]) or (getattr(ins[1], "name", "t") if len(ins) > 1 else "t")
    name_ehs = pick_name(ins, ["encoder_hidden_states", "hidden", "context", "cond"]) or (
        getattr(ins[2], "name", "encoder_hidden_states") if len(ins) > 2 else "encoder_hidden_states"
    )

    if timestep_dtype == "int32":
        t_arr = np.array([timestep_val], dtype=np.int32)
    else:
        t_arr = np.array([float(timestep_val)], dtype=np.float32)

    return {
        name_sample: sample.astype(np.float32, copy=False),
        name_t: t_arr,
        name_ehs: ehs.astype(np.float32, copy=False),
    }


def run_unet_cfg(unet: InferenceSession, sample_np: np.ndarray, timestep_val: int,
                uncond_ehs: np.ndarray, cond_ehs: np.ndarray,
                guidance: float, timestep_dtype: str) -> np.ndarray:
    feed_u = build_unet_feed(unet, sample_np, timestep_val, uncond_ehs, timestep_dtype)
    feed_c = build_unet_feed(unet, sample_np, timestep_val, cond_ehs, timestep_dtype)

    out_u = unet.run(output_names=None, input_feed=feed_u)[0]
    out_c = unet.run(output_names=None, input_feed=feed_c)[0]

    if out_u.dtype != np.float32:
        out_u = out_u.astype(np.float32)
    if out_c.dtype != np.float32:
        out_c = out_c.astype(np.float32)

    return out_u + guidance * (out_c - out_u)


def run_vae_decode(vae: InferenceSession, latents_np: np.ndarray, debug=False) -> np.ndarray:
    ins = vae.get_inputs()
    outs = vae.get_outputs()

    in_name = pick_name(ins, ["x", "latent"]) or getattr(ins[0], "name", "x")
    out_name = getattr(outs[0], "name", None)

    if debug:
        print("[DBG] VAE in_name:", in_name, "out_name:", out_name)
        print("[DBG]", stat("vae.latents_in", latents_np))

    y = vae.run(
        output_names=[out_name] if out_name else None,
        input_feed={in_name: latents_np.astype(np.float32, copy=False)},
    )[0]

    if y.dtype != np.float32:
        y = y.astype(np.float32)
    return y


def vae_to_pil(img_nchw: np.ndarray) -> Image.Image:
    x = img_nchw[0]  # CHW
    x = (x / 2.0 + 0.5)
    x = np.clip(x, 0.0, 1.0)
    x = (np.transpose(x, (1, 2, 0)) * 255.0).round().astype(np.uint8)
    return Image.fromarray(x, "RGB")


def save_png_with_metadata(pil_img: Image.Image, out_path: str, metadata: dict, embed: bool = True):
    ensure_dir(os.path.dirname(out_path) or ".")
    if not embed:
        pil_img.save(out_path)
        return

    pnginfo = PngInfo()
    pnginfo.add_text("sd_run_json", json.dumps(metadata, ensure_ascii=False))
    pnginfo.add_text("prompt", metadata.get("prompt", ""))
    pnginfo.add_text("negative", metadata.get("negative", ""))
    pnginfo.add_text("seed", str(metadata.get("seed", "")))
    pnginfo.add_text("steps", str(metadata.get("steps", "")))
    pnginfo.add_text("guidance", str(metadata.get("guidance", "")))
    pil_img.save(out_path, pnginfo=pnginfo)


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--tokenizer_dir", required=True)

    ap.add_argument("--te", default="sd15_text_encoder_sim.axmodel")
    ap.add_argument("--unet", default="unet.axmodel")
    ap.add_argument("--vae", default="vae_decoder.axmodel")

    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)

    ap.add_argument("--timestep_dtype", choices=["int32", "float32"], default="int32")
    ap.add_argument("--provider", default="AXCLRTExecutionProvider")

    ap.add_argument("--vae_config", default="./support/vae/config.json",
                    help="Path to VAE config.json (for scaling_factor)")
    ap.add_argument("--scheduler_dir", default="./support/scheduler",
                    help="Folder containing scheduler_config.json")

    ap.add_argument("--out", required=True)

    # Progress / ETA
    ap.add_argument("--log_every", type=int, default=5, help="Print progress every N steps (default 5)")
    ap.add_argument("--eta_window", type=int, default=8, help="Rolling window for ETA smoothing (default 8)")
    ap.add_argument("--warmup", type=int, default=0, help="Run UNet warmup iterations before denoise (default 0)")

    # Faster startup
    ap.add_argument("--no_md5", action="store_true", help="Skip md5 (faster startup)")

    # Metadata output
    ap.add_argument("--metadata_json", default="", help="If set, write run metadata JSON to this path")
    ap.add_argument("--embed_metadata", action="store_true", help="Embed metadata into the output PNG")

    # NPU: compact live line
    ap.add_argument("--npu_live", action="store_true",
                    help="Poll axcl-smi and print compact NPU stats inline with step logs")
    ap.add_argument("--npu_poll_every", type=int, default=0,
                    help="Poll NPU every N steps (0 = same as log_every)")
    ap.add_argument("--npu_record", action="store_true",
                    help="Record NPU samples into metadata JSON (only at poll points)")
    ap.add_argument("--npu_smi_cmd", default="axcl-smi",
                    help="Command to run for NPU stats (default: axcl-smi)")

    # Debug / intermediate
    ap.add_argument("--debug_io", action="store_true")
    ap.add_argument("--save_intermediate_every", type=int, default=0)

    args = ap.parse_args()

    torch.set_num_threads(1)

    te_path = os.path.join(args.weights_dir, args.te)
    unet_path = os.path.join(args.weights_dir, args.unet)
    vae_path = os.path.join(args.weights_dir, args.vae)

    for p in [te_path, unet_path, vae_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    scaling_factor = load_scaling_factor(args.vae_config)

    # Scheduler: load from your pack to reduce mismatch risk
    scheduler = EulerDiscreteScheduler.from_pretrained(args.scheduler_dir)
    scheduler.set_timesteps(args.steps)

    h8 = args.height // 8
    w8 = args.width // 8

    run_meta = {
        "run_started": datetime.now().isoformat(timespec="seconds"),
        "provider": args.provider,
        "timestep_dtype": args.timestep_dtype,
        "height": int(args.height),
        "width": int(args.width),
        "latent_h": int(h8),
        "latent_w": int(w8),
        "steps": int(args.steps),
        "guidance": float(args.guidance),
        "seed": int(args.seed),
        "prompt": args.prompt,
        "negative": args.negative,
        "scaling_factor": float(scaling_factor),
        "scheduler": scheduler.__class__.__name__,
        "weights": {
            "te": os.path.basename(te_path),
            "unet": os.path.basename(unet_path),
            "vae": os.path.basename(vae_path),
        },
    }

    # NPU poll setup
    npu_cmd = args.npu_smi_cmd.split()
    poll_every = args.npu_poll_every if args.npu_poll_every > 0 else max(1, args.log_every)
    run_meta["npu"] = {
        "live": bool(args.npu_live),
        "poll_every": int(poll_every),
        "cmd": " ".join(npu_cmd),
        # We can reliably show VNPU type from axengine prints already, but not "number of NPUs"
        # without extra documented APIs. We'll record what we *can* read from axcl-smi.
    }
    if args.npu_record:
        run_meta["npu_samples"] = []

    # Header (keep clean)
    try:
        avail = InferenceSession.get_available_providers() if hasattr(InferenceSession, "get_available_providers") else None
        if avail is not None:
            print("[INFO] Available providers: ", avail)
    except Exception:
        pass

    print("[INFO] provider:", args.provider)
    print(f"[INFO] size: {args.width}x{args.height}  latents: (1,4,{h8},{w8})")
    print("[INFO] steps:", args.steps, "guidance:", args.guidance, "seed:", args.seed)
    print("[INFO] scheduler:", run_meta["scheduler"])
    print("[INFO] VAE scaling_factor:", scaling_factor, "(decode with latents / scaling_factor)")

    print("[INFO] model files:")
    for p in [te_path, unet_path, vae_path]:
        if args.no_md5:
            print(f"  - {p} size={os.path.getsize(p)}")
        else:
            print(f"  - {p} size={os.path.getsize(p)} md5={md5sum(p)}")

    tok = CLIPTokenizer.from_pretrained(args.tokenizer_dir)
    cond_ids = tokenize(tok, args.prompt)
    uncond_ids = tokenize(tok, args.negative if args.negative else "")

    te = InferenceSession(path_or_bytes=te_path, providers=[args.provider])
    unet = InferenceSession(path_or_bytes=unet_path, providers=[args.provider])
    vae = InferenceSession(path_or_bytes=vae_path, providers=[args.provider])

    if args.debug_io:
        describe_session("TEXT_ENCODER", te)
        describe_session("UNET", unet)
        describe_session("VAE_DECODER", vae)

    # TEXT ENCODER
    print("[INFO] running text encoder (cond + uncond)...")
    t0 = time.perf_counter()
    cond_ehs = run_text_encoder(te, cond_ids, debug=args.debug_io)
    uncond_ehs = run_text_encoder(te, uncond_ids, debug=args.debug_io)
    te_s = time.perf_counter() - t0
    print(f"[INFO] TE done in {te_s:.3f}s  cond={cond_ehs.shape} uncond={uncond_ehs.shape}")
    run_meta["time_text_encoder_s"] = float(te_s)

    # INIT LATENTS
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    latents = torch.randn((1, 4, h8, w8), generator=g, dtype=torch.float32)
    latents = latents * scheduler.init_noise_sigma

    out_path = args.out
    out_dir = os.path.dirname(out_path) or "."
    ensure_dir(out_dir)

    # OPTIONAL WARMUP
    if args.warmup > 0:
        print(f"[INFO] warmup: running UNet {args.warmup} iteration(s) to stabilize timing...")
        t_w = int(scheduler.timesteps[0].item())
        latent_in = scheduler.scale_model_input(latents, scheduler.timesteps[0])
        sample_np = latent_in.cpu().numpy().astype(np.float32, copy=False)
        for _ in range(args.warmup):
            _ = run_unet_cfg(
                unet=unet,
                sample_np=sample_np,
                timestep_val=t_w,
                uncond_ehs=uncond_ehs,
                cond_ehs=cond_ehs,
                guidance=float(args.guidance),
                timestep_dtype=args.timestep_dtype,
            )
        print("[INFO] warmup done.")

    # DENOISE WITH ETA + compact NPU line
    print("[INFO] denoising...")
    t_start = time.perf_counter()
    step_times = deque(maxlen=max(1, int(args.eta_window)))

    last_npu = None
    last_npu_step = 0

    def maybe_poll_npu(step_idx_1based, elapsed_s):
        nonlocal last_npu, last_npu_step
        if not args.npu_live:
            return None
        if (step_idx_1based == 1) or (step_idx_1based - last_npu_step >= poll_every) or (step_idx_1based == args.steps):
            d = get_npu_compact(npu_cmd)
            last_npu = d
            last_npu_step = step_idx_1based
            if args.npu_record:
                run_meta["npu_samples"].append({
                    "step": int(step_idx_1based),
                    "elapsed_s": float(elapsed_s),
                    "stats": d,
                    "time": datetime.now().isoformat(timespec="seconds")
                })
        return last_npu

    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            step_idx = i + 1
            step_t0 = time.perf_counter()

            t_int = int(t.item())
            latent_in = scheduler.scale_model_input(latents, t)

            eps = run_unet_cfg(
                unet=unet,
                sample_np=latent_in.cpu().numpy().astype(np.float32, copy=False),
                timestep_val=t_int,
                uncond_ehs=uncond_ehs,
                cond_ehs=cond_ehs,
                guidance=float(args.guidance),
                timestep_dtype=args.timestep_dtype,
            )

            eps_t = torch.from_numpy(eps).to(dtype=torch.float32)
            step_out = scheduler.step(eps_t, t, latents)
            latents = step_out.prev_sample

            dt = time.perf_counter() - step_t0
            step_times.append(dt)

            do_log = (i == 0) or (step_idx % max(1, args.log_every) == 0) or (step_idx == args.steps)
            if do_log:
                avg = sum(step_times) / len(step_times)
                sps = (1.0 / avg) if avg > 1e-9 else 0.0
                remaining = args.steps - step_idx
                eta_s = remaining * avg
                elapsed_s = time.perf_counter() - t_start

                npu_d = maybe_poll_npu(step_idx, elapsed_s)
                npu_txt = f" | {fmt_npu_compact(npu_d)}" if args.npu_live else ""

                print(
                    f"[INFO] step {step_idx:02d}/{args.steps} t={t_int} "
                    f"dt={dt:.3f}s avg={avg:.3f}s ({sps:.2f} step/s) "
                    f"elapsed={fmt_hms(elapsed_s)} ETA={fmt_hms(eta_s)} "
                    f"latents mean={latents.mean().item():.6f} std={latents.std().item():.6f}"
                    f"{npu_txt}"
                )

            # Optional intermediate decode
            if args.save_intermediate_every and (step_idx % args.save_intermediate_every == 0):
                lat_np = (latents / scaling_factor).cpu().numpy().astype(np.float32, copy=False)
                img_np = run_vae_decode(vae, lat_np)
                pil_mid = vae_to_pil(img_np)
                mid_path = os.path.join(out_dir, f"intermediate_{step_idx:03d}.png")
                save_png_with_metadata(
                    pil_mid,
                    mid_path,
                    {**run_meta, "intermediate_step": int(step_idx)},
                    embed=args.embed_metadata,
                )

    denoise_s = time.perf_counter() - t_start
    print(f"[INFO] denoise total: {denoise_s:.3f}s")
    run_meta["time_denoise_s"] = float(denoise_s)

    # VAE DECODE
    print("[INFO] decoding with VAE (latents / scaling_factor)...")
    t1 = time.perf_counter()
    latents_np = (latents / scaling_factor).cpu().numpy().astype(np.float32, copy=False)
    img_np = run_vae_decode(vae, latents_np, debug=args.debug_io)
    vae_s = time.perf_counter() - t1
    run_meta["time_vae_decode_s"] = float(vae_s)
    print(f"[INFO] VAE decode done in {vae_s:.3f}s")

    if args.debug_io:
        print("[DBG]", stat("vae_out", img_np))

    pil = vae_to_pil(img_np)

    # Write JSON metadata if requested
    if args.metadata_json:
        ensure_dir(os.path.dirname(args.metadata_json) or ".")
        with open(args.metadata_json, "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2, ensure_ascii=False)
        print("[OK] saved metadata:", args.metadata_json)

    # Save output (optionally with embedded metadata)
    save_png_with_metadata(pil, out_path, run_meta, embed=args.embed_metadata)
    print("[OK] saved:", out_path)


if __name__ == "__main__":
    main()
