#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import contextlib
import numpy as np
from PIL import Image

import torch
from diffusers import EulerDiscreteScheduler

import axengine

SD15_LATENT_SCALING = 0.18215

# LOCKED model IO names (from your probe)
TE_IN_INPUT_IDS = "input_ids"
UNET_IN_SAMPLE = "sample"
UNET_IN_T = "t"  # MUST be int32 [1]
UNET_IN_EHS = "encoder_hidden_states"
VAE_DEC_IN_X = "x"
VAE_ENC_IN_IMAGE = "image_sample"


# ----------------- small logging helpers -----------------
def now_ms() -> float:
    return time.perf_counter() * 1000.0


def fmt_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms/1000.0:.2f} s"


def log(msg: str):
    print(msg, flush=True)


@contextlib.contextmanager
def suppress_stdout_stderr(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
# ---------------------------------------------------------


def load_scheduler(scheduler_config_path: str) -> EulerDiscreteScheduler:
    with open(scheduler_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return EulerDiscreteScheduler.from_config(cfg)


def preprocess_image_512(path: str) -> np.ndarray:
    """Returns float32 image in [-1,1], NCHW [1,3,512,512]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((512, 512), resample=Image.BICUBIC)

    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr * 2.0) - 1.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr[None, ...]  # NCHW


def postprocess_image(img_nchw: np.ndarray) -> Image.Image:
    """img_nchw float32 [-1,1] [1,3,512,512] -> PIL RGB."""
    x = np.clip(img_nchw, -1.0, 1.0)
    x = (x + 1.0) * 0.5
    x = (x * 255.0).round().astype(np.uint8)
    x = np.transpose(x[0], (1, 2, 0))  # HWC
    return Image.fromarray(x, mode="RGB")


def simple_tokenize_77(tokenizer_dir: str, prompt: str) -> np.ndarray:
    """Loads CLIPTokenizer from local folder support/tokenizer and returns input_ids int32 [1,77]."""
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(tokenizer_dir)
    ids = tok(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="np",
    )["input_ids"].astype(np.int32)
    return ids


def run_text_encoder_sess(te_sess, input_ids_i32: np.ndarray) -> np.ndarray:
    out = te_sess.run(None, {TE_IN_INPUT_IDS: input_ids_i32.astype(np.int32)})
    return out[0].astype(np.float32)  # [1,77,768]


def encode_vae_latents_sess(ve_sess, image_nchw: np.ndarray, seed: int) -> np.ndarray:
    """
    VAE encoder outputs latent_sample [1,8,64,64] float32 (mean+logvar).
    We sample z = mean + std * eps, then scale by 0.18215.
    """
    out = ve_sess.run(None, {VAE_ENC_IN_IMAGE: image_nchw.astype(np.float32)})
    z = out[0].astype(np.float32)  # [1,8,64,64]

    if z.ndim != 4 or z.shape[1] != 8:
        raise RuntimeError(f"Unexpected VAE encoder output: {z.shape} (expected [1,8,64,64])")

    mean, logvar = np.split(z, 2, axis=1)  # each [1,4,64,64]
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(mean.shape).astype(np.float32)
    std = np.exp(0.5 * logvar).astype(np.float32)
    latents = mean + std * eps  # [1,4,64,64]

    latents = (latents * SD15_LATENT_SCALING).astype(np.float32)
    return latents


def run_unet_sess(unet_sess, sample: np.ndarray, t_i32: np.ndarray, ehs: np.ndarray) -> np.ndarray:
    out = unet_sess.run(None, {
        UNET_IN_SAMPLE: sample.astype(np.float32),
        UNET_IN_T: t_i32.astype(np.int32),  # MUST be int32 [1]
        UNET_IN_EHS: ehs.astype(np.float32),
    })
    return out[0].astype(np.float32)  # [1,4,64,64]


def decode_vae_sess(vd_sess, latents_scaled: np.ndarray) -> np.ndarray:
    latents_in = (latents_scaled / SD15_LATENT_SCALING).astype(np.float32)
    out = vd_sess.run(None, {VAE_DEC_IN_X: latents_in})
    return out[0].astype(np.float32)  # [1,3,512,512] in [-1,1]


def safe_close(sess):
    # If axengine exposes close(), use it; otherwise do nothing
    try:
        if sess is not None and hasattr(sess, "close"):
            sess.close()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=os.path.expanduser("~/axm1_sd/sd15_euler512"))
    ap.add_argument("--init_image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--strength", type=float, default=0.45)
    ap.add_argument("--guidance_scale", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=93)
    ap.add_argument("--out", default="img2img_out.png")
    ap.add_argument("--log_every", type=int, default=1, help="Print progress every N denoise steps.")
    ap.add_argument("--quiet_axengine", action="store_true", help="Suppress axengine [INFO] spam.")
    args = ap.parse_args()

    assert 0.0 < args.strength <= 1.0
    assert args.steps >= 1

    base = args.base_dir
    axmodels = os.path.join(base, "axmodels")
    support = os.path.join(base, "support")

    text_encoder_path = os.path.join(axmodels, "sd15_text_encoder_sim.axmodel")
    unet_path = os.path.join(axmodels, "unet.axmodel")
    vae_decoder_path = os.path.join(axmodels, "vae_decoder.axmodel")
    vae_encoder_path = os.path.join(axmodels, "vae_encoder.axmodel")

    tokenizer_dir = os.path.join(support, "tokenizer")
    sched_cfg = os.path.join(support, "scheduler", "scheduler_config.json")

    t0 = now_ms()
    log("== IMG2IMG AX-M1 (AX8850) / Euler SD1.5 ==")
    log(f"init_image: {args.init_image}")
    log(f"steps={args.steps} strength={args.strength} cfg={args.guidance_scale} seed={args.seed}")
    log(f"out: {args.out}")

    # CPU-only math uses torch (scheduler)
    torch.manual_seed(args.seed)

    # Scheduler
    t_s = now_ms()
    scheduler = load_scheduler(sched_cfg)
    scheduler.set_timesteps(args.steps)
    timesteps = scheduler.timesteps  # torch tensor
    log(f"[1/7] scheduler loaded in {fmt_ms(now_ms()-t_s)}")

    # Tokenize
    t_tok = now_ms()
    cond_ids = simple_tokenize_77(tokenizer_dir, args.prompt)
    uncond_ids = simple_tokenize_77(tokenizer_dir, args.negative if args.negative else "")
    log(f"[2/7] tokenize done in {fmt_ms(now_ms()-t_tok)}")

    # Preprocess image
    t_img = now_ms()
    image_nchw = preprocess_image_512(args.init_image)
    log(f"[3/7] preprocess image done in {fmt_ms(now_ms()-t_img)}")

    te_sess = unet_sess = ve_sess = vd_sess = None
    try:
        # Create sessions ONCE
        t_load = now_ms()
        with suppress_stdout_stderr(args.quiet_axengine):
            te_sess = axengine.InferenceSession(text_encoder_path, providers=["AXCLRTExecutionProvider"])
            ve_sess = axengine.InferenceSession(vae_encoder_path, providers=["AXCLRTExecutionProvider"])
            unet_sess = axengine.InferenceSession(unet_path, providers=["AXCLRTExecutionProvider"])
            vd_sess = axengine.InferenceSession(vae_decoder_path, providers=["AXCLRTExecutionProvider"])
        log(f"[4/7] axengine sessions loaded in {fmt_ms(now_ms()-t_load)}")

        # Text embeddings
        t_te = now_ms()
        cond = run_text_encoder_sess(te_sess, cond_ids)       # [1,77,768]
        uncond = run_text_encoder_sess(te_sess, uncond_ids)   # [1,77,768]
        log(f"[5/7] text_encoder done in {fmt_ms(now_ms()-t_te)}")

        # Encode init image -> latents (scaled)
        t_ve = now_ms()
        latents = encode_vae_latents_sess(ve_sess, image_nchw, seed=args.seed)  # [1,4,64,64]
        log(f"[6/7] vae_encoder done in {fmt_ms(now_ms()-t_ve)}")

        # Start timestep based on strength
        start_idx = int((1.0 - args.strength) * args.steps)
        start_idx = max(0, min(args.steps - 1, start_idx))
        t_start = timesteps[start_idx]  # torch scalar

        # Add noise at t_start (use exact timestep value; do NOT int-cast)
        t_noise = now_ms()
        rng = np.random.default_rng(args.seed)
        noise = rng.standard_normal(latents.shape).astype(np.float32)

        latents_t = torch.from_numpy(latents)
        noise_t = torch.from_numpy(noise)
        latents = scheduler.add_noise(latents_t, noise_t, t_start.clone().reshape(1)).numpy().astype(np.float32)
        log(f"[7/7] add_noise done in {fmt_ms(now_ms()-t_noise)}")
        log("== denoise loop ==")

        # Denoise loop with progress + timing
        denoise_ms_total = 0.0
        steps_run = args.steps - start_idx
        for k, i in enumerate(range(start_idx, args.steps), start=1):
            step_t0 = now_ms()
            t = timesteps[i]  # torch scalar (often float)
            t_i32 = np.array([int(t.item())], dtype=np.int32)  # for UNet AXCLRT

            # EulerDiscreteScheduler expects scale_model_input before UNet
            latents_scaled_t = scheduler.scale_model_input(torch.from_numpy(latents), t)
            latents_in = latents_scaled_t.numpy().astype(np.float32)

            # CFG with batch1 UNet: run uncond + cond separately
            noise_pred_uncond = run_unet_sess(unet_sess, latents_in, t_i32, uncond)
            noise_pred_text = run_unet_sess(unet_sess, latents_in, t_i32, cond)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Euler update
            latents_step = torch.from_numpy(latents)
            noise_pred_step = torch.from_numpy(noise_pred.astype(np.float32))
            step_out = scheduler.step(noise_pred_step, t, latents_step)
            latents = step_out.prev_sample.numpy().astype(np.float32)

            step_ms = now_ms() - step_t0
            denoise_ms_total += step_ms

            if args.log_every > 0 and (k == 1 or k == steps_run or (k % args.log_every) == 0):
                avg = denoise_ms_total / k
                eta = avg * (steps_run - k)
                log(f"  step {k:>3}/{steps_run}  t={float(t.item()):.4g}  step={fmt_ms(step_ms)}  avg={fmt_ms(avg)}  eta={fmt_ms(eta)}")

        log(f"denoise total: {fmt_ms(denoise_ms_total)}  avg/step: {fmt_ms(denoise_ms_total/max(1,steps_run))}")

        # Decode
        t_dec = now_ms()
        img_out = decode_vae_sess(vd_sess, latents)
        log(f"decode: {fmt_ms(now_ms()-t_dec)}")

        # Save
        t_save = now_ms()
        im = postprocess_image(img_out)
        im.save(args.out)
        log(f"save: {fmt_ms(now_ms()-t_save)}")
        log(f"WROTE: {args.out}")
        log(f"TOTAL: {fmt_ms(now_ms()-t0)}")

        # Clean shutdown to avoid axengine __del__ spam:
        # 1) try close() if available
        safe_close(te_sess); safe_close(ve_sess); safe_close(unet_sess); safe_close(vd_sess)
        te_sess = ve_sess = unet_sess = vd_sess = None

        # 2) hard-exit on success so buggy __del__ can't run during interpreter teardown
        os._exit(0)

    finally:
        # If we errored, do best-effort cleanup without hard exit (so you see traceback)
        try:
            safe_close(te_sess); safe_close(ve_sess); safe_close(unet_sess); safe_close(vd_sess)
        except Exception:
            pass


if __name__ == "__main__":
    main()
