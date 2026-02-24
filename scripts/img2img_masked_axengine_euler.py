#!/usr/bin/env python3
import argparse, json, os, time, contextlib
import numpy as np
from PIL import Image

import torch
from diffusers import EulerDiscreteScheduler
import axengine

SD15_LATENT_SCALING = 0.18215

TE_IN_INPUT_IDS = "input_ids"
UNET_IN_SAMPLE = "sample"
UNET_IN_T = "t"
UNET_IN_EHS = "encoder_hidden_states"
VAE_DEC_IN_X = "x"
VAE_ENC_IN_IMAGE = "image_sample"

def now_ms(): return time.perf_counter()*1000.0
def fmt_ms(ms): return f"{ms:.1f} ms" if ms < 1000 else f"{ms/1000:.2f} s"
def log(s): print(s, flush=True)

@contextlib.contextmanager
def suppress_stdout_stderr(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

def load_scheduler(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return EulerDiscreteScheduler.from_config(cfg)

def preprocess_image_512(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((512,512), resample=Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr * 2.0) - 1.0
    arr = np.transpose(arr, (2,0,1))
    return arr[None, ...]  # [1,3,512,512]

def load_mask_512(path: str, blur: int = 3) -> np.ndarray:
    """
    Returns mask in latent space resolution [1,1,64,64] float32 in [0,1].
    White=edit, Black=keep.
    """
    m = Image.open(path).convert("L").resize((512,512), resample=Image.BILINEAR)
    if blur and blur > 0:
        # simple blur using resize trick (fast + no extra deps)
        small = m.resize((512//blur, 512//blur), resample=Image.BILINEAR)
        m = small.resize((512,512), resample=Image.BILINEAR)

    arr = np.array(m).astype(np.float32) / 255.0  # [0,1]
    # downsample to latent resolution
    m64 = Image.fromarray((arr*255).astype(np.uint8)).resize((64,64), resample=Image.BILINEAR)
    arr64 = np.array(m64).astype(np.float32) / 255.0
    arr64 = arr64[None, None, ...]  # [1,1,64,64]
    return np.clip(arr64, 0.0, 1.0)

def postprocess(img_nchw: np.ndarray) -> Image.Image:
    x = np.clip(img_nchw, -1, 1)
    x = (x + 1) * 0.5
    x = (x*255).round().astype(np.uint8)
    x = np.transpose(x[0], (1,2,0))
    return Image.fromarray(x, "RGB")

def tokenize(tokenizer_dir, prompt):
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(tokenizer_dir)
    ids = tok(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np")["input_ids"]
    return ids.astype(np.int32)

def run_te(te, ids):
    return te.run(None, {TE_IN_INPUT_IDS: ids.astype(np.int32)})[0].astype(np.float32)

def run_ve(ve, img_nchw, seed):
    z = ve.run(None, {VAE_ENC_IN_IMAGE: img_nchw.astype(np.float32)})[0].astype(np.float32)  # [1,8,64,64]
    mean, logvar = np.split(z, 2, axis=1)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(mean.shape).astype(np.float32)
    std = np.exp(0.5*logvar).astype(np.float32)
    lat = mean + std*eps
    return (lat * SD15_LATENT_SCALING).astype(np.float32)

def run_unet(unet, sample, t_i32, ehs):
    return unet.run(None, {
        UNET_IN_SAMPLE: sample.astype(np.float32),
        UNET_IN_T: t_i32.astype(np.int32),
        UNET_IN_EHS: ehs.astype(np.float32),
    })[0].astype(np.float32)

def run_vd(vd, lat_scaled):
    lat_in = (lat_scaled / SD15_LATENT_SCALING).astype(np.float32)
    return vd.run(None, {VAE_DEC_IN_X: lat_in})[0].astype(np.float32)

def safe_close(sess):
    try:
        if sess is not None and hasattr(sess, "close"):
            sess.close()
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=os.path.expanduser("~/axm1_sd/sd15_euler512"))
    ap.add_argument("--init_image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--strength", type=float, default=0.75)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=93)
    ap.add_argument("--mask_blur", type=int, default=3)
    ap.add_argument("--out", default="img2img_masked_out.png")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--quiet_axengine", action="store_true")
    args = ap.parse_args()

    t0 = now_ms()
    log("== MASKED IMG2IMG AX-M1 (AX8850) / Euler SD1.5 ==")
    log(f"init_image: {args.init_image}")
    log(f"mask: {args.mask}  (white=edit, black=keep) blur={args.mask_blur}")
    log(f"steps={args.steps} strength={args.strength} cfg={args.guidance_scale} seed={args.seed}")
    log(f"out: {args.out}")

    axmodels = os.path.join(args.base_dir, "axmodels")
    support  = os.path.join(args.base_dir, "support")

    te_path = os.path.join(axmodels, "sd15_text_encoder_sim.axmodel")
    unet_path = os.path.join(axmodels, "unet.axmodel")
    ve_path = os.path.join(axmodels, "vae_encoder.axmodel")
    vd_path = os.path.join(axmodels, "vae_decoder.axmodel")

    tokenizer_dir = os.path.join(support, "tokenizer")
    sched_cfg = os.path.join(support, "scheduler", "scheduler_config.json")

    torch.manual_seed(args.seed)

    scheduler = load_scheduler(sched_cfg)
    scheduler.set_timesteps(args.steps)
    timesteps = scheduler.timesteps

    # start timestep
    start_idx = int((1.0 - args.strength) * args.steps)
    start_idx = max(0, min(args.steps - 1, start_idx))
    t_start = timesteps[start_idx]

    # inputs
    img_nchw = preprocess_image_512(args.init_image)
    mask64 = load_mask_512(args.mask, blur=args.mask_blur)  # [1,1,64,64]
    mask64 = np.repeat(mask64, 4, axis=1)                   # [1,4,64,64]
    inv_mask64 = 1.0 - mask64

    cond_ids = tokenize(tokenizer_dir, args.prompt)
    uncond_ids = tokenize(tokenizer_dir, args.negative if args.negative else "")

    te = unet = ve = vd = None
    try:
        with suppress_stdout_stderr(args.quiet_axengine):
            te = axengine.InferenceSession(te_path, providers=["AXCLRTExecutionProvider"])
            ve = axengine.InferenceSession(ve_path, providers=["AXCLRTExecutionProvider"])
            unet = axengine.InferenceSession(unet_path, providers=["AXCLRTExecutionProvider"])
            vd = axengine.InferenceSession(vd_path, providers=["AXCLRTExecutionProvider"])

        cond = run_te(te, cond_ids)       # [1,77,768]
        uncond = run_te(te, uncond_ids)

        # base latents from image
        latents0 = run_ve(ve, img_nchw, seed=args.seed)  # [1,4,64,64]

        # noise at t_start
        rng = np.random.default_rng(args.seed)
        noise = rng.standard_normal(latents0.shape).astype(np.float32)

        lat0_t = torch.from_numpy(latents0)
        noise_t = torch.from_numpy(noise)
        latents = scheduler.add_noise(lat0_t, noise_t, t_start.clone().reshape(1)).numpy().astype(np.float32)

        # IMPORTANT: also compute the "noised original latents" each step for blending
        # We will enforce outside-mask region to remain from original image at the same noise level.
        log("== denoise loop (masked latent blend) ==")
        denoise_ms = 0.0
        steps_run = args.steps - start_idx

        for k, i in enumerate(range(start_idx, args.steps), start=1):
            st0 = now_ms()
            t = timesteps[i]
            t_i32 = np.array([int(t.item())], dtype=np.int32)

            # compute original latents at current t for preservation outside mask
            lat0_noised = scheduler.add_noise(lat0_t, noise_t, t.clone().reshape(1)).numpy().astype(np.float32)

            # scale input for Euler
            lat_scaled = scheduler.scale_model_input(torch.from_numpy(latents), t).numpy().astype(np.float32)

            # CFG two passes (batch1)
            eps_u = run_unet(unet, lat_scaled, t_i32, uncond)
            eps_c = run_unet(unet, lat_scaled, t_i32, cond)
            eps = eps_u + args.guidance_scale * (eps_c - eps_u)

            # step
            out = scheduler.step(torch.from_numpy(eps), t, torch.from_numpy(latents))
            latents = out.prev_sample.numpy().astype(np.float32)

            # MASKED BLEND: keep original outside mask, allow edits inside mask
            latents = (latents * mask64) + (lat0_noised * inv_mask64)

            step_ms = now_ms() - st0
            denoise_ms += step_ms
            if args.log_every > 0 and (k == 1 or k == steps_run or (k % args.log_every) == 0):
                avg = denoise_ms / k
                eta = avg * (steps_run - k)
                log(f"  step {k:>3}/{steps_run}  step={fmt_ms(step_ms)}  avg={fmt_ms(avg)}  eta={fmt_ms(eta)}")

        # decode
        img_out = run_vd(vd, latents)
        im = postprocess(img_out)
        im.save(args.out)
        log(f"WROTE: {args.out}")
        log(f"TOTAL: {fmt_ms(now_ms()-t0)}")

        # clean shutdown (avoid __del__ spam)
        safe_close(te); safe_close(ve); safe_close(unet); safe_close(vd)
        te = ve = unet = vd = None
        os._exit(0)

    finally:
        try:
            safe_close(te); safe_close(ve); safe_close(unet); safe_close(vd)
        except Exception:
            pass


if __name__ == "__main__":
    main()
