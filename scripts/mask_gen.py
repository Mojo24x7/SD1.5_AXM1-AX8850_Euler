#!/usr/bin/env python3
"""
mask_gen.py (SYSTEM PYTHON ONLY)

Purpose:
  Generate an inpainting-style mask (white=edit, black=keep) using system-installed OpenCV.

Outputs (inside run folder typically):
  - mask.png        : final mask used for diffusion (512x512 L)
  - mask_debug.png  : debug mask before/after refine
  - overlay.png     : visualization overlay of mask on image (for UI thumbnail)

Important:
  - This script must run with /usr/bin/python3 (system cv2).
  - No torch/diffusers/axengine dependencies.

Usage examples:
  /usr/bin/python3 mask_gen.py --init_image input.png --task keep_face_change_clothes --out_dir ./out/run1
  /usr/bin/python3 mask_gen.py --init_image input.png --method grabcut --rect 80,40,350,460 --invert --out_dir ./out/run2
"""

import argparse
import os
import re
import json
import time
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image

import cv2


# ------------------------- helpers -------------------------

def now_ms() -> float:
    return time.perf_counter() * 1000.0

def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def parse_rect(s: str) -> Tuple[int, int, int, int]:
    # "x,y,w,h"
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"rect must be 'x,y,w,h' got: {s}")
    x, y, w, h = [int(float(p)) for p in parts]
    return x, y, w, h

def parse_hsv(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"hsv must be 'H,S,V' got: {s}")
    return np.array([int(float(p)) for p in parts], dtype=np.uint8)

def load_image_rgb_512(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((512, 512), resample=Image.BICUBIC)
    return np.array(img, dtype=np.uint8)  # RGB

def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def save_mask_u8(path: str, mask_u8: np.ndarray):
    Image.fromarray(mask_u8, mode="L").save(path)

def make_overlay(rgb_512: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """
    Create an overlay preview: mask in a bright tint over image.
    """
    m = (mask_u8.astype(np.float32) / 255.0)[..., None]
    rgb = rgb_512.astype(np.float32)
    tint = np.array([255.0, 255.0, 255.0], dtype=np.float32)  # keep neutral (UI can colorize)
    out = rgb * (1.0 - 0.35 * m) + tint * (0.35 * m)
    return np.clip(out, 0, 255).astype(np.uint8)

def detect_face_boxes(img_bgr_512: np.ndarray) -> List[Tuple[int,int,int,int]]:
    gray = cv2.cvtColor(img_bgr_512, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if faces is None:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

def apply_face_protect(mask_u8: np.ndarray, img_bgr_512: np.ndarray, pad: float) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
    faces = detect_face_boxes(img_bgr_512)
    m = mask_u8.copy()
    for (x, y, w, h) in faces:
        px = int(w * pad)
        py = int(h * pad)
        x0 = clamp(x - px, 0, 512)
        y0 = clamp(y - py, 0, 512)
        x1 = clamp(x + w + px, 0, 512)
        y1 = clamp(y + h + py, 0, 512)
        m[y0:y1, x0:x1] = 0
    return m, faces

def morph_refine(mask_u8: np.ndarray, erode: int, dilate: int, blur: int) -> np.ndarray:
    m = mask_u8.copy()

    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
        m = cv2.erode(m, k, iterations=1)

    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        m = cv2.dilate(m, k, iterations=1)

    if blur > 0:
        k = max(3, blur | 1)  # odd
        m = cv2.GaussianBlur(m, (k, k), 0)

    return m

def mask_from_preset(preset: str) -> np.ndarray:
    """
    Simple presets (512x512):
      - shirt: lower torso
      - body: larger lower half
      - hair: upper portion
      - lower / upper
      - full: full image
    """
    m = np.zeros((512, 512), dtype=np.uint8)
    preset = preset.lower().strip()

    if preset == "shirt":
        m[280:512, 0:512] = 255
    elif preset == "body":
        m[240:512, 0:512] = 255
    elif preset == "hair":
        m[0:260, 0:512] = 255
    elif preset == "lower":
        m[256:512, 0:512] = 255
    elif preset == "upper":
        m[0:256, 0:512] = 255
    elif preset == "full":
        m[:, :] = 255
    else:
        raise ValueError(f"Unknown preset: {preset}")
    return m

def mask_from_rect(rect: Tuple[int,int,int,int]) -> np.ndarray:
    x, y, w, h = rect
    m = np.zeros((512, 512), dtype=np.uint8)
    x0 = clamp(x, 0, 512); y0 = clamp(y, 0, 512)
    x1 = clamp(x + w, 0, 512); y1 = clamp(y + h, 0, 512)
    m[y0:y1, x0:x1] = 255
    return m

def mask_grabcut(img_bgr_512: np.ndarray, rect: Tuple[int,int,int,int], iters: int) -> np.ndarray:
    x, y, w, h = rect
    mask = np.zeros(img_bgr_512.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img_bgr_512,
        mask,
        (x, y, w, h),
        bgdModel,
        fgdModel,
        int(iters),
        cv2.GC_INIT_WITH_RECT
    )
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    return out

def mask_color_hsv(img_bgr_512: np.ndarray, hsv_low: np.ndarray, hsv_high: np.ndarray, roi: Optional[Tuple[int,int,int,int]]) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr_512, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, hsv_low, hsv_high)  # u8

    if roi is not None:
        x, y, w, h = roi
        roi_m = np.zeros_like(m)
        x0 = clamp(x, 0, 512); y0 = clamp(y, 0, 512)
        x1 = clamp(x + w, 0, 512); y1 = clamp(y + h, 0, 512)
        roi_m[y0:y1, x0:x1] = 255
        m = cv2.bitwise_and(m, roi_m)

    return m

def mask_from_file(path: str) -> np.ndarray:
    m = Image.open(path).convert("L").resize((512, 512), resample=Image.BILINEAR)
    return np.array(m, dtype=np.uint8)


# -------------------- human task presets --------------------

@dataclass
class TaskDefaults:
    method: str
    preset: str = ""
    invert: bool = False
    rect: str = "0,0,512,512"
    grabcut_iters: int = 5
    hsv_low: str = "0,0,200"
    hsv_high: str = "179,80,255"
    roi: str = ""
    protect_face: bool = False
    face_pad: float = 0.25
    erode: int = 0
    dilate: int = 3
    blur: int = 7

def defaults_for_task(task: str) -> TaskDefaults:
    """
    Human-friendly tasks -> mask defaults.
    You can override everything via CLI flags.
    """
    t = (task or "").lower().strip()

    if t in ["keep_face_change_clothes", "change_clothes", "clothes"]:
        return TaskDefaults(method="preset", preset="shirt", protect_face=True, face_pad=0.30, dilate=3, blur=7)

    if t in ["keep_face_change_body", "change_body"]:
        return TaskDefaults(method="preset", preset="body", protect_face=True, face_pad=0.30, dilate=3, blur=7)

    if t in ["change_hair_only", "hair"]:
        return TaskDefaults(method="preset", preset="hair", protect_face=True, face_pad=0.20, dilate=2, blur=5)

    if t in ["keep_person_change_background", "change_background", "background"]:
        # Make mask for PERSON via grabcut, then invert to edit background
        return TaskDefaults(method="grabcut", rect="80,40,350,460", grabcut_iters=5,
                            invert=True, protect_face=True, face_pad=0.25, dilate=2, blur=9)

    if t in ["remove_object", "object_removal"]:
        # Typically manual/rect/import, but default to rect; user should provide --rect or --mask_file
        return TaskDefaults(method="rect", rect="180,180,160,160",
                            protect_face=False, dilate=1, blur=5)

    if t in ["replace_object", "object_replace"]:
        return TaskDefaults(method="rect", rect="180,180,160,160",
                            protect_face=False, dilate=2, blur=5)

    # fallback: user chooses method
    return TaskDefaults(method="preset", preset="full", protect_face=False, dilate=0, blur=0)


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate 512x512 edit mask using system OpenCV (white=edit).")
    ap.add_argument("--init_image", required=True, help="Input image path")
    ap.add_argument("--out_dir", required=True, help="Output directory for mask.png/mask_debug.png/overlay.png")

    # Human task layer
    ap.add_argument("--task", default="", help="Human goal (e.g., keep_face_change_clothes, keep_person_change_background, remove_object)")

    # Raw method layer
    ap.add_argument("--method", default="", choices=["", "grabcut", "file", "rect", "color", "preset"],
                    help="Mask method override. If empty, uses task defaults.")

    ap.add_argument("--mask_file", default="", help="Mask file path when method=file")
    ap.add_argument("--rect", default="", help="x,y,w,h (for grabcut/rect/roi). If empty uses task defaults.")
    ap.add_argument("--roi", default="", help="Optional ROI x,y,w,h for color mode")
    ap.add_argument("--grabcut_iters", type=int, default=-1)

    ap.add_argument("--hsv_low", default="", help="H,S,V (opencv H 0..179)")
    ap.add_argument("--hsv_high", default="", help="H,S,V")

    ap.add_argument("--preset", default="", help="shirt/body/hair/upper/lower/full")

    ap.add_argument("--invert", action="store_true", help="Invert mask (edit opposite region)")

    # Refinement
    ap.add_argument("--erode", type=int, default=-1)
    ap.add_argument("--dilate", type=int, default=-1)
    ap.add_argument("--blur", type=int, default=-1)

    # Protection
    ap.add_argument("--protect_face", action="store_true")
    ap.add_argument("--face_pad", type=float, default=-1.0)

    # Output names
    ap.add_argument("--mask_out", default="mask.png")
    ap.add_argument("--mask_debug_out", default="mask_debug.png")
    ap.add_argument("--overlay_out", default="overlay.png")

    args = ap.parse_args()

    t0 = now_ms()
    ensure_dir(args.out_dir)

    # Resolve defaults from task
    d = defaults_for_task(args.task)

    method = args.method.strip().lower() if args.method else d.method
    rect_s = args.rect.strip() if args.rect else d.rect
    preset = args.preset.strip().lower() if args.preset else d.preset
    grabcut_iters = args.grabcut_iters if args.grabcut_iters >= 0 else d.grabcut_iters
    hsv_low = args.hsv_low.strip() if args.hsv_low else d.hsv_low
    hsv_high = args.hsv_high.strip() if args.hsv_high else d.hsv_high
    roi_s = args.roi.strip() if args.roi else d.roi

    invert = bool(args.invert) if args.invert else bool(d.invert)

    erode = args.erode if args.erode >= 0 else d.erode
    dilate = args.dilate if args.dilate >= 0 else d.dilate
    blur = args.blur if args.blur >= 0 else d.blur

    protect_face = bool(args.protect_face) if args.protect_face else bool(d.protect_face)
    face_pad = args.face_pad if args.face_pad >= 0 else d.face_pad

    print("== mask_gen ==")
    print(f"init_image: {args.init_image}")
    print(f"task: {args.task or '(none)'}")
    print(f"method: {method} preset={preset} invert={invert}")
    print(f"rect: {rect_s} roi: {roi_s or '(none)'} iters={grabcut_iters}")
    print(f"hsv: low={hsv_low} high={hsv_high}")
    print(f"refine: erode={erode} dilate={dilate} blur={blur}")
    print(f"protect_face: {protect_face} face_pad={face_pad}")

    rgb = load_image_rgb_512(args.init_image)
    bgr = rgb_to_bgr(rgb)

    # Build base mask
    if method == "file":
        if not args.mask_file:
            raise SystemExit("--mask_file required for method=file")
        mask_u8 = mask_from_file(args.mask_file)

    elif method == "rect":
        rect = parse_rect(rect_s)
        mask_u8 = mask_from_rect(rect)

    elif method == "preset":
        if not preset:
            raise SystemExit("--preset required for method=preset (or provide --task)")
        mask_u8 = mask_from_preset(preset)

    elif method == "color":
        low = parse_hsv(hsv_low)
        high = parse_hsv(hsv_high)
        roi = parse_rect(roi_s) if roi_s else None
        mask_u8 = mask_color_hsv(bgr, low, high, roi)

    elif method == "grabcut":
        rect = parse_rect(rect_s)
        mask_u8 = mask_grabcut(bgr, rect, iters=grabcut_iters)

    else:
        raise SystemExit(f"Unknown method: {method}")

    # Face protect
    faces = []
    if protect_face:
        mask_u8, faces = apply_face_protect(mask_u8, bgr, pad=float(face_pad))
        print(f"faces_detected: {len(faces)}")

    # Invert if requested
    if invert:
        mask_u8 = (255 - mask_u8)

    # Save debug before refine (optional)
    debug_path = os.path.join(args.out_dir, args.mask_debug_out)
    save_mask_u8(debug_path, mask_u8)

    # Morphology refine
    mask_u8 = morph_refine(mask_u8, erode=erode, dilate=dilate, blur=blur)

    # Write final mask + overlay
    mask_path = os.path.join(args.out_dir, args.mask_out)
    save_mask_u8(mask_path, mask_u8)

    overlay = make_overlay(rgb, mask_u8)
    overlay_path = os.path.join(args.out_dir, args.overlay_out)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)

    # Write a small JSON summary (helps runner/UI)
    summary = {
        "init_image": args.init_image,
        "task": args.task,
        "method": method,
        "preset": preset,
        "invert": invert,
        "rect": rect_s,
        "roi": roi_s,
        "grabcut_iters": grabcut_iters,
        "hsv_low": hsv_low,
        "hsv_high": hsv_high,
        "refine": {"erode": erode, "dilate": dilate, "blur": blur},
        "protect_face": protect_face,
        "face_pad": face_pad,
        "faces_detected": len(faces),
        "outputs": {
            "mask": mask_path,
            "mask_debug": debug_path,
            "overlay": overlay_path,
        },
        "elapsed_ms": (now_ms() - t0),
    }
    with open(os.path.join(args.out_dir, "mask_gen.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"WROTE: {mask_path}")
    print(f"WROTE: {debug_path}")
    print(f"WROTE: {overlay_path}")
    print(f"DONE in {(now_ms()-t0)/1000.0:.3f}s")


if __name__ == "__main__":
    main()
