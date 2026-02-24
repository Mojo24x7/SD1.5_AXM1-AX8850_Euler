# SD1.5_AXM1-AX8850_Euler

Stable Diffusion (Realistic Vision, SD1.5-based) inference runtime for **Radxa AI Core AX-M1 (AX8850)** using **Euler / EulerDiscreteScheduler**.

This GitHub repository contains **only code + lightweight runtime assets** (tokenizer, scheduler config, VAE config).  
All heavy **`.axmodel`** binaries are hosted separately on Hugging Face.

**Hugging Face weights (AXMODEL):** https://huggingface.co/Mojo24x7/sd15-axm1-euler512-axmodels

---

## Hardware target

- **Accelerator:** Radxa AI Core **AX-M1 (AX8850)**
- **Host (tested / supported style):**
  - Rock 5B / Rock 5B+ (PCIe host)
  - **Raspberry Pi 5 can also be the host** (as long as AX-M1 is connected and AXCLRT runtime works on that OS)
- **Runtime provider:** AXCLRT / axengine (ONNX Runtime Execution Provider: `AXCLRTExecutionProvider`)
- **Scheduler:** Euler (EulerDiscreteScheduler)
- **Resolution:** 512×512 (latent 64×64)

> Note: The **AX-M1** does the model inference. The host (Rock 5B/5B+/Pi 5) orchestrates tokenization, scheduling loop, and I/O.

---

## What this repository contains

- AX-M1 Stable Diffusion 1.5 runtime scripts:
  - **txt2img** (Euler)
  - **img2img**
  - **masked img2img** (mask-based editing workflow)
- Lightweight Web UI (Flask-based) + job runner
- Tokenizer and scheduler config required for runtime

This repository **does NOT** store large binaries (`.axmodel`) or generated outputs.

---

## Repository structure (after setup)

```
SD1.5_AXM1-AX8850_Euler/
├── axmodels/                       # (downloaded from HF, NOT in git)
│   ├── sd15_text_encoder_sim.axmodel
│   ├── unet.axmodel
│   ├── vae_decoder.axmodel
│   └── vae_encoder.axmodel         # optional (img2img)
├── scripts/
│   ├── txt2img_axengine_euler.py
│   ├── img2img_axengine_euler.py
│   ├── img2img_masked_axengine_euler.py
│   ├── mask_gen.py
│   └── runner_allinone.py
├── support/
│   ├── tokenizer/                  # CLIP tokenizer assets
│   ├── scheduler/scheduler_config.json
│   └── vae/config.json
├── ui/
│   ├── ui_server.py
│   ├── ui_app.py
│   ├── ui_page.py
│   └── api_*.py
└── run_ui.py
```

---

## Installation

Clone:

```bash
git clone https://github.com/Mojo24x7/SD1.5_AXM1-AX8850_Euler
cd SD1.5_AXM1-AX8850_Euler
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install Python requirements (choose one approach):

- If you already have a working venv for axengine/onnxruntime + AXCLRT provider, install only what you need for the scripts/UI.
- Otherwise, install your known-good requirement set for this runtime.

> Tip: Keep your “known working” environment consistent with the AXCLRT/axengine packages you used to validate inference.

---

## Download models (mandatory)

Download the **AXMODEL** weights from Hugging Face and place them in `./axmodels/`:

### Option A — Git LFS (recommended)

```bash
git lfs install
git clone https://huggingface.co/Mojo24x7/sd15-axm1-euler512-axmodels hf_axmodels
mkdir -p axmodels
cp -v hf_axmodels/*.axmodel axmodels/
```

### Option B — Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Mojo24x7/sd15-axm1-euler512-axmodels   --local-dir hf_axmodels
mkdir -p axmodels
cp -v hf_axmodels/*.axmodel axmodels/
```

Required files:
- `sd15_text_encoder_sim.axmodel`
- `unet.axmodel`
- `vae_decoder.axmodel`

Optional (img2img / masked workflows):
- `vae_encoder.axmodel`

---

## Usage examples

### Text-to-Image (Euler)

```bash
python3 scripts/txt2img_axengine_euler.py   --prompt "a cinematic ultra realistic portrait photo"   --steps 30
```

### Image-to-Image

```bash
python3 scripts/img2img_axengine_euler.py   --init-image images/input.png   --prompt "highly detailed, realistic photo"   --steps 30
```

### Masked edit (inpainting-style workflow)

```bash
python3 scripts/img2img_masked_axengine_euler.py   --init-image images/input.png   --mask images/mask.png   --prompt "In masked area ONLY: lush green grass"   --negative-prompt "snow, white"   --steps 30
```

---

## Web UI

Start the UI:

```bash
python3 run_ui.py
```

Then open in your browser:

- `http://<host-ip>:<port>/`

> If you run the UI on the same machine: use `http://127.0.0.1:<port>/`.

(Your exact port is defined in `run_ui.py` / `ui/ui_server.py`.)

---

## Model details

- Base architecture: Stable Diffusion 1.5
- Checkpoint style: **Realistic Vision** (SD1.5-based)
- Scheduler: Euler (EulerDiscreteScheduler)
- Target: AX-M1 / AX8850 via AXCLRT provider
- Primary resolution: 512×512

---

## Expected model I/O (Euler 512)

- **Text encoder**
  - input: `input_ids` `[1,77]` `int32`
  - output: `last_hidden_state` `[1,77,768]` `fp32`

- **UNet**
  - inputs:
    - `sample` `[1,4,64,64]` `fp32`
    - `timestep` `[1]` `int32`
    - `encoder_hidden_states` `[1,77,768]` `fp32`
  - output: `[1,4,64,64]` `fp32`

- **VAE decoder**
  - input: `latent` `[1,4,64,64]` `fp32`
  - output: `[1,3,512,512]` `fp32` (commonly in `[-1..1]` before postprocess)

---

## Runtime notes / gotchas

- Ensure `timestep` is **int32** (int64 often fails in AX pipelines).
- Ensure `input_ids` are **int32**.
- Typical pipeline:
  1) tokenize → text encoder
  2) scheduler loop → UNet
  3) VAE decode → image postprocess

---

## Credits and acknowledgements

This project builds upon the work of:

- **Stable Diffusion 1.5** by CompVis and Stability AI
- **Realistic Vision** (SD1.5-based checkpoint/style)
- **Diffusers** scheduler concepts (EulerDiscreteScheduler)
- **AXERA / Radxa AX-M1 (AX8850)** toolchain and runtime (AXCLRT/axengine)

All model weights remain under their original upstream licenses and terms.  
This repository provides the **runtime orchestration code**; the compiled `.axmodel` binaries are hosted on Hugging Face.

---
###REQUIREMENTS : 

-- see requirements.txt

---

## Disclaimer

This project is intended for research, experimentation, and personal use.

The author is not affiliated with Stability AI, CompVis, AXERA, Radxa, or Hugging Face.
