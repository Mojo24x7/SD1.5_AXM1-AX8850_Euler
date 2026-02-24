# SD1.5_AXM1-AX8850_Euler

Stable Diffusion (Realistic Vision, SD1.5-based) inference runtime for **Radxa AI Core AX-M1 (AX8850)** using **Euler / EulerDiscreteScheduler**.

This GitHub repository contains **only code + lightweight runtime assets** (tokenizer, scheduler config, VAE config).  
All heavy **`.axmodel`** binaries are hosted separately on Hugging Face.

- **Hugging Face weights (AXMODEL):** https://huggingface.co/Mojo24x7/sd15-axm1-euler512-axmodels  
- **Radxa AX-M1 docs (getting started):** https://docs.radxa.com/en/aicore/ax-m1/getting-started  
- **Radxa AX-M1 environment setup (driver + axcl-smi):** https://docs.radxa.com/en/aicore/ax-m1/getting-started/env_install  
- **Radxa AX-M1 hardware install:** https://docs.radxa.com/en/aicore/ax-m1/getting-started/hardware_install  

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

```text
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
mkdir -p /home/rock/venvs
python3 -m venv .venv_sd15
source .venv_sd15/bin/activate
pip install -U pip
```

Install Python requirements:

```bash
pip install -r requirements.txt
```

> Note: `axengine` / AXCLRT Execution Provider is typically installed via Radxa/AXERA packages (not always from PyPI).  
> Follow Radxa’s environment setup first:  
> https://docs.radxa.com/en/aicore/ax-m1/getting-started/env_install

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
huggingface-cli download Mojo24x7/sd15-axm1-euler512-axmodels --local-dir hf_axmodels
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

## Quickstart (txt2img Euler 512)

From repo root:

```bash
python3 scripts/txt2img_axengine_euler.py \
  --weights_dir ./axmodels \
  --tokenizer_dir ./support/tokenizer \
  --scheduler_dir ./support/scheduler \
  --vae_config ./support/vae/config.json \
  --prompt "a cinematic ultra realistic portrait photo" \
  --steps 30 \
  --out ./out/txt2img_euler_512.png
```

### Example output

![Example 512×512 output](examples/output_512.png)

### Example runtime (trimmed)

```text
(.venv_sd15) rock@rock-5b-plus-2:~/axm1_sd/sd15_euler512$ python3 scripts/txt2img_axengine_euler.py   --weights_dir ./axmodels   --tokenizer_dir ./support/tokenizer   --scheduler_dir ./support/scheduler   --vae_config ./support/vae/config.json   --prompt "a cinematic ultra realistic portrait photo, 85mm lens, natural lighting"   --steps 30   --out ./out/txt2img_euler_512.png
[INFO] Available providers:  ['AXCLRTExecutionProvider']
[INFO] provider: AXCLRTExecutionProvider
[INFO] size: 512x512  latents: (1,4,64,64)
[INFO] steps: 30 guidance: 7.5 seed: 1
[INFO] scheduler: EulerDiscreteScheduler
[INFO] VAE scaling_factor: 0.18215 (decode with latents / scaling_factor)
[INFO] model files:
  - ./axmodels/sd15_text_encoder_sim.axmodel size=247719309 md5=e10b751d76a0428a1f04eaa91ad0b6ec
  - ./axmodels/unet.axmodel size=942067291 md5=9e7e19157dfd3efa884e33e1dfe774a3
  - ./axmodels/vae_decoder.axmodel size=69426024 md5=56da785b5d967509ee9ce258d0bfd98c
[INFO] Using provider: AXCLRTExecutionProvider
[INFO] SOC Name: AX8850
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Compiler version: 5.1-patch1 cd6c30b4
[INFO] Using provider: AXCLRTExecutionProvider
[INFO] SOC Name: AX8850
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Compiler version: 5.1-patch1 cd6c30b4
[INFO] Using provider: AXCLRTExecutionProvider
[INFO] SOC Name: AX8850
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Compiler version: 5.1-patch1 cd6c30b4
[INFO] running text encoder (cond + uncond)...
[INFO] TE done in 0.023s  cond=(1, 77, 768) uncond=(1, 77, 768)
[INFO] denoising...
[INFO] step 01/30 t=999 dt=0.866s avg=0.866s (1.15 step/s) elapsed=00:01 ETA=00:25 latents mean=-0.053721 std=11.926263
[INFO] step 05/30 t=861 dt=0.864s avg=0.865s (1.16 step/s) elapsed=00:04 ETA=00:22 latents mean=-0.005770 std=5.781946
[INFO] step 10/30 t=688 dt=0.865s avg=0.867s (1.15 step/s) elapsed=00:09 ETA=00:17 latents mean=0.003259 std=2.899466
[INFO] step 15/30 t=516 dt=0.870s avg=0.867s (1.15 step/s) elapsed=00:13 ETA=00:13 latents mean=0.004263 std=1.781489
[INFO] step 20/30 t=344 dt=0.866s avg=0.865s (1.16 step/s) elapsed=00:17 ETA=00:09 latents mean=0.002296 std=1.301835
[INFO] step 25/30 t=172 dt=0.865s avg=0.863s (1.16 step/s) elapsed=00:22 ETA=00:04 latents mean=-0.000190 std=1.086818
[INFO] step 30/30 t=0 dt=0.863s avg=0.863s (1.16 step/s) elapsed=00:26 ETA=00:00 latents mean=-0.002385 std=1.004734
[INFO] denoise total: 25.956s
[INFO] decoding with VAE (latents / scaling_factor)...
[INFO] VAE decode done in 0.906s
[OK] saved: ./out/txt2img_euler_512.png

```

Full log: `examples/run_log.txt`

---

## Web UI

Start the UI:

```bash
python3 run_ui.py
```

Then open in your browser:

- `http://<host-ip>:<port>/`

> If you run the UI on the same machine: use `http://127.0.0.1:<port>/`.  
> Your exact port is defined in `run_ui.py` / `ui/ui_server.py`.

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
- **Radxa AICore AX-M1 (AX8850)** documentation and examples
- **AXERA AXCLRT** runtime ecosystem

All model weights remain under their original upstream licenses and terms.  
This repository provides the **runtime orchestration code**; the compiled `.axmodel` binaries are hosted on Hugging Face.

---

## Disclaimer

This project is intended for research, experimentation, and personal use.

The author is not affiliated with Stability AI, CompVis, AXERA, Radxa, or Hugging Face.
