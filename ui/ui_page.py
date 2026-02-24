#!/usr/bin/env python3
# ui_page.py - Single-page UI (HTML/CSS/JS) for AX-M1 SD15 UI
# - Professional dark-grey theme (ChatGPT/Grok-ish)
# - Fully responsive (mobile/laptop)
# - File browser with thumbnails
# - Live NPU metrics + progress
# - Live token count (prompt, negative, total) using /api/tokens/count (CLIP 77 limit)
# - Click main image to open full-res

import json
from flask import Blueprint, Response, current_app

bp_ui = Blueprint("ui", __name__)

UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
  <title>AX-M1 SD15 UI</title>
  <style>
    :root{
      --bg:#0b0f14;
      --panel:#0f1620;
      --panel2:#121b26;
      --text:#e6edf3;
      --muted:#9fb0c0;
      --line:#1f2b3a;
      --accent:#4aa3ff;
      --ok:#35d07f;
      --warn:#ffcc66;
      --err:#ff5c7a;
      --chip:#182333;
      --shadow: 0 8px 24px rgba(0,0,0,.35);
      --radius: 16px;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      background:linear-gradient(180deg,#0a0f15 0%, #070b10 100%);
      color:var(--text);
      font: 14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial;
    }
    a{color:var(--accent); text-decoration:none}
    .topbar{
      position:sticky; top:0; z-index:20;
      backdrop-filter: blur(12px);
      background: rgba(8,12,18,.75);
      border-bottom:1px solid var(--line);
    }
    .topbar-inner{
      max-width:1200px; margin:0 auto;
      padding:14px 14px;
      display:flex; align-items:center; gap:12px;
    }
    .brand{
      display:flex; align-items:center; gap:10px;
      font-weight:800; letter-spacing:.2px;
    }
    .dot{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 18px rgba(74,163,255,.55)}
    .tabs{display:flex; gap:8px; margin-left:auto; flex-wrap:wrap; justify-content:flex-end}
    .tab{
      padding:8px 12px; border:1px solid var(--line); border-radius:999px;
      color:var(--muted); background:transparent; cursor:pointer;
    }
    .tab.active{color:var(--text); border-color:rgba(74,163,255,.45); background:rgba(74,163,255,.12)}
    .container{max-width:1200px; margin:0 auto; padding:14px}
    .grid{
      display:grid; gap:14px;
      grid-template-columns: 420px 1fr;
      align-items:start;
    }
    @media (max-width: 980px){
      .grid{grid-template-columns:1fr}
      .tab{padding:8px 10px}
    }
    .card{
      background:linear-gradient(180deg,var(--panel) 0%, var(--panel2) 100%);
      border:1px solid var(--line);
      border-radius:var(--radius);
      box-shadow: var(--shadow);
      overflow:hidden;
    }
    .card h3{
      margin:0; padding:14px 14px 10px;
      font-size:14px; color:var(--text);
      border-bottom:1px solid var(--line);
      background:rgba(255,255,255,.02);
    }
    .card .body{padding:14px}
    .row{display:flex; gap:10px; flex-wrap:wrap}
    .row > *{flex:1}
    label{display:block; color:var(--muted); font-size:12px; margin-bottom:6px}
    input[type="text"], input[type="number"], textarea, select{
      width:100%;
      background:#0b121b;
      border:1px solid #223146;
      color:var(--text);
      padding:10px 10px;
      border-radius:12px;
      outline:none;
    }
    textarea{min-height:90px; resize:vertical}
    .btn{
      border:1px solid #2a3a52;
      background:#0b121b;
      color:var(--text);
      padding:10px 12px;
      border-radius:12px;
      cursor:pointer;
      font-weight:700;
      white-space:nowrap;
    }
    .btn.primary{
      background:rgba(74,163,255,.16);
      border-color: rgba(74,163,255,.35);
    }
    .btn.danger{background:rgba(255,92,122,.12); border-color:rgba(255,92,122,.35)}
    .btn:disabled{opacity:.5; cursor:not-allowed}
    .chip{
      display:inline-flex; align-items:center; gap:6px;
      padding:6px 10px;
      border-radius:999px;
      border:1px solid var(--line);
      background:rgba(255,255,255,.03);
      color:var(--muted);
      font-size:12px;
      user-select:none;
      white-space:nowrap;
    }
    .chip.ok{border-color:rgba(53,208,127,.35); background:rgba(53,208,127,.09); color:#b9f3d4}
    .chip.warn{border-color:rgba(255,204,102,.35); background:rgba(255,204,102,.10); color:#ffe4ad}
    .chip.err{border-color:rgba(255,92,122,.45); background:rgba(255,92,122,.12); color:#ffd0da}
    .hint{color:var(--muted); font-size:12px; margin-top:6px}
    .progress{
      height:10px; border-radius:999px;
      background:#0b121b; border:1px solid #223146;
      overflow:hidden;
    }
    .bar{height:100%; width:0%; background: linear-gradient(90deg, rgba(74,163,255,.3), rgba(74,163,255,.9))}
    .metrics{
      display:grid; grid-template-columns: repeat(4, 1fr); gap:10px;
    }
    @media (max-width: 980px){
      .metrics{grid-template-columns: repeat(2, 1fr)}
    }
    .metric{
      padding:10px;
      border:1px solid var(--line);
      background:rgba(255,255,255,.02);
      border-radius:14px;
    }
    .metric .k{color:var(--muted); font-size:12px}
    .metric .v{font-size:16px; font-weight:900; margin-top:2px}
    .metric .mini{margin-top:8px}
    .preview{
      display:grid; gap:14px;
      grid-template-columns: 1fr;
    }
    .imgbox{
      position:relative;
      border-radius:14px;
      border:1px solid var(--line);
      overflow:hidden;
      background:#0b121b;
      min-height:240px;
      cursor: zoom-in; /* NEW */
    }
    .imgbox img{
      width:100%;
      height:auto;
      display:block;
    }
    .overlay{
      position:absolute; inset:0;
      display:flex; align-items:center; justify-content:center;
      color:rgba(230,237,243,.9);
      font-weight:900;
      letter-spacing:.2px;
      text-shadow:0 2px 20px rgba(0,0,0,.7);
      pointer-events:none;
    }
    .thumbstrip{
      display:flex; gap:10px;
      overflow:auto;
      padding-bottom:4px;
    }
    .thumb{
      width:92px; height:92px;
      border-radius:12px;
      border:1px solid var(--line);
      background:#0b121b;
      overflow:hidden;
      flex:0 0 auto;
      cursor:pointer;
    }
    .thumb img{width:100%; height:100%; object-fit:cover; display:block}
    .logs{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size:12px;
      background:#080c12;
      border:1px solid #223146;
      border-radius:14px;
      padding:10px;
      max-height:280px;
      overflow:auto;
      white-space:pre-wrap;
    }
    .split{
      display:grid; grid-template-columns: 1fr 1fr; gap:12px;
    }
    @media (max-width: 980px){
      .split{grid-template-columns:1fr}
    }

    /* Modal file browser */
    .modal{position:fixed; inset:0; background:rgba(0,0,0,.55); display:none; align-items:center; justify-content:center; padding:14px; z-index:50}
    .modal.on{display:flex}
    .modal-card{width:min(980px, 100%); max-height:90vh; overflow:hidden}
    .modal-head{padding:12px 14px; display:flex; gap:10px; align-items:center; border-bottom:1px solid var(--line); background:rgba(255,255,255,.02)}
    .modal-head input{flex:1}
    .modal-body{display:grid; grid-template-columns: 1fr 1fr; gap:0; height:70vh}
    @media (max-width: 980px){
      .modal-body{grid-template-columns: 1fr; height:75vh}
    }
    .pane{padding:12px; overflow:auto; border-right:1px solid var(--line)}
    .pane:last-child{border-right:none}
    .entry{padding:8px 10px; border:1px solid transparent; border-radius:12px; cursor:pointer; display:flex; gap:8px; align-items:center}
    .entry:hover{background:rgba(255,255,255,.03); border-color:rgba(255,255,255,.04)}
    .entry .icon{width:18px; opacity:.9}
    .gridthumbs{display:grid; grid-template-columns: repeat(4, 1fr); gap:10px}
    @media (max-width: 980px){ .gridthumbs{grid-template-columns: repeat(3, 1fr)} }
    @media (max-width: 520px){ .gridthumbs{grid-template-columns: repeat(2, 1fr)} }
    .gridthumbs .thumb{width:100%; height:120px}

    /* token chips */
    .tokrow{display:flex; gap:8px; flex-wrap:wrap; margin-top:8px}
    .tokrow .chip{font-weight:800}
    
    .toktools{display:flex; gap:10px; flex-wrap:wrap; margin-top:10px}
    .toktools .btn{padding:8px 10px; font-size:12px}
    .tokpanel{
      margin-top:10px;
      border:1px solid var(--line);
      background:rgba(255,255,255,.02);
      border-radius:14px;
      padding:10px;
      display:none;
      max-height:220px;
      overflow:auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size:12px;
      white-space:pre;
    }
    .tokpanel.on{display:block}

  </style>
</head>

<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div class="brand"><span class="dot"></span> AX-M1 SD15 UI</div>
      <div class="tabs">
        <button class="tab active" data-tab="txt2img">TXT2IMG</button>
        <button class="tab" data-tab="img2img">IMG2IMG</button>
        <button class="tab" data-tab="mask">MASK STUDIO</button>
      </div>
    </div>
  </div>

  <div class="container">
    <div class="grid">
      <!-- LEFT: controls -->
      <div class="card">
        <h3 id="panelTitle">TXT2IMG Controls</h3>
        <div class="body" id="controls"></div>
      </div>

      <!-- RIGHT: preview + progress + logs -->
      <div class="preview">
        <div class="card">
          <h3>Preview</h3>
          <div class="body">
            <div class="split">
              <div>
                <div class="imgbox" id="mainImgBox" title="Click to open full image">
                  <img id="mainImg" src="" alt="" style="display:none"/>
                  <div class="overlay" id="mainOverlay">No image yet</div>
                </div>
                <div class="hint" id="imgHint">Output will appear here.</div>
                <div style="height:10px"></div>
                <div class="thumbstrip" id="thumbstrip"></div>
              </div>
              <div>
                <div class="row" style="align-items:center">
                  <span class="chip" id="statusChip">idle</span>
                  <span class="chip" id="etaChip">ETA: --</span>
                  <span class="chip" id="stepChip">Step: --</span>
                </div>

                <div style="height:10px"></div>
                <div class="progress"><div class="bar" id="progBar"></div></div>
                <div class="hint" id="progHint">Progress</div>

                <div style="height:14px"></div>
                <div class="metrics">
                  <div class="metric">
                    <div class="k">Temp</div>
                    <div class="v" id="mTemp">--</div>
                    <div class="mini"><div class="progress"><div class="bar" id="bTemp"></div></div></div>
                  </div>
                  <div class="metric">
                    <div class="k">NPU</div>
                    <div class="v" id="mNpu">--</div>
                    <div class="mini"><div class="progress"><div class="bar" id="bNpu"></div></div></div>
                  </div>
                  <div class="metric">
                    <div class="k">Mem</div>
                    <div class="v" id="mMem">--</div>
                    <div class="mini"><div class="progress"><div class="bar" id="bMem"></div></div></div>
                  </div>
                  <div class="metric">
                    <div class="k">CMM</div>
                    <div class="v" id="mCmm">--</div>
                    <div class="mini"><div class="progress"><div class="bar" id="bCmm"></div></div></div>
                  </div>
                </div>

                <div style="height:14px"></div>
                <div class="row">
                  <button class="btn primary" id="btnRun">Run</button>
                  <button class="btn danger" id="btnCancel" disabled>Cancel</button>
                  <button class="btn" id="btnOpenRuns">Runs</button>
                </div>
                <div class="hint">Runs are saved in your selected output folder → run name directory.</div>
              </div>
            </div>

            <div style="height:14px"></div>
            <div class="logs" id="logs"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- File browser modal -->
  <div class="modal" id="modal">
    <div class="card modal-card">
      <div class="modal-head">
        <button class="btn" id="btnUp">Up</button>
        <input type="text" id="pathBox" value="/"/>
        <button class="btn" id="btnGo">Go</button>
        <button class="btn primary" id="btnSelect">Select</button>
        <button class="btn" id="btnClose">Close</button>
      </div>
      <div class="modal-body">
        <div class="pane" id="paneDirs"></div>
        <div class="pane" id="paneFiles"></div>
      </div>
    </div>
  </div>

<script>
  // ---------- cache busting ----------
  function bust(){
    return Date.now(); // unique per request
  }

  function thumbUrl(path, w, h){
    return `/api/fs/thumb?path=${encodeURIComponent(path)}&w=${w}&h=${h}&t=${bust()}`;
  }

  function fileUrl(path){
    return `${FULL_FILE_ENDPOINT}${encodeURIComponent(path)}&t=${bust()}`;
  }

  // injected by server
  window.__OUT_ROOT__ = __OUT_ROOT_JSON__;

  // If your backend uses a different endpoint for full file, change this:
  const FULL_FILE_ENDPOINT = "/api/fs/file?path=";

  // ---------- utilities ----------
  const $ = (id)=>document.getElementById(id);
  const logsEl = $("logs");
  function log(line){
    logsEl.textContent += line + "\\n";
    logsEl.scrollTop = logsEl.scrollHeight;
  }
  function fmtPct01(x){ return Math.round(x*100) + "%"; }
  function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }

  // ---------- state ----------
  let activeTab = "txt2img";
  let jobId = null;
  let evtSrc = null;

  // current image shown in preview (so clicking main opens full-res)
  let currentMainPath = null;

  // File browser state
  let fbPurpose = null; // "init_image" | "mask_path" | "output_root"
  let fbSelectedPath = null;
  let fbSelectedIsDir = false;

  // Blur effect
  let blurBase = 14; // px

  // Token debounce
  let tokTimer = null;

  // ---------- token UI helpers ----------
  
  async function clipTrimPrompt(){
    const p = $("prompt")?.value ?? "";
    if(!p.trim()) return;

    try{
      const r = await fetch("/api/tokens/trim", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ text: p })
      });
      const data = await r.json();
      if(!data.ok) return;

      if(data.changed){
        $("prompt").value = data.trimmed || "";
        log(`[TOK] trimmed to ${data.tokens}/${data.max}. dropped: ${Array.isArray(data.dropped) ? data.dropped.length : 0}`);
      }else{
        log(`[TOK] already CLIP-safe (${data.tokens}/${data.max})`);
      }
      fetchTokenCount();
      // if panel is open, refresh analysis
      if($("tokCostPanel")?.classList.contains("on")){
        await showTokenCost();
      }
    }catch(e){}
  }

  async function showTokenCost(){
    const panel = $("tokCostPanel");
    if(!panel) return;
    const p = $("prompt")?.value ?? "";
    if(!p.trim()){
      panel.textContent = "No prompt.";
      panel.classList.add("on");
      return;
    }

    try{
      const r = await fetch("/api/tokens/analyze", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ text: p })
      });
      const data = await r.json();
      if(!data.ok) return;

      const rows = data.words || [];
      let out = `TOTAL: ${data.total_tokens}/${data.max}  ${data.over ? "(OVER)" : ""}\n`;
      out += "----------------------------------------\n";
      // show top 80 tokens/words to keep it readable
      rows.slice(0, 120).forEach((x,i)=>{
        const w = String(x.word ?? "");
        const c = String(x.cost ?? "");
        out += `${String(i+1).padStart(3," ")}  +${c.padStart(2," ")}  ${w}\n`;
      });
      if(rows.length > 120) out += `... (${rows.length-120} more)\n`;

      panel.textContent = out;
      panel.classList.add("on");
    }catch(e){}
  }

  function toggleTokenPanel(){
    const panel = $("tokCostPanel");
    if(!panel) return;
    const on = panel.classList.toggle("on");
    if(on) showTokenCost();
  }


  function setChip(id, text, cls){
    const el = $(id);
    if(!el) return;
    el.textContent = text;
    el.classList.remove("ok","warn","err");
    if(cls) el.classList.add(cls);
  }

  function classifyTokens(t){
    // t: {tokens, max, over}
    if(!t) return "chip";
    if(t.over) return "err";
    // warn if close to max
    if(t.max && t.tokens >= Math.max(1, t.max-4)) return "warn";
    return "ok";
  }

  async function fetchTokenCount(){
    if(!(activeTab==="txt2img" || activeTab==="img2img")) return;

    const p = $("prompt")?.value ?? "";
    const n = $("negative")?.value ?? "";

    try{
      const r = await fetch("/api/tokens/count", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ prompt: p, negative: n })
      });
      const data = await r.json();
      if(!data.ok) return;

      const tp = data.prompt;
      const tn = data.negative;
      const total = (tp?.tokens ?? 0) + (tn?.tokens ?? 0);
      const max = tp?.max ?? 77;

      setChip("chipTokPrompt", `Prompt: ${tp.tokens}/${max}`, classifyTokens(tp));
      setChip("chipTokNeg", `Negative: ${tn.tokens}/${max}`, classifyTokens(tn));

      let totalCls = "ok";
      if(total > (2*max)) totalCls = "err";
      else if(total > (2*max - 10)) totalCls = "warn";
      setChip("chipTokTotal", `Total: ${total}`, totalCls);
    }catch(e){}
  }

  function scheduleTokenCount(){
    clearTimeout(tokTimer);
    tokTimer = setTimeout(fetchTokenCount, 250);
  }

  // ---------- tab controls templates ----------
  function tokenRowHtml(){
    return `
      <div class="tokrow">
        <span class="chip" id="chipTokPrompt">Prompt: --</span>
        <span class="chip" id="chipTokNeg">Negative: --</span>
        <span class="chip" id="chipTokTotal">Total: --</span>
      </div>

      <div class="toktools">
        <button class="btn" type="button" id="btnClipTrim">CLIP-safe trim ≤77</button>
        <button class="btn" type="button" id="btnTokCost">Token cost</button>
      </div>

      <div class="tokpanel" id="tokCostPanel"></div>

      <div class="hint">Token count uses CLIP BPE (BOS/EOS included in the 77 limit).</div>
    `;
  }


  function controls_txt2img(){
    return `
      <div class="row">
        <div>
          <label>Output folder</label>
          <div class="row">
            <input type="text" id="outRoot" value="${window.__OUT_ROOT__ || ""}"/>
            <button class="btn" onclick="openFileBrowser('output_root')">Browse</button>
          </div>
          <div class="hint">Full filesystem access. Choose where runs are saved.</div>
        </div>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div>
          <label>Run name</label>
          <input type="text" id="runName" value="txt2img_${new Date().toISOString().slice(0,19).replaceAll(':','').replace('T','_')}"/>
          <div class="hint">Creates: output_root/run_name/</div>
        </div>
      </div>

      <div style="height:10px"></div>
      <div>
        <label>Prompt</label>
        <textarea id="prompt" placeholder="Describe what you want..."></textarea>
        ${tokenRowHtml()}
      </div>

      <div style="height:10px"></div>
      <div>
        <label>Negative prompt</label>
        <textarea id="negative" placeholder="What to avoid..."></textarea>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div>
          <label>Steps</label>
          <input type="number" id="steps" value="30" min="1" max="80"/>
        </div>
        <div>
          <label>CFG</label>
          <input type="number" id="cfg" value="7.5" step="0.1" min="1" max="20"/>
        </div>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div>
          <label>Seed</label>
          <input type="number" id="seed" value="1001" min="0"/>
        </div>
        <div>
          <label>NPU poll every N steps</label>
          <input type="number" id="npuPoll" value="5" min="1" max="30"/>
        </div>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div class="chip"><input type="checkbox" id="npuLive" checked style="margin-right:8px"/> NPU live</div>
        <div class="chip"><input type="checkbox" id="npuRecord" checked style="margin-right:8px"/> Record NPU</div>
        <div class="chip"><input type="checkbox" id="embedMeta" checked style="margin-right:8px"/> Embed metadata</div>
      </div>
    `;
  }

  function controls_img2img(){
    return `
      <div class="row">
        <div>
          <label>Output folder</label>
          <div class="row">
            <input type="text" id="outRoot" value="${window.__OUT_ROOT__ || ""}"/>
            <button class="btn" onclick="openFileBrowser('output_root')">Browse</button>
          </div>
        </div>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div>
          <label>Run name</label>
          <input type="text" id="runName" value="img2img_${new Date().toISOString().slice(0,19).replaceAll(':','').replace('T','_')}"/>
        </div>
      </div>

      <div style="height:10px"></div>
      <div>
        <label>Init image</label>
        <div class="row">
          <input type="text" id="initImage" placeholder="/path/to/input.png"/>
          <button class="btn" onclick="openFileBrowser('init_image')">Browse</button>
        </div>
      </div>

      <div style="height:10px"></div>
      <div>
        <label>Mask</label>
        <div class="row">
          <input type="text" id="maskPath" placeholder="/path/to/mask.png"/>
          <button class="btn" onclick="openFileBrowser('mask_path')">Browse</button>
        </div>
        <div class="hint">Mask white = edit, black = keep.</div>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div>
          <label>Strength</label>
          <input type="number" id="strength" value="0.85" step="0.01" min="0.05" max="0.99"/>
        </div>
        <div>
          <label>Mask blur</label>
          <input type="number" id="maskBlur" value="3" min="0" max="32"/>
        </div>
      </div>

      <div style="height:10px"></div>
      <div>
        <label>Prompt</label>
        <textarea id="prompt" placeholder="Example: keep same face/identity, change clothes, change background..."></textarea>
        ${tokenRowHtml()}
      </div>

      <div style="height:10px"></div>
      <div>
        <label>Negative prompt</label>
        <textarea id="negative" placeholder="Example: different identity, face change, deformed..."></textarea>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div>
          <label>Steps</label>
          <input type="number" id="steps" value="30" min="1" max="80"/>
        </div>
        <div>
          <label>CFG</label>
          <input type="number" id="cfg" value="7.5" step="0.1" min="1" max="20"/>
        </div>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div>
          <label>Seed</label>
          <input type="number" id="seed" value="93" min="0"/>
        </div>
        <div class="chip">
          <input type="checkbox" id="blurFx" checked style="margin-right:8px"/> Blurred generating effect
        </div>
      </div>

      <div class="hint">During denoise, preview shows a blur that clears as progress increases.</div>
    `;
  }

  function controls_mask(){
    return `
      <div class="row">
        <div>
          <label>Output folder</label>
          <div class="row">
            <input type="text" id="outRoot" value="${window.__OUT_ROOT__ || ""}"/>
            <button class="btn" onclick="openFileBrowser('output_root')">Browse</button>
          </div>
        </div>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div>
          <label>Run name</label>
          <input type="text" id="runName" value="mask_${new Date().toISOString().slice(0,19).replaceAll(':','').replace('T','_')}"/>
        </div>
      </div>

      <div style="height:10px"></div>
      <div>
        <label>Init image</label>
        <div class="row">
          <input type="text" id="initImage" placeholder="/path/to/input.png"/>
          <button class="btn" onclick="openFileBrowser('init_image')">Browse</button>
        </div>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div>
          <label>Mask mode</label>
          <select id="maskMode">
            <option value="grabcut">GrabCut</option>
            <option value="color">HSV Color</option>
            <option value="preset">Preset</option>
            <option value="rect">Rect</option>
          </select>
        </div>
        <div>
          <label>Preset</label>
          <select id="preset">
            <option value="shirt">shirt</option>
            <option value="body">body</option>
            <option value="hair">hair</option>
            <option value="lower">lower</option>
            <option value="upper">upper</option>
          </select>
        </div>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div>
          <label>Rect (x,y,w,h)</label>
          <input type="text" id="rect" value="0,280,512,232"/>
        </div>
        <div>
          <label>GrabCut iters</label>
          <input type="number" id="grabIters" value="5" min="1" max="20"/>
        </div>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div>
          <label>HSV low (H,S,V)</label>
          <input type="text" id="hsvLow" value="0,0,200"/>
        </div>
        <div>
          <label>HSV high (H,S,V)</label>
          <input type="text" id="hsvHigh" value="179,80,255"/>
        </div>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div>
          <label>ROI (optional x,y,w,h)</label>
          <input type="text" id="roi" value=""/>
        </div>
        <div class="chip" style="align-self:end">
          <input type="checkbox" id="invertMask" style="margin-right:8px"/> Invert mask
        </div>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div>
          <label>Erode</label>
          <input type="number" id="erode" value="0" min="0" max="32"/>
        </div>
        <div>
          <label>Dilate</label>
          <input type="number" id="dilate" value="3" min="0" max="32"/>
        </div>
        <div>
          <label>Blur</label>
          <input type="number" id="blur" value="7" min="0" max="63"/>
        </div>
      </div>

      <div style="height:10px"></div>
      <div class="row">
        <div class="chip">
          <input type="checkbox" id="protectFace" style="margin-right:8px"/> Protect face
        </div>
        <div>
          <label>Face pad</label>
          <input type="number" id="facePad" value="0.25" step="0.05" min="0.0" max="1.0"/>
        </div>
      </div>

      <div class="hint">Mask generation uses system python (/usr/bin/python3) for cv2.</div>
    `;
  }

  function setControls(tab){
    $("controls").innerHTML =
      tab==="txt2img" ? controls_txt2img() :
      (tab==="img2img" ? controls_img2img() : controls_mask());

    $("panelTitle").textContent =
      tab==="txt2img" ? "TXT2IMG Controls" :
      (tab==="img2img" ? "IMG2IMG Controls" : "MASK STUDIO Controls");

    if(tab==="txt2img" || tab==="img2img"){
      $("prompt")?.addEventListener("input", scheduleTokenCount);
      $("negative")?.addEventListener("input", scheduleTokenCount);
      fetchTokenCount();
    }
      // token tools buttons (only exist on txt2img/img2img)
    if(tab==="txt2img" || tab==="img2img"){
      $("btnClipTrim")?.addEventListener("click", clipTrimPrompt);
      $("btnTokCost")?.addEventListener("click", toggleTokenPanel);
    }

  }

  // ---------- tabs ----------
  document.querySelectorAll(".tab").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active"));
      btn.classList.add("active");
      activeTab = btn.dataset.tab;
      setControls(activeTab);
      resetUI();
    });
  });

  // ---------- file browser ----------
  async function fsList(path){
    const r = await fetch(`/api/fs/list?path=${encodeURIComponent(path)}`);
    return await r.json();
  }

  function openFileBrowser(purpose){
    fbPurpose = purpose;
    fbSelectedPath = null;
    fbSelectedIsDir = false;
    const cur =
      purpose==="output_root" ? ($("outRoot")?.value || "/") :
      (purpose==="init_image" ? ($("initImage")?.value || "/") :
      ($("maskPath")?.value || "/"));
    $("pathBox").value = cur && cur.trim() ? cur : "/";
    $("modal").classList.add("on");
    loadBrowser($("pathBox").value);
  }

  async function loadBrowser(path){
    const data = await fsList(path);
    if(!data.ok){ log("[FS] "+data.error); return; }

    $("pathBox").value = data.path;

    const dirs = data.entries.filter(e=>e.is_dir);
    $("paneDirs").innerHTML = dirs.map(d=>`
      <div class="entry" onclick="loadBrowser('${d.path.replaceAll("'","\\'")}')">
        <div class="icon">📁</div><div>${d.name}</div>
      </div>
    `).join("") || `<div class="hint">No subfolders</div>`;

    const imgs = data.entries.filter(e=>e.is_file && e.is_image);
    const others = data.entries.filter(e=>e.is_file && !e.is_image);

    let html = "";
    if(imgs.length){
      html += `<div class="gridthumbs">` + imgs.map(f=>`
        <div class="thumb" onclick="selectPath('${f.path.replaceAll("'","\\'")}', false)">
          <img src="${thumbUrl(f.path, 256, 256)}" alt=""/>

        </div>
      `).join("") + `</div>`;
      html += `<div style="height:12px"></div>`;
    }
    if(others.length){
      html += others.slice(0,120).map(f=>`
        <div class="entry" onclick="selectPath('${f.path.replaceAll("'","\\'")}', false)">
          <div class="icon">📄</div><div>${f.name}</div>
        </div>
      `).join("");
    }
    if(!html) html = `<div class="hint">No files here</div>`;
    $("paneFiles").innerHTML = html;

    if(fbPurpose==="output_root"){
      selectPath(data.path, true);
    }
  }

  function selectPath(path, isDir){
    fbSelectedPath = path;
    fbSelectedIsDir = isDir;
    log(`[FS] selected: ${path}`);
  }

  $("btnUp").onclick = ()=>{
    const p = $("pathBox").value;
    const up = p.endsWith("/") ? p.slice(0,-1) : p;
    const parent = up.split("/").slice(0,-1).join("/") || "/";
    loadBrowser(parent);
  };
  $("btnGo").onclick = ()=> loadBrowser($("pathBox").value);
  $("btnClose").onclick = ()=> $("modal").classList.remove("on");
  $("btnSelect").onclick = ()=>{
    if(!fbSelectedPath){
      log("[FS] Select something first");
      return;
    }
    if(fbPurpose==="output_root"){
      $("outRoot").value = fbSelectedPath;
    } else if(fbPurpose==="init_image"){
      $("initImage").value = fbSelectedPath;
      showInputThumb(fbSelectedPath);
    } else if(fbPurpose==="mask_path"){
      $("maskPath").value = fbSelectedPath;
      showInputThumb(fbSelectedPath);
    }
    $("modal").classList.remove("on");
  };

  function showInputThumb(path){
    if(!path) return;
    currentMainPath = path; // NEW
    $("mainImg").src = thumbUrl(path, 1024, 1024);

    $("mainImg").style.display = "block";
    $("mainOverlay").textContent = "Input selected";
    $("imgHint").textContent = path;
  }

  // ---------- run management ----------
  function resetUI(){
    $("statusChip").textContent = "idle";
    $("etaChip").textContent = "ETA: --";
    $("stepChip").textContent = "Step: --";
    $("progBar").style.width = "0%";
    $("progHint").textContent = "Progress";
    $("thumbstrip").innerHTML = "";
    $("logs").textContent = "";
    setNpu(null);
    $("mainOverlay").textContent = "No image yet";
    $("mainImg").style.display = "none";
    $("imgHint").textContent = "Output will appear here.";
    $("btnCancel").disabled = true;
    $("btnRun").disabled = false;
    currentMainPath = null; // NEW

    if(activeTab==="txt2img" || activeTab==="img2img"){
      setChip("chipTokPrompt","Prompt: --",null);
      setChip("chipTokNeg","Negative: --",null);
      setChip("chipTokTotal","Total: --",null);
      fetchTokenCount();
    }
  }

  function setNpu(n){
    const temp = n?.temp_c ?? null;
    const npu = n?.npu_pct ?? null;
    const mu  = n?.mem_used ?? null;
    const mt  = n?.mem_total ?? null;
    const cu  = n?.cmm_used ?? null;
    const ct  = n?.cmm_total ?? null;

    $("mTemp").textContent = temp===null ? "--" : `${temp}°C`;
    $("mNpu").textContent  = npu===null ? "--" : `${npu}%`;
    $("mMem").textContent  = (mu===null||mt===null) ? "--" : `${mu}/${mt} MiB`;
    $("mCmm").textContent  = (cu===null||ct===null) ? "--" : `${cu}/${ct} MiB`;

    $("bTemp").style.width = temp===null ? "0%" : `${clamp((temp/85)*100, 0, 100)}%`;
    $("bNpu").style.width  = npu===null  ? "0%" : `${clamp(npu, 0, 100)}%`;
    $("bMem").style.width  = (mu===null||mt===null||mt===0) ? "0%" : `${clamp((mu/mt)*100,0,100)}%`;
    $("bCmm").style.width  = (cu===null||ct===null||ct===0) ? "0%" : `${clamp((cu/ct)*100,0,100)}%`;
  }

  function applyBlurFx(pct){
    const b = clamp(blurBase * (1 - pct), 0, blurBase);
    $("mainImg").style.filter = `blur(${b}px)`;
  }
  function clearBlurFx(){ $("mainImg").style.filter = "none"; }

  async function runActiveTab(){
    const outRoot = $("outRoot").value.trim();
    const runName = $("runName").value.trim();

    if(!outRoot){
      alert("Select output folder");
      return;
    }

    let endpoint = null;
    let payload = { output_root: outRoot, run_name: runName };

    if(activeTab==="txt2img"){
      endpoint = "/api/run/txt2img";
      payload.prompt = $("prompt").value;
      payload.negative = $("negative").value;
      payload.steps = parseInt($("steps").value || "30", 10);
      payload.cfg = parseFloat($("cfg").value || "7.5");
      payload.seed = parseInt($("seed").value || "1", 10);
      payload.npu_live = $("npuLive").checked;
      payload.npu_poll_every = parseInt($("npuPoll").value || "5", 10);
      payload.npu_record = $("npuRecord").checked;
      payload.embed_metadata = $("embedMeta").checked;
      $("mainOverlay").textContent = "Generating...";
    }

    if(activeTab==="img2img"){
      endpoint = "/api/run/img2img";
      payload.init_image = $("initImage").value.trim();
      payload.mask_path = $("maskPath").value.trim();
      payload.prompt = $("prompt").value;
      payload.negative = $("negative").value;
      payload.steps = parseInt($("steps").value || "30", 10);
      payload.cfg = parseFloat($("cfg").value || "7.5");
      payload.seed = parseInt($("seed").value || "93", 10);
      payload.strength = parseFloat($("strength").value || "0.85");
      payload.mask_blur = parseInt($("maskBlur").value || "3", 10);

      if(!payload.init_image){ alert("Select init image"); return; }
      if(!payload.mask_path){ alert("Select mask path"); return; }

      currentMainPath = payload.init_image; // NEW
      $("mainImg").src = thumbUrl(payload.init_image, 1024, 1024);

      $("mainImg").style.display = "block";
      $("mainOverlay").textContent = "Editing...";
      $("imgHint").textContent = payload.init_image;

      if($("blurFx").checked){
        applyBlurFx(0.0);
      }
    }

    if(activeTab==="mask"){
      endpoint = "/api/run/mask";
      payload.init_image = $("initImage").value.trim();
      payload.mask_mode = $("maskMode").value;
      payload.rect = $("rect").value.trim();
      payload.grabcut_iters = parseInt($("grabIters").value || "5", 10);
      payload.hsv_low = $("hsvLow").value.trim();
      payload.hsv_high = $("hsvHigh").value.trim();
      payload.roi = $("roi").value.trim();
      payload.preset = $("preset").value;
      payload.invert_mask = $("invertMask").checked;
      payload.erode = parseInt($("erode").value || "0", 10);
      payload.dilate = parseInt($("dilate").value || "3", 10);
      payload.blur = parseInt($("blur").value || "7", 10);
      payload.protect_face = $("protectFace").checked;
      payload.face_pad = parseFloat($("facePad").value || "0.25");

      if(!payload.init_image){ alert("Select init image"); return; }

      currentMainPath = payload.init_image; // NEW
      $("mainImg").src = `/api/fs/thumb?path=${encodeURIComponent(payload.init_image)}&w=1024&h=1024`;
      $("mainImg").style.display = "block";
      $("mainOverlay").textContent = "Generating mask...";
      $("imgHint").textContent = payload.init_image;
    }

    $("btnRun").disabled = true;
    $("btnCancel").disabled = false;
    $("statusChip").textContent = "running";
    log("[UI] launching...");

    const r = await fetch(endpoint, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)});
    const data = await r.json();
    if(!data.ok){
      log("[UI] ERROR: " + (data.error || "unknown"));
      $("btnRun").disabled = false;
      $("btnCancel").disabled = true;
      $("statusChip").textContent = "error";
      return;
    }

    jobId = data.job_id;
    subscribe(jobId);
  }

  function subscribe(jobId){
    if(evtSrc) evtSrc.close();
    evtSrc = new EventSource(`/api/jobs/events?job_id=${encodeURIComponent(jobId)}`);

    evtSrc.onmessage = (ev)=>{
      const msg = JSON.parse(ev.data);

      if(msg.type==="snapshot"){
        if(msg.npu) setNpu(msg.npu);
        if(msg.progress){
          const pct = msg.progress.pct || 0;
          $("progBar").style.width = fmtPct01(pct);
        }
        return;
      }

      if(msg.type==="log"){
        log(msg.line);
      }

      if(msg.type==="progress"){
        const p = msg.progress || {};
        const pct = p.pct || 0;
        $("progBar").style.width = fmtPct01(pct);
        $("etaChip").textContent = "ETA: " + (p.eta || "--");
        $("stepChip").textContent = `Step: ${p.step||"--"}/${p.steps||"--"}`;
        $("progHint").textContent = `Progress: ${Math.round(pct*100)}%`;

        if(activeTab==="img2img" && $("blurFx")?.checked){
          applyBlurFx(pct);
        }
      }

      if(msg.type==="npu"){
        setNpu(msg.npu);
      }

      if(msg.type==="artifacts"){
        const imgs = msg.artifacts?.images || [];
        renderThumbs(imgs);
        if(imgs.length){
          showMainImage(imgs[0], "Output ready");
        }
      }

      if(msg.type==="status"){
        $("statusChip").textContent = msg.status;
        if(msg.status==="done"){
          $("btnRun").disabled = false;
          $("btnCancel").disabled = true;
          $("mainOverlay").textContent = "Done";
          clearBlurFx();
          evtSrc.close();
        }
        if(msg.status==="error" || msg.status==="cancelled"){
          $("btnRun").disabled = false;
          $("btnCancel").disabled = true;
          $("mainOverlay").textContent = msg.status.toUpperCase();
          clearBlurFx();
          evtSrc.close();
        }
      }
    };

    evtSrc.onerror = ()=>{
      try{ evtSrc.close(); }catch(e){}
    };
  }

  function renderThumbs(paths){
    const strip = $("thumbstrip");
    strip.innerHTML = "";
    paths.forEach(p=>{
      const d = document.createElement("div");
      d.className = "thumb";
      d.innerHTML = `<img src="${thumbUrl(p, 256, 256)}" alt=""/>`;

      d.onclick = ()=> showMainImage(p, p);
      strip.appendChild(d);
    });
  }

  function showMainImage(path, label){
    currentMainPath = path; // NEW
    $("mainImg").src = thumbUrl(path, 1400, 1400);

    $("mainImg").style.display = "block";
    $("mainOverlay").textContent = label || "";
    $("imgHint").textContent = path;
  }

  async function cancelJob(){
    if(!jobId) return;
    await fetch("/api/jobs/cancel", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({job_id: jobId})});
  }

  async function openRuns(){
    const r = await fetch("/api/runs/list");
    const data = await r.json();
    const runs = data.runs || [];
    if(!runs.length){
      alert("No runs found yet.");
      return;
    }
    const imgs = runs[0].images || [];
    if(imgs.length){
      renderThumbs(imgs);
      showMainImage(imgs[0], "Latest run");
    }
    log(`[UI] latest run: ${runs[0].name} (${runs[0].path})`);
  }

  // ---------- init ----------
  $("btnRun").onclick = runActiveTab;
  $("btnCancel").onclick = cancelJob;
  $("btnOpenRuns").onclick = openRuns;

  window.openFileBrowser = openFileBrowser;
  window.loadBrowser = loadBrowser;
  window.selectPath = selectPath;

  // NEW: click-to-open full res
  $("mainImgBox").onclick = ()=>{
    if(!currentMainPath) return;
    const url = fileUrl(currentMainPath);

    window.open(url, "_blank");
  };

  setControls(activeTab);
  resetUI();
</script>

</body>
</html>
"""

@bp_ui.get("/")
def index():
    out_root = current_app.config["OUT_ROOT"]
    html = UI_HTML.replace("__OUT_ROOT_JSON__", json.dumps(out_root))
    return Response(html, mimetype="text/html")
