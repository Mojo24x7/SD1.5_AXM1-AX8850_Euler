import os, re, time, uuid, queue, shlex, signal, threading, subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

RE_STEP = re.compile(r"\bstep\s+(\d+)\s*/\s*(\d+)\b", re.IGNORECASE)
RE_ETA  = re.compile(r"\bETA=([0-9:]+)\b")

# IMPORTANT: NPU info is split over lines in your logs.
RE_TEMP = re.compile(r"\b(?:T|Temp)\s*[:=]\s*(\d+)\s*°?\s*C\b", re.IGNORECASE)

RE_CPU  = re.compile(r"\bCPU\s*[:=]\s*(\d+)\s*%", re.IGNORECASE)
RE_NPU  = re.compile(r"\bNPU\s*[:=]\s*(\d+)\s*%", re.IGNORECASE)

RE_MEMCMM = re.compile(
    r"\bMem=(\d+)\s*/\s*(\d+)MiB\b.*?\bCMM=(\d+)\s*/\s*(\d+)MiB\b",
    re.IGNORECASE
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS

def find_rep_images(run_dir: str, limit: int = 12) -> List[str]:
    imgs = []
    try:
        for root, _, files in os.walk(run_dir):
            for fn in files:
                p = os.path.join(root, fn)
                if is_image(p):
                    imgs.append(p)
    except Exception:
        pass
    imgs.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
    return imgs[:limit]

class Job:
    def __init__(self, job_id: str, kind: str, run_dir: str, cmd: List[str], env: Optional[Dict[str, str]] = None):
        self.job_id = job_id
        self.kind = kind
        self.run_dir = run_dir
        self.cmd = cmd
        self.env = env or {}
        self.created = now_iso()
        self.started = None
        self.ended = None
        self.returncode = None
        self.status = "queued"

        self.progress = {"step": 0, "steps": 0, "eta": "", "pct": 0.0}
        self.npu = {
            "temp_c": None, "cpu_pct": None, "npu_pct": None,
            "mem_used": None, "mem_total": None, "cmm_used": None, "cmm_total": None,
        }

        self._q = queue.Queue(maxsize=5000)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def push(self, msg: Dict[str, Any]):
        try:
            self._q.put_nowait(msg)
        except queue.Full:
            pass

    def pop(self, timeout=1.0):
        return self._q.get(timeout=timeout)

    def set_proc(self, proc: subprocess.Popen):
        with self._lock:
            self._proc = proc

    def kill(self):
        with self._lock:
            proc = self._proc
        if not proc:
            return
        try:
            if proc.poll() is not None:
                return
        except Exception:
            pass

        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            pgid = None

        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            pass

        time.sleep(0.4)

        try:
            if proc.poll() is None:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
        except Exception:
            pass

JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()

def parse_progress(line: str) -> Optional[Dict[str, Any]]:
    prog = None

    m = RE_STEP.search(line)
    if m:
        s = int(m.group(1))
        S = int(m.group(2))
        pct = (float(s) / float(S)) if S > 0 else 0.0
        prog = {"step": s, "steps": S, "pct": pct}

    m2 = RE_ETA.search(line)
    if m2:
        if prog is None:
            prog = {}
        prog["eta"] = m2.group(1)

    return prog

def parse_npu_partial(line: str) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}

    mt = RE_TEMP.search(line)
    if mt:
        out["temp_c"] = int(mt.group(1))

    mc = RE_CPU.search(line)
    if mc:
        out["cpu_pct"] = int(mc.group(1))

    mn = RE_NPU.search(line)
    if mn:
        out["npu_pct"] = int(mn.group(1))

    m2 = RE_MEMCMM.search(line)
    if m2:
        out["mem_used"] = int(m2.group(1))
        out["mem_total"] = int(m2.group(2))
        out["cmm_used"] = int(m2.group(3))
        out["cmm_total"] = int(m2.group(4))

    return out or None


def _write_command_txt(job: Job):
    try:
        with open(os.path.join(job.run_dir, "command.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(shlex.quote(x) for x in job.cmd) + "\n")
            if job.env:
                f.write("\n# env overrides:\n")
                for k, v in job.env.items():
                    f.write(f"{k}={v}\n")
    except Exception:
        pass

def run_job(job: Job):
    job.status = "running"
    job.started = now_iso()
    job.push({"type": "status", "status": job.status, "started": job.started})

    safe_mkdir(job.run_dir)
    _write_command_txt(job)

    env = os.environ.copy()
    env.update(job.env or {})
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            job.cmd,
            cwd=job.run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
            start_new_session=True,
        )
        job.set_proc(proc)

        job.push({"type": "log", "line": f"[UI] started: {now_iso()}"})
        job.push({"type": "log", "line": f"[UI] cwd: {job.run_dir}"})
        job.push({"type": "log", "line": f"[UI] cmd: {' '.join(job.cmd)}"})

        assert proc.stdout is not None

        for raw in iter(proc.stdout.readline, ""):
            if raw == "":
                break
            line = raw.rstrip("\n")
            if not line:
                continue

            job.push({"type": "log", "line": line})

            prog = parse_progress(line)
            if prog:
                job.progress.update(prog)
                job.push({"type": "progress", "progress": job.progress})

            npu_part = parse_npu_partial(line)
            if npu_part:
                job.npu.update(npu_part)
                job.push({"type": "npu", "npu": job.npu})

        rc = proc.wait()
        job.returncode = rc
        job.ended = now_iso()
        job.status = "done" if rc == 0 else "error"

        artifacts = {"run_dir": job.run_dir, "images": find_rep_images(job.run_dir)}
        job.push({"type": "artifacts", "artifacts": artifacts})
        job.push({"type": "status", "status": job.status, "ended": job.ended, "returncode": rc})

    except Exception as ex:
        job.status = "error"
        job.ended = now_iso()
        job.push({"type": "log", "line": f"[UI] ERROR: {ex}"})
        job.push({"type": "status", "status": job.status, "ended": job.ended, "returncode": -1})

def spawn_job(kind: str, run_dir: str, cmd: List[str], env: Optional[Dict[str, str]] = None) -> Job:
    job_id = str(uuid.uuid4())
    job = Job(job_id=job_id, kind=kind, run_dir=run_dir, cmd=cmd, env=env)
    with JOBS_LOCK:
        JOBS[job_id] = job
    t = threading.Thread(target=run_job, args=(job,), daemon=True)
    t.start()
    return job
