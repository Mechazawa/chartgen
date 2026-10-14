"""Remote web UI: paste links or upload audio from any device, get the chart back.

    python -m chartgen.web                 # serves on http://127.0.0.1:8471
    python -m chartgen.web --host 0.0.0.0  # also reachable over the LAN / Tailscale

Runs on the PC with the GPU. Two ways to reach it:

- Behind a Cloudflare Tunnel + Cloudflare Access (the intended setup for use
  from anywhere): Access authenticates the visitor and forwards their email
  in the `Cf-Access-Authenticated-User-Email` header. Each visitor then sees
  only their own queue, and finished songs download to their device
  automatically. Emails listed in CHARTGEN_WEB_OWNERS (comma-separated) are
  the PC's owners: their songs are saved to the library by default. Anyone
  else is a guest: their songs are charted in a scratch folder, handed over
  as a zip, and deleted afterwards - nothing of theirs stays on the PC unless
  they tick "save to this PC's library".
- Over a private network with no Access in front (Tailscale, LAN): there is
  no login, every visitor is treated as the owner, and the page behaves like
  a shared queue. Never port-forward this to the open internet.

One worker thread owns the GPU, mirroring the desktop app's batch behaviour
(per-song metadata, skip-if-charted).
"""
import argparse
import io
import os
import queue as queuemod
import shutil
import threading
import time
import uuid
import zipfile
from argparse import Namespace
from collections import deque
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file

from . import pipeline, youtube
from .app import GRIDS, load_settings  # reuse the desktop defaults

app = Flask("chartgen")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # one upload; audio never needs more

ACCESS_HEADER = "Cf-Access-Authenticated-User-Email"
OWNERS = {e.strip().lower() for e in os.environ.get("CHARTGEN_WEB_OWNERS", "").split(",") if e.strip()}
GUEST_TTL_S = 24 * 3600  # scratch folders a guest never downloaded

JOBS: "queuemod.Queue[dict]" = queuemod.Queue()
LOG = deque(maxlen=400)
STATE = {"current": None, "jobs": []}  # jobs: newest last, each a dict (see new_job)
LOCK = threading.Lock()


def log(message: str) -> None:
    with LOCK:
        LOG.append(f"{time.strftime('%H:%M:%S')}  {message}")


def outdir() -> Path:
    saved = load_settings()
    return Path(saved.get("outdir", str(Path("out").resolve())))


def guest_root() -> Path:
    return outdir() / "_guest"


def base_opts(target: Path) -> Namespace:
    saved = load_settings()
    return Namespace(
        audio=None, outdir=target,
        subdiv=GRIDS.get(saved.get("grid"), 4),
        min_variety=0.80,
        hopos=bool(saved.get("hopos", True)),
        no_star_power=not saved.get("star_power", True),
        no_sections=not saved.get("sections", True),
        no_sustains=not saved.get("sustains", True),
        lyrics=bool(saved.get("lyrics", True)),
        ghl=bool(saved.get("ghl", True)),
        target_diff=None, skip_existing=True,
        name=None, artist="Unknown", album="", genre="", year="",
    )


# ------------------------------------------------------------------ identity
def visitor() -> str:
    """Who is asking: the Access email, or 'local' when nothing is in front."""
    return (request.headers.get(ACCESS_HEADER) or "local").strip().lower()


def is_owner(who: str) -> bool:
    return who == "local" or who in OWNERS


def new_job(who: str, source, label: str, keep: bool) -> dict:
    job = {
        "id": uuid.uuid4().hex[:12], "owner": who, "input": source, "label": label,
        "keep": keep, "status": "queued", "song": None, "song_dir": None,
        "error": None, "added": time.time(),
    }
    with LOCK:
        STATE["jobs"].append(job)
        if len(STATE["jobs"]) > 500:
            del STATE["jobs"][:-500]
    JOBS.put(job)
    return job


def public(job: dict) -> dict:
    return {k: job[k] for k in ("id", "label", "keep", "status", "song", "error")}


# -------------------------------------------------------------------- worker
def worker() -> None:
    import copy

    while True:
        job = JOBS.get()
        target = outdir() if job["keep"] else guest_root() / job["id"]
        with LOCK:
            STATE["current"] = job
            job["status"] = "charting"
        opts = copy.copy(base_opts(target))
        opts.audio = job["input"]
        try:
            result = pipeline.run(opts, progress=log)
            with LOCK:
                job["song_dir"] = str(result["song_dir"])
                job["song"] = Path(result["song_dir"]).name
                job["status"] = "skipped" if result.get("skipped") else "done"
        except Exception as error:
            log(f"FAILED: {type(error).__name__}: {error}")
            with LOCK:
                job["status"] = "failed"
                job["error"] = f"{type(error).__name__}: {error}"
        finally:
            with LOCK:
                STATE["current"] = None
            if not job["keep"] and isinstance(job["input"], Path):
                # A guest's upload has done its job; do not keep their audio.
                try:
                    job["input"].unlink()
                except OSError:
                    pass
            sweep_guests()


def sweep_guests() -> None:
    """Drop scratch folders nobody downloaded within GUEST_TTL_S."""
    root = guest_root()
    if not root.is_dir():
        return
    with LOCK:
        live = {j["id"] for j in STATE["jobs"] if j["status"] in ("queued", "charting")}
    for folder in root.iterdir():
        try:
            if folder.name not in live and time.time() - folder.stat().st_mtime > GUEST_TTL_S:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass


def zip_folder(folder: Path, name: str) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in folder.iterdir():
            if f.is_file():
                zf.write(f, arcname=f"{name}/{f.name}")
    buffer.seek(0)
    return buffer


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>chartgen</title><style>
:root{
  --bg:#0d0f12; --panel:#15181d; --panel-2:#1b1f26; --line:#262b33;
  --text:#e8eaed; --muted:#8b939e; --accent:#34d17b; --accent-dim:#1f7a4b;
  --amber:#e8b84b; --red:#e2665e; --radius:12px;
}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);
  color:var(--text);margin:0;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:680px;margin:0 auto;padding:0 20px 60px}
header{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.bar{max-width:680px;margin:0 auto;padding:14px 20px;display:flex;align-items:center;gap:12px}
.logo{width:26px;height:26px;flex:none}
h1{font-size:17px;font-weight:650;margin:0;letter-spacing:-.01em}
h1 small{color:var(--muted);font-weight:400;margin-left:8px;font-size:13px}
.pill{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:13px;
  padding:5px 12px;border-radius:99px;border:1px solid var(--line);color:var(--muted);
  background:var(--panel);white-space:nowrap}
.pill.busy{color:var(--amber);border-color:color-mix(in srgb,var(--amber) 35%,var(--line))}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted);flex:none}
.busy .dot{background:var(--amber);animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{50%{opacity:.35}}
section{margin-top:28px}
.label{font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);margin-bottom:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px}
textarea{width:100%;min-height:96px;resize:vertical;background:var(--panel-2);
  color:var(--text);border:1px solid var(--line);border-radius:8px;padding:12px;
  font:inherit;font-size:15px}
textarea::placeholder{color:var(--muted)}
textarea:focus,button:focus-visible{outline:2px solid var(--accent-dim);outline-offset:1px;border-color:transparent}
.row{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
button{font:inherit;font-size:14px;font-weight:600;border:none;border-radius:8px;
  padding:10px 18px;cursor:pointer;background:var(--accent);color:#08130c;
  transition:filter .15s}
button:hover{filter:brightness(1.1)}
button:disabled{opacity:.45;cursor:default;filter:none}
button.ghost{background:var(--panel-2);color:var(--text);border:1px solid var(--line)}
.or{color:var(--muted);font-size:13px}
.drop{margin-top:12px;border:1.5px dashed var(--line);border-radius:8px;padding:18px;
  text-align:center;color:var(--muted);font-size:14px;cursor:pointer;transition:border-color .15s,background .15s}
.drop:hover,.drop.over{border-color:var(--accent-dim);background:var(--panel-2);color:var(--text)}
.now{display:flex;align-items:center;gap:12px;padding:14px 16px}
.eq{display:flex;gap:3px;align-items:flex-end;height:18px;flex:none}
.eq span{width:4px;border-radius:2px;background:var(--accent);animation:eq 1s ease-in-out infinite}
.eq span:nth-child(2){animation-delay:.2s}.eq span:nth-child(3){animation-delay:.4s}
@keyframes eq{0%,100%{height:6px}50%{height:18px}}
.now .what{min-width:0}
.now .name{font-weight:600;font-size:14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.now .sub{color:var(--muted);font-size:12.5px}
details{margin-top:10px}
summary{cursor:pointer;color:var(--muted);font-size:13px;user-select:none}
pre{background:var(--panel-2);border:1px solid var(--line);margin:10px 0 0;padding:12px;
  border-radius:8px;font-size:12px;line-height:1.6;max-height:240px;overflow:auto;
  white-space:pre-wrap;color:var(--muted);font-family:ui-monospace,Consolas,monospace}
.song{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--line)}
.song:last-child{border-bottom:none}
.song .name{flex:1;min-width:0;font-size:14.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.song .note{color:var(--muted);font-size:12.5px}
.song.fail .name{color:var(--red);white-space:normal;font-size:13px}
a.dl{flex:none;display:flex;align-items:center;gap:6px;text-decoration:none;font-size:13px;
  font-weight:600;color:var(--accent);border:1px solid var(--line);border-radius:8px;
  padding:7px 12px;transition:background .15s}
a.dl:hover{background:var(--panel-2)}
.empty{padding:22px 16px;color:var(--muted);font-size:14px;text-align:center}
.cardlist{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden}
.count{color:var(--muted);font-weight:400}
.who{font-size:12.5px;color:var(--muted);margin-left:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:40vw}
label.keep{display:flex;align-items:center;gap:8px;font-size:13.5px;color:var(--muted);cursor:pointer;margin-left:auto}
label.keep input{accent-color:var(--accent);width:16px;height:16px;margin:0}
.song .st{flex:none;font-size:12.5px;color:var(--muted)}
.song.busy .st{color:var(--amber)}
.song.gone .name{color:var(--muted)}
.hint{color:var(--muted);font-size:12.5px;margin-top:10px}
</style></head><body>
<header><div class="bar">
<svg class="logo" viewBox="0 0 24 24" fill="none" aria-hidden="true">
  <rect x="2" y="3" width="20" height="18" rx="4" fill="#1f7a4b"/>
  <rect x="5.5" y="7" width="3" height="10" rx="1.5" fill="#34d17b"/>
  <rect x="10.5" y="10" width="3" height="7" rx="1.5" fill="#e8b84b"/>
  <rect x="15.5" y="6" width="3" height="11" rx="1.5" fill="#e2665e"/>
</svg>
<h1>chartgen<small>Clone Hero chart generator</small></h1><span class="who" id="who"></span>
<div class="pill" id="pill"><span class="dot"></span><span id="pilltext">Idle</span></div>
</div></header>
<div class="wrap">

<section>
  <div class="label">Add songs</div>
  <div class="card">
    <form id="f">
      <textarea id="links" placeholder="Paste YouTube links or a playlist URL — one per line"></textarea>
      <div class="row">
        <button type="submit" id="chartbtn">Chart them</button>
        <span class="or">or drop audio files below</span>
        <label class="keep"><input type="checkbox" id="keep"> save to this PC's library</label>
      </div>
    </form>
    <div class="drop" id="drop">Click to choose audio files — or drag them here</div>
    <input type="file" id="file" accept="audio/*" multiple hidden>
    <div class="hint" id="hint"></div>
  </div>
</section>

<section id="activity" hidden>
  <div class="label">Now charting</div>
  <div class="card now">
    <div class="eq" aria-hidden="true"><span></span><span></span><span></span></div>
    <div class="what">
      <div class="name" id="curname"></div>
      <div class="sub" id="cursub"></div>
    </div>
  </div>
  <details><summary>Show log</summary><pre id="log"></pre></details>
</section>

<section>
  <div class="label">Your songs <span class="count" id="donecount"></span></div>
  <div class="cardlist" id="done"><div class="empty">Nothing charted yet — paste a link above.</div></div>
</section>

</div>
<script>
const $=id=>document.getElementById(id);
const f=$('f'), drop=$('drop'), file=$('file'), btn=$('chartbtn'), keep=$('keep');
let me={who:null,owner:true};
// Songs this browser already pulled down, so a refresh never re-downloads them.
const got=new Set(JSON.parse(localStorage.getItem('cg_got')||'[]'));
function remember(id){got.add(id);localStorage.setItem('cg_got',JSON.stringify([...got].slice(-200)));}

async function whoami(){
  try{me=await(await fetch('/me')).json();}catch(e){}
  keep.checked=me.owner;
  $('who').textContent=me.who?me.who:'';
  $('hint').textContent=me.owner
    ?'Finished songs download to this device automatically and stay in the library on the charting PC.'
    :'Finished songs download to this device automatically. Untick "save" is the default for guests: your song is charted, handed to you, and deleted from the PC.';
}

f.onsubmit=async e=>{e.preventDefault();
  const links=$('links').value.trim(); if(!links)return;
  btn.disabled=true;
  try{
    const r=await fetch('/jobs',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({links:links.split(/\\s+/).filter(Boolean),keep:keep.checked})});
    if(r.ok) $('links').value='';
    else alert((await r.json()).error||'Could not queue those links.');
  }finally{btn.disabled=false;}
  refresh();};

drop.onclick=()=>file.click();
['dragover','dragenter'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('over');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('over');}));
drop.addEventListener('drop',e=>sendFiles(e.dataTransfer.files));
file.onchange=()=>sendFiles(file.files);
async function sendFiles(list){
  for(const item of list){const fd=new FormData();fd.append('file',item);fd.append('keep',keep.checked);
    const r=await fetch('/upload',{method:'POST',body:fd});
    if(!r.ok) alert('Upload failed: '+((await r.json().catch(()=>({}))).error||r.status));}
  file.value='';refresh();}

function download(job){
  remember(job.id);
  const a=document.createElement('a');a.href='/jobs/'+job.id+'.zip';a.download='';
  document.body.appendChild(a);a.click();a.remove();}

const STATUS={queued:'queued',charting:'charting…',done:'',skipped:'already charted',
  failed:'failed',collected:'downloaded'};
function songRow(job){
  const row=document.createElement('div');row.className='song';
  if(job.status==='charting'||job.status==='queued')row.classList.add('busy');
  if(job.status==='collected')row.classList.add('gone');
  if(job.status==='failed')row.classList.add('fail');
  const n=document.createElement('div');n.className='name';
  n.textContent=job.status==='failed'?(job.label+' — '+job.error):(job.song||job.label);
  row.appendChild(n);
  const st=document.createElement('span');st.className='st';st.textContent=STATUS[job.status]||job.status;
  if(st.textContent)row.appendChild(st);
  if((job.status==='done'||job.status==='skipped')&&(job.keep||!got.has(job.id))){
    const a=document.createElement('a');a.className='dl';a.href='/jobs/'+job.id+'.zip';
    a.innerHTML='&#8681; zip';a.title='Download to this device';
    a.onclick=()=>remember(job.id);row.appendChild(a);}
  return row;}

async function refresh(){
  let s;try{s=await(await fetch('/status')).json();}catch(e){
    $('pilltext').textContent='PC unreachable';$('pill').classList.remove('busy');return;}
  const busy=!!s.current||s.queued>0;
  $('pill').classList.toggle('busy',busy);
  $('pilltext').textContent=s.current?('Working · '+s.queued+' queued'):(s.queued?s.queued+' queued':'Idle');
  $('activity').hidden=!busy;
  if(s.current){$('curname').textContent=s.current;
    $('cursub').textContent=s.queued?s.queued+' more in queue':'last one in the queue';}
  $('log').textContent=s.log.join('\\n');
  const done=$('done');done.innerHTML='';
  const jobs=s.jobs.slice().reverse();
  $('donecount').textContent=jobs.length?('· '+jobs.length):'';
  if(!jobs.length){done.innerHTML='<div class="empty">Nothing charted yet — paste a link above.</div>';}
  else for(const j of jobs) done.appendChild(songRow(j));
  // Auto-download: the first finished song this browser has not collected yet.
  // One per poll, so a batch arrives as separate saves rather than a burst.
  const fresh=jobs.find(j=>(j.status==='done'||j.status==='skipped')&&!got.has(j.id));
  if(fresh) download(fresh);
}
whoami().then(refresh);setInterval(refresh,2500);
</script></body></html>"""


@app.get("/")
def index() -> Response:
    return Response(PAGE, mimetype="text/html")


@app.get("/me")
def me():
    who = visitor()
    return jsonify({"who": who if who != "local" else None, "owner": is_owner(who)})


def keep_flag(default: bool) -> bool:
    payload = request.get_json(silent=True) or {}
    raw = payload.get("keep", request.form.get("keep"))
    if raw is None:
        return default
    return str(raw).lower() in ("1", "true", "yes", "on")


@app.post("/jobs")
def add_jobs():
    who = visitor()
    links = (request.get_json(silent=True) or {}).get("links", [])
    links = [l.strip() for l in links if l.strip()]
    bad = [l for l in links if not youtube.is_youtube_url(l)]
    if bad:
        return jsonify({"error": f"not YouTube links: {bad}"}), 400
    try:
        expanded = youtube.expand_inputs(links, log)
    except ValueError as error:
        return jsonify({"error": str(error)}), 502
    keep = keep_flag(is_owner(who))
    ids = [new_job(who, url, url, keep)["id"] for url in expanded]
    log(f"queued {len(expanded)} link(s) for {who}")
    return jsonify({"queued": len(expanded), "ids": ids})


@app.post("/upload")
def upload():
    who = visitor()
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "no file"}), 400
    dest = outdir() / "_uploads"
    dest.mkdir(parents=True, exist_ok=True)
    safe = Path(file.filename).name
    path = dest / safe
    if path.exists():  # two people uploading "song.mp3" must not clobber each other
        path = dest / f"{path.stem}-{uuid.uuid4().hex[:6]}{path.suffix}"
    file.save(path)
    job = new_job(who, path, safe, keep_flag(is_owner(who)))
    log(f"queued upload {safe} for {who}")
    return jsonify({"queued": 1, "ids": [job["id"]]})


@app.get("/status")
def status():
    who = visitor()
    with LOCK:
        mine = [public(j) for j in STATE["jobs"] if j["owner"] == who]
        current = STATE["current"]
        current_mine = current is not None and current["owner"] == who
        return jsonify({
            "current": (current["label"] if current_mine
                        else ("another visitor's song" if current else None)),
            "current_mine": current_mine,
            "queued": JOBS.qsize(),
            "jobs": mine,
            # The log is the charting PC's progress stream; only the person whose
            # song is on the GPU gets to read it.
            "log": list(LOG)[-60:] if current_mine else [],
        })


@app.get("/jobs/<job_id>.zip")
def job_zip(job_id: str):
    who = visitor()
    with LOCK:
        job = next((j for j in STATE["jobs"] if j["id"] == job_id), None)
        if job is None or job["owner"] != who or not job["song_dir"]:
            abort(404)
        folder, name, keep = Path(job["song_dir"]), job["song"], job["keep"]
    if not folder.is_dir():
        abort(410)  # already collected and deleted
    buffer = zip_folder(folder, name)
    if not keep:
        # Guest song: the zip in memory is now the only copy, by design.
        shutil.rmtree(folder.parent, ignore_errors=True)
        with LOCK:
            job["status"] = "collected"
    return send_file(buffer, download_name=f"{name}.zip", mimetype="application/zip")


@app.get("/songs/<name>.zip")
def song_zip(name: str):
    """Any library song, for the PC's owner (e.g. an older chart)."""
    if not is_owner(visitor()):
        abort(404)
    folder = (outdir() / name).resolve()
    if not str(folder).startswith(str(outdir().resolve())) or not folder.is_dir():
        abort(404)
    return send_file(zip_folder(folder, name), download_name=f"{name}.zip",
                     mimetype="application/zip")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 for a Cloudflare Tunnel on this PC (default); "
                         "0.0.0.0 to reach it over Tailscale / the LAN")
    ap.add_argument("--port", type=int, default=8471)
    args = ap.parse_args()

    threading.Thread(target=worker, daemon=True).start()
    log(f"chartgen web up — output folder: {outdir()}")
    if OWNERS:
        log(f"owners: {', '.join(sorted(OWNERS))}; everyone else charts as a guest")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
