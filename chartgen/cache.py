"""On-disk cache for the two expensive, deterministic per-song analyses:
stem separation and Basic Pitch transcriptions.

Both are pure functions of the audio bytes (and the model that ran), and
both were measured bit-identical run to run (the SW separator CLI twice on
the same clip: every stem equal; two full deterministic pipeline runs: the
same chart). Caching them therefore cannot change a chart - it only skips
recomputing what would come out the same. Stems are stored as float32,
exactly as the separator produced them: float16 or FLAC would quantise the
values and that WOULD change downstream transcriptions.

Layout under CHARTGEN_CACHE_DIR (default $XDG_CACHE_HOME/chartgen/cache):
    stems/<key>.npz            raw separator outputs, one mono float32 per stem
    transcriptions/<key>.json  [(start, end, pitch, amplitude), ...]

A song's stems are ~230 MB, so the stems store is capped (CHARTGEN_STEM_CACHE_GB,
default 6) with least-recently-used eviction; CHARTGEN_STEM_CACHE=0 disables it.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

_AUDIO_KEYS: dict[tuple, str] = {}


def cache_dir() -> Path:
    override = os.environ.get("CHARTGEN_CACHE_DIR")
    if override:
        return Path(override)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "chartgen" / "cache"


def enabled() -> bool:
    return os.environ.get("CHARTGEN_STEM_CACHE", "1") not in ("0", "false", "no")


def size_cap_bytes() -> int:
    try:
        return int(float(os.environ.get("CHARTGEN_STEM_CACHE_GB", "6")) * 2**30)
    except ValueError:
        return 6 * 2**30


def audio_key(audio_path: str) -> str:
    """sha256 of the file bytes (memoised per path + size + mtime)."""
    p = Path(audio_path)
    st = p.stat()
    memo = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if memo in _AUDIO_KEYS:
        return _AUDIO_KEYS[memo]
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    key = h.hexdigest()[:32]
    _AUDIO_KEYS[memo] = key
    return key


def _safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key)


# ------------------------------------------------------------------- stems
def load_stems(key: str) -> dict | None:
    if not enabled():
        return None
    path = cache_dir() / "stems" / f"{_safe(key)}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            out = {name: np.ascontiguousarray(data[name], dtype=np.float32)
                   for name in data.files}
        os.utime(path)  # LRU touch
        return out
    except Exception:
        return None


def save_stems(key: str, raw: dict) -> None:
    """raw: {stem: mono float32 array}. Never raises."""
    if not enabled():
        return
    try:
        folder = cache_dir() / "stems"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{_safe(key)}.npz"
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **{k: np.asarray(v, dtype=np.float32) for k, v in raw.items()})
        tmp.replace(path)
        _evict(folder, size_cap_bytes(), keep=path)
    except Exception:
        pass


def _evict(folder: Path, cap: int, keep: Path) -> None:
    files = [p for p in folder.glob("*.npz") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    for p in sorted(files, key=lambda p: p.stat().st_mtime):
        if total <= cap:
            break
        if p == keep:
            continue
        try:
            size = p.stat().st_size
            p.unlink()
            total -= size
        except OSError:
            pass


# ---------------------------------------------------------- transcriptions
def load_transcription(key: str) -> list | None:
    if not enabled():
        return None
    path = cache_dir() / "transcriptions" / f"{_safe(key)}.json"
    if not path.is_file():
        return None
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
        return [(float(s), float(e), int(p), float(a)) for s, e, p, a in events]
    except Exception:
        return None


def save_transcription(key: str, events: list) -> None:
    if not enabled():
        return
    try:
        folder = cache_dir() / "transcriptions"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{_safe(key)}.json"
        tmp = path.with_suffix(".tmp")
        # json round-trips Python floats exactly (repr), so a hit reproduces
        # the same tuples the model returned.
        tmp.write_text(json.dumps([[s, e, p, a] for s, e, p, a in events]),
                       encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass
