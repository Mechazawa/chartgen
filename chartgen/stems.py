"""Source separation shared by lyrics, solos, taps, rescue and guitar texture.

Two backends behind one contract. Callers always receive mono float32 at
`SR` under the four Demucs-era names (drums, bass, other, vocals), so no
consumer needs to know which model ran.

- "sw"     BS-RoFormer SW, six stems, THE DEFAULT. Better on every stem we consume
           (vocals 11.3 vs 8.3 SDR, bass 14.6 vs 12.1, drums 14.1 vs 11.2)
           and it adds a real guitar stem (9.05), which is what made
           "is a strummed guitar playing chords here" answerable at all:
           on the local library, guitar songs measure 32-33% guitar-stem
           energy against 0.0% for synth songs, where every chart-level
           signal had measured identical. Runs ~34-48s per song on an
           RTX 3060 Ti via the bs-roformer-infer CLI (MIT code, ~700MB
           weights auto-cached under ~/.cache/bs-roformer-infer).
- "demucs" htdemucs, four stems, ~8s per song. The fallback, and the
           reference the original thresholds were calibrated against.

SW became the DEFAULT on 2026-09-02 after a controlled overnight
recalibration - same songs, same human-chart truth, same sweep code, only
the separator differing - cleared every consumer:
- taps (799 songs, 4733 sections): SW 42%p/44%r vs Demucs 40%p/40%r at
  each backend's optimum, 42/18 vs 41/16 at the production thresholds;
  the tapped-vs-untapped share gap is wider (0.079 vs 0.056) and the
  production rule behaves the same under both, so nothing moved
  underneath calibrated features;
- solos: a wash on the old lead rule (62/11 vs 61/11), but the guitar stem
  SW alone provides gives chartgen.solo a rule at 82% precision held-out
  with ~1 quiet fire per 50 solo-less songs, where the Demucs-era family
  could not pass 2% recall without firing on 14-22 of 100;
- vocals and stem rescue: 0.99 vs 0.96 AUC and 0.28 vs 0.13 precision in
  the 12-song A/B.
Cost: ~37s vs ~8s per song, paid once - the guitar pass no longer
separates a second time.

THE COMPATIBILITY POINT: SW carves guitar and piano OUT of "other", so its
"other" is not the same thing Demucs called "other" - a solo detector
z-scoring "other" energy would silently stop seeing lead guitar. So the
shim rebuilds a Demucs-equivalent residual, other = other+guitar+piano,
and exposes the parts separately as extra keys. Same semantics, better
components; consumers that want the guitar ask for it by name.

Separation failing must never fail a chart - callers treat None as "no
stems, use the old path".
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_SEPARATOR = None
_CACHE: dict[tuple, dict] = {}
_CACHE_LIMIT = 2

SR = 44100  # both backends are native 44.1k
DEMUCS_STEMS = ("drums", "bass", "other", "vocals")
SW_EXTRA = ("guitar", "piano")
# Set CHARTGEN_STEMS=demucs|sw to pin a backend (the A/B harness uses this).
BACKEND_ENV = "CHARTGEN_STEMS"
SW_MODEL = "roformer-model-bs-roformer-sw-by-jarredou"
_SW = None  # (model, config, device) once loaded in-process


def _sw_model():
    """The SW separator, loaded once per process and kept resident.

    Mirrors the CLI's construction exactly (same config loader, same
    checkpoint, cudnn.benchmark on, eval) so the in-process path produces
    the stems the subprocess produced - verified bit-identical on a 30 s
    clip and a full song when this landed. Loading once saves the ~19 s
    per song the subprocess spent on interpreter start, torch import, CUDA
    init and re-reading the 700 MB checkpoint.
    """
    global _SW
    if _SW is not None:
        return _SW
    import contextlib
    import io

    import torch
    import yaml
    from bs_roformer import ensure_model_assets, get_model_from_config
    from bs_roformer.inference import SafeLoaderWithTuple
    from ml_collections import ConfigDict

    # The package prints download and progress text; keep the app log clean.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model_path, config_path = ensure_model_assets(SW_MODEL)
        with open(config_path) as fh:
            config = ConfigDict(yaml.load(fh, Loader=SafeLoaderWithTuple))
        model = get_model_from_config("bs_roformer", config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    _SW = (model, config, device)
    return _SW


def _separate_sw_inprocess(y, progress) -> dict | None:
    """Six raw SW stems from a (2, n) float32 mix at SR, via the resident
    model. Returns {stem: mono float32} or None when the model cannot load
    (the caller then falls back to the CLI subprocess).

    The input goes through an in-memory 16-bit WAV exactly as the file
    the CLI used to read (PCM_16 write, float64 read), and the outputs are
    averaged to mono from the same float32 arrays the CLI wrote to disk,
    so nothing about the numbers changes - only where they travel.
    """
    import contextlib
    import io

    import soundfile as sf
    import torch
    from bs_roformer import demix_track

    try:
        model, config, device = _sw_model()
    except Exception as error:
        progress(f"      SW in-process load failed ({type(error).__name__}: "
                 f"{str(error)[:60]}); using the CLI")
        return None
    buf = io.BytesIO()
    sf.write(buf, y.T, SR, format="WAV")
    buf.seek(0)
    mix, _ = sf.read(buf)
    mixture = torch.tensor(mix.T, dtype=torch.float32)
    progress("      separating stems (BS-RoFormer SW)")
    # The CLI ran with cudnn.benchmark on; keep that for the separation
    # itself but restore the process-wide setting afterwards. Left on, it
    # changed the algorithm choice of the lyric aligner's convolutions
    # (torchaudio MMS on CUDA) and moved a word boundary by ~0.1 s - the
    # one output difference the in-process path produced.
    benchmark_before = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res, _ = demix_track(config, model, mixture, device)
    finally:
        torch.backends.cudnn.benchmark = benchmark_before
    return {name: res[name].T.mean(axis=1) for name in DEMUCS_STEMS + SW_EXTRA}


def _demucs():
    global _SEPARATOR
    if _SEPARATOR is None:
        import torch
        import demucs.api

        _SEPARATOR = demucs.api.Separator(
            model="htdemucs",
            device="cuda" if torch.cuda.is_available() else "cpu")
    return _SEPARATOR


def _separate_demucs(audio_path: str, progress) -> dict | None:
    try:
        progress("      separating stems (Demucs)")
        _, stems = _demucs().separate_audio_file(audio_path)
    except Exception as error:
        progress(f"      stem separation unavailable "
                 f"({type(error).__name__}: {error})")
        return None
    return {name: wav.mean(dim=0).cpu().numpy().astype(np.float32)
            for name, wav in stems.items()}


def _separate_sw(audio_path: str, progress) -> dict | None:
    """Six SW stems, remapped onto the Demucs contract (see module docs)."""
    try:
        import librosa
        import soundfile as sf

        y, _ = librosa.load(str(audio_path), sr=SR, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])
        raw = _separate_sw_inprocess(y, progress)
    except Exception as error:
        progress(f"      SW in-process separation unavailable "
                 f"({type(error).__name__}: {error})")
        raw = None
    if raw is None:
        raw = _separate_sw_cli(y, progress)
    if raw is None:
        return None
    return _shim(raw)


def _shim(raw: dict) -> dict:
    n = min(len(v) for v in raw.values())
    out = {name: raw[name][:n] for name in DEMUCS_STEMS}
    # Demucs-equivalent residual: guitar and piano belong in "other" for
    # every consumer calibrated before SW existed.
    out["other"] = raw["other"][:n] + raw["guitar"][:n] + raw["piano"][:n]
    out["guitar"] = raw["guitar"][:n]
    out["piano"] = raw["piano"][:n]
    out["sw_other"] = raw["other"][:n]  # the narrow residual, for new work
    return out


def _separate_sw_cli(y, progress) -> dict | None:
    """The original subprocess path, kept as the fallback."""
    try:
        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="chartgen_sw_") as tmp:
            tmp = Path(tmp)
            (tmp / "in").mkdir()
            sf.write(str(tmp / "in" / "song.wav"), y.T, SR)
            progress("      separating stems (BS-RoFormer SW, subprocess)")
            run = subprocess.run(
                [str(Path(sys.executable).parent / "bs-roformer-infer"),
                 "--input_folder", str(tmp / "in"),
                 "--store_dir", str(tmp / "out")],
                capture_output=True, text=True, timeout=1800)
            if run.returncode != 0:
                tail = (run.stderr.strip().splitlines() or ["separator failed"])[-1]
                progress(f"      SW separation failed ({tail[:80]})")
                return None
            raw = {}
            for name in DEMUCS_STEMS + SW_EXTRA:
                path = tmp / "out" / f"song_{name}.wav"
                if not path.is_file():
                    progress(f"      SW output missing {name}; using Demucs")
                    return None
                data, _ = sf.read(str(path), dtype="float32", always_2d=True)
                raw[name] = data.mean(axis=1)
    except Exception as error:
        progress(f"      SW separation unavailable "
                 f"({type(error).__name__}: {error})")
        return None
    return raw


def separate(audio_path: str, progress=lambda m: None,
             backend: str | None = None) -> dict | None:
    """{'vocals': mono float32 @44100, 'other': ..., ...} or None on failure.

    With SW available the dict also carries 'guitar', 'piano' and
    'sw_other'. backend: 'sw', 'demucs', or None for the env/default.
    """
    backend = backend or os.environ.get(BACKEND_ENV) or "sw"
    key = (str(audio_path), backend)
    if key in _CACHE:
        return _CACHE[key]

    from . import cache as diskcache

    disk_key = None
    mono = None
    if backend == "sw":
        try:
            disk_key = f"sw:{SW_MODEL}:{diskcache.audio_key(str(audio_path))}"
            raw = diskcache.load_stems(disk_key)
        except Exception:
            raw = None
        if raw is not None and all(k in raw for k in DEMUCS_STEMS + SW_EXTRA):
            progress("      stems: cached separation reused")
            mono = _shim(raw)
        else:
            mono = _separate_sw(audio_path, progress)
            if mono is not None and disk_key:
                diskcache.save_stems(disk_key, {k: mono[k] for k in
                                                ("drums", "bass", "vocals", "guitar", "piano")}
                                     | {"other": mono["sw_other"]})
        if mono is None:
            progress("      falling back to Demucs stems")
    if mono is None:
        mono = _separate_demucs(audio_path, progress)
    if mono is None:
        return None
    mono["_key"] = disk_key or f"demucs:{diskcache.audio_key(str(audio_path))}"

    while len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = mono
    return mono


def activity(stem: np.ndarray, sr: int = SR, frame_s: float = 0.05):
    """(times, rms) envelope of a stem, for singing/lead detection."""
    hop = max(1, int(frame_s * sr))
    n = len(stem) // hop
    frames = stem[:n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    times = (np.arange(n) + 0.5) * frame_s
    return times, rms


def singing_share(vocal: np.ndarray, sr: int, t0: float, t1: float,
                  floor_ratio: float = 0.1) -> float:
    """Share of [t0, t1] where the vocal stem is audibly singing.

    The threshold is relative to the song's own loud singing (p90 of the
    vocal envelope), so a breathy verse still counts and stem bleed in an
    instrumental does not.
    """
    times, rms = activity(vocal, sr)
    loud = float(np.percentile(rms, 90)) or 1.0
    window = (times >= t0) & (times < t1)
    if not window.any():
        return 0.0
    return float(np.mean(rms[window] > floor_ratio * loud))
