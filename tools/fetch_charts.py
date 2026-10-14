"""Pull quality-filtered human charts from Chorus Encore for calibration.

Every threshold in chartgen is calibrated against the local library, which
is fewer than 200 charts of uneven quality. Encore indexes ~87k community
charts and its API carries the quality evidence needed to filter BEFORE
downloading: per-difficulty note counts (a full Easy-Expert ladder means the
charter did the whole job), the scanner's chartIssues list, taps/solos
flags, and NPS statistics.

Charts land in a calibration folder, NOT the Clone Hero songs folder - they
are measurement data, not a playlist. Audio is kept (a few GB) so future
features (e.g. stem separation) never need a re-download; a manifest
records every chart's API metadata for provenance and reproducibility.

    python tools/fetch_charts.py -o data/calibration -n 400

Downloads are throttled and the client identifies itself: Encore is a free
community service, not a CDN to hammer.
"""
import argparse
import io
import json
import random
import re
import struct
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

API = "https://api.enchor.us/search"
FILES = "https://files.enchor.us"
USER_AGENT = "chartgen/1.0 (calibration research; github.com/ErnestSarna/chartgen)"
THROTTLE_S = 1.0

# Freeform genre strings normalised into buckets, so a per-bucket cap can
# force variety: the index itself is rock/metal-heavy (the audio2chart
# paper measured its 10k-chart training set at 72% metal+rock), and
# thresholds calibrated on a skewed sample would inherit the skew.
GENRE_BUCKETS = (
    ("metal", ("metal", "metalcore", "deathcore", "djent", "thrash")),
    ("rock", ("rock", "grunge", "hard rock", "prog")),
    ("punk", ("punk", "hardcore", "emo", "ska")),
    ("electronic", ("edm", "electronic", "dance", "house", "techno", "trance",
                    "dubstep", "drum", "dnb", "synth", "eurobeat", "hardstyle")),
    ("pop", ("pop", "disco", "funk", "soul", "r&b", "rnb")),
    ("hiphop", ("hip", "rap", "trap")),
    ("jpop-anime", ("j-pop", "jpop", "k-pop", "kpop", "anime", "vocaloid",
                    "touhou", "j-rock", "jrock")),
    ("game", ("video game", "videogame", "vgm", "game", "chiptune", "8-bit")),
    ("country-folk", ("country", "folk", "bluegrass", "acoustic")),
    ("jazz-classical", ("jazz", "blues", "classical", "orchestral", "swing")),
    ("indie-alt", ("indie", "alternative", "alt ")),
)


def genre_bucket(genre: str | None) -> str:
    text = (genre or "").lower()
    for bucket, needles in GENRE_BUCKETS:
        if any(needle in text for needle in needles):
            return bucket
    return "other"


# What "worth calibrating against" means, in API terms.
MIN_LENGTH_MS, MAX_LENGTH_MS = 90_000, 480_000
LADDER = ("expert", "hard", "medium", "easy")
AUDIO_EXT = (".opus", ".ogg", ".mp3", ".wav", ".flac")
KEEP_EXT = (".chart",) + AUDIO_EXT  # art/video/backgrounds are dead weight


def api_search(page: int, per_page: int = 100) -> dict:
    body = json.dumps({"search": "", "page": page, "per_page": per_page,
                       "instrument": "guitar", "difficulty": None}).encode()
    request = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wanted(row: dict) -> str | None:
    """None if this chart earns a download; else the reason it does not."""
    notes = row.get("notesData") or {}
    if "guitar" not in (notes.get("instruments") or []):
        return "no guitar track"
    if notes.get("chartIssues"):
        return "scanner flagged issues"
    counts = {(c["instrument"], c["difficulty"]): c["count"]
              for c in notes.get("noteCounts") or []}
    if not all(counts.get(("guitar", d), 0) >= 50 for d in LADDER):
        return "not a full difficulty ladder"
    length = row.get("song_length") or 0
    if not (MIN_LENGTH_MS <= length <= MAX_LENGTH_MS):
        return "length out of range"
    if not row.get("md5"):
        return "no file hash"
    return None


def read_sng(raw: bytes) -> tuple[dict, list[tuple[str, bytes]]]:
    """(metadata, [(filename, contents)]) from a .sng bundle.

    Format per the SngFileFormat spec: 'SNGPKG' magic, version, a 16-byte
    XOR mask, metadata and file tables, then file data where each byte i is
    masked with mask[i % 16] ^ (i & 0xFF) - verified against a live file
    (decodes to the UTF-8 BOM and '[Song]').
    """
    buf = io.BytesIO(raw)
    if buf.read(6) != b"SNGPKG":
        raise ValueError("not a .sng file")
    struct.unpack("<I", buf.read(4))
    mask = buf.read(16)

    _, meta_count = struct.unpack("<QQ", buf.read(16))
    meta = {}
    for _ in range(meta_count):
        klen = struct.unpack("<i", buf.read(4))[0]
        key = buf.read(klen).decode("utf-8", "replace")
        vlen = struct.unpack("<i", buf.read(4))[0]
        meta[key] = buf.read(vlen).decode("utf-8", "replace")

    _, file_count = struct.unpack("<QQ", buf.read(16))
    entries = []
    for _ in range(file_count):
        nlen = struct.unpack("<B", buf.read(1))[0]
        name = buf.read(nlen).decode("utf-8", "replace")
        clen, cidx = struct.unpack("<QQ", buf.read(16))
        entries.append((name, clen, cidx))

    files = []
    for name, clen, cidx in entries:
        data = bytearray(raw[cidx:cidx + clen])
        for i in range(len(data)):
            data[i] ^= mask[i % 16] ^ (i & 0xFF)
        files.append((name, bytes(data)))
    return meta, files


def folder_name(row: dict) -> str:
    text = f"{row.get('artist') or 'Unknown'} - {row.get('name') or 'Untitled'} ({row.get('charter') or 'unknown'})"
    return re.sub(r'[<>:"/\\|?*]', "", text).strip(" -.")[:120] or row["md5"]


def library_keys(library: Path) -> set[str]:
    """artist|title pairs already owned, so calibration does not double-count
    a song the local library measures anyway."""
    keys = set()
    if not library or not library.is_dir():
        return keys
    for ini in library.rglob("song.ini"):
        try:
            text = ini.read_text(encoding="utf-8", errors="ignore").lower()
            artist = re.search(r"^artist\s*=\s*(.+)$", text, re.M)
            name = re.search(r"^name\s*=\s*(.+)$", text, re.M)
            if artist and name:
                keys.add(f"{artist.group(1).strip()}|{name.group(1).strip()}")
        except OSError:
            continue
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("data/calibration"))
    ap.add_argument("-n", "--count", type=int, default=400)
    ap.add_argument("--library", type=Path,
                    default=Path.home() / ".clonehero/Songs",
                    help="skip songs already in this library")
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--genre-cap", type=int, default=None,
                    help="max charts per genre bucket (default: 15%% of -n); "
                         "makes the sample genre-balanced instead of "
                         "inheriting the index's rock/metal skew")
    ap.add_argument("--seed", type=int, default=None,
                    help="shuffle the page order with this seed so the pull "
                         "samples the whole index, not its front pages")
    args = ap.parse_args(argv)
    genre_cap = args.genre_cap or max(30, int(0.15 * args.count))

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.outdir / "manifest.jsonl"
    have_md5 = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                have_md5.add(json.loads(line)["md5"])
            except (ValueError, KeyError):
                continue
    owned = library_keys(args.library)
    print(f"{len(have_md5)} already fetched; {len(owned)} songs in local library",
          flush=True)

    fetched = 0
    seen_songs = set()
    skips: dict[str, int] = {}
    genres: Counter = Counter()
    started = time.time()

    # Shuffled page order: Encore's default ordering front-loads whatever it
    # front-loads; a random walk over the whole index makes the pull a real
    # sample of every era and community, not of the first pages.
    first = api_search(args.start_page)
    per_page = 100
    total_pages = max(1, (first.get("found") or 0) // per_page + 1)
    page_order = list(range(1, total_pages + 1))
    random.Random(args.seed).shuffle(page_order)
    print(f"{first.get('found')} charts indexed; sampling {total_pages} pages "
          f"in random order, genre cap {genre_cap}", flush=True)

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for page in page_order:
            if fetched >= args.count:
                break
            try:
                result = api_search(page)
            except Exception as error:
                print(f"search page {page} failed ({type(error).__name__}); "
                      f"waiting 30s", flush=True)
                time.sleep(30)
                continue
            rows = result.get("data") or []
            if not rows:
                continue

            for row in rows:
                if fetched >= args.count:
                    break
                md5 = row.get("md5")
                song_key = (f"{(row.get('artist') or '').lower().strip()}|"
                            f"{(row.get('name') or '').lower().strip()}")
                reason = wanted(row)
                if md5 in have_md5 or song_key in seen_songs:
                    reason = "duplicate"
                elif song_key in owned:
                    reason = "already in local library"
                bucket = genre_bucket(row.get("genre"))
                if not reason and genres[bucket] >= genre_cap:
                    reason = f"genre cap ({bucket})"
                if reason:
                    skips[reason] = skips.get(reason, 0) + 1
                    continue

                try:
                    request = urllib.request.Request(
                        f"{FILES}/{md5}.sng", headers={"User-Agent": USER_AGENT})
                    raw = urllib.request.urlopen(request, timeout=120).read()
                    _, files = read_sng(raw)
                except Exception as error:
                    print(f"  download failed {row.get('name')!r}: "
                          f"{type(error).__name__}: {error}", flush=True)
                    skips["download failed"] = skips.get("download failed", 0) + 1
                    time.sleep(THROTTLE_S)
                    continue

                names = [n.lower() for n, _ in files]
                if not any(n.endswith(".chart") for n in names) or \
                        not any(n.endswith(AUDIO_EXT) for n in names):
                    skips["mid-only or no audio"] = skips.get("mid-only or no audio", 0) + 1
                    time.sleep(THROTTLE_S)
                    continue

                folder = args.outdir / folder_name(row)
                folder.mkdir(parents=True, exist_ok=True)
                kept = 0
                for name, data in files:
                    if name.lower().endswith(KEEP_EXT):
                        (folder / Path(name).name).write_bytes(data)
                        kept += 1
                manifest.write(json.dumps(row) + "\n")
                manifest.flush()
                have_md5.add(md5)
                seen_songs.add(song_key)
                genres[bucket] += 1
                fetched += 1
                print(f"  [{fetched}/{args.count}] {bucket:<14} "
                      f"{folder.name[:54]:<56} {time.time() - started:.0f}s",
                      flush=True)
                time.sleep(THROTTLE_S)

    print(f"\nfetched {fetched} charts in {time.time() - started:.0f}s")
    print("genre spread: " + ", ".join(f"{b} {n}" for b, n in genres.most_common()))
    for reason, count in sorted(skips.items(), key=lambda s: -s[1]):
        print(f"  skipped {count}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
