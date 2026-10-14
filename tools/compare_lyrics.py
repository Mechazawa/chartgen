"""Score our transcribed lyrics against real synced lyrics from LRCLIB.

Playtest said lyrics "cut off" and grow "phantom words near the end". Both
are claims about coverage, and neither can be measured without a reference.
LRCLIB publishes community LRC files keyed by artist/title/duration, which
gives one: how many words the song really has, when its singing really
starts and stops, and therefore whether we stopped early or invented a tail.

    python tools/compare_lyrics.py ~/.clonehero/Songs
"""
import argparse
import configparser
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from study_solos import blocks, RES_RE

EVENT_RE = re.compile(r"(\d+)\s*=\s*E\s+(.+)")
LRC_LINE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)")
UA = "chartgen/0.1 (https://github.com/ErnestSarna/chartgen)"


def lrclib_search(artist: str, title: str):
    q = urllib.parse.urlencode({"artist_name": artist, "track_name": title})
    req = urllib.request.Request("https://lrclib.net/api/search?" + q,
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_lrc(text: str):
    out = []
    for line in text.splitlines():
        m = LRC_LINE.match(line.strip())
        if m and m.group(3).strip():
            out.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
    return sorted(out)


def our_lyrics(chart: Path):
    b = blocks(chart.read_text(encoding="utf-8", errors="ignore"))
    found = RES_RE.search(b.get("Song", ""))
    res = int(found.group(1)) if found else 192
    words = []
    for tick, text in EVENT_RE.findall(b.get("Events", "")):
        t = text.strip().strip('"')
        if t.lower().startswith("lyric "):
            words.append((int(tick) / res, t[6:].strip()))
    return sorted(words)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library", type=Path)
    ap.add_argument("--filter", default="chartgen")
    args = ap.parse_args(argv)

    for chart in sorted(args.library.rglob("*.chart")):
        folder = chart.parent
        if args.filter.lower() not in folder.name.lower():
            continue
        ini = folder / "song.ini"
        if not ini.exists():
            continue
        cfg = configparser.ConfigParser(strict=False)
        try:
            cfg.read(ini, encoding="utf-8")
        except (OSError, configparser.Error):
            continue
        sec = cfg.sections()[0] if cfg.sections() else None
        if not sec:
            continue
        artist = cfg[sec].get("artist", "")
        name = cfg[sec].get("name", "")
        length_ms = cfg[sec].getint("song_length", fallback=0)

        ours = our_lyrics(chart)
        try:
            hits = lrclib_search(artist, re.sub(r"\s*\[chartgen\]", "", name))
        except Exception as error:
            print(f"{folder.name}: LRCLIB error {error}")
            continue

        synced = [h for h in hits if h.get("syncedLyrics")]
        if not synced:
            print(f"{folder.name}: no synced lyrics on LRCLIB "
                  f"(ours: {len(ours)} words)")
            continue
        if length_ms:
            synced.sort(key=lambda h: abs((h.get("duration") or 0) - length_ms / 1000))
        best = synced[0]
        lines = parse_lrc(best["syncedLyrics"])
        ref_words = sum(len(t.split()) for _, t in lines)

        print(f"\n{folder.name}")
        print(f"  LRCLIB: {best['artistName']} - {best['trackName']} "
              f"({best.get('duration')}s vs ours {length_ms / 1000:.0f}s), "
              f"{len(lines)} lines / {ref_words} words")
        print(f"  ours: {len(ours)} words  ->  "
              f"{100.0 * len(ours) / ref_words:.0f}% of reference")
        if lines and ours:
            print(f"  singing ends at {lines[-1][0]:.0f}s (reference)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
