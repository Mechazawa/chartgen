"""Add album.png to existing generated song folders.

For each folder matching --match without art: embedded tag art from the
folder's audio if it survived (copied-through mp3/ogg keep tags; transcoded
opus does not), else a YouTube search for "<artist> <name>" and that video's
thumbnail.

    python tools/add_art.py ~/.clonehero/Songs --match chartgen
"""
import argparse
import configparser
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartgen import art  # noqa: E402

AUDIO_EXTS = (".opus", ".ogg", ".mp3", ".wav")


def ini_meta(folder: Path) -> tuple[str, str]:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        parser.read_string((folder / "song.ini").read_text(encoding="utf-8"))
        section = next(s for s in parser.sections() if s.lower() == "song")
        return parser[section].get("artist", ""), parser[section].get("name", folder.name)
    except Exception:
        return "", folder.name


def thumbnail_by_search(artist: str, name: str) -> str | None:
    from yt_dlp import YoutubeDL

    import re

    query = re.sub(r"\[chartgen[^\]]*\]", "", f"{artist} {name}").strip()
    try:
        with YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
        entry = (info.get("entries") or [None])[0]
        return entry.get("thumbnail") if entry else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--match", default="chartgen")
    args = ap.parse_args()

    for folder in sorted(args.root.iterdir()):
        if not folder.is_dir() or args.match not in folder.name:
            continue
        if (folder / "album.png").is_file() or (folder / "album.jpg").is_file():
            print(f"{folder.name[:52]:<54} already has art")
            continue
        audio = next((folder / f"song{e}" for e in AUDIO_EXTS
                      if (folder / f"song{e}").is_file()), None)
        image = art.embedded_art(audio) if audio else None
        source = "embedded tags"
        if image is None:
            artist, name = ini_meta(folder)
            url = thumbnail_by_search(artist, name)
            image = art.fetch(url) if url else None
            source = "YouTube thumbnail"
        if image and art.write_album(folder, image):
            print(f"{folder.name[:52]:<54} art from {source}")
        else:
            print(f"{folder.name[:52]:<54} NO ART FOUND")
    return 0


if __name__ == "__main__":
    sys.exit(main())
