"""Fetch the official 6-fret charts (Guitar Hero Live on disc and GHTV)
from Chorus Encore into data/ghl-official/encore/<icon>/.

Encore has no "official" flag. The game rips carry the `ghl` or `ghtv`
icon; rows filed under "Archived Charts" are community recharts with that
icon, and rows a user uploaded through the app are edits. Both are left
out. Video is dropped; chart, audio and song.ini are kept. Songs already
in --skip-root (the on-disc rips from another source) are not fetched again.

    python -m tools.fetch_official_ghl
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from tools.fetch_charts import API, FILES, THROTTLE_S, USER_AGENT, folder_name, read_sng

OFFICIAL_ICONS = ("ghl", "ghtv")
KEEP_EXT = (".mid", ".chart", ".opus", ".ogg", ".mp3", ".wav", ".flac")


def search(page: int) -> dict:
    body = json.dumps({"search": "", "page": page, "per_page": 100,
                       "instrument": "guitarghl", "difficulty": None}).encode()
    request = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def official_rows() -> list[dict]:
    rows, page = {}, 1
    while True:
        found = search(page)
        for row in found.get("data") or []:
            rows[row["md5"]] = row
        if page * 100 >= (found.get("found") or 0):
            break
        page += 1
        time.sleep(THROTTLE_S)
    return [row for row in rows.values()
            if row.get("icon") in OFFICIAL_ICONS and row.get("packName") != "Archived Charts"
            and not row.get("applicationUsername")]


def song_key(artist: str, name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", f"{artist}|{name}".lower())


def owned_keys(root: Path) -> set[str]:
    keys = set()
    for ini in root.rglob("song.ini"):
        text = ini.read_text(encoding="utf-8", errors="ignore")
        artist = re.search(r"^artist\s*=\s*(.+)$", text, re.M)
        name = re.search(r"^name\s*=\s*(.+)$", text, re.M)
        if artist and name:
            keys.add(song_key(artist.group(1), name.group(1)))
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("data/ghl-official/encore"))
    ap.add_argument("--skip-root", type=Path, default=Path("data/ghl-official"),
                    help="on-disc (ghl) songs found here are not fetched again")
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = args.outdir / "manifest.jsonl"
    have = {json.loads(line)["md5"] for line in manifest.read_text().splitlines()} \
        if manifest.exists() else set()
    owned = owned_keys(args.skip_root)
    rows = official_rows()
    todo = [row for row in rows if row["md5"] not in have and not (
        row["icon"] == "ghl" and song_key(row["artist"], row["name"]) in owned)]
    print(f"{len(rows)} official charts, {len(have)} fetched, {len(todo)} to fetch", flush=True)

    with manifest.open("a", encoding="utf-8") as log:
        for index, row in enumerate(todo, 1):
            try:
                request = urllib.request.Request(f"{FILES}/{row['md5']}.sng",
                                                 headers={"User-Agent": USER_AGENT})
                meta, files = read_sng(urllib.request.urlopen(request, timeout=300).read())
            except Exception as error:
                print(f"  failed {row['name']!r}: {type(error).__name__}: {error}", flush=True)
                time.sleep(THROTTLE_S)
                continue
            folder = args.outdir / row["icon"] / folder_name(row)
            folder.mkdir(parents=True, exist_ok=True)
            for name, data in files:
                if name.lower().endswith(KEEP_EXT):
                    (folder / Path(name).name).write_bytes(data)
            # A .sng keeps song.ini as header metadata, not as a file.
            (folder / "song.ini").write_text(
                "[song]\n" + "".join(f"{key} = {value}\n" for key, value in meta.items()),
                encoding="utf-8")
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[{index}/{len(todo)}] {row['icon']} {row['artist']} - {row['name']}", flush=True)
            time.sleep(THROTTLE_S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
