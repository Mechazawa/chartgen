"""Measure how human charters actually mark guitar solos.

Our rule is a guess: a chroma section with no transcribed lyrics that plays
busier than average. This measures the real thing in a human library - how
many songs carry solos, how long they run, where they sit, how much busier
they actually are, and (the number that decides whether our premise holds)
whether a lyric-free gap is even a useful predictor of a solo.

    python tools/study_solos.py ~/.clonehero/Songs
"""
import argparse
import re
import statistics
import sys
from itertools import pairwise
from pathlib import Path

TIERS = ("ExpertSingle", "HardSingle", "MediumSingle", "EasySingle")
NOTE_RE = re.compile(r"(\d+)\s*=\s*N\s+(\d+)\s+(\d+)")
EVENT_RE = re.compile(r"(\d+)\s*=\s*E\s+(.+)")
RES_RE = re.compile(r"Resolution\s*=\s*(\d+)")
OPEN = 7


def blocks(text: str) -> dict[str, str]:
    out, name = {}, None
    depth = 0
    buf: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            name, buf = s[1:-1], []
        elif s == "{":
            depth = 1
        elif s == "}" and depth:
            if name:
                out[name] = "\n".join(buf)
            depth, name = 0, None
        elif depth and name:
            buf.append(s)
    return out


def is_generated(blocks_dict: dict) -> bool:
    """True when this chart was written by chartgen itself.

    Folder-name tagging came late; the Charter field was always signed.
    Ground truth polluted by our own old detector is how Run Boy Run's
    machine-made solo marker spent a day masquerading as the calibration
    suite's best human-validated catch.
    """
    song = blocks_dict.get("Song", "")
    return "chartgen" in song.lower()


def solo_spans(body: str) -> list[tuple[int, int]]:
    """[(start, end)] from the local `E solo` / `E soloend` track events."""
    spans, open_at = [], None
    for tick, text in sorted(((int(t), e.strip()) for t, e in EVENT_RE.findall(body))):
        low = text.lower()
        if low == "solo":
            open_at = tick
        elif low in ("soloend", "solo_end") and open_at is not None:
            spans.append((open_at, tick))
            open_at = None
    return spans


def note_ticks(body: str) -> list[int]:
    return sorted({int(t) for t, lane, _ in NOTE_RE.findall(body)
                   if int(lane) < 5 or int(lane) == OPEN})


def lyric_ticks(events: str) -> list[int]:
    """Lyric word positions from the global [Events] block."""
    out = []
    for tick, text in EVENT_RE.findall(events):
        t = text.strip().strip('"')
        if t.lower().startswith("lyric "):
            out.append(int(tick))
    return sorted(out)


def section_ticks(events: str) -> list[tuple[int, str]]:
    out = []
    for tick, text in EVENT_RE.findall(events):
        t = text.strip().strip('"')
        if t.lower().startswith("section "):
            out.append((int(tick), t[8:].strip()))
    return sorted(out)


def study(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    b = blocks(text)
    found = RES_RE.search(b.get("Song", ""))
    res = int(found.group(1)) if found else 192
    expert = b.get("ExpertSingle", "")
    ticks = note_ticks(expert)
    if len(ticks) < 50:
        return None

    events = b.get("Events", "")
    spans = solo_spans(expert)
    per_tier = {t: len(solo_spans(b[t])) for t in TIERS if t in b}
    words = lyric_ticks(events)
    sections = section_ticks(events)

    last = ticks[-1]
    overall = len(ticks) / max(1.0, last / res)

    rows = []
    for start, end in spans:
        inside = [t for t in ticks if start <= t <= end]
        beats = max(1e-6, (end - start) / res)
        rows.append({
            "beats": beats,
            "notes": len(inside),
            "density_ratio": (len(inside) / beats) / overall if overall else 0,
            "pos": start / last if last else 0,
            "words_inside": sum(1 for w in words if start <= w <= end),
            "sec_offset": min((abs(start - s) / res for s, _ in sections), default=None),
            "named_solo": any(abs(start - s) <= 4 * res and "solo" in n.lower()
                              for s, n in sections),
        })

    # How well does "a gap in the lyrics" predict a solo? Only meaningful in
    # songs that have both lyrics and at least one marked solo.
    gap_stats = None
    if words and spans and len(words) >= 20:
        gaps = []
        edges = [0] + words + [last]
        for a, b2 in pairwise(edges):
            if (b2 - a) / res >= 8:  # >= 2 bars of no singing
                gaps.append((a, b2))
        hit = sum(1 for a, b2 in gaps
                  if any(a <= s <= b2 or a <= e <= b2 or (s <= a and b2 <= e)
                         for s, e in spans))
        gap_stats = {"gaps": len(gaps), "gaps_with_solo": hit, "solos": len(spans)}

    return {
        "name": path.parent.name,
        "solos": len(spans),
        "rows": rows,
        "per_tier": per_tier,
        "has_lyrics": len(words) >= 20,
        "gap_stats": gap_stats,
        "named_sections": sum(1 for _, n in sections if "solo" in n.lower()),
        "sections": len(sections),
    }


def pct(x, n):
    return f"{100.0 * x / n:.0f}%" if n else "n/a"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library", type=Path)
    args = ap.parse_args(argv)

    charts = sorted(args.library.rglob("*.chart"))
    print(f"{len(charts)} charts under {args.library}\n")
    studied = [s for s in (study(p) for p in charts) if s]
    if not studied:
        sys.exit("no usable charts")

    n = len(studied)
    with_solo = [s for s in studied if s["solos"]]
    print(f"usable: {n}")
    print(f"charts with solo markers: {len(with_solo)} ({pct(len(with_solo), n)})")
    if with_solo:
        counts = [s["solos"] for s in with_solo]
        print(f"solos per song (of those): median {statistics.median(counts):.0f}  "
              f"mean {statistics.mean(counts):.1f}  max {max(counts)}")

    rows = [r for s in with_solo for r in s["rows"]]
    if rows:
        def dist(key, fmt="{:.2f}"):
            vals = sorted(r[key] for r in rows if r[key] is not None)
            if not vals:
                return "n/a"
            q = statistics.quantiles(vals, n=4) if len(vals) > 3 else [vals[0]] * 3
            return (f"median {fmt.format(statistics.median(vals))}  "
                    f"p25 {fmt.format(q[0])}  p75 {fmt.format(q[2])}")
        print(f"\nsolo length, beats:      {dist('beats', '{:.0f}')}")
        print(f"solo length, bars:       "
              f"median {statistics.median([r['beats'] / 4 for r in rows]):.1f}")
        print(f"notes in solo:           {dist('notes', '{:.0f}')}")
        print(f"density vs song average: {dist('density_ratio')}")
        print(f"position in song (0-1):  {dist('pos')}")
        print(f"distance to nearest section marker, beats: {dist('sec_offset', '{:.1f}')}")
        on_sec = sum(1 for r in rows if r["sec_offset"] is not None and r["sec_offset"] <= 1)
        print(f"  solos starting within 1 beat of a section: {on_sec}/{len(rows)} "
              f"({pct(on_sec, len(rows))})")
        named = sum(1 for r in rows if r["named_solo"])
        print(f"  solos whose section is NAMED 'solo': {named}/{len(rows)} "
              f"({pct(named, len(rows))})")
        denser = sum(1 for r in rows if r["density_ratio"] >= 1.2)
        print(f"  solos at least 1.2x average density: {denser}/{len(rows)} "
              f"({pct(denser, len(rows))})")
        quiet = sum(1 for r in rows if r["words_inside"] == 0)
        print(f"  solos with NO lyric word inside: {quiet}/{len(rows)} "
              f"({pct(quiet, len(rows))})")

    # Does a lyric gap predict a solo?
    gs = [s["gap_stats"] for s in studied if s["gap_stats"]]
    print(f"\ncharts carrying lyrics AND solos: {len(gs)}")
    if gs:
        gaps = sum(g["gaps"] for g in gs)
        hits = sum(g["gaps_with_solo"] for g in gs)
        solos = sum(g["solos"] for g in gs)
        print(f"  lyric-free gaps >= 2 bars: {gaps}")
        print(f"  of those, containing a solo: {hits} -> PRECISION {pct(hits, gaps)}")
        print(f"  solos found inside such a gap: {hits}/{solos} -> RECALL {pct(hits, solos)}")

    # Tier consistency: is the marker copied to every difficulty?
    same = sum(1 for s in with_solo
               if len(set(s["per_tier"].values())) == 1 and len(s["per_tier"]) > 1)
    multi = sum(1 for s in with_solo if len(s["per_tier"]) > 1)
    print(f"\nsame solo count on every charted tier: {same}/{multi} ({pct(same, multi)})")

    named_any = sum(1 for s in studied if s["named_sections"])
    print(f"charts with a section NAMED 'solo': {named_any}/{n} ({pct(named_any, n)})")
    marked_and_named = sum(1 for s in studied if s["named_sections"] and s["solos"])
    print(f"  of those, also carrying solo MARKERS: {marked_and_named} "
          f"(named but unmarked: {named_any - marked_and_named})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
