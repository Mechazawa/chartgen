"""Desktop UI for chartgen.

    python -m chartgen.app

Tkinter on purpose: it ships with Python, so the UI adds almost nothing to
bundle. The look comes from sv-ttk (a pure-Python Fluent/Win11 theme, ~50 kB);
if it is missing the app still runs on the stock theme. Generation runs on a
worker thread and reports through a queue — Tk widgets are only ever touched
from the main thread.
"""
import json
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from argparse import Namespace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import pipeline

SETTINGS = (Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
            / "chartgen" / "settings.json")
GRIDS = {"16th notes (default)": 4, "8th notes (sparser)": 2, "Triplet 8ths": 3}
# Transcribing vocals invents words over instrumental outros - Whisper falls
# back on the YouTube captions it was trained on, and a real chart ended with
# "Thank you for watching". Looking the song up first avoids the guesswork
# entirely when someone has already synced it.
LYRIC_SOURCES = {
    "Online, else listen": "auto",
    "Online only": "online",
    "Listen": "transcribe",
}


def fmt_elapsed(seconds: float) -> str:
    """42s -> '42s', 754s -> '12m 34s', 5025s -> '1h 23m'."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {seconds % 3600 // 60}m"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(values: dict) -> None:
    try:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(values, indent=2), encoding="utf-8")
    except OSError:
        pass  # settings are a convenience; never fail a run over them


# Set by apply_theme(); the App reads these so a machine without sv-ttk still
# gets readable colours on the stock light theme.
DARK = False
MUTED = "#666666"
LOG_BG, LOG_FG = "#ffffff", "#222222"


def apply_theme(root: tk.Tk) -> None:
    global DARK, MUTED, LOG_BG, LOG_FG
    try:
        import sv_ttk

        sv_ttk.set_theme("dark", root)
        DARK = True
        MUTED = "#9aa0a6"
        LOG_BG, LOG_FG = "#161616", "#d6d6d6"
    except Exception:
        ttk.Style().theme_use("clam")


class Tooltip:
    """Hover text for a widget. Tk ships no tooltip, so: a borderless
    Toplevel that appears under the pointer after a short delay and dies on
    leave/click. One instance per widget; themed to match the app."""

    DELAY_MS = 450
    WRAP_PX = 320

    def __init__(self, widget, text: str):
        self.widget, self.text = widget, text
        self.tip = None
        self.after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        for event in ("<Leave>", "<ButtonPress>"):
            widget.bind(event, self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self.after_id = self.widget.after(self.DELAY_MS, self._show)

    def _cancel(self):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def _show(self):
        if self.tip or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        bg, fg = ("#2a2a2a", "#e8e8e8") if DARK else ("#ffffe1", "#222222")
        tk.Label(self.tip, text=self.text, justify="left", wraplength=self.WRAP_PX,
                 background=bg, foreground=fg, relief="solid", borderwidth=1,
                 font=("sans-serif", 9), padx=8, pady=6).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self.tip:
            self.tip.destroy()
            self.tip = None


TIPS = {
    "audio": "A song file (mp3/flac/wav/ogg, 30s+), a YouTube link, or a "
             "whole playlist link. Paste several links at once - Enter or "
             "Add stacks them in the queue below, and each becomes its own "
             "song folder. The queue shrinks as songs finish.",
    "outdir": "Each song becomes its own folder here with notes.chart, "
              "audio and album art. Point it at your Clone Hero songs "
              "folder and new charts appear after a rescan.",
    "target_diff": "Caps the difficulty badge (0-6). The chart is thinned "
                   "until it rates at or below this, and the whole "
                   "Easy-Expert ladder scales down with it. Auto keeps "
                   "whatever the song supports. Notes are never invented "
                   "to raise a rating.",
    "sustains": "Long notes you hold for bonus points. Placed where the "
                "audio actually holds a sound; share calibrated against "
                "human charts (~9-15% by tier).",
    "star_power": "The glowing phrase groups that charge 2x score. About "
                  "one phrase every 30-40 seconds, avoiding the very end "
                  "where they could never be spent.",
    "sections": "Named practice-mode markers (Section 1, 2...) at the "
                "song's real structural boundaries, for CH's practice "
                "seek menu.",
    "hopos": "Hammer-ons/pull-offs: close-spaced notes play without "
             "strumming, shown with a white ring. Uses CH's natural "
             "spacing rule, like human charts.",
    "lyrics": "Words scroll above the highway as they're sung. Real "
              "synced lyrics are looked up online first; if the song "
              "isn't in the database, the vocals are transcribed "
              "locally. Instrumentals cleanly get none.",
    "solos": "E solo sections that award bonus points per note hit. "
             "Detected from the audio - a lead line that stands out from "
             "the rest of the song. Deliberately conservative: most songs "
             "get none (70% of human charts have none either); a marker "
             "that does appear is almost always a real lead break.",
    "ghl": "Also writes a 6-fret part for Guitar Hero Live controllers "
           "(three white and three black buttons) in the same song. The "
           "buttons come from a model trained on the official Guitar Hero "
           "Live charts; the note timing is the 5-fret chart's.",
    "opens": "The no-fret purple strum: lone notes clearly below the "
             "melody - bass drops, chugs, pedal tones - play as a bare "
             "strum with no button held. 60% of human charts use them as "
             "punctuation.",
    "taps": "Marks soft phrases - piano lines, plucks, gentle synth runs - "
            "as tap notes, playable without strumming. It's a stylistic "
            "choice, and human charters only half-agree on when to tap - "
            "turn it off if you'd rather strum everything.",
    "grid": "The finest rhythm notes can land on. 16ths fit almost "
            "everything; 8ths force a sparser, easier chart; triplet 8ths "
            "suit shuffle/swing songs.",
    "lyric_source": "Look up online: fetch hand-synced lyrics from LRCLIB "
                    "(free, no account) - the right words with human "
                    "timing. Listen: transcribe the vocals locally with "
                    "Whisper, which can mishear or invent words over "
                    "instrumental outros. Default tries the lookup and "
                    "only listens when the song isn't in the database.",
}


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.batch_pos: tuple[int, int] | None = None
        self.cancel = threading.Event()
        self.result = None
        self.started = 0.0
        saved = load_settings()

        root.title("chartgen — Clone Hero chart generator")
        # Pixel sizes don't scale with DPI, fonts do; scale the frame so the
        # layout isn't cramped at 125/150 % display scaling.
        scale = root.winfo_fpixels("1i") / 96
        root.minsize(int(720 * scale), int(620 * scale))
        root.geometry(f"{int(760 * scale)}x{int(790 * scale)}")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(9, weight=1)

        self._build_header()
        self._section("Songs", row=1)
        self._build_input(saved)
        self._section("Options", row=3)
        self._build_options(saved)
        self._build_actions()
        self._section("Progress", row=8)
        self._build_output()
        root.after(100, self._drain)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- layout
    @staticmethod
    def _tip(key: str, *widgets):
        """One tooltip text on the option's label and its control alike."""
        for widget in widgets:
            Tooltip(widget, TIPS[key])

    def _section(self, text: str, row: int):
        ttk.Label(self.root, text=text.upper(), foreground=MUTED,
                  font=("sans-serif", 8, "bold")).grid(
            row=row, column=0, sticky="w", padx=22, pady=(14, 4))

    def _card(self, row: int, expand=False) -> ttk.Frame:
        frame = ttk.Frame(self.root, style="Card.TFrame", padding=14)
        frame.grid(row=row, column=0, sticky="nsew" if expand else "ew", padx=16)
        return frame

    def _build_header(self):
        head = ttk.Frame(self.root)
        head.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 0))
        bg = ttk.Style().lookup("TFrame", "background") or self.root.cget("bg")
        logo = tk.Canvas(head, width=30, height=30, highlightthickness=0, bg=bg)
        for x, height, color in ((4, 16, "#34d17b"), (13, 9, "#e8b84b"),
                                 (22, 13, "#e2665e")):
            logo.create_rectangle(x, 26 - height, x + 5, 26, fill=color, width=0)
        logo.pack(side="left", padx=(0, 10))
        ttk.Label(head, text="chartgen",
                  font=("sans-serif", 15, "bold")).pack(side="left")
        ttk.Label(head, text="Clone Hero chart generator", foreground=MUTED,
                  font=("sans-serif", 10)).pack(side="left", padx=(10, 0), pady=(4, 0))

    def _build_input(self, saved):
        frame = self._card(row=2)
        frame.columnconfigure(1, weight=1)

        self.audio = tk.StringVar()
        label = ttk.Label(frame, text="Audio / YouTube")
        label.grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(frame, textvariable=self.audio)
        entry.grid(row=0, column=1, sticky="ew", padx=10)
        entry.bind("<Return>", self._add_from_entry)
        self._tip("audio", label, entry)
        buttons = ttk.Frame(frame)
        buttons.grid(row=0, column=2)
        ttk.Button(buttons, text="Browse…", command=self._pick_audio).pack(
            side="left", padx=(0, 4))
        ttk.Button(buttons, text="Add", width=5,
                   command=self._add_from_entry).pack(side="left")

        # The queue ladder: every added file/link is a row here, and rows
        # disappear one by one as their song finishes — a per-song progress
        # indicator on top of the stage bar. Hidden entirely when empty.
        self.queue_items: list[str] = []
        self.queue_labels: dict[str, str] = {}
        self.run_items: list[str] = []
        self.run_done = 0
        self.queue_frame = ttk.Frame(frame)
        self.queue_frame.grid(row=2, column=1, sticky="ew", padx=10, pady=(6, 0))
        self.queue_frame.columnconfigure(0, weight=1)
        # Fixed height: the dark box spans the scroller's full run instead
        # of shrinking to one centred sliver on short queues. Newest entries
        # display at the TOP; processing stays first-added-first, so the
        # ladder drains from the bottom.
        self.queue_box = tk.Listbox(
            self.queue_frame, height=4, activestyle="none", relief="flat",
            selectmode="extended", bg=LOG_BG, fg=LOG_FG,
            highlightthickness=0, font=("sans-serif", 9))
        self.queue_box.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self.queue_frame, orient="vertical",
                               command=self.queue_box.yview)
        self.queue_box.configure(yscrollcommand=scroll.set)
        scroll.grid(row=0, column=1, sticky="ns")
        self.queue_box.bind("<Delete>", self._queue_remove)
        self.queue_box.bind("<Double-Button-1>", self._queue_remove)
        Tooltip(self.queue_box, "Double-click or press Delete to remove a row.")
        footer = ttk.Frame(self.queue_frame)
        footer.grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Button(footer, text="Clear", width=6,
                   command=self._queue_clear).pack(side="left", padx=(0, 10))
        self.queue_count = ttk.Label(footer, text="", foreground=MUTED)
        self.queue_count.pack(side="left")
        self.queue_frame.grid_remove()

        # No artist/title fields: the pipeline fills them itself — YouTube
        # metadata for links, audio tags for files, filename as last resort.
        self.outdir = tk.StringVar(value=saved.get("outdir", str(Path("out").resolve())))
        label = ttk.Label(frame, text="Save to")
        label.grid(row=3, column=0, sticky="w", pady=(8, 0))
        entry = ttk.Entry(frame, textvariable=self.outdir)
        entry.grid(row=3, column=1, sticky="ew", padx=10, pady=(8, 0))
        self._tip("outdir", label, entry)
        ttk.Button(frame, text="Browse…", command=self._pick_outdir).grid(
            row=3, column=2, sticky="w", pady=(8, 0))
        ttk.Label(frame, text="Tip: point this at your Clone Hero songs folder.",
                  foreground=MUTED).grid(row=4, column=1, sticky="w", padx=10)

    def _build_options(self, saved):
        """The choices that change what kind of chart you get."""
        frame = self._card(row=4)
        for col in (1, 3, 5):
            frame.columnconfigure(col, weight=1)

        # One row of three pickers: the former Advanced panel had only two
        # knobs left once the neural engine's settings went, so they sit
        # beside Max difficulty instead of behind a collapsed header.
        self.target_diff = tk.StringVar(value=saved.get("target_diff", "Auto"))
        label = ttk.Label(frame, text="Max difficulty")
        label.grid(row=0, column=0, sticky="w")
        combo = ttk.Combobox(frame, textvariable=self.target_diff,
                             values=["Auto"] + [str(n) for n in range(7)],
                             state="readonly", width=6)
        combo.grid(row=0, column=1, sticky="w", padx=10)
        self._tip("target_diff", label, combo)

        self.grid_choice = tk.StringVar(value=saved.get("grid", list(GRIDS)[0]))
        label = ttk.Label(frame, text="Note grid")
        label.grid(row=0, column=2, sticky="w")
        combo = ttk.Combobox(frame, textvariable=self.grid_choice, values=list(GRIDS),
                             state="readonly", width=20)
        combo.grid(row=0, column=3, sticky="w", padx=10)
        self._tip("grid", label, combo)

        # Labels were shortened 2026-09-07; a saved long label falls back to
        # the default rather than showing stale text the map cannot resolve.
        saved_lyrics = saved.get("lyric_source")
        self.lyric_source = tk.StringVar(
            value=saved_lyrics if saved_lyrics in LYRIC_SOURCES else list(LYRIC_SOURCES)[0])
        label = ttk.Label(frame, text="Lyrics")
        label.grid(row=0, column=4, sticky="w")
        combo = ttk.Combobox(frame, textvariable=self.lyric_source,
                             values=list(LYRIC_SOURCES), state="readonly", width=18)
        combo.grid(row=0, column=5, sticky="w", padx=10)
        self._tip("lyric_source", label, combo)

        toggles = ttk.Frame(frame)
        toggles.grid(row=1, column=0, columnspan=6, sticky="w", pady=(12, 0))
        self.sustains = tk.BooleanVar(value=saved.get("sustains", True))
        self.star_power = tk.BooleanVar(value=saved.get("star_power", True))
        self.sections = tk.BooleanVar(value=saved.get("sections", True))
        self.hopos = tk.BooleanVar(value=saved.get("hopos", True))
        self.lyrics = tk.BooleanVar(value=saved.get("lyrics", True))
        # "taps_v2": taps became on-by-default on 2026-09-05; the old key holds the
        # old default (False) in every existing settings.json, so it is ignored.
        self.taps = tk.BooleanVar(value=saved.get("taps_v2", True))
        self.solos = tk.BooleanVar(value=saved.get("solos", True))
        self.opens = tk.BooleanVar(value=saved.get("opens", True))
        self.ghl = tk.BooleanVar(value=saved.get("ghl", True))
        # Two rows: one row of nine needs ~800px and the card offers ~700,
        # which clips the last toggles down to anonymous checkboxes.
        for index, (text, var, key) in enumerate((
                ("Sustains", self.sustains, "sustains"),
                ("Star power", self.star_power, "star_power"),
                ("Sections", self.sections, "sections"),
                ("HOPOs", self.hopos, "hopos"),
                ("Lyrics", self.lyrics, "lyrics"),
                ("Solos", self.solos, "solos"),
                ("Opens", self.opens, "opens"),
                ("Taps", self.taps, "taps"),
                ("6-fret", self.ghl, "ghl"))):
            box = ttk.Checkbutton(toggles, text=text, variable=var)
            box.grid(row=index // 5, column=index % 5, sticky="w", padx=(0, 14), pady=(0, 4))
            self._tip(key, box)

    def _build_actions(self):
        frame = ttk.Frame(self.root)
        frame.grid(row=7, column=0, sticky="ew", padx=16, pady=(14, 0))
        frame.columnconfigure(2, weight=1)
        self.go = ttk.Button(frame, text="Generate chart", command=self._start,
                             style="Accent.TButton")
        self.go.grid(row=0, column=0)
        self.stop = ttk.Button(frame, text="Cancel", command=self._request_cancel,
                               state="disabled")
        self.stop.grid(row=0, column=1, padx=10)
        self.bar = ttk.Progressbar(frame, mode="determinate", maximum=6)
        self.bar.grid(row=0, column=2, sticky="ew", padx=10)
        self.status = ttk.Label(frame, text="Ready", anchor="e",
                                foreground=MUTED)
        self.status.grid(row=0, column=3)

    def _build_output(self):
        frame = self._card(row=9, expand=True)
        frame.configure(padding=6)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.log = tk.Text(frame, height=6, wrap="none", font=("monospace", 9),
                           bg=LOG_BG, fg=LOG_FG, insertbackground=LOG_FG,
                           relief="flat", borderwidth=0, padx=10, pady=8,
                           selectbackground="#2f4f77" if DARK else "#cce4ff")
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set, state="disabled")

        actions = ttk.Frame(self.root)
        actions.grid(row=10, column=0, sticky="ew", padx=16, pady=(12, 16))
        self.open_button = ttk.Button(actions, text="Open song folder",
                                      command=self._open_folder, state="disabled")
        self.open_button.pack(side="left")

    # ---------------------------------------------------------------- events
    def _display(self, item) -> str:
        text = str(item)
        if text in self.queue_labels:
            return self.queue_labels[text]
        return text if text.lower().startswith(("http", "www.")) else Path(text).name

    def _add_from_entry(self, _event=None):
        """Entry -> queue: one or more links (whitespace-separated) or a path."""
        from . import youtube

        text = self.audio.get().strip()
        if not text:
            return "break"
        tokens = text.split()
        if all(youtube.is_youtube_url(t) for t in tokens):
            self._queue_add(tokens)
        elif Path(text).is_file():
            self._queue_add([text])
        else:
            messagebox.showerror(
                "chartgen", f"Not a file or YouTube link:\n{text}")
            return "break"
        self.audio.set("")
        return "break"

    def _queue_add(self, items):
        from . import youtube

        fresh = []
        for item in items:
            item = str(item)
            if item not in self.queue_items:
                self.queue_items.append(item)
                self.queue_box.insert(0, f"  {self._display(item)}")
                if youtube.is_youtube_url(item):
                    fresh.append(item)
        self._queue_refresh()
        if fresh:
            threading.Thread(target=self._resolve_titles, args=(fresh,),
                             daemon=True).start()

    def _resolve_titles(self, links):
        """Fetch video titles so ladder rows read as songs, not URLs."""
        import yt_dlp

        options = {"quiet": True, "no_warnings": True, "extract_flat": True,
                   "skip_download": True}
        for link in links:
            try:
                with yt_dlp.YoutubeDL(options) as ydl:
                    info = ydl.extract_info(link, download=False)
                title = info.get("title")
                if not title:
                    continue
                link_key = link
                if info.get("_type") == "playlist" or info.get("entries"):
                    count = len(list(info.get("entries") or []))
                    label = f"{title} (playlist, {count} songs)" if count                         else f"{title} (playlist)"
                else:
                    from . import youtube as yt
                    artist, name = yt.parse_title(title)
                    label = f"{artist} - {name}" if artist else title
                self.events.put(("queue_title", (link_key, label)))
            except Exception:
                continue

    def _render_run_ladder(self):
        """Rebuild the ladder from the in-flight batch: remaining items,
        newest-first, titles where known."""
        remaining = self.run_items[self.run_done:]
        self.queue_box.delete(0, "end")
        for item in reversed(remaining):
            self.queue_box.insert("end", f"  {self._display(item)}")
        self._queue_refresh()

    def _queue_remove(self, _event=None):
        if self.worker and self.worker.is_alive():
            return
        for i in reversed(list(self.queue_box.curselection())):
            self.queue_box.delete(i)
            # Display is newest-first; the backing list is oldest-first.
            del self.queue_items[len(self.queue_items) - 1 - i]
        self._queue_refresh()

    def _queue_clear(self):
        if self.worker and self.worker.is_alive():
            return
        self.queue_items.clear()
        self.queue_box.delete(0, "end")
        self._queue_refresh()

    def _queue_refresh(self):
        n = self.queue_box.size()
        if n:
            self.queue_frame.grid()
            self.queue_count.configure(
                text=f"{n} song{'s' if n != 1 else ''} queued")
        else:
            self.queue_frame.grid_remove()

    def _pick_audio(self):
        paths = filedialog.askopenfilenames(
            title="Choose songs (at least 30 seconds each; multi-select works)",
            filetypes=[("Audio", "*.mp3 *.flac *.wav *.ogg *.opus *.m4a"), ("All", "*.*")])
        if paths:
            self._queue_add(paths)

    def _pick_outdir(self):
        path = filedialog.askdirectory(title="Where to save the song folder")
        if path:
            self.outdir.set(path)

    def _ask_metadata(self, path: Path):
        """Prompt for artist/title when a file carries no usable tags.

        Returns (artist, title) — either may be None, meaning let the
        pipeline fall back — or None entirely if the user cancelled the run.
        Only called for a single local file; links and batches resolve their
        own metadata.
        """
        from . import tags

        artist, name = tags.guess_metadata(str(path))
        if artist and name:
            return artist, name

        win = tk.Toplevel(self.root)
        win.title("Song details")
        win.resizable(False, False)
        win.transient(self.root)
        frame = ttk.Frame(win, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=f"{path.name} has no usable tags — how should "
                              f"this song be listed?").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        artist_var = tk.StringVar(value=artist or "")
        name_var = tk.StringVar(value=name or path.stem)
        ttk.Label(frame, text="Artist").grid(row=1, column=0, sticky="w")
        artist_entry = ttk.Entry(frame, textvariable=artist_var, width=36)
        artist_entry.grid(row=1, column=1, sticky="ew", padx=(10, 0))
        ttk.Label(frame, text="Title").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=name_var, width=36).grid(
            row=2, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))

        result = {}
        def ok(_event=None):
            result["v"] = (artist_var.get().strip() or None,
                           name_var.get().strip() or None)
            win.destroy()
        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(16, 0))
        ttk.Button(buttons, text="Cancel", command=win.destroy).pack(
            side="left", padx=(0, 8))
        ttk.Button(buttons, text="Chart it", style="Accent.TButton",
                   command=ok).pack(side="left")
        win.bind("<Return>", ok)
        win.bind("<Escape>", lambda e: win.destroy())

        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width()
                                       - win.winfo_reqwidth()) // 2
        win.geometry(f"+{max(x, 0)}+{self.root.winfo_rooty() + 160}")
        win.grab_set()
        artist_entry.focus_set()
        self.root.wait_window(win)
        return result.get("v")

    def _write(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self):
        import copy

        from . import youtube

        # Anything still sitting in the entry joins the queue first.
        if self.audio.get().strip():
            self._add_from_entry()
            if self.audio.get().strip():
                return  # entry content was invalid; the error dialog showed
        if not self.queue_items:
            messagebox.showerror(
                "chartgen", "Choose audio file(s) or paste YouTube link(s) first.")
            return
        batch = [i for i in self.queue_items] if len(self.queue_items) > 1 else []
        audio = self.queue_items[0]
        is_url = youtube.is_youtube_url(audio)
        meta_artist = meta_name = None
        if not batch and not is_url:
            asked = self._ask_metadata(Path(audio))
            if asked is None:
                return  # user closed the prompt: don't chart
            meta_artist, meta_name = asked
        save_settings(self._settings())
        base = Namespace(
            # URLs must stay strings; Path("https://…") collapses the //.
            audio=audio if is_url else (Path(audio) if not batch else None),
            outdir=Path(self.outdir.get()),
            subdiv=GRIDS[self.grid_choice.get()],
            min_variety=0.80,
            hopos=self.hopos.get(), no_star_power=not self.star_power.get(),
            no_sections=not self.sections.get(), no_sustains=not self.sustains.get(),
            target_diff=(None if self.target_diff.get() == "Auto"
                         else int(self.target_diff.get())),
            lyrics=self.lyrics.get(),
            taps=self.taps.get(),
            opens=self.opens.get(),
            no_solos=not self.solos.get(),
            ghl=self.ghl.get(),
            lyric_source=LYRIC_SOURCES.get(self.lyric_source.get(), "auto"),
            name=meta_name,
            artist=meta_artist or "Unknown",
            album="", genre="", year="",
        )
        queue_items = batch or [base.audio]

        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.cancel.clear()
        self.result = None
        self.batch_pos = None
        self.queue_items = []
        self.started = time.monotonic()
        self.bar["value"] = 0
        self.bar["maximum"] = (7 if is_url else 6) * len(queue_items)
        self.go.configure(state="disabled")
        self.stop.configure(state="normal")
        self.open_button.configure(state="disabled")
        self.status.configure(text="Working…")
        if len(queue_items) > 1:
            self._write(f"Batch of {len(queue_items)} songs — each gets its own "
                        f"folder, named from its tags.\n")
        else:
            self._write("This takes a few minutes. The window stays responsive.\n")

        def work():
            done = failed = 0
            last_result = None
            try:
                try:
                    items = youtube.expand_inputs(
                        queue_items, lambda m: self.events.put(("log", m)))
                except ValueError as error:
                    self.events.put(("error", str(error)))
                    return
                if len(items) != len(queue_items):
                    self.events.put(("max", (7 if is_url else 6) * len(items)))
                # Playlists may have expanded: show the real ladder.
                self.events.put(("queue_set", [str(i) for i in items]))
                def chart_one(item, position, total):
                    opts = copy.copy(base)
                    opts.audio = item
                    if total > 1:
                        self.events.put(("song", (position, total)))
                        self.events.put(("log", f"\n=== [{position}/{total}] "
                                                f"{Path(str(item)).name} ==="))
                        # Per-song metadata from each file's own tags; songs
                        # already charted are skipped, so re-pasting a batch
                        # never redoes finished work.
                        opts.name, opts.artist = None, "Unknown"
                        opts.skip_existing = True
                    return pipeline.run(
                        opts, progress=lambda m: self.events.put(("log", m)),
                        should_cancel=self.cancel.is_set)

                retry = []
                for i, item in enumerate(items, 1):
                    try:
                        result = chart_one(item, i, len(items))
                        if not result.get("skipped"):
                            last_result = result
                            done += 1
                    except (ValueError, OSError) as error:
                        if len(items) == 1:
                            raise
                        self.events.put(("log", f"FAILED: {error}"))
                        retry.append(item)
                    finally:
                        # The ladder shrinks as each song finishes, whatever
                        # its outcome — the log carries the verdicts.
                        self.events.put(("item_done", None))
                # Second chance for songs YouTube 403-blocked mid-batch; the
                # throttle usually lifts while the rest of the queue charts.
                if retry and not self.cancel.is_set():
                    self.events.put(("log", f"\nRetrying {len(retry)} failed…"))
                    for i, item in enumerate(retry, 1):
                        try:
                            result = chart_one(item, i, len(retry))
                            if not result.get("skipped"):
                                last_result = result
                                done += 1
                        except (ValueError, OSError) as error:
                            failed += 1
                            self.events.put(("log", f"FAILED again: {error}"))
                if len(items) > 1:
                    self.events.put(("log", f"\nBatch finished: {done} ok, {failed} failed."))
                self.events.put(("done", last_result))
            except pipeline.Cancelled:
                self.events.put(("cancelled", None))
            except Exception as error:  # surface everything; there is no console
                self.events.put(("error", f"{type(error).__name__}: {error}"))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _request_cancel(self):
        self.cancel.set()
        self.status.configure(text="Cancelling…")
        self._write("cancelling after the current stage…")

    def _drain(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "max":
                    self.bar["maximum"] = payload
                elif kind == "song":
                    self.batch_pos = payload
                elif kind == "queue_set":
                    self.run_items = list(payload)
                    self.run_done = 0
                    unknown = [i for i in self.run_items
                               if str(i).lower().startswith(("http", "www."))
                               and str(i) not in self.queue_labels]
                    if unknown:
                        threading.Thread(target=self._resolve_titles,
                                         args=(unknown,), daemon=True).start()
                    self._render_run_ladder()
                elif kind == "queue_title":
                    link, label = payload
                    self.queue_labels[str(link)] = label
                    if self.worker and self.worker.is_alive():
                        self._render_run_ladder()
                    else:
                        needle = f"  {link}"
                        for i, row in enumerate(self.queue_box.get(0, "end")):
                            if row == needle:
                                self.queue_box.delete(i)
                                self.queue_box.insert(i, f"  {label}")
                                break
                elif kind == "item_done":
                    self.run_done += 1
                    self._render_run_ladder()
                elif kind == "log":
                    self._write(payload)
                    if payload.startswith("["):
                        self.bar["value"] = self.bar["value"] + 1
                elif kind == "done":
                    self._finish(payload)
                elif kind == "cancelled":
                    self._write("\nCancelled.")
                    self._reset("Cancelled")
                elif kind == "error":
                    self._write(f"\nFAILED: {payload}")
                    self._reset("Failed")
                    messagebox.showerror("chartgen", payload)
        except queue.Empty:
            pass
        # The clock lives here, not in the log handler: stages can run for
        # minutes without emitting a line, and the elapsed time should tick
        # every second regardless.
        if self.worker and self.worker.is_alive() and not self.cancel.is_set():
            elapsed = time.monotonic() - self.started
            count = (f"  ·  {self.batch_pos[0]}/{self.batch_pos[1]}"
                     if self.batch_pos else "")
            self.status.configure(text=f"Working… {fmt_elapsed(elapsed)}{count}")
        self.root.after(100, self._drain)

    def _finish(self, result):
        self.result = result
        self.bar["value"] = self.bar["maximum"]
        if result is None or "summary" not in result:
            # Batch where every song failed or was skipped: no table to show.
            self._reset("Done")
            return
        rows = result["summary"]
        self._write("")
        self._write(f"{'tier':<14}{'notes':>7}{'sustains':>10}{'star pow':>10}"
                    f"{'variety':>9}{'walk':>7}")
        for tier, row in rows.items():
            flag = "  <- weak" if min(row["variety"], row["walk"]) < 0.80 else ""
            self._write(f"{tier:<14}{row['notes']:>7}{row['sustains']:>10}"
                        f"{row['star_power']:>10}{row['variety']:>9.2f}"
                        f"{row['walk']:>7.2f}{flag}")
        self._write(f"\nSaved to: {result['song_dir']}")
        self._write("Rescan songs in Clone Hero to see it.")
        self._reset(f"Done in {fmt_elapsed(time.monotonic() - self.started)}")
        self.open_button.configure(state="normal")

    def _reset(self, status):
        self.go.configure(state="normal")
        self.stop.configure(state="disabled")
        self.status.configure(text=status)

    def _open_folder(self):
        if self.result:
            path = str(self.result["song_dir"])
            subprocess.Popen(["xdg-open", path])

    def _settings(self) -> dict:
        return {
            "outdir": self.outdir.get(),
            "grid": self.grid_choice.get(), "sustains": self.sustains.get(),
            "star_power": self.star_power.get(), "sections": self.sections.get(),
            "hopos": self.hopos.get(),
            "target_diff": self.target_diff.get(), "lyrics": self.lyrics.get(),
            "lyric_source": self.lyric_source.get(),
            "taps_v2": self.taps.get(),
            "solos": self.solos.get(),
            "opens": self.opens.get(),
            "ghl": self.ghl.get(),
        }

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askokcancel("chartgen", "A chart is still generating. Quit?"):
                return
            self.cancel.set()
        # Settings used to persist only after a successful Generate; pointing
        # the output folder somewhere and closing lost it. Closing saves too.
        save_settings(self._settings())
        self.root.destroy()


def main():
    root = tk.Tk()
    apply_theme(root)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
