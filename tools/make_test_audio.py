"""Synthesize a 40s guitar-ish riff at a known BPM.

Used as a fixture: because the BPM is known exactly, it also validates the
tempo-detection stage (see chartgen/tempo.py).
"""
import argparse
import numpy as np
import soundfile as sf

SR = 44100
# E-minor riff spanning most of a guitar's range, one entry per eighth note.
# None is a rest; a tuple is a chord. The rests at index 1 and 5 are load-bearing
# for the sustain checks: 2 of every 6 onsets precede a one-beat gap, so Expert
# should sustain ~33% of its notes.
RIFF_HZ = [
    82.41,            # E2, low open
    None,
    (98.00, 146.83),  # G2 + D3 power chord
    220.00,           # A3
    329.63,           # E4
    None,
    (493.88, 659.26),  # B4 + E5, up near the top
    246.94,           # B3
]


def pluck(freq, dur, sr=SR):
    """Karplus-Strong-lite: decaying harmonic stack. Good enough for onset tests.

    Deliberately not a great guitar model. It exercises timing reliably, but the
    lane the model picks for a sine stack means little — judge fret variety on
    real recordings, not on this.
    """
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    env = np.exp(-4.0 * t / dur)
    wave = sum(np.sin(2 * np.pi * freq * h * t) / h for h in (1, 2, 3, 4))
    return (wave * env).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--out", default="test_audio.wav")
    args = ap.parse_args()

    eighth = 60.0 / args.bpm / 2
    total = int(SR * args.seconds)
    buf = np.zeros(total + SR, dtype=np.float32)

    n_eighths = int(args.seconds / eighth) + 1
    for i in range(n_eighths):
        entry = RIFF_HZ[i % len(RIFF_HZ)]
        if entry is None:
            continue
        start = int(i * eighth * SR)
        for freq in (entry if isinstance(entry, tuple) else (entry,)):
            # slight ring-out past the note slot, like a real guitar
            note = pluck(freq, eighth * 1.6)
            buf[start:start + len(note)] += note

    # steady kick on every beat so beat-trackers have something to lock onto
    beat = 60.0 / args.bpm
    kick = pluck(55.0, 0.12) * 1.5
    for i in range(int(args.seconds / beat) + 1):
        start = int(i * beat * SR)
        buf[start:start + len(kick)] += kick

    buf = buf[:total]
    buf /= np.max(np.abs(buf)) * 1.05
    sf.write(args.out, buf, SR)
    print(f"wrote {args.out}: {args.seconds:.0f}s @ {args.bpm:g} BPM")


if __name__ == "__main__":
    main()
