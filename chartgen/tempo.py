"""Beat/tempo detection and musical time->tick conversion.

Why this exists: audio2chart writes every chart at a fixed 200 BPM and places
notes at whatever tick the raw onset time happens to land on. Measured on a test
clip, 2 of 180 notes fell on a beat. Difficulty reducers decide what to keep by
asking "is this note on a beat?", so with an off-grid chart that logic never
fires and Easy/Medium collapse into the same track. Everything here exists to
put notes on a real musical grid so that stays meaningful.
"""
import functools
from dataclasses import dataclass

import librosa
import numpy as np
from scipy.special import expit

RESOLUTION = 480  # ticks per beat; divisible by 4 (16ths) and 3 (triplets)
# librosa's onset envelope peaks after the transient: beats tracked on it sat
# 19-37 ms late on click tracks, and the official Guitar Hero Live notes a
# median 29 ms before them. Envelope lookups against the beat grid shift by this.
ONSET_LAG_S = 0.03
TRACKER_SR = 44100  # madmom's beat networks are trained on 44.1 kHz audio
SMOOTH_NEIGHBOURS = 2


@dataclass
class TempoMap:
    """Maps audio time to chart ticks, anchored so beat N lands on tick N*resolution."""

    beat_times: np.ndarray  # detected beat positions, seconds
    resolution: int = RESOLUTION
    pickup_beats: int = 1  # beats between tick 0 and the first detected beat

    @property
    def bpm(self) -> float:
        """Overall tempo, least-squares fitted across every beat.

        Not the median of beat gaps: trackers snap beats to analysis frames,
        and with librosa's ~23 ms frames that jitter biased the median enough
        to read a 120.0 BPM click track as 117.5. Fitting the whole run
        averages the jitter out.
        """
        idx = np.arange(len(self.beat_times))
        slope = np.polyfit(idx, self.beat_times, 1)[0]
        return float(60.0 / slope)

    def _tail_step(self) -> float:
        """Seconds per beat to carry past the last detected beat.

        The single final interval is the wrong thing to use: beats detected
        during a fade-out are noisy, and one short gap inflated the outro so
        badly that charts ran 20+ seconds past the end of the audio. The
        median of the last few intervals is stable and still local enough to
        follow a song that genuinely slows down.
        """
        bt = self.beat_times
        if len(bt) < 2:
            return 0.0
        gaps = np.diff(bt[-9:]) if len(bt) >= 3 else np.diff(bt)
        return float(np.median(gaps))

    def beat_to_time(self, beat: float) -> float:
        """Inverse of time_to_beat: audio timestamp of a fractional beat."""
        bt = self.beat_times
        if beat <= self.pickup_beats:
            return bt[0] * (beat / self.pickup_beats) if self.pickup_beats else 0.0
        idx = beat - self.pickup_beats
        if idx > len(bt) - 1 and len(bt) >= 2:
            step = self._tail_step()
            if step > 0:
                return float(bt[-1] + (idx - (len(bt) - 1)) * step)
        return float(np.interp(idx, np.arange(len(bt)), bt))

    def time_to_beat(self, t: float) -> float:
        """Fractional beat number for an audio timestamp, tick 0 == t 0."""
        bt = self.beat_times
        if t <= bt[0]:
            # Inside the pickup: linear from tick 0 to the first beat.
            return self.pickup_beats * (t / bt[0]) if bt[0] > 0 else 0.0
        if t > bt[-1] and len(bt) >= 2:
            # Beat trackers stop once a song fades out, and
            # np.interp CLAMPS beyond its last beat — so every note in the
            # outro quantized onto one tick and the chart simply stopped.
            # Measured: charts ended up to 25s early, at exactly the last
            # detected beat, on both engines. Carry the final tempo forward.
            step = self._tail_step()
            if step > 0:
                return (len(bt) - 1) + (t - bt[-1]) / step + self.pickup_beats
        # np.interp handles tempo drift by interpolating between real beats
        # instead of assuming one constant BPM.
        idx = np.interp(t, bt, np.arange(len(bt)))
        return float(idx) + self.pickup_beats

    def quantize(self, t: float, subdiv: int = 4) -> int:
        """Snap an audio timestamp to the nearest 1/subdiv note and return ticks."""
        step = self.resolution // subdiv
        return int(round(self.time_to_beat(t) * self.resolution / step)) * step

    def sync_track(self, max_drift_s: float = 0.025) -> list[tuple[int, int]]:
        """[(tick, bpm*1000)] events reproducing the detected beat times.

        Greedily extends one constant-tempo run for as long as every beat in it
        stays within max_drift_s of where that tempo would place it, then starts
        a new event. This distinguishes the two things that look alike in the
        detected beats: frame-quantization jitter (absorbed, so a steady song
        collapses to a single event) and genuine tempo drift (followed).

        Each run is fitted against the time the emitted map actually reaches, so
        rounding error does not accumulate across runs.

        ponytail: O(n^2) over beats; a few ms for song-length input. Segment the
        beat list first if that ever matters.
        """
        bt = self.beat_times
        res = self.resolution
        events: list[tuple[int, int]] = []

        if self.pickup_beats > 0 and bt[0] > 0.02:
            events.append((0, int(round(60.0 * self.pickup_beats / bt[0] * 1000))))
        map_time = float(bt[0])  # playback time the emitted map reaches at beat 0

        i = 0
        while i < len(bt) - 1:
            fit = None
            j = i + 1
            while j < len(bt):
                span = np.arange(1, j - i + 1)
                offsets = bt[i + 1:j + 1] - map_time
                # Least-squares seconds-per-beat through the origin at map_time.
                step = float((span * offsets).sum() / (span * span).sum())
                if step <= 0:
                    break
                bpm_milli = max(1, int(round(60.0 / step * 1000)))
                step_q = 60.0 / (bpm_milli / 1000.0)  # tempo as the game sees it
                if np.max(np.abs(offsets - span * step_q)) > max_drift_s:
                    break
                fit = (j, bpm_milli, step_q)
                j += 1

            if fit is None:  # a single beat too far off to fit: pin it exactly
                step_q = max(float(bt[i + 1]) - map_time, 1e-3)
                bpm_milli = max(1, int(round(60.0 / step_q * 1000)))
                fit = (i + 1, bpm_milli, 60.0 / (bpm_milli / 1000.0))

            j, bpm_milli, step_q = fit
            # A repeat of the current tempo is a no-op event; drop it.
            if not events or events[-1][1] != bpm_milli:
                events.append(((i + self.pickup_beats) * res, bpm_milli))
            map_time += (j - i) * step_q
            i = j

        return events or [(0, int(round(self.bpm * 1000)))]


def load(audio_path: str):
    """Decode audio once; callers reuse it for beats, sections and duration."""
    return librosa.load(audio_path, mono=True)


@functools.cache
def _madmom():
    """madmom's offline beat tracker, with its eight BLSTM networks stacked.

    madmom runs the networks one after another, frame by frame in Python: 20 s
    on a 3-minute song. They share one architecture, so stacking their weights
    steps all eight, both directions, through each frame together. The
    weights, features and DBN decoding are madmom's own.
    """
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

    features, ensemble = RNNBeatProcessor().processors
    nets = ensemble.processors[0].processors
    layers = []
    for depth in range(len(nets[0].layers) - 1):
        lstms = [lstm for net in nets
                 for lstm in (net.layers[depth].fwd_layer, net.layers[depth].bwd_layer)]
        gates = [(lstm.input_gate, lstm.forget_gate, lstm.cell, lstm.output_gate) for lstm in lstms]
        stacked = [np.stack([np.hstack([getattr(gate, part) for gate in four]) for four in gates])
                   for part in ("weights", "bias", "recurrent_weights")]
        peephole = np.stack([[lstm.input_gate.peephole_weights, lstm.forget_gate.peephole_weights,
                              lstm.output_gate.peephole_weights] for lstm in lstms])
        layers.append((*stacked, peephole))
    head = (np.stack([net.layers[-1].weights for net in nets]),
            np.stack([net.layers[-1].bias for net in nets]))
    return features, layers, head, DBNBeatTrackingProcessor(fps=100)


def _beat_activation(y44: np.ndarray) -> np.ndarray:
    """Beat probability per 10 ms frame: the mean of madmom's eight networks."""
    features, layers, (head_w, head_b), _ = _madmom()
    x = np.asarray(features(y44), dtype=np.float32)
    frames, n_nets = len(x), len(head_w)
    inputs = np.broadcast_to(x, (n_nets, *x.shape))
    for weights, bias, recurrent, peephole in layers:
        hidden = recurrent.shape[1]
        # (net, direction, frame, gate): the input term of every gate for all frames
        drive = np.matmul(inputs[:, None], weights.reshape(n_nets, 2, *weights.shape[1:]))
        drive += bias.reshape(n_nets, 2, 1, -1)
        drive[:, 1] = drive[:, 1, ::-1]
        drive = np.ascontiguousarray(drive.reshape(2 * n_nets, frames, -1).transpose(1, 0, 2))
        out = np.empty((frames, 2 * n_nets, hidden), dtype=np.float32)
        prev = np.zeros((2 * n_nets, hidden), dtype=np.float32)
        state = np.zeros_like(prev)
        for t in range(frames):
            gate = drive[t] + np.matmul(prev[:, None], recurrent)[:, 0]
            in_gate = expit(gate[:, :hidden] + state * peephole[:, 0])
            forget = expit(gate[:, hidden:2 * hidden] + state * peephole[:, 1])
            state = np.tanh(gate[:, 2 * hidden:3 * hidden]) * in_gate + state * forget
            prev = out[t] = np.tanh(state) * expit(gate[:, 3 * hidden:] + state * peephole[:, 2])
        out = out.reshape(frames, n_nets, 2, hidden)
        out[:, :, 1] = out[::-1, :, 1]
        inputs = out.transpose(1, 0, 2, 3).reshape(n_nets, frames, 2 * hidden)
    return expit(np.matmul(inputs, head_w) + head_b[:, None]).mean(axis=0)[:, 0]


def _smooth(beats: np.ndarray, half: int = SMOOTH_NEIGHBOURS) -> np.ndarray:
    """Move every beat onto the straight line through it and `half` neighbours
    each side: this removes the DBN's 10 ms frame jitter and still follows
    tempo changes. On 100 official Guitar Hero Live songs, 2 neighbours put
    86.8% of the notes on the 16th grid within 20 ms, against 85.3% raw;
    4 and 8 neighbours scored lower."""
    out = beats.copy()
    for i in range(len(beats)):
        lo, hi = max(0, i - half), min(len(beats), i + half + 1)
        slope, intercept = np.polyfit(np.arange(lo, hi), beats[lo:hi], 1)
        out[i] = intercept + slope * i
    return out


def _halve(beats: np.ndarray, y, sr) -> np.ndarray:
    """Drop every other beat, keeping the parity that lands on stronger onsets.

    Halving the tempo label means half the detected beats become off-beats;
    the ones to keep as beats are wherever the accents actually are.
    """
    env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(env, sr=sr)
    even, odd = beats[0::2], beats[1::2]
    strength = [float(np.interp(b + ONSET_LAG_S, times, env).mean()) for b in (even, odd)]
    return even if strength[0] >= strength[1] else odd


def _double(beats: np.ndarray) -> np.ndarray:
    """Insert a beat at every midpoint: the label doubles, the grid does not move."""
    mids = (beats[:-1] + beats[1:]) / 2
    return np.sort(np.concatenate([beats, mids]))


def detect(y, sr, resolution: int = RESOLUTION, bpm_mult: str = "1") -> TempoMap:
    """Beat grid from madmom's RNN + DBN tracker. bpm_mult '0.5' or '2'
    forces the other tempo octave; anything else keeps the detected one.

    The octave matters even though the physical grid is identical either way:
    the label decides tick spacing, and therefore whether close notes fall
    inside the natural HOPO window (a 16th at 95 BPM HOPOs, the same gap
    labelled 8th-at-190 strums). There is no automatic octave rule: the
    official Guitar Hero Live charts run 69-258 BPM, madmom alone picks
    their octave on 86% of 100 songs, and folding its tempo into a fixed
    range lowered that. madmom's beats sit on the transient (measured lag
    -1.5 ms), so they get no shift.
    """
    y44 = y if sr == TRACKER_SR else librosa.resample(y, orig_sr=sr, target_sr=TRACKER_SR)
    beats = np.asarray(_madmom()[3](_beat_activation(y44)), dtype=float)
    if len(beats) < 2:
        raise ValueError("could not detect a beat grid in this audio")
    beats = _smooth(beats)

    if bpm_mult == "0.5" and len(beats) >= 4:
        beats = _halve(beats, y, sr)
    elif bpm_mult == "2":
        beats = _double(beats)

    beat_dur = float(np.median(np.diff(beats)))
    pickup = max(1, int(round(beats[0] / beat_dur))) if beats[0] > 0.02 else 0
    return TempoMap(beat_times=beats, resolution=resolution, pickup_beats=pickup)
