import argparse, json, math, os, subprocess, shutil, random, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import namedtuple

import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d
import cv2
from tqdm import tqdm

def run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}):\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr[:8000]}")

def has_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found. Install ffmpeg (sudo apt install ffmpeg).")

def ffprobe_duration(path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=nk=1:nw=1", path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(p.stdout.strip())
    except Exception:
        return 0.0

def superflux_onset_env(
    y: np.ndarray,
    sr: int,
    hop: int,
    n_fft: int,
    n_mels: int = 128,
    smooth_frames: int = 3
) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=1.0)
    S = np.log1p(S)
    D = np.diff(S, axis=1)
    D = np.maximum(D, 0.0)

    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)

    def band_mask(lo, hi):
        return (mel_freqs >= lo) & (mel_freqs < hi)

    low  = D[band_mask(20,   200),  :]
    mid  = D[band_mask(200,  1500), :]
    high = D[band_mask(1500, 8000), :]

    env = (0.9*np.mean(low,  axis=0, keepdims=True) if low.size  else 0) \
        + (1.2*np.mean(mid,  axis=0, keepdims=True) if mid.size  else 0) \
        + (1.1*np.mean(high, axis=0, keepdims=True) if high.size else 0)
    env = env.ravel()

    if smooth_frames > 1:
        env = uniform_filter1d(env, size=smooth_frames)

    return np.pad(env, (1, 0), mode="edge")

def detect_onsets_percussive(
    y_perc: np.ndarray,
    sr: int,
    hop: int,
    n_fft: int,
    delta: float,
    pre_max: int,
    post_max: int,
    pre_avg: int,
    post_avg: int,
    backtrack: bool
) -> np.ndarray:
    env = superflux_onset_env(y_perc, sr, hop, n_fft)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=env,
        sr=sr,
        hop_length=hop,
        units="frames",
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        backtrack=backtrack
    )
    return onset_frames

def track_beats(
    onset_env: np.ndarray,
    sr: int,
    hop: int,
    start_bpm: float = 120.0,
    tempo_min: float = 70.0,
    tempo_max: float = 180.0,
    tracker: str = "dynamic"
) -> Tuple[float, np.ndarray, np.ndarray]:
    def fold_octave(bpm: float) -> float:
        if bpm <= 0:
            return start_bpm
        while bpm < tempo_min - 1e-6:
            bpm *= 2.0
        while bpm > tempo_max + 1e-6:
            bpm /= 2.0
        return bpm

    if tracker == "plp":
        try:
            pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr, hop_length=hop)
            beat_frames = np.flatnonzero(librosa.util.localmax(pulse) & (pulse > np.median(pulse)))
            if beat_frames.size >= 2:
                beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
                tempo = 60.0 / float(np.median(np.diff(beat_times)))
                return fold_octave(tempo), beat_frames, beat_times
        except Exception:
            pass

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop, start_bpm=start_bpm, units="frames", trim=False
    )
    tempo = float(np.atleast_1d(tempo)[0])
    corrected = fold_octave(tempo)
    if abs(corrected - tempo) > 1.0:
        try:
            tempo2, beat_frames2 = librosa.beat.beat_track(
                onset_envelope=onset_env, sr=sr, hop_length=hop,
                start_bpm=corrected, bpm=corrected, units="frames", trim=False
            )
            if np.size(beat_frames2) >= 2:
                beat_frames = beat_frames2
                tempo = float(np.atleast_1d(tempo2)[0])
            else:
                tempo = corrected
        except Exception:
            tempo = corrected
    else:
        tempo = corrected

    beat_times = librosa.frames_to_time(np.asarray(beat_frames), sr=sr, hop_length=hop)
    return float(tempo), np.asarray(beat_frames), beat_times

def _complete_beat_grid(beat_times: np.ndarray, tempo: float, duration: float) -> np.ndarray:
    bt = np.unique(np.asarray(beat_times, dtype=float))
    bt = bt[(bt >= 0.0) & (bt <= duration)]
    if tempo and tempo > 0:
        period = 60.0 / float(tempo)
    elif bt.size >= 2:
        period = float(np.median(np.diff(bt)))
    else:
        return bt
    if period <= 1e-3:
        return bt
    if bt.size == 0:
        return np.arange(0.0, duration, period, dtype=float)
    filled = [float(bt[0])]
    for a, b in zip(bt[:-1], bt[1:]):
        gap = float(b - a)
        n_missing = int(round(gap / period)) - 1
        if n_missing >= 1 and gap > 1.5 * period:
            local = gap / (n_missing + 1)
            filled.extend(float(a + local * k) for k in range(1, n_missing + 1))
        filled.append(float(b))
    lead_in: List[float] = []
    t = filled[0] - period
    while t >= 0.0:
        lead_in.append(t)
        t -= period
    tail: List[float] = []
    t = filled[-1] + period
    while t < duration - 1e-3:
        tail.append(t)
        t += period
    return np.array(sorted(lead_in) + filled + tail, dtype=float)

def _beat_strengths(env: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(beat_frames, dtype=int)
    if frames.size == 0 or env.size == 0:
        return np.zeros(frames.size, dtype=float)
    v  = env[np.clip(frames, 0, env.size - 1)]
    hi = float(np.percentile(v, 95)) + 1e-9
    return np.clip(v / hi, 0.0, 1.0)

def _merge_min_seg(segs: List[Tuple[float, float]], min_seg: float, duration: float) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    carry = None
    for s, e in segs:
        if carry is None:
            if (e - s) < min_seg and len(segs) > 1:
                carry = [s, e]
            else:
                merged.append((s, e))
        else:
            carry[1] = e
            if (carry[1] - carry[0]) >= min_seg:
                merged.append((carry[0], carry[1]))
                carry = None
    if carry is not None:
        if merged and (carry[1] - carry[0]) < min_seg:
            merged[-1] = (merged[-1][0], carry[1])
        else:
            merged.append((carry[0], carry[1]))
    return [(max(0.0, s), min(duration, e)) for s, e in merged if e > s + 1e-6]

def _snap_segments(
    segs: List[Tuple[float, float]],
    beat_times: np.ndarray,
    tol_s: float,
    duration: float
) -> List[Tuple[float, float]]:
    if beat_times is None or len(beat_times) == 0 or tol_s <= 0 or len(segs) < 2:
        return segs
    bts = np.asarray(beat_times, dtype=float)
    boundaries = [0.0] + [float(e) for _, e in segs]
    snapped = [0.0]
    for b in boundaries[1:-1]:
        j  = int(np.argmin(np.abs(bts - b)))
        nb = float(bts[j])
        snapped.append(nb if abs(nb - b) <= tol_s else b)
    snapped.append(boundaries[-1])

    cleaned = sorted(set(min(max(0.0, x), duration) for x in snapped))
    return [(cleaned[i], cleaned[i + 1]) for i in range(len(cleaned) - 1) if cleaned[i + 1] > cleaned[i] + 1e-6]

def choose_segments(
    y: np.ndarray,
    sr: int,
    hop: int,
    n_fft: int,
    min_seg: float,
    mode: str,
    onset_kwargs: Dict,
    y_perc: Optional[np.ndarray] = None,
    beat_times: Optional[np.ndarray] = None,
    beat_frames: Optional[np.ndarray] = None,
    env_perc: Optional[np.ndarray] = None,
    tempo: float = 0.0,
    snap: bool = True,
    snap_tol_beats: float = 0.5,
    beats_per_bar: int = 4,
    bpm: float = 0.0,
    subdiv: int = 1,
    beat_refine_s: float = 0.06
) -> Tuple[List[Tuple[float, float]], float]:
    duration = len(y) / sr

    if mode == "grid":
        if not bpm or bpm <= 0:
            raise RuntimeError("segment-mode grid requires --bpm > 0.")
        subdiv = int(max(1, subdiv))
        step = (60.0 / float(bpm)) / float(subdiv)
        if step <= 1e-6:
            return [(0.0, duration)], duration
        times = np.arange(0.0, duration + 1e-9, step, dtype=float)
        if times.size < 2:
            return [(0.0, duration)], duration
        segs = [(float(times[i]), float(min(duration, times[i+1]))) for i in range(times.size - 1)]
        return _merge_min_seg(segs, min_seg, duration), duration

    if y_perc is None:
        _, y_perc = librosa.effects.hpss(y, margin=(1.0, 2.0))
    if env_perc is None:
        env_perc = superflux_onset_env(y_perc, sr, hop, n_fft)
    if beat_frames is None or beat_times is None:
        tempo, beat_frames, beat_times = track_beats(env_perc, sr, hop)
    beat_frames = np.asarray(beat_frames)
    beat_times  = np.asarray(beat_times, dtype=float)

    onset_frames = detect_onsets_percussive(y_perc, sr, hop, n_fft, **onset_kwargs)
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)

    if beat_frames.size and env_perc.size:
        accent_at_beats = env_perc[np.clip(beat_frames, 0, env_perc.size - 1)]
    else:
        accent_at_beats = np.array([])

    def segments_from_onsets(times: np.ndarray) -> List[Tuple[float, float]]:
        if len(times) < 2:
            return [(0.0, duration)]
        times = np.clip(times, 0, duration)
        t = np.concatenate([[0.0], times, [duration]])
        return [(float(t[i]), float(t[i + 1])) for i in range(len(t) - 1)]

    def segments_from_bars(bt: np.ndarray, accent: np.ndarray, bpb: int = 4) -> List[Tuple[float, float]]:
        n = len(bt)
        if n < bpb:
            return [(0.0, duration)]
        best_phase, best_score = 0, -1.0
        for phase in range(bpb):
            idxs = np.arange(phase, n, bpb)
            sc = float(np.mean(accent[idxs])) if accent.size and idxs.size else 0.0
            if sc > best_score:
                best_score, best_phase = sc, phase
        downbeats = [float(bt[i]) for i in range(best_phase, n, bpb)]
        bars = sorted(set([0.0] + downbeats + [duration]))
        return [(bars[i], bars[i + 1]) for i in range(len(bars) - 1)]

    if mode == "beats":
        grid = np.asarray(beat_times, dtype=float)
        grid = grid[(grid > 1e-3) & (grid < duration - 1e-3)]
        subdiv = int(max(1, subdiv))
        if subdiv > 1 and grid.size >= 2:
            fine: List[float] = []
            for a, b in zip(grid[:-1], grid[1:]):
                fine.extend(float(a + (b - a) * k / subdiv) for k in range(subdiv))
            fine.append(float(grid[-1]))
            grid = np.array(fine, dtype=float)
        if beat_refine_s > 0 and grid.size:
            peak_kwargs = dict(onset_kwargs)
            peak_kwargs["backtrack"] = False
            peak_frames = detect_onsets_percussive(y_perc, sr, hop, n_fft, **peak_kwargs)
            peaks = librosa.frames_to_time(peak_frames, sr=sr, hop_length=hop)
            if peaks.size:
                near = peaks[np.argmin(np.abs(peaks[None, :] - grid[:, None]), axis=1)]
                grid = np.where(np.abs(near - grid) <= beat_refine_s, near, grid)
        cuts = [0.0]
        for g in sorted(set(float(x) for x in grid)):
            if g > cuts[-1] + 1e-3 and g < duration - 1e-3:
                cuts.append(g)
        cuts.append(duration)
        segs = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i] + 1e-6]
        return _merge_min_seg(segs, min_seg, duration), duration

    if mode == "onsets":
        segs = segments_from_onsets(onset_times)
    elif mode == "bars":
        segs = segments_from_bars(beat_times, accent_at_beats, beats_per_bar)
    else:
        segs_on = segments_from_onsets(onset_times)
        med_on  = np.median([e - s for s, e in segs_on]) if segs_on else 0
        if len(onset_times) >= 12 and med_on >= 0.25:
            segs = segs_on
        else:
            segs = segments_from_bars(beat_times, accent_at_beats, beats_per_bar)

    if snap and beat_times.size:
        if tempo and tempo > 0:
            beat_period = 60.0 / float(tempo)
        elif beat_times.size >= 2:
            beat_period = float(np.median(np.diff(beat_times)))
        else:
            beat_period = 0.0
        segs = _snap_segments(segs, beat_times, snap_tol_beats * beat_period, duration)

    return _merge_min_seg(segs, min_seg, duration), duration

def key_mode_from_chroma(chroma_vec: np.ndarray) -> Tuple[str, float]:
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
    c = chroma_vec / (np.sum(chroma_vec) + 1e-9)

    rot_M = np.array([np.roll(major_profile, k) for k in range(12)])
    rot_m = np.array([np.roll(minor_profile, k) for k in range(12)])
    sM = float(np.max(rot_M @ c))
    sN = float(np.max(rot_m @ c))

    total = sM + sN + 1e-9
    if sM >= sN:
        return "major", sM / total
    else:
        return "minor", sN / total

QUIET_QUANTILE = 0.35

def detect_sections(
    mfcc: np.ndarray,
    chroma: np.ndarray,
    hop_times: np.ndarray,
    duration: float,
    k: int,
    min_section_s: float,
    beat_times: Optional[np.ndarray] = None,
    beat_period: float = 0.0
) -> List[float]:
    n = min(mfcc.shape[1], chroma.shape[1], hop_times.size)
    if n < 8 or duration < 2.0 * min_section_s:
        return [0.0]
    frame_dt = float(hop_times[1] - hop_times[0]) if n > 1 else 0.0116
    stride   = max(1, int(round(0.25 / frame_dt)))
    blocks   = np.arange(0, n, stride)
    X  = np.vstack([mfcc[:, :n], chroma[:, :n]])
    Xs = librosa.util.sync(X, blocks, aggregate=np.mean)
    Xs = (Xs - Xs.mean(axis=1, keepdims=True)) / (Xs.std(axis=1, keepdims=True) + 1e-9)
    if k <= 0:
        k = int(np.clip(round(duration / 20.0), 2, 16))
    k = int(min(k, max(1, Xs.shape[1] // 4)))
    if k < 2:
        return [0.0]
    bounds = librosa.segment.agglomerative(Xs, k)
    times  = [float(hop_times[min(n - 1, int(blocks[min(int(b), blocks.size - 1)]))]) for b in bounds]
    if beat_times is not None and len(beat_times) and beat_period > 0:
        bts = np.asarray(beat_times, dtype=float)
        snapped = []
        for t in times:
            j  = int(np.argmin(np.abs(bts - t)))
            nb = float(bts[j])
            snapped.append(nb if abs(nb - t) <= 0.5 * beat_period else t)
        times = snapped
    kept = [0.0]
    for t in sorted(set(times)):
        if t - kept[-1] >= min_section_s and (duration - t) >= 0.5 * min_section_s:
            kept.append(float(t))
    return kept

def _snap_to_segment_starts(times: List[float], segs: List[Tuple[float, float]]) -> List[float]:
    if not segs:
        return [0.0]
    starts = np.array([s for s, _ in segs], dtype=float)
    out = [0.0]
    for t in times[1:]:
        j  = int(np.argmin(np.abs(starts - t)))
        st = float(starts[j])
        if st > out[-1] + 1e-6:
            out.append(st)
    return out

def _merge_quiet_segments(
    segs: List[Tuple[float, float]],
    rms_s: np.ndarray,
    hop_times: np.ndarray,
    hold_len: float,
    section_times: List[float]
) -> List[Tuple[float, float]]:
    if len(segs) < 3 or hold_len <= 0:
        return segs
    energy = []
    for s, e in segs:
        i0 = int(np.searchsorted(hop_times, s, side="left"))
        i1 = int(np.searchsorted(hop_times, e, side="left"))
        energy.append(float(np.mean(rms_s[i0:i1])) if i1 > i0 else 0.0)
    energy = np.array(energy, dtype=float)
    thr    = float(np.quantile(energy, QUIET_QUANTILE))
    starts = set(round(t, 6) for t in section_times)
    out: List[Tuple[float, float]] = []
    i = 0
    while i < len(segs):
        s, e = segs[i]
        if energy[i] <= thr:
            j = i + 1
            while (j < len(segs) and energy[j] <= thr
                   and (segs[j][1] - s) <= hold_len + 1e-6
                   and round(segs[j][0], 6) not in starts):
                e = segs[j][1]
                j += 1
            out.append((s, e))
            i = j
        else:
            out.append((s, e))
            i += 1
    return out

def _section_index_for_segments(segs: List[Tuple[float, float]], section_times: List[float]) -> List[int]:
    st = np.array(section_times, dtype=float)
    return [int(max(0, np.searchsorted(st, s + 1e-6, side="right") - 1)) for s, _ in segs]

def analyze_audio_segments(
    audio_path: str,
    min_seg: float = 0.35,
    hop: int = 256,
    n_fft: int = 1024,
    percussive_boost: float = 1.0,
    segment_mode: str = "auto",
    onset_delta: float = 0.08,
    onset_pre_max: int = 12,
    onset_post_max: int = 12,
    onset_pre_avg: int = 50,
    onset_post_avg: int = 50,
    onset_backtrack: bool = True,
    bpm: float = 0.0,
    subdiv: int = 1,
    compute_f0: bool = False,
    start_bpm: float = 120.0,
    tempo_min: float = 70.0,
    tempo_max: float = 180.0,
    beat_tracker: str = "dynamic",
    snap: bool = True,
    snap_tol_beats: float = 0.5,
    beats_per_bar: int = 4,
    sections: int = 0,
    quiet_hold_beats: float = 0.0,
    beat_refine_s: float = 0.06
) -> Dict:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    def lufs_short(y_arr, rate):
        try:
            import pyloudnorm as pyln
        except Exception:
            return None
        meter = pyln.Meter(rate)
        try:
            return float(meter.integrated_loudness(y_arr.astype(np.float64)))
        except Exception:
            return None

    track_lufs = lufs_short(y, sr)

    y_h, y_p = librosa.effects.hpss(y, margin=(1.0, 2.0))

    if percussive_boost is not None and percussive_boost > 0:
        y = librosa.util.normalize(y_h + float(percussive_boost) * y_p)
    else:
        y = librosa.util.normalize(y)

    onset_kwargs = dict(
        delta=onset_delta,
        pre_max=onset_pre_max,
        post_max=onset_post_max,
        pre_avg=onset_pre_avg,
        post_avg=onset_post_avg,
        backtrack=onset_backtrack
    )

    env_perc = superflux_onset_env(y_p, sr, hop, n_fft)
    tempo, beat_frames, beat_times = track_beats(
        env_perc, sr, hop, start_bpm=start_bpm,
        tempo_min=tempo_min, tempo_max=tempo_max, tracker=beat_tracker
    )
    beat_times     = _complete_beat_grid(np.asarray(beat_times, dtype=float), tempo, len(y) / sr)
    beat_frames    = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop)
    beat_strengths = _beat_strengths(env_perc, beat_frames)

    segments, duration = choose_segments(
        y=y,
        sr=sr,
        hop=hop,
        n_fft=n_fft,
        min_seg=min_seg,
        mode=segment_mode,
        onset_kwargs=onset_kwargs,
        y_perc=y_p,
        beat_times=beat_times,
        beat_frames=beat_frames,
        env_perc=env_perc,
        tempo=tempo,
        snap=snap,
        snap_tol_beats=snap_tol_beats,
        beats_per_bar=beats_per_bar,
        bpm=bpm,
        subdiv=subdiv,
        beat_refine_s=beat_refine_s
    )

    S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True, window="hann")
    S_mag = np.abs(S_complex)
    Pxx   = S_mag ** 2

    mel    = librosa.feature.melspectrogram(S=Pxx, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    rms  = librosa.feature.rms(S=S_mag, frame_length=n_fft, hop_length=hop, center=True)[0]
    cen  = librosa.feature.spectral_centroid(S=S_mag, sr=sr, hop_length=hop, center=True)[0]
    zcr  = librosa.feature.zero_crossing_rate(y=y, hop_length=hop, center=True)[0]

    onset_env = librosa.onset.onset_strength(S=mel, sr=sr, hop_length=hop, center=True)

    rolloff85 = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, roll_percent=0.85, hop_length=hop)[0]
    rolloff95 = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, roll_percent=0.95, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S_mag, sr=sr, hop_length=hop, center=True)[0]
    flatness  = librosa.feature.spectral_flatness(S=S_mag, hop_length=hop, center=True)[0]

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    hfc   = np.sum((S_mag.T * freqs).T, axis=0) / (np.sum(S_mag, axis=0) + 1e-9)

    mfcc   = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=13)
    dmfcc  = librosa.feature.delta(mfcc)
    ddmfcc = librosa.feature.delta(mfcc, order=2)

    H   = librosa.feature.rms(y=y_h, hop_length=hop, center=True)[0]
    P   = librosa.feature.rms(y=y_p, hop_length=hop, center=True)[0]
    hpr = (H + 1e-9) / (P + 1e-9)

    chroma     = librosa.feature.chroma_stft(S=S_mag, sr=sr, hop_length=hop, n_fft=n_fft)
    chroma_dom = np.max(chroma, axis=0) / (np.sum(chroma, axis=0) + 1e-9)

    tonnetz = librosa.feature.tonnetz(chroma=chroma)

    beat_idx  = np.clip(beat_frames, 0, len(onset_env) - 1) if len(beat_frames) else beat_frames
    beat_conf = float(np.mean(onset_env[beat_idx])) if len(beat_frames) else 0.0

    onset_frames_all = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="frames")
    onset_times_all  = librosa.frames_to_time(onset_frames_all, sr=sr, hop_length=hop)

    if compute_f0:
        try:
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                             sr=sr, frame_length=n_fft, hop_length=hop)
            voiced_mask = np.isfinite(f0) & (f0 > 0)
        except Exception:
            f0 = None
            voiced_mask = None
    else:
        f0 = None
        voiced_mask = None

    mel_freqs = librosa.mel_frequencies(n_mels=mel.shape[0], fmin=0, fmax=sr/2)
    sub_mask  = mel_freqs < 80
    low_mask  = (mel_freqs >= 80)  & (mel_freqs < 300)
    mid_mask  = (mel_freqs >= 300) & (mel_freqs < 2500)
    high_mask = mel_freqs >= 2500

    mel_norm = mel / (np.maximum(np.sum(mel, axis=0, keepdims=True), 1e-9))
    flux     = np.maximum(0.0, np.diff(mel_norm, axis=1))
    novelty  = np.sum(flux, axis=0)
    novelty  = np.pad(novelty, (1, 0), mode="edge")

    def smooth(a):
        return uniform_filter1d(a, size=3)

    rms_s, cen_s, zcr_s, on_s = map(smooth, (rms, cen, zcr, onset_env))
    roll85_s     = smooth(rolloff85)
    roll95_s     = smooth(rolloff95)
    bw_s         = smooth(bandwidth)
    flat_s       = smooth(flatness)
    hfc_s        = smooth(hfc)
    hpr_s        = smooth(hpr)
    chroma_dom_s = smooth(chroma_dom)
    novelty_s    = smooth(novelty)

    n = min(*[len(a) for a in (rms_s, cen_s, zcr_s, on_s, roll85_s, roll95_s, bw_s, flat_s, hfc_s, hpr_s, chroma_dom_s, novelty_s)])
    rms_s, cen_s, zcr_s, on_s = (a[:n] for a in (rms_s, cen_s, zcr_s, on_s))
    roll85_s, roll95_s, bw_s, flat_s, hfc_s, hpr_s, chroma_dom_s, novelty_s = (
        a[:n] for a in (roll85_s, roll95_s, bw_s, flat_s, hfc_s, hpr_s, chroma_dom_s, novelty_s)
    )

    mfcc = mfcc[:, :n]; dmfcc = dmfcc[:, :n]; ddmfcc = ddmfcc[:, :n]
    if tonnetz.shape[1] > n:
        tonnetz = tonnetz[:, :n]

    hop_times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)

    beat_times_arr = np.asarray(beat_times, dtype=float) if len(beat_frames) else np.array([])
    beat_period    = (60.0 / float(tempo)) if tempo and tempo > 0 else 0.0
    min_section_s  = max(6.0, 2.0 * beats_per_bar * beat_period)
    section_times  = detect_sections(mfcc, chroma[:, :n], hop_times, duration, sections, min_section_s,
                                     beat_times_arr, beat_period)
    section_times  = _snap_to_segment_starts(section_times, segments)
    if quiet_hold_beats and quiet_hold_beats > 0 and beat_period > 0:
        segments = _merge_quiet_segments(segments, rms_s, hop_times, float(quiet_hold_beats) * beat_period, section_times)
    section_idx = _section_index_for_segments(segments, section_times)

    chroma_mean_track           = np.mean(chroma[:, :n], axis=1)
    global_mode, global_mode_conf = key_mode_from_chroma(chroma_mean_track)

    out = []
    for k, (s, e) in enumerate(segments):
        i0   = int(np.searchsorted(hop_times, s, side='left'))
        i1   = int(np.searchsorted(hop_times, e, side='left'))
        sl   = slice(i0, i1)
        dur  = max(1e-9, float(e - s))
        n_sl = i1 - i0

        def seg_mean(a): return float(np.mean(a[sl])) if n_sl > 0 else 0.0
        def seg_std(a):  return float(np.std(a[sl]))  if n_sl > 0 else 0.0

        rms_mean    = seg_mean(rms_s)
        rms_slope   = float(np.polyfit(np.arange(n_sl, dtype=float), rms_s[sl], 1)[0] * n_sl / (rms_mean + 1e-9)) if n_sl >= 2 else 0.0
        cen_mean    = seg_mean(cen_s)
        zcr_mean    = seg_mean(zcr_s)
        on_mean     = seg_mean(on_s);     on_std     = seg_std(on_s)
        roll85_mean = seg_mean(roll85_s); roll85_std = seg_std(roll85_s)
        roll95_mean = seg_mean(roll95_s); roll95_std = seg_std(roll95_s)
        bw_mean     = seg_mean(bw_s);     bw_std     = seg_std(bw_s)
        flat_mean   = seg_mean(flat_s);   flat_std   = seg_std(flat_s)
        hfc_mean    = seg_mean(hfc_s);    hfc_std    = seg_std(hfc_s)
        hpr_mean    = seg_mean(hpr_s)
        chroma_dom_mean = seg_mean(chroma_dom_s)
        novelty_mean    = seg_mean(novelty_s); nov_std = seg_std(novelty_s)

        mfcc_seg   = mfcc[:, sl]
        dmfcc_seg  = dmfcc[:, sl]
        ddmfcc_seg = ddmfcc[:, sl]
        if n_sl > 0:
            mfcc_means   = mfcc_seg.mean(axis=1).tolist()
            mfcc_stds    = mfcc_seg.std(axis=1).tolist()
            dmfcc_means  = dmfcc_seg.mean(axis=1).tolist()
            dmfcc_stds   = dmfcc_seg.std(axis=1).tolist()
            ddmfcc_means = ddmfcc_seg.mean(axis=1).tolist()
            ddmfcc_stds  = ddmfcc_seg.std(axis=1).tolist()
        else:
            zeros13 = [0.0] * 13
            mfcc_means = list(zeros13); mfcc_stds    = list(zeros13)
            dmfcc_means= list(zeros13); dmfcc_stds   = list(zeros13)
            ddmfcc_means=list(zeros13); ddmfcc_stds  = list(zeros13)

        mel_seg  = mel[:, sl]
        mel_tot  = float(np.sum(mel_seg) + 1e-9)
        sub_ratio  = float(np.sum(mel_seg[sub_mask,  :]) / mel_tot) if mel_seg.size else 0.0
        low_ratio  = float(np.sum(mel_seg[low_mask,  :]) / mel_tot) if mel_seg.size else 0.0
        mid_ratio  = float(np.sum(mel_seg[mid_mask,  :]) / mel_tot) if mel_seg.size else 0.0
        high_ratio = float(np.sum(mel_seg[high_mask, :]) / mel_tot) if mel_seg.size else 0.0

        on_in      = onset_times_all[(onset_times_all >= s) & (onset_times_all < e)]
        onset_rate = float(len(on_in) / dur)
        ioi_std    = float(np.std(np.diff(on_in))) if len(on_in) >= 2 else 0.0

        if beat_times_arr.size:
            local_beats = (beat_times_arr >= s) & (beat_times_arr < e)
            local_conf  = float(np.mean(onset_env[beat_idx][local_beats])) if np.any(local_beats) else beat_conf
        else:
            local_conf = 0.0

        if f0 is not None:
            f0_seg     = f0[sl]
            vm         = voiced_mask[sl] if voiced_mask is not None else None
            voiced_pct = float(np.mean(vm)) if vm is not None and vm.size else 0.0
            f0_med     = float(np.median(f0_seg[np.isfinite(f0_seg)])) if f0_seg.size and np.any(np.isfinite(f0_seg)) else 0.0
        else:
            voiced_pct, f0_med = 0.0, 0.0

        chroma_seg = chroma[:, sl]
        if chroma_seg.size and np.sum(chroma_seg) > 1e-6:
            mode_lbl, mode_conf = key_mode_from_chroma(np.mean(chroma_seg, axis=1))
        else:
            mode_lbl, mode_conf = global_mode, global_mode_conf

        if tonnetz.shape[1] == len(hop_times):
            tn = tonnetz[:, sl]
            tonnetz_std = float(np.mean(np.std(tn, axis=1))) if tn.size else 0.0
        else:
            tonnetz_std = 0.0

        segment_samples = y[int(s*sr):int(e*sr)]
        peak  = float(np.max(np.abs(segment_samples))) if segment_samples.size else 0.0
        crest = float(peak / (rms_mean + 1e-9))

        out.append({
            "start": float(s),
            "end": float(e),
            "duration": float(dur),
            "section": int(section_idx[k]),
            "features": {
                "rms": rms_mean,
                "rms_slope": rms_slope,
                "centroid": cen_mean,
                "zcr": zcr_mean,
                "onset_avg": on_mean,
                "track_lufs": track_lufs,
                "crest_factor": crest,
                "tempo_bpm": float(tempo),
                "beat_conf": float(local_conf),
                "onset_rate": onset_rate,
                "ioi_std": ioi_std,
                "hfc_mean": hfc_mean,
                "hfc_std": hfc_std,
                "rolloff85_mean": roll85_mean,
                "rolloff85_std": roll85_std,
                "rolloff95_mean": roll95_mean,
                "rolloff95_std": roll95_std,
                "bandwidth_mean": bw_mean,
                "bandwidth_std": bw_std,
                "flatness_mean": flat_mean,
                "flatness_std": flat_std,
                "hpr": hpr_mean,
                "mfcc_means": mfcc_means,
                "mfcc_stds": mfcc_stds,
                "dmfcc_means": dmfcc_means,
                "dmfcc_stds": dmfcc_stds,
                "ddmfcc_means": ddmfcc_means,
                "ddmfcc_stds": ddmfcc_stds,
                "chroma_dominance": chroma_dom_mean,
                "mode": mode_lbl,
                "mode_conf": float(mode_conf),
                "tonnetz_std": tonnetz_std,
                "novelty": novelty_mean,
                "sub_ratio": sub_ratio,
                "low_ratio": low_ratio,
                "mid_ratio": mid_ratio,
                "high_ratio": high_ratio,
                "voiced_pct": voiced_pct,
                "f0_median_hz": f0_med,
                "onset_std": on_std,
                "novelty_std": nov_std
            }
        })

    curves = {
        "hop_times": hop_times.astype(float),
        "onset_env": on_s.astype(float),
        "novelty":   novelty_s.astype(float),
        "rms":       rms_s.astype(float),
    }

    sections_out = [
        {"index": i, "start": float(t),
         "end": float(section_times[i + 1] if i + 1 < len(section_times) else duration)}
        for i, t in enumerate(section_times)
    ]

    return {"sr": sr, "duration": duration, "tempo": float(tempo), "beat_times": beat_times_arr,
            "beat_strengths": np.asarray(beat_strengths, dtype=float), "segments": out, "curves": curves,
            "sections": sections_out}

def _prep(frame: np.ndarray, target_w: int = 320) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > target_w:
        scale = target_w / float(w)
        frame = cv2.resize(frame, (target_w, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return frame

_RADIAL_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

def _radial_basis(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    basis = _RADIAL_CACHE.get(shape)
    if basis is None:
        h, w = shape
        ys, xs = np.mgrid[0:h, 0:w]
        rx = xs.astype(np.float32) - (w - 1) / 2.0
        ry = ys.astype(np.float32) - (h - 1) / 2.0
        r  = np.sqrt(rx * rx + ry * ry) + 1e-6
        basis = (rx / r, ry / r)
        _RADIAL_CACHE[shape] = basis
    return basis

def _flow_stats(prev_gray: np.ndarray, gray: np.ndarray) -> Tuple[float, float]:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=2, winsize=15,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
    )
    mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    ux, uy = _radial_basis(gray.shape)
    div = float(np.mean(ux * flow[..., 0] + uy * flow[..., 1]))
    return float(np.mean(mag)), div

def _diff_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(gray, prev_gray)))

def _appearance(small_bgr: np.ndarray, gray: np.ndarray) -> Tuple[float, float, float, float]:
    luma     = float(np.mean(gray)) / 255.0
    contrast = float(np.std(gray)) / 255.0
    sat      = float(np.mean(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)[..., 1])) / 255.0
    edges    = cv2.Canny(gray, 100, 200)
    return luma, contrast, sat, float(np.count_nonzero(edges)) / float(edges.size)

TIMELINE_KEYS = ("scores", "diff", "luma", "contrast", "sat", "edges", "div")

def _empty_timeline() -> Dict:
    tl = {k: np.array([], dtype=float) for k in TIMELINE_KEYS}
    tl["times"] = np.array([], dtype=float)
    tl["dur"]   = 0.0
    return tl

def motion_timeline(
    path: str,
    step_s: float = 0.25,
    max_samples: int = 1200,
    method: str = "diff",
    target_w: int = 320
) -> Dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return _empty_timeline()

    fps    = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    dur    = ffprobe_duration(path)
    if dur <= 0 and fps > 0 and frames > 0:
        dur = frames / fps
    if dur <= 0:
        cap.release()
        return _empty_timeline()
    if fps <= 0:
        fps = (frames / dur) if dur > 0 else 30.0

    step    = max(0.03, float(step_s))
    n_steps = max(1, min(int(math.ceil(dur / step)), int(max_samples)))
    times   = np.linspace(0, dur, num=n_steps, endpoint=False).astype(float)
    cols    = {k: np.zeros(n_steps, dtype=float) for k in TIMELINE_KEYS}
    target_frames = np.round(times * fps).astype(int)

    prev_gray = None
    sample_i  = 0
    idx       = -1
    while sample_i < target_frames.size:
        if not cap.grab():
            break
        idx += 1
        if idx < target_frames[sample_i]:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        small = _prep(frame, target_w)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        luma, contrast, sat, edges = _appearance(small, gray)
        motion = diff = div = 0.0
        if prev_gray is not None:
            diff = _diff_score(prev_gray, gray)
            if method == "flow":
                motion, div = _flow_stats(prev_gray, gray)
            else:
                motion = diff
        while sample_i < target_frames.size and target_frames[sample_i] <= idx:
            cols["scores"][sample_i]   = motion
            cols["diff"][sample_i]     = diff
            cols["luma"][sample_i]     = luma
            cols["contrast"][sample_i] = contrast
            cols["sat"][sample_i]      = sat
            cols["edges"][sample_i]    = edges
            cols["div"][sample_i]      = div
            sample_i += 1
        prev_gray = gray

    cap.release()
    if n_steps > 3:
        cols["scores"] = uniform_filter1d(cols["scores"], size=3)
    tl = dict(cols)
    tl["times"] = times
    tl["dur"]   = float(dur)
    return tl

def _timeline_worker(args: Tuple) -> Tuple[str, Dict]:
    path, step_s, max_samples, method = args
    return path, motion_timeline(path, step_s=step_s, max_samples=max_samples, method=method)

def score_all_videos(
    videos_dir: str,
    motion_method: str = "diff",
    vt_step: float = 0.25,
    vt_max_samples: int = 1200,
    workers: Optional[int] = None
) -> List[Dict]:
    exts  = (".mp4", ".mov", ".mkv", ".webm")
    paths = sorted(str(p) for p in Path(videos_dir).glob("*") if p.suffix and p.suffix.lower() in exts)
    if not paths:
        raise RuntimeError("No video files found in videos_dir.")

    videos: List[Dict] = []
    tasks = [(p, vt_step, vt_max_samples, motion_method) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path, tl in tqdm(ex.map(_timeline_worker, tasks), total=len(tasks),
                             desc=f"Analyzing video timelines ({motion_method})"):
            n    = int(tl["times"].size)
            step = float(tl["times"][1] - tl["times"][0]) if n > 1 else float(vt_step)
            ms   = float(np.mean(tl["scores"])) if n else 0.0
            videos.append({"path": path, "dur": float(tl["dur"]), "step": step,
                           "timeline": tl, "motion_score": ms})
    return videos

FEATURE_DIMS = ("pace", "burst", "groove", "brightness", "noise", "bass", "density", "warmth", "slope")
MOTION_DIMS  = ("pace", "burst", "groove")
BASE_WEIGHTS = {"pace": 1.0, "burst": 1.0, "groove": 0.25, "brightness": 0.6, "noise": 0.4,
                "bass": 0.4, "density": 0.3, "warmth": 0.3, "slope": 0.3}
_DIM = {name: i for i, name in enumerate(FEATURE_DIMS)}

POOL_PENALTY   = 0.20
GAP_PENALTY    = 0.80
PALETTE_WEIGHT = 0.20

def _rolling_mean(x: np.ndarray, size: int) -> np.ndarray:
    return uniform_filter1d(np.asarray(x, dtype=float), size=max(1, int(size)), mode="nearest")

def _rolling_std(x: np.ndarray, size: int) -> np.ndarray:
    x  = np.asarray(x, dtype=float)
    m  = _rolling_mean(x, size)
    m2 = _rolling_mean(x * x, size)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))

def _context_groove(x: np.ndarray, ctx: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        return np.zeros(n, dtype=float)
    ctx  = int(max(8, min(ctx, n)))
    W    = np.lib.stride_tricks.sliding_window_view(x, ctx)
    Wc   = W - W.mean(axis=1, keepdims=True)
    mags = np.abs(np.fft.rfft(Wc, axis=1))[:, 1:]
    g    = mags.max(axis=1) / (mags.sum(axis=1) + 1e-9)
    pad_l = (n - g.size) // 2
    return np.pad(g, (pad_l, n - g.size - pad_l), mode="edge")

def _cut_flags(diff: np.ndarray) -> np.ndarray:
    diff = np.asarray(diff, dtype=float)
    if diff.size < 3:
        return np.zeros_like(diff)
    d   = diff[1:]
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) * 1.4826
    thr = max(med + 4.0 * mad, 10.0)
    return (diff > thr).astype(float)

def _raw_video_features(v: Dict) -> Optional[Dict[str, np.ndarray]]:
    tl = v["timeline"]
    if tl["times"].size == 0:
        return None
    step   = float(v["step"])
    motion = np.asarray(tl["scores"], dtype=float)
    cuts   = _cut_flags(tl["diff"])
    return {
        "pace":       motion,
        "burst":      _rolling_std(motion, round(0.5 / step)),
        "groove":     _context_groove(motion, round(4.0 / step)),
        "brightness": np.asarray(tl["luma"], dtype=float),
        "contrast":   np.asarray(tl["contrast"], dtype=float),
        "noise":      np.asarray(tl["edges"], dtype=float),
        "density":    _rolling_mean(cuts, round(2.0 / step)) / step,
        "warmth":     np.asarray(tl["sat"], dtype=float),
        "slope":      np.asarray(tl["div"], dtype=float),
    }

def _rank01(x: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if sorted_ref.size == 0:
        return np.full(x.shape, 0.5, dtype=float)
    lo = np.searchsorted(sorted_ref, x, side="left")
    hi = np.searchsorted(sorted_ref, x, side="right")
    return (lo + hi) / (2.0 * sorted_ref.size)

def build_library_features(videos: List[Dict]) -> Dict:
    raws = [_raw_video_features(v) for v in videos]
    live = [r for r in raws if r is not None]
    for r in live:
        r["bass"] = np.zeros_like(r["brightness"])
    refs: Dict[str, np.ndarray] = {}
    for k in [k for k in FEATURE_DIMS if k != "bass"] + ["contrast"]:
        refs[k] = np.sort(np.concatenate([r[k] for r in live])) if live else np.array([], dtype=float)
    for r in live:
        r["bass"] = 0.5 * (1.0 - _rank01(r["brightness"], refs["brightness"])) + 0.5 * (1.0 - _rank01(r["contrast"], refs["contrast"]))
    refs["bass"] = np.sort(np.concatenate([r["bass"] for r in live])) if live else np.array([], dtype=float)

    def spread(k: str) -> bool:
        ref = refs[k]
        return ref.size > 0 and (float(ref[-1]) - float(ref[0])) > 1e-9

    active = [spread(k) for k in FEATURE_DIMS]
    bounds = {k: ((float(refs[k][0]), float(refs[k][-1])) if refs[k].size else (0.0, 0.0)) for k in refs}

    for v, r in zip(videos, raws):
        if r is None:
            v["feat"]    = None
            v["profile"] = None
            continue
        feat = np.stack([_rank01(r[k], refs[k]) for k in FEATURE_DIMS], axis=1)
        v["feat"]    = feat
        v["profile"] = feat.mean(axis=0)
    return {"videos": videos, "active": np.array(active, dtype=bool), "bounds": bounds}

def classify_audio_segment(feat: Dict, rms_q: Tuple[float, float], onset_q: Tuple[float, float]) -> str:
    energy, trans = feat["rms"], feat["onset_avg"]
    if energy >= rms_q[1] or trans >= onset_q[1]:
        return "fast"
    if energy <= rms_q[0] and trans <= onset_q[0]:
        return "slow"
    return "medium"

def _minmax01(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 1e-9:
        return np.zeros_like(x, dtype=float), lo, hi
    return (x - lo) / (hi - lo), lo, hi

def build_segment_profiles(segments: List[Dict]) -> List[Dict]:
    arr = lambda k: np.array([s["features"].get(k, 0.0) for s in segments], dtype=float)

    rms       = arr("rms")
    onset_rt  = arr("onset_rate")
    beat_cf   = arr("beat_conf")
    roll85    = arr("rolloff85_mean")
    flatn     = arr("flatness_mean")
    subr      = arr("sub_ratio")
    nov       = arr("novelty")
    onset_std = arr("onset_std")
    nov_std   = arr("novelty_std")
    slope     = arr("rms_slope")
    warm_raw  = np.array([
        s["features"].get("mode_conf", 0.5) if s["features"].get("mode", "major") == "major"
        else 1.0 - s["features"].get("mode_conf", 0.5)
        for s in segments
    ], dtype=float)

    rms01,      _, _ = _minmax01(rms)
    onset01,    _, _ = _minmax01(onset_rt)
    beat01,     _, _ = _minmax01(beat_cf)
    bright01,   _, _ = _minmax01(roll85)
    noise01,    _, _ = _minmax01(flatn)
    bass01,     _, _ = _minmax01(subr)
    nov01,      _, _ = _minmax01(nov)
    onsetstd01, _, _ = _minmax01(onset_std)
    novstd01,   _, _ = _minmax01(nov_std)
    slope01,    _, _ = _minmax01(slope)
    warm01,     _, _ = _minmax01(warm_raw)

    profiles = []
    for i in range(len(segments)):
        pace  = float(0.45*rms01[i] + 0.35*onset01[i] + 0.20*beat01[i])
        burst = float(0.5*onsetstd01[i] + 0.5*novstd01[i])
        profiles.append({
            "pace":       pace,
            "burst":      burst,
            "groove":     float(beat01[i]),
            "brightness": float(bright01[i]),
            "noise":      float(noise01[i]),
            "bass":       float(bass01[i]),
            "novelty":    float(nov01[i]),
            "density":    float(onset01[i]),
            "warmth":     float(warm01[i]),
            "slope":      float(slope01[i])
        })
    for k in FEATURE_DIMS:
        vals = np.array([p[k] for p in profiles], dtype=float)
        ranks = _rank01(vals, np.sort(vals))
        for p, x in zip(profiles, ranks):
            p[k] = float(x)
    return profiles

def profile_vector(p: Dict) -> np.ndarray:
    return np.array([float(p.get(k, 0.5)) for k in FEATURE_DIMS], dtype=float)

def feature_weights(visual_weight: float, active: np.ndarray, pairing: str) -> np.ndarray:
    vw = max(0.0, float(visual_weight))
    w  = np.array([BASE_WEIGHTS[k] * (1.0 if k in MOTION_DIMS else vw) for k in FEATURE_DIMS], dtype=float)
    w *= np.asarray(active, dtype=float)
    if pairing != "smart":
        w[:] = 0.0
    return w

def _interp_curve(times_src: np.ndarray, values_src: np.ndarray, times_q: np.ndarray) -> np.ndarray:
    if times_src.size == 0 or values_src.size == 0 or times_q.size == 0:
        return np.zeros_like(times_q, dtype=float)
    return np.interp(times_q, times_src.astype(float), values_src.astype(float),
                     left=float(values_src[0]), right=float(values_src[-1]))

def build_audio_curve_for_segments(
    curves: Dict,
    segments: List[Dict],
    vt_step: float,
    curve_method: str = "novelty"
) -> List[np.ndarray]:
    hop_times = curves["hop_times"]
    if curve_method == "onset":
        base = curves["onset_env"]
    elif curve_method == "rms":
        base = curves["rms"]
    else:
        base = curves["novelty"]

    base01, _, _ = _minmax01(base.astype(float))
    base01 = uniform_filter1d(base01, size=3)

    seg_curves: List[np.ndarray] = []
    for s in segments:
        start = float(s["start"])
        end   = float(s["end"])
        dur   = max(1e-9, end - start)

        n = max(2, int(math.ceil(dur / max(1e-6, vt_step))))
        t = start + np.arange(n, dtype=float) * float(vt_step)
        t = np.clip(t, start, max(start, end - 1e-6))

        seg_curve = _interp_curve(hop_times, base01, t)
        seg_curves.append(seg_curve.astype(float))
    return seg_curves

def score_windows(
    v: Dict,
    win: int,
    seg_vec: np.ndarray,
    weights: np.ndarray,
    seg_curve: Optional[np.ndarray],
    curve_weight: float,
    align_weight: float,
    means_cache: Dict,
    impact_weight: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, int]:
    feat = v["feat"]
    n    = feat.shape[0]
    win  = int(max(1, min(win, n)))
    key  = (v["path"], win)
    means = means_cache.get(key)
    if means is None:
        cs    = np.vstack([np.zeros((1, feat.shape[1])), np.cumsum(feat, axis=0)])
        means = (cs[win:] - cs[:-win]) / float(win)
        means_cache[key] = means
    n_w  = means.shape[0]
    wsum = float(np.sum(weights))
    if wsum > 0:
        d_feat = np.sum(((means - seg_vec) ** 2) * weights, axis=1) / wsum
    else:
        d_feat = np.zeros(n_w, dtype=float)

    d_curve = np.ones(n_w, dtype=float)
    d_align = np.zeros(n_w, dtype=float)
    if seg_curve is not None and seg_curve.size >= 2 and win > 1:
        motion = np.asarray(v["timeline"]["scores"], dtype=float)[:n]
        W = np.lib.stride_tricks.sliding_window_view(motion, win)
        c = _interp_curve(np.linspace(0.0, 1.0, seg_curve.size), seg_curve, np.linspace(0.0, 1.0, win))
        if win >= 3:
            cz   = c - c.mean()
            cn   = float(np.linalg.norm(cz))
            Wc   = W - W.mean(axis=1, keepdims=True)
            Wn   = np.linalg.norm(Wc, axis=1)
            corr = (Wc @ cz) / (Wn * cn + 1e-9)
            d_curve = 1.0 - np.clip(corr, -1.0, 1.0)
        if align_weight > 0:
            peak    = float(np.argmax(c)) / float(win - 1)
            d_align = (peak - W.argmax(axis=1) / float(win - 1)) ** 2

    d_impact = np.zeros(n_w, dtype=float)
    if impact_weight > 0 and win > 1:
        pace  = feat[:, _DIM["pace"]]
        k     = int(max(1, min(win - 1, round(0.1 / max(1e-6, float(v["step"]))))))
        cs_p  = np.concatenate([[0.0], np.cumsum(pace)])
        front = (cs_p[1 + k : 1 + k + n_w] - cs_p[1 : 1 + n_w]) / float(k)
        d_impact = 0.5 * (1.0 - (front - means[:, _DIM["pace"]]))
    eff_curve = curve_weight * min(1.0, max(0.0, (win - 3) / 13.0))
    cost = (1.0 - eff_curve) * d_feat + eff_curve * d_curve + align_weight * d_align + impact_weight * d_impact
    return cost, means, win

def section_pools(
    videos: List[Dict],
    seg_vecs: List[np.ndarray],
    sections: List[int],
    weights: np.ndarray,
    pool_frac: float
) -> Dict[int, set]:
    n_v  = len(videos)
    size = n_v if pool_frac >= 1.0 else min(n_v, max(3, int(math.ceil(pool_frac * n_v))))
    wsum = float(np.sum(weights))
    pools: Dict[int, set] = {}
    for sec in sorted(set(sections)):
        members = [seg_vecs[i] for i, s in enumerate(sections) if s == sec]
        sec_vec = np.mean(members, axis=0)
        dists   = np.array([np.sum(((v["profile"] - sec_vec) ** 2) * weights) / wsum for v in videos], dtype=float)
        order   = np.argsort(dists, kind="stable")
        pools[sec] = set(int(i) for i in order[:size])
    return pools

Cand = namedtuple("Cand", "vi j win cost bright warm pace vec")

class _Hyp:
    __slots__ = ("cost", "node", "use", "last", "used", "lastc", "sec")

    def __init__(self, cost, node, use, last, used, lastc, sec):
        self.cost  = cost
        self.node  = node
        self.use   = use
        self.last  = last
        self.used  = used
        self.lastc = lastc
        self.sec   = sec

def choose_clips_for_segments(
    segments: List[Dict],
    library: Dict,
    audio_seg_curves: Optional[List[np.ndarray]] = None,
    clip_strategy: str = "timeline",
    no_reuse: float = 1.25,
    jitter: float = 0.15,
    diversity_cooldown: int = 3,
    pairing: str = "smart",
    curve_weight: float = 0.60,
    align_weight: float = 0.25,
    min_source_gap: int = 0,
    clip_topk: int = 4,
    reuse_penalty: float = 0.30,
    beam_width: int = 8,
    visual_weight: float = 1.0,
    pool_frac: float = 0.5,
    impact_weight: float = 0.25
) -> List[Dict]:
    videos = [v for v in library["videos"] if v.get("feat") is not None and v["feat"].shape[0] > 0]
    if not videos:
        raise RuntimeError("No analyzable videos available to choose from.")

    rng = random.Random(1337)
    n_v = len(videos)
    if min_source_gap > 0:
        gap_req = min(min_source_gap, n_v - 1)
    else:
        gap_req = max(diversity_cooldown, (n_v + 1) // 2)
    gap_req = max(0, min(gap_req, n_v - 1))

    curve_weight = float(max(0.0, min(1.0, curve_weight)))
    align_weight = float(max(0.0, align_weight))
    beam_width   = max(1, int(beam_width))
    clip_topk    = max(1, int(clip_topk))
    max_cands    = max(16, 4 * beam_width)

    weights = feature_weights(visual_weight, library["active"], pairing)
    if clip_strategy != "timeline":
        weights = np.zeros_like(weights)
    profiles = build_segment_profiles(segments)
    seg_vecs = [profile_vector(p) for p in profiles]
    sections = [int(s.get("section", 0)) for s in segments]
    pools    = None
    if float(np.sum(weights)) > 0 and pool_frac < 1.0:
        pools = section_pools(videos, seg_vecs, sections, weights, pool_frac)

    bi, wi, pi = _DIM["brightness"], _DIM["warmth"], _DIM["pace"]

    def make_cand(vi: int, j: int, win: int, cost_j: float, pool_pen: float, means: np.ndarray) -> Cand:
        return Cand(vi, j, win, float(cost_j) + pool_pen,
                    float(means[j, bi]), float(means[j, wi]), float(means[j, pi]), means[j].copy())

    means_cache: Dict = {}
    beam = [_Hyp(0.0, None, {}, {}, {}, None, None)]

    for t, seg in enumerate(segments):
        target  = float(seg["duration"])
        sec     = sections[t]
        seg_vec = seg_vecs[t]
        seg_curve = None
        if clip_strategy == "timeline" and audio_seg_curves is not None and t < len(audio_seg_curves):
            seg_curve = audio_seg_curves[t]

        primary: List[Cand]   = []
        secondary: List[Cand] = []
        for vi, v in enumerate(videos):
            win = max(1, int(round(target / v["step"])))
            cost, means, win = score_windows(v, win, seg_vec, weights, seg_curve, curve_weight, align_weight, means_cache,
                                             impact_weight if clip_strategy == "timeline" else 0.0)
            pool_pen = POOL_PENALTY if (pools is not None and vi not in pools[sec]) else 0.0
            if clip_strategy != "timeline":
                j = rng.randint(0, cost.size - 1)
                primary.append(make_cand(vi, j, win, cost[j], pool_pen, means))
                continue
            order = np.argsort(cost, kind="stable")
            taken: List[int] = []
            for j in order[: max(50, 25 * clip_topk)]:
                j = int(j)
                if any(abs(j - q) < win for q in taken):
                    continue
                taken.append(j)
                c = make_cand(vi, j, win, cost[j], pool_pen, means)
                (primary if len(taken) == 1 else secondary).append(c)
                if len(taken) >= clip_topk:
                    break
        secondary.sort(key=lambda c: (c.cost, c.vi, c.j))
        cands = primary + secondary[: max(0, max_cands - len(primary))]

        scored: List[Tuple[float, int, Cand]] = []
        for h_i, h in enumerate(beam):
            prev     = h.lastc
            boundary = prev is not None and sec != h.sec
            for c in cands:
                if prev is not None and c.vi == prev.vi and n_v > 1:
                    continue
                used = h.used.get(c.vi)
                if used:
                    nr = no_reuse / videos[c.vi]["step"]
                    if any(c.j < ue + nr and c.j + c.win > us - nr for us, ue in used):
                        continue
                total = h.cost + c.cost + reuse_penalty * h.use.get(c.vi, 0)
                li = h.last.get(c.vi)
                if li is not None and (t - li) < gap_req:
                    total += GAP_PENALTY * (gap_req - (t - li)) / float(max(1, gap_req))
                if prev is not None and not boundary:
                    total += PALETTE_WEIGHT * 0.5 * (abs(c.bright - prev.bright) + abs(c.warm - prev.warm))
                scored.append((total, h_i, c))
        if not scored:
            for h_i, h in enumerate(beam):
                prev = h.lastc
                pool = [c for c in cands if prev is None or c.vi != prev.vi] or cands
                c = min(pool, key=lambda c: (c.cost, c.vi, c.j))
                scored.append((h.cost + c.cost, h_i, c))
        scored.sort(key=lambda x: (x[0], x[1], x[2].vi, x[2].j))

        new_beam: List[_Hyp] = []
        for total, h_i, c in scored[:beam_width]:
            h    = beam[h_i]
            use  = dict(h.use)
            last = dict(h.last)
            used = {k: list(vv) for k, vv in h.used.items()}
            use[c.vi] = use.get(c.vi, 0) + 1
            last[c.vi] = t
            used.setdefault(c.vi, []).append((c.j, c.j + c.win))
            new_beam.append(_Hyp(total, (c, h.node), use, last, used, c, sec))
        beam = new_beam

    best  = min(beam, key=lambda h: h.cost)
    picks: List[Cand] = []
    node  = best.node
    while node is not None:
        c, node = node
        picks.append(c)
    picks.reverse()

    chosen: List[Dict] = []
    for t, c in enumerate(picks):
        v      = videos[c.vi]
        target = float(segments[t]["duration"])
        start  = c.j * v["step"]
        if jitter > 0:
            start += rng.uniform(-jitter, jitter)
        if v["dur"] > target:
            start = min(start, v["dur"] - target)
        start  = max(0.0, start)
        bucket = "slow" if c.pace < 1.0 / 3.0 else ("medium" if c.pace < 2.0 / 3.0 else "fast")
        chosen.append({
            "path":         v["path"],
            "motion_score": float(v["motion_score"]),
            "bucket":       bucket,
            "start":        float(start),
            "src_dur":      float(v["dur"]),
            "section":      sections[t],
            "match_cost":   float(c.cost),
            "segment_profile": {k: float(profiles[t].get(k, 0.5)) for k in FEATURE_DIMS},
            "window_profile":  {k: float(x) for k, x in zip(FEATURE_DIMS, c.vec)}
        })
    return chosen

def _ff_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")

def build_reactive_commands(
    curves: Dict,
    beat_times: np.ndarray,
    beat_strengths: np.ndarray,
    tempo: float,
    duration: float,
    fps: int,
    cut_lead_frames: int,
    pulse_strength: float,
    react_strength: float,
    path: str
) -> bool:
    pulse_strength = float(max(0.0, pulse_strength))
    react_strength = float(max(0.0, react_strength))
    if pulse_strength <= 0 and react_strength <= 0:
        return False
    n_frames = int(math.ceil(duration * fps)) + 1
    t_video  = np.arange(n_frames, dtype=float) / float(fps)
    t_audio  = t_video + max(0, int(cut_lead_frames)) / float(fps)
    sat      = np.ones(n_frames, dtype=float)
    bright   = np.zeros(n_frames, dtype=float)

    beats = np.asarray(beat_times, dtype=float)
    if pulse_strength > 0 and beats.size:
        if tempo and tempo > 0:
            period = 60.0 / float(tempo)
        elif beats.size >= 2:
            period = float(np.median(np.diff(beats)))
        else:
            period = 0.5
        tau       = max(1.0 / fps, period / 3.0)
        strengths = np.asarray(beat_strengths, dtype=float)
        if strengths.size != beats.size:
            strengths = np.ones(beats.size, dtype=float)
        idx   = np.searchsorted(beats, t_audio, side="right") - 1
        valid = idx >= 0
        dt    = t_audio[valid] - beats[idx[valid]]
        amp   = pulse_strength * (0.5 + 0.5 * strengths[idx[valid]])
        sat[valid] = 1.0 + amp * np.exp(-dt / tau)

    if react_strength > 0:
        hop_times = np.asarray(curves.get("hop_times", np.array([])), dtype=float)
        env       = np.asarray(curves.get("onset_env", np.array([])), dtype=float)
        if hop_times.size and env.size:
            hi      = float(np.percentile(env, 95)) + 1e-9
            sampled = _interp_curve(hop_times, np.clip(env / hi, 0.0, 1.0), t_audio)
            release = math.exp(-1.0 / (fps * 0.12))
            level   = 0.0
            for k in range(n_frames):
                level     = max(float(sampled[k]), level * release)
                bright[k] = react_strength * level

    with open(path, "w", encoding="utf-8") as f:
        for k in range(n_frames):
            t_cmd = max(0.0, (k - 0.5) / float(fps))
            f.write(f"{t_cmd:.6f} [enter] eq@react saturation {sat[k]:.4f}, eq@react brightness {bright[k]:.4f};\n")
    return True

def build_ffmpeg_graph(
    audio_path: str,
    segments: List[Dict],
    chosen: List[Dict],
    fps: int,
    width: int,
    height: int,
    crf: int,
    preset: str,
    out_path: str,
    zoom_depths: Optional[List[float]] = None,
    zoom_in: Optional[List[bool]] = None,
    react_cmds: Optional[str] = None,
    trail_frames: int = 6,
    trail_decay: float = 0.70,
    trail_opacity: float = 0.35,
    trail_mode: str = "screen",
    cut_lead_frames: int = 0
):
    cut_lead_frames = int(max(0, cut_lead_frames))
    seg_frames: List[int] = []
    cum_time   = 0.0
    prev_frame = 0
    for seg in segments:
        cum_time  += max(0.0, float(seg["duration"]))
        cur_frame  = int(round(cum_time * fps)) - cut_lead_frames
        cur_frame  = max(prev_frame + 1, cur_frame)
        seg_frames.append(cur_frame - prev_frame)
        prev_frame = cur_frame

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", audio_path]
    for c, seg in zip(chosen, segments):
        start       = max(0.0, float(c.get("start", 0.0)))
        read_window = max(0.001, float(seg["duration"])) + 0.5
        cmd += ["-ss", f"{start:.6f}", "-t", f"{read_window:.6f}", "-i", c["path"]]

    rng  = random.Random(1337)
    pans = ["", "left", "right", "up", "down"]

    trail_frames   = int(max(0, trail_frames))
    trail_decay    = float(max(0.0, min(1.0, trail_decay)))
    trail_opacity  = float(max(0.0, min(1.0, trail_opacity)))
    enable_trails  = (trail_frames > 1 and trail_opacity > 0.0)

    def trail_weights(frames: int, decay: float) -> str:
        w = [decay**k for k in range(frames)]
        return " ".join(f"{x:.6f}" for x in w)

    filters, labels = [], []
    for i, (seg, ch) in enumerate(zip(segments, chosen), start=1):
        N        = max(1, int(seg_frames[i-1]))
        target   = N / float(fps)
        zoom_amt = float(zoom_depths[i-1]) if zoom_depths is not None and i-1 < len(zoom_depths) else 0.015
        inward   = bool(zoom_in[i-1]) if zoom_in is not None and i-1 < len(zoom_in) else True
        pan      = rng.choice(pans)
        prog     = f"min(1,on/{N})"

        if inward:
            z_expr = f"1+{zoom_amt:.4f}*{prog}"
        else:
            z_expr = f"1+{zoom_amt:.4f}*(1-{prog})"

        pan_frac_x = 0.0
        pan_frac_y = 0.0
        if pan == "left":  pan_frac_x = -0.02
        if pan == "right": pan_frac_x =  0.02
        if pan == "up":    pan_frac_y = -0.02
        if pan == "down":  pan_frac_y =  0.02

        x_expr = f"(iw-iw/zoom)/2+({pan_frac_x})*iw*{prog}"
        y_expr = f"(ih-ih/zoom)/2+({pan_frac_y})*ih*{prog}"

        chain  = f"[{i}:v]setpts=PTS-STARTPTS,fps={fps}"
        chain += f",tpad=stop_mode=clone:stop_duration={target + 0.5:.6f}"
        chain += f",trim=end_frame={N},setpts=PTS-STARTPTS"

        chain += f",scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos"
        chain += f",crop={width}:{height},setsar=1"
        chain += f",zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d=1:s={width}x{height}:fps={fps}"
        chain += f",eq=saturation=1.04:gamma=1.02"

        midlab  = f"seg{i}"
        chain  += f"[{midlab}]"
        filters.append(chain)

        if enable_trails:
            w      = trail_weights(trail_frames, trail_decay)
            trails = (
                f"[{midlab}]format=gbrp,split=2[{midlab}a][{midlab}b];"
                f"[{midlab}b]tmix=frames={trail_frames}:weights='{w}'[{midlab}t];"
                f"[{midlab}a][{midlab}t]blend=all_mode={trail_mode}:all_opacity={trail_opacity:.3f},format=yuv420p[v{i}]"
            )
            filters.append(trails)
            labels.append(f"[v{i}]")
        else:
            labels.append(f"[{midlab}]")

    if react_cmds:
        filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[vcat]")
        filters.append(f"[vcat]sendcmd=f='{_ff_escape(react_cmds)}',eq@react=eval=frame[vout]")
    else:
        filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[vout]")
    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-map", "0:a:0",
        "-shortest",
        "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", preset, "-crf", str(crf),
        "-c:a", "aac",
        out_path
    ]
    return cmd

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma")

def find_audio_files(input_dir: str) -> List[str]:
    p = Path(input_dir)
    if not p.is_dir():
        return []
    return sorted(str(f) for f in p.glob("*") if f.suffix.lower() in AUDIO_EXTS)

def process_one(
    audio_path: str,
    out_path: Path,
    args,
    library: Dict,
    manifest_path: Optional[Path] = None,
    label: str = ""
) -> None:
    random.seed(1337)
    np.random.seed(1337)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{label}Analyzing audio: {audio_path}")
    audio_info = analyze_audio_segments(
        audio_path,
        min_seg=args.min_seg,
        hop=args.hop,
        n_fft=args.n_fft,
        percussive_boost=args.percussive_boost,
        segment_mode=args.segment_mode,
        onset_delta=args.onset_delta,
        onset_pre_max=args.onset_pre_max,
        onset_post_max=args.onset_post_max,
        onset_pre_avg=args.onset_pre_avg,
        onset_post_avg=args.onset_post_avg,
        onset_backtrack=args.onset_backtrack,
        bpm=args.bpm,
        subdiv=args.subdiv,
        compute_f0=args.f0,
        start_bpm=args.start_bpm,
        tempo_min=args.tempo_min,
        tempo_max=args.tempo_max,
        beat_tracker=args.beat_tracker,
        snap=args.snap,
        snap_tol_beats=args.snap_tol_beats,
        beats_per_bar=args.beats_per_bar,
        sections=args.sections,
        quiet_hold_beats=args.quiet_hold,
        beat_refine_s=max(0.0, args.beat_refine_ms) / 1000.0
    )

    segments = audio_info["segments"]
    if not segments:
        raise RuntimeError(f"No segments found in {audio_path}.")
    sections = audio_info.get("sections", [])
    print(f"{label}{len(segments)} segments in {len(sections)} section(s), tempo {audio_info['tempo']:.1f} BPM")

    rms_vals   = np.array([s["features"]["rms"]       for s in segments], dtype=float)
    onset_vals = np.array([s["features"]["onset_avg"] for s in segments], dtype=float)

    def safe_quantiles(x, qs=(0.33, 0.66)):
        if len(x) < 3:
            m = float(np.median(x)) if len(x) else 0.0
            return (m, m)
        q = np.quantile(x, qs)
        return float(q[0]), float(q[1])

    rms_q   = safe_quantiles(rms_vals)
    onset_q = safe_quantiles(onset_vals)

    for s in segments:
        s["bucket"] = classify_audio_segment(s["features"], rms_q, onset_q)

    print(f"{label}Building audio segment curves ({args.curve_method})...")
    audio_seg_curves = None
    if args.clip_strategy == "timeline":
        audio_seg_curves = build_audio_curve_for_segments(
            curves=audio_info["curves"],
            segments=segments,
            vt_step=args.vt_step,
            curve_method=args.curve_method
        )

    print(f"{label}Matching clip windows (beam width {args.beam_width})...")
    chosen = choose_clips_for_segments(
        segments=segments,
        library=library,
        audio_seg_curves=audio_seg_curves,
        clip_strategy=args.clip_strategy,
        no_reuse=args.no_reuse,
        jitter=args.clip_jitter,
        diversity_cooldown=args.diversity_cooldown,
        pairing=args.pairing,
        curve_weight=args.curve_weight,
        align_weight=args.align_weight,
        min_source_gap=args.min_source_gap,
        clip_topk=args.clip_topk,
        reuse_penalty=args.reuse_penalty,
        beam_width=args.beam_width,
        visual_weight=args.visual_weight,
        pool_frac=args.pool_frac,
        impact_weight=args.impact_weight
    )
    distinct = len({c["path"] for c in chosen})
    avg_cost = float(np.mean([c["match_cost"] for c in chosen])) if chosen else 0.0
    print(f"{label}Used {distinct} distinct source(s) across {len(chosen)} cuts, mean match cost {avg_cost:.3f}.")

    beat_times     = np.asarray(audio_info.get("beat_times", np.array([])), dtype=float)
    beat_strengths = np.asarray(audio_info.get("beat_strengths", np.array([])), dtype=float)
    if args.segment_mode == "grid" and args.bpm and args.bpm > 0:
        beat_times     = np.arange(0.0, float(audio_info["duration"]), 60.0 / float(args.bpm), dtype=float)
        beat_strengths = np.ones(beat_times.size, dtype=float)
    profiles    = build_segment_profiles(segments)
    zoom_depths = [0.01 + 0.01 * float(p.get("pace", 0.5)) for p in profiles]
    zoom_in     = [float(s["features"].get("rms_slope", 0.0)) >= 0.0 for s in segments]

    if manifest_path is None:
        manifest_path = out_path.with_suffix(".json")
    meta = {
        "audio_path":     audio_path,
        "videos_dir":     args.videos_dir,
        "output_path":    str(out_path),
        "sr":             audio_info["sr"],
        "audio_duration": audio_info["duration"],
        "tempo_bpm":      audio_info.get("tempo", 0.0),
        "num_segments":   len(segments),
        "segment_mode":   args.segment_mode,
        "bpm":            args.bpm,
        "subdiv":         args.subdiv,
        "beat_tracker":   args.beat_tracker,
        "snap":           args.snap,
        "snap_tol_beats": args.snap_tol_beats,
        "beats_per_bar":  args.beats_per_bar,
        "align_weight":   args.align_weight,
        "cut_lead":       args.cut_lead,
        "min_source_gap": args.min_source_gap,
        "clip_topk":      args.clip_topk,
        "reuse_penalty":  args.reuse_penalty,
        "beam_width":     args.beam_width,
        "visual_weight":  args.visual_weight,
        "pool_frac":      args.pool_frac,
        "quiet_hold":     args.quiet_hold,
        "fps":            args.fps,
        "pulse_strength": args.pulse_strength,
        "react_strength": args.react_strength,
        "impact_weight":  args.impact_weight,
        "beat_refine_ms": args.beat_refine_ms,
        "beat_times":     [round(float(b), 4) for b in beat_times],
        "clip_strategy":  args.clip_strategy,
        "motion_method":  args.motion_method,
        "curve_method":   args.curve_method,
        "curve_weight":   args.curve_weight,
        "sections":       sections,
        "placements": [
            {
                "segment_start":    s["start"],
                "segment_end":      s["end"],
                "segment_duration": s["duration"],
                "section":          s.get("section", 0),
                "audio_features":   s["features"],
                "desired_bucket":   s["bucket"],
                "chosen_clip":      c["path"],
                "chosen_motion_score": c["motion_score"],
                "chosen_bucket":    c["bucket"],
                "clip_start_offset": c["start"],
                "match_cost":       c["match_cost"],
                "segment_profile":  c["segment_profile"],
                "window_profile":   c["window_profile"]
            }
            for s, c in zip(segments, chosen)
        ],
        "seed": 1337
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"{label}Wrote manifest: {manifest_path}")

    react_fd, react_path = tempfile.mkstemp(prefix="avsync_react_", suffix=".txt")
    os.close(react_fd)
    try:
        has_react = build_reactive_commands(
            audio_info["curves"], beat_times, beat_strengths, float(audio_info.get("tempo", 0.0)),
            float(audio_info["duration"]), args.fps, args.cut_lead,
            args.pulse_strength, args.react_strength, react_path
        )
        print(f"{label}Building ffmpeg graph...")
        cmd = build_ffmpeg_graph(
            audio_path=audio_path,
            segments=segments,
            chosen=chosen,
            fps=args.fps,
            width=args.width,
            height=args.height,
            crf=args.crf,
            preset=args.preset,
            out_path=str(out_path),
            zoom_depths=zoom_depths,
            zoom_in=zoom_in,
            react_cmds=react_path if has_react else None,
            trail_frames=args.trail_frames,
            trail_decay=args.trail_decay,
            trail_opacity=args.trail_opacity,
            trail_mode=args.trail_mode,
            cut_lead_frames=args.cut_lead
        )

        print(f"{label}Rendering with ffmpeg...")
        run(cmd)
    finally:
        try:
            os.remove(react_path)
        except OSError:
            pass
    print(f"{label}Done -> {out_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Beat/onset-synced visual editor using ffmpeg (no MoviePy). "
                    "Run with no arguments to render every audio file in ./input "
                    "against ./videos, writing one video per track to ./output."
    )
    ap.add_argument("--audio", default=None,
                    help="Single audio file to render. If omitted, every audio file "
                         "in --input-dir is rendered (one output video each).")
    ap.add_argument("--input-dir", "--input_dir", dest="input_dir", default="input",
                    help="Directory scanned for audio files when --audio is omitted (default: input).")
    ap.add_argument("--videos-dir", "--videos_dir", dest="videos_dir", default="videos",
                    metavar="VIDEOS_DIR",
                    help="Directory of source video clips (default: videos).")
    ap.add_argument("--out", default=None,
                    help="Output path (single-file mode only). In batch mode one file "
                         "per input is written to --out-dir.")
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default="output",
                    help="Directory for rendered videos and manifests (default: output).")

    ap.add_argument("--segment-mode", choices=["beats", "auto", "onsets", "bars", "grid"], default="beats",
                    help="'beats' cuts on every detected beat, refined to the nearest onset (default). "
                         "'auto'/'onsets' cut on onsets, 'bars' on downbeats, 'grid' on a fixed --bpm grid.")
    ap.add_argument("--bpm",    type=float, default=0.0,
                    help="Required for segment-mode=grid; in grid mode the beat pulse follows this grid too.")
    ap.add_argument("--subdiv", type=int,   default=1,   help="For beats/grid modes: cuts per beat (1=every beat, 2=half-beats, 4=quarter-beats).")
    ap.add_argument("--beat-refine-ms", type=float, default=60.0,
                    help="In beats mode, move each beat cut onto the nearest onset peak within this many ms (0 disables).")

    ap.add_argument("--min-seg",          type=float, default=0.20,
                    help="Minimum segment length (s). Lower = tighter, faster cuts (default: 0.20).")
    ap.add_argument("--hop",              type=int,   default=256)
    ap.add_argument("--n-fft",            type=int,   default=1024, help="FFT size (default 1024; use 2048 for higher freq resolution).")
    ap.add_argument("--percussive-boost", type=float, default=1.0)

    ap.add_argument("--onset-delta",    type=float, default=0.08)
    ap.add_argument("--onset-pre-max",  type=int,   default=12)
    ap.add_argument("--onset-post-max", type=int,   default=12)
    ap.add_argument("--onset-pre-avg",  type=int,   default=50)
    ap.add_argument("--onset-post-avg", type=int,   default=50)
    ap.add_argument("--onset-backtrack", action=argparse.BooleanOptionalAction, default=True,
                    help="Backtrack onsets to the local energy minimum (attack start) for tighter cuts. On by default; use --no-onset-backtrack to disable.")

    ap.add_argument("--start-bpm",   type=float, default=120.0, help="Tempo prior for beat tracking.")
    ap.add_argument("--tempo-min",   type=float, default=70.0,  help="Lower bound for tempo octave correction.")
    ap.add_argument("--tempo-max",   type=float, default=180.0, help="Upper bound for tempo octave correction.")
    ap.add_argument("--beat-tracker", choices=["dynamic", "plp"], default="dynamic",
                    help="'dynamic' = beat_track with octave correction; 'plp' follows tempo drift.")
    ap.add_argument("--snap", action=argparse.BooleanOptionalAction, default=True,
                    help="Snap onset/bar cut points onto the beat grid. On by default.")
    ap.add_argument("--snap-tol-beats", type=float, default=0.5,
                    help="Max distance (in beats) a cut may be moved to reach a beat.")
    ap.add_argument("--beats-per-bar",  type=int, default=4, help="Beats per bar for downbeat detection / bars mode.")
    ap.add_argument("--cut-lead", type=int, default=1,
                    help="Shift cuts this many frames earlier so the new shot leads the beat.")

    ap.add_argument("--sections",   type=int,   default=0,
                    help="Number of musical sections to detect from timbre and harmony (0 = auto from duration).")
    ap.add_argument("--quiet-hold", type=float, default=0.0,
                    help="Opt-in: merge cuts in quiet passages into holds up to this many beats long. "
                         "Default 0 keeps every cut, so each beat interval gets its own clip.")

    ap.add_argument("--f0", action="store_true",
                    help="Enable F0/pitch analysis (adds voiced_pct & f0_median_hz features; slow).")

    ap.add_argument("--clip-strategy",    choices=["timeline", "random"], default="timeline",
                    help="'timeline' matches clip windows to each segment's features; 'random' picks random windows and keeps only structure and diversity rules.")
    ap.add_argument("--vt-step",          type=float, default=0.0625,
                    help="Video motion timeline step (s). Keep well below the typical cut length so "
                         "curve matching has several points per segment (default: 0.0625).")
    ap.add_argument("--vt-max-samples",   type=int,   default=1200)
    ap.add_argument("--motion-method",    choices=["flow", "diff"], default="diff",
                    help="Motion scoring method. 'diff' is ~20x faster; 'flow' also measures zoom direction for matching rising/falling energy.")
    ap.add_argument("--workers",          type=int, default=0,
                    help="Parallel processes for video scoring (0 = all CPU cores).")
    ap.add_argument("--clip-jitter",      type=float, default=0.10)
    ap.add_argument("--no-reuse",         type=float, default=1.25)
    ap.add_argument("--diversity-cooldown", type=int, default=3,
                    help="Minimum distinct sources between reuses (lower bound when --min-source-gap is auto).")
    ap.add_argument("--min-source-gap",   type=int, default=0,
                    help="Min distinct sources that must intervene before a source repeats. 0 = auto (~half the library).")
    ap.add_argument("--clip-topk",        type=int, default=4,
                    help="Best non-overlapping windows kept per source as match candidates for each segment.")
    ap.add_argument("--reuse-penalty",    type=float, default=0.30,
                    help="Match-score penalty per prior use of a source, so unused clips win close matches (0 disables).")
    ap.add_argument("--pairing",          choices=["classic", "smart"], default="smart",
                    help="'smart' matches audio and video features; 'classic' uses only curve shape, structure and diversity.")
    ap.add_argument("--beam-width",       type=int, default=8,
                    help="Hypotheses kept while matching clips to segments (1 = greedy).")
    ap.add_argument("--impact-weight",    type=float, default=0.25,
                    help="Reward for windows whose motion rises in the first 100 ms after the cut (0 disables).")
    ap.add_argument("--visual-weight",    type=float, default=1.0,
                    help="Weight of brightness, texture, color and cut-rate matching relative to motion (0 = motion only).")
    ap.add_argument("--pool-frac",        type=float, default=0.5,
                    help="Fraction of the library preferred within each musical section (1 = no section pools).")

    ap.add_argument("--curve-method", choices=["novelty", "onset", "rms"], default="novelty")
    ap.add_argument("--curve-weight", type=float, default=0.60,
                    help="0..1 weight for curve correlation vs feature match.")
    ap.add_argument("--align-weight", type=float, default=0.30,
                    help="Weight for aligning the clip's motion peak to the segment's musical accent (0 disables).")

    ap.add_argument("--fps",    type=int, default=60,
                    help="Output frame rate. 60 halves cut-timing jitter versus 30 at roughly double the encode time (default: 60).")
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--crf",    type=int, default=18)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--manifest", default=None)

    ap.add_argument("--pulse-strength", type=float, default=0.15,
                    help="Saturation pulse on each detected beat: one-frame attack, third-of-a-beat decay, "
                         "amplitude scaled by beat strength (0 disables; default: 0.15).")
    ap.add_argument("--react-strength", type=float, default=0.08,
                    help="Per-frame brightness lift following the audio onset envelope with a 120 ms release. "
                         "Keep below 0.10 to stay under the luminance flash threshold (0 disables; default: 0.08).")
    ap.add_argument("--trail-frames",   type=int,   default=6)
    ap.add_argument("--trail-decay",    type=float, default=0.70)
    ap.add_argument("--trail-opacity",  type=float, default=0.35)
    ap.add_argument("--trail-mode",     choices=["screen", "lighten", "addition", "overlay"], default="screen")

    args = ap.parse_args()
    has_ffmpeg()
    if args.audio:
        audio_files = [args.audio]
    else:
        audio_files = find_audio_files(args.input_dir)
        if not audio_files:
            raise RuntimeError(
                f"No audio files found in '{args.input_dir}'. Add one "
                f"(e.g. {args.input_dir}/song.mp3) or pass --audio PATH."
            )
        print(f"Found {len(audio_files)} audio file(s) in '{args.input_dir}'.")

    single = len(audio_files) == 1
    if args.out and not single:
        print(f"Note: --out is ignored in batch mode; writing one file per input to '{args.out_dir}'.")

    jobs: List[Tuple[str, Path]] = []
    for af in audio_files:
        if single and args.out:
            out_path = Path(args.out)
        else:
            out_path = Path(args.out_dir) / (Path(af).stem + ".mp4")
        jobs.append((af, out_path))

    print("Analyzing candidate videos...")
    workers = args.workers if args.workers and args.workers > 0 else None
    videos = score_all_videos(
        args.videos_dir,
        motion_method=args.motion_method,
        vt_step=args.vt_step,
        vt_max_samples=args.vt_max_samples,
        workers=workers
    )
    library = build_library_features(videos)
    active  = [k for k, a in zip(FEATURE_DIMS, library["active"]) if a]
    print(f"Library: {len(videos)} clip(s); matching on {', '.join(active)}.")

    failures = 0
    for i, (audio_path, out_path) in enumerate(jobs, start=1):
        label = f"[{i}/{len(jobs)}] " if len(jobs) > 1 else ""
        manifest_path = Path(args.manifest) if (args.manifest and single) else None
        try:
            process_one(audio_path, out_path, args, library,
                        manifest_path=manifest_path, label=label)
        except Exception as e:
            if single:
                raise
            failures += 1
            print(f"{label}ERROR processing {audio_path}: {e}")

    if single:
        print("All done.")
    else:
        rendered = len(jobs) - failures
        print(f"All done. Rendered {rendered}/{len(jobs)} video(s) to '{args.out_dir}'.")

if __name__ == "__main__":
    main()
