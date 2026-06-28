import argparse, json, math, subprocess, shutil, random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

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
        onset_envelope=onset_env, sr=sr, hop_length=hop, start_bpm=start_bpm, units="frames"
    )
    tempo = float(np.atleast_1d(tempo)[0])
    corrected = fold_octave(tempo)
    if abs(corrected - tempo) > 1.0:
        try:
            tempo2, beat_frames2 = librosa.beat.beat_track(
                onset_envelope=onset_env, sr=sr, hop_length=hop,
                start_bpm=corrected, bpm=corrected, units="frames"
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
    subdiv: int = 1
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
    beats_per_bar: int = 4
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
        subdiv=subdiv
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

    chroma_mean_track           = np.mean(chroma[:, :n], axis=1)
    global_mode, global_mode_conf = key_mode_from_chroma(chroma_mean_track)

    beat_times_arr = np.asarray(beat_times, dtype=float) if len(beat_frames) else np.array([])

    out = []
    for s, e in segments:
        i0   = int(np.searchsorted(hop_times, s, side='left'))
        i1   = int(np.searchsorted(hop_times, e, side='left'))
        sl   = slice(i0, i1)
        dur  = max(1e-9, float(e - s))
        n_sl = i1 - i0

        def seg_mean(a): return float(np.mean(a[sl])) if n_sl > 0 else 0.0
        def seg_std(a):  return float(np.std(a[sl]))  if n_sl > 0 else 0.0

        rms_mean    = seg_mean(rms_s)
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
            "features": {
                "rms": rms_mean,
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

    return {"sr": sr, "duration": duration, "tempo": float(tempo), "segments": out, "curves": curves}

def _prep(frame: np.ndarray, target_w: int = 320) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > target_w:
        scale = target_w / float(w)
        frame = cv2.resize(frame, (target_w, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def _flow_mag(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=2, winsize=15,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
    )
    mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    return float(np.mean(mag))

def _diff_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    diff = cv2.absdiff(gray, prev_gray)
    return float(np.mean(diff))

def motion_score_video(
    path: str,
    sample_every: int = 10,
    max_samples: int = 300,
    method: str = "diff",
    target_w: int = 320
) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0

    scores: List[float] = []
    prev_gray  = None
    frame_idx  = 0
    taken      = 0

    while taken < max_samples:
        if not cap.grab():
            break
        if frame_idx % sample_every == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = _prep(frame, target_w)
            if prev_gray is not None:
                if method == "flow":
                    scores.append(_flow_mag(prev_gray, gray))
                else:
                    scores.append(_diff_score(prev_gray, gray))
            prev_gray = gray
            taken += 1
        frame_idx += 1

    cap.release()
    return float(np.mean(scores)) if scores else 0.0

def _timeline_worker(args: Tuple) -> Tuple[str, Dict[str, np.ndarray], float]:
    path, step_s, max_samples, method = args
    tl = motion_timeline(path, step_s=step_s, max_samples=max_samples, method=method)
    ms = float(np.mean(tl["scores"])) if tl["scores"].size else 0.0
    return path, tl, ms

def _score_worker(args: Tuple) -> Tuple[str, float]:
    path, method = args
    return path, motion_score_video(path, method=method)

def score_all_videos(
    videos_dir: str,
    motion_method: str = "diff",
    clip_strategy: str = "timeline",
    vt_step: float = 0.25,
    vt_max_samples: int = 1200,
    workers: Optional[int] = None
) -> Tuple[List[Dict], Dict[str, Dict[str, np.ndarray]]]:
    exts  = (".mp4", ".mov", ".mkv", ".webm")
    paths = [str(p) for p in Path(videos_dir).glob("*") if p.suffix and p.suffix.lower() in exts]
    if not paths:
        raise RuntimeError("No video files found in videos_dir.")

    results: List[Dict] = []
    timeline_cache: Dict[str, Dict[str, np.ndarray]] = {}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        if clip_strategy == "timeline":
            tasks = [(p, vt_step, vt_max_samples, motion_method) for p in paths]
            for path, tl, ms in tqdm(ex.map(_timeline_worker, tasks), total=len(tasks),
                                     desc=f"Scoring video motion ({motion_method})"):
                timeline_cache[path] = tl
                results.append({"path": path, "motion_score": ms})
        else:
            tasks = [(p, motion_method) for p in paths]
            for path, ms in tqdm(ex.map(_score_worker, tasks), total=len(tasks),
                                 desc=f"Scoring video motion ({motion_method})"):
                results.append({"path": path, "motion_score": ms})

    scores = np.array([r["motion_score"] for r in results], dtype=float)
    q33, q66 = np.quantile(scores, [0.33, 0.66]) if len(scores) else (0.0, 0.0)

    for r in results:
        r["motion_bucket"] = "slow" if r["motion_score"] < q33 else ("medium" if r["motion_score"] < q66 else "fast")

    results.sort(key=lambda d: d["motion_score"])
    return results, timeline_cache

def motion_timeline(
    path: str,
    step_s: float = 0.25,
    max_samples: int = 1200,
    method: str = "diff",
    target_w: int = 320
) -> Dict[str, np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"times": np.array([]), "scores": np.array([])}

    fps    = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    dur    = ffprobe_duration(path)
    if dur <= 0 and fps > 0 and frames > 0:
        dur = frames / fps
    if dur <= 0:
        cap.release()
        return {"times": np.array([]), "scores": np.array([])}

    if fps <= 0:
        fps = (frames / dur) if dur > 0 else 30.0

    step   = max(0.03, float(step_s))
    n_steps = min(int(math.ceil(dur / step)), int(max_samples))
    times  = np.linspace(0, dur, num=n_steps, endpoint=False).astype(float)
    scores = np.zeros_like(times, dtype=float)

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
        gray = _prep(frame, target_w)

        while sample_i < target_frames.size and target_frames[sample_i] <= idx:
            if prev_gray is not None:
                if method == "flow":
                    scores[sample_i] = _flow_mag(prev_gray, gray)
                else:
                    scores[sample_i] = _diff_score(prev_gray, gray)
            sample_i += 1
        prev_gray = gray

    cap.release()

    if len(scores) > 3:
        scores = uniform_filter1d(scores, size=3)

    return {"times": times, "scores": scores}

def classify_audio_segment(feat: Dict, rms_q: Tuple[float, float], onset_q: Tuple[float, float]) -> str:
    energy, trans = feat["rms"], feat["onset_avg"]
    if energy >= rms_q[1] or trans >= onset_q[1]:
        return "fast"
    if energy <= rms_q[0] and trans <= onset_q[0]:
        return "slow"
    return "medium"

def build_bucket_index(videos_scored: List[Dict]) -> Dict[str, List[Dict]]:
    idx = {"slow": [], "medium": [], "fast": []}
    for v in videos_scored:
        idx[v["motion_bucket"]].append(v)
    for k in idx:
        idx[k] = sorted(idx[k], key=lambda d: d["motion_score"])
    return idx

def _minmax01(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 1e-9:
        return np.zeros_like(x, dtype=float), lo, hi
    return (x - lo) / (hi - lo), lo, hi

def build_segment_profiles(segments: List[Dict]) -> List[Dict]:
    arr = lambda k: np.array([s["features"].get(k, 0.0) for s in segments], dtype=float)

    rms      = arr("rms")
    onset_rt = arr("onset_rate")
    beat_cf  = arr("beat_conf")
    roll85   = arr("rolloff85_mean")
    flatn    = arr("flatness_mean")
    subr     = arr("sub_ratio")
    nov      = arr("novelty")
    onset_std = arr("onset_std")
    nov_std   = arr("novelty_std")

    rms01,      _, _ = _minmax01(rms)
    onset01,    _, _ = _minmax01(onset_rt)
    beat01,     _, _ = _minmax01(beat_cf)
    bright01,   _, _ = _minmax01(roll85)
    noise01,    _, _ = _minmax01(flatn)
    bass01,     _, _ = _minmax01(subr)
    nov01,      _, _ = _minmax01(nov)
    onsetstd01, _, _ = _minmax01(onset_std)
    novstd01,   _, _ = _minmax01(nov_std)

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
            "novelty":    float(nov01[i])
        })
    return profiles

def compute_video_groove_raw(scores: np.ndarray) -> float:
    if scores.size < 8:
        return 0.0
    x    = scores - float(np.mean(scores))
    fft  = np.fft.rfft(x)
    mags = np.abs(fft)
    if mags.size <= 1:
        return 0.0
    return float(np.max(mags[1:]) / (np.sum(mags[1:]) + 1e-9))

def build_video_metrics(timeline: Dict[str, np.ndarray], motion_score: float) -> Dict:
    times  = timeline.get("times",  np.array([]))
    scores = timeline.get("scores", np.array([]))
    if times.size == 0 or scores.size == 0:
        return {"pace_raw": float(motion_score), "burst_raw": 0.0, "groove_raw": 0.0}
    return {"pace_raw": float(np.mean(scores)), "burst_raw": float(np.std(scores)), "groove_raw": compute_video_groove_raw(scores)}

def av_match_score_multi(seg_prof: Dict, vm: Dict, reuse_penalty: float, recent_penalty: float) -> float:
    d2  = (vm["pace01"]  - seg_prof["pace"])**2
    d2 += (vm["burst01"] - seg_prof["burst"])**2
    if "groove01" in vm:
        d2 += 0.25*(vm["groove01"] - seg_prof.get("groove", 0.0))**2
    return d2 + reuse_penalty + recent_penalty

def _zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    m = float(np.mean(x))
    s = float(np.std(x))
    if s <= 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - m) / s

def _corr_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 1.0
    n  = min(a.size, b.size)
    a  = a[:n]; b = b[:n]
    az = _zscore(a)
    bz = _zscore(b)
    denom = float(np.linalg.norm(az) * np.linalg.norm(bz)) + 1e-9
    corr  = float(np.dot(az, bz) / denom)
    corr  = max(-1.0, min(1.0, corr))
    return float(1.0 - corr)

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

def pick_region_by_timeline_windowed(
    timeline: Dict[str, np.ndarray],
    target_dur: float,
    seg_prof: Dict,
    seg_curve: Optional[np.ndarray],
    no_reuse: float,
    used_windows: List[Tuple[float, float]],
    jitter: float,
    curve_weight: float = 0.60,
    align_weight: float = 0.25
) -> float:
    times  = timeline["times"]
    scores = timeline["scores"]
    if times.size == 0 or target_dur <= 0:
        return 0.0

    step = float(times[1] - times[0]) if times.size > 1 else 0.1
    win  = max(1, int(round(target_dur / max(step, 1e-6))))
    if win > scores.size:
        win = scores.size
    if win <= 1:
        w_idx = np.array([0], dtype=int)
    else:
        w_idx = np.arange(scores.size - win + 1, dtype=int)

    if win <= 1:
        w_mean = np.array([float(np.mean(scores))], dtype=float)
        w_std  = np.array([float(np.std(scores))],  dtype=float)
    else:
        cumsum = np.cumsum(np.insert(scores, 0, 0.0))
        mean   = (cumsum[win:] - cumsum[:-win]) / win
        sq     = scores * scores
        cumsq  = np.cumsum(np.insert(sq, 0, 0.0))
        var    = (cumsq[win:] - cumsq[:-win]) / win - mean * mean
        var    = np.maximum(var, 0.0)
        std    = np.sqrt(var)
        w_mean, w_std = mean.astype(float), std.astype(float)

    m01, _, _ = _minmax01(w_mean)
    s01, _, _ = _minmax01(w_std)

    curve_weight = float(max(0.0, min(1.0, curve_weight)))
    base_weight  = 1.0 - curve_weight
    align_weight = float(max(0.0, align_weight))

    seg_peak_frac = None
    if seg_curve is not None and seg_curve.size >= 2:
        seg_peak_frac = float(np.argmax(seg_curve)) / float(seg_curve.size - 1)

    best = None
    for i, idx in enumerate(w_idx):
        start = float(times[idx])

        too_close = any((start < (s2 + no_reuse) and (start + target_dur) > (s2 - no_reuse)) for s2, _ in used_windows)
        if too_close:
            continue

        d_p     = (float(m01[i]) - float(seg_prof.get("pace",  0.5)))
        d_b     = (float(s01[i]) - float(seg_prof.get("burst", 0.5)))
        d_basic = d_p*d_p + d_b*d_b

        win_scores = scores[idx:idx + win].astype(float)

        d_curve = 1.0
        if seg_curve is not None and seg_curve.size >= 2:
            w01, _, _  = _minmax01(win_scores)
            d_curve    = _corr_distance(seg_curve.astype(float), w01.astype(float))

        d_align = 0.0
        if align_weight > 0 and seg_peak_frac is not None and win > 1:
            win_peak_frac = float(np.argmax(win_scores)) / float(win - 1)
            d_align = (seg_peak_frac - win_peak_frac) ** 2

        score = base_weight * d_basic + curve_weight * d_curve + align_weight * d_align

        if best is None or score < best[0]:
            best = (score, start)

    if best is None:
        dur_est = float(times[-1] if times.size else target_dur * 2)
        for _ in range(20):
            start     = random.uniform(0.0, max(0.0, dur_est - target_dur))
            too_close = any((start < (s2 + no_reuse) and (start + target_dur) > (s2 - no_reuse)) for s2, _ in used_windows)
            if not too_close:
                best = (0.0, start)
                break

    start = 0.0 if best is None else best[1]
    if jitter > 0:
        start = max(0.0, start + random.uniform(-jitter, jitter))
    return start

def choose_clips_for_segments(
    segments: List[Dict],
    bucket_index: Dict[str, List[Dict]],
    audio_seg_curves: Optional[List[np.ndarray]] = None,
    clip_strategy: str = "timeline",
    vt_step: float = 0.25,
    vt_max_samples: int = 1200,
    motion_method: str = "diff",
    no_reuse: float = 1.25,
    jitter: float = 0.15,
    diversity_cooldown: int = 3,
    pairing: str = "smart",
    curve_weight: float = 0.60,
    align_weight: float = 0.25,
    min_source_gap: int = 0,
    clip_topk: int = 4,
    timeline_cache: Optional[Dict[str, Dict[str, np.ndarray]]] = None
) -> List[Dict]:
    video_meta: Dict[str, Dict] = {}
    timeline_cache = timeline_cache or {}

    def ensure_meta(vpath: str, motion_score: float):
        if vpath in video_meta:
            return video_meta[vpath]
        if clip_strategy == "timeline":
            tl = timeline_cache.get(vpath)
            if tl is None:
                tl = motion_timeline(vpath, step_s=vt_step, max_samples=vt_max_samples, method=motion_method)
        else:
            tl = {"times": np.array([]), "scores": np.array([])}
        vm = {"dur": ffprobe_duration(vpath), "timeline": tl, "used": [], "motion_score": float(motion_score)}
        vm.update(build_video_metrics(tl, vm["motion_score"]))
        video_meta[vpath] = vm
        return vm

    use_count = defaultdict(int)
    last_idx: Dict[str, int] = {}
    rng       = random.Random(1337)

    seg_profiles = build_segment_profiles(segments) if pairing == "smart" else [{} for _ in segments]
    chosen: List[Dict] = []

    all_cands = []
    for lst in bucket_index.values():
        all_cands.extend(lst)

    seen         = set()
    unique_cands = []
    for c in all_cands:
        if c["path"] not in seen:
            seen.add(c["path"])
            unique_cands.append(c)

    for c in unique_cands:
        ensure_meta(c["path"], c["motion_score"])

    pace_raw  = np.array([video_meta[c["path"]]["pace_raw"]  for c in unique_cands], dtype=float) if unique_cands else np.array([0.0])
    burst_raw = np.array([video_meta[c["path"]]["burst_raw"] for c in unique_cands], dtype=float) if unique_cands else np.array([0.0])
    groove_raw= np.array([video_meta[c["path"]]["groove_raw"]for c in unique_cands], dtype=float) if unique_cands else np.array([0.0])

    p01, _, _ = _minmax01(pace_raw)
    b01, _, _ = _minmax01(burst_raw)
    g01, _, _ = _minmax01(groove_raw)

    for i, c in enumerate(unique_cands):
        vm = video_meta[c["path"]]
        vm["pace01"]  = float(p01[i])
        vm["burst01"] = float(b01[i])
        vm["groove01"]= float(g01[i])

    for seg_idx, s in enumerate(segments):
        bucket = s["bucket"]
        cands  = bucket_index.get(bucket) or bucket_index.get("medium") or sum(bucket_index.values(), [])
        if not cands:
            raise RuntimeError("No scored videos available to choose from.")

        n_pool = len({c["path"] for c in cands})
        if min_source_gap > 0:
            gap_req = min(min_source_gap, n_pool - 1)
        else:
            gap_req = max(diversity_cooldown, (n_pool + 1) // 2)
        gap_req = max(0, min(gap_req, n_pool - 1))

        def gap_of(path: str) -> int:
            li = last_idx.get(path)
            return (seg_idx - li) if li is not None else 1_000_000

        if pairing == "smart":
            seg_prof_sel = seg_profiles[seg_idx]
            def match_of(cand):
                return av_match_score_multi(seg_prof_sel, video_meta[cand["path"]], 0.0, 0.0)
        else:
            def match_of(cand):
                return 0.0

        eligible = [c for c in cands if gap_of(c["path"]) >= gap_req]
        if not eligible:
            max_gap  = max(gap_of(c["path"]) for c in cands)
            eligible = [c for c in cands if gap_of(c["path"]) == max_gap]

        eligible.sort(key=lambda c: (match_of(c), use_count[c["path"]], c["path"]))
        k    = max(1, min(clip_topk, len(eligible)))
        pick = rng.choice(eligible[:k])

        use_count[pick["path"]] += 1
        last_idx[pick["path"]]   = seg_idx

        meta   = video_meta[pick["path"]]
        target = float(s["duration"])

        seg_curve = None
        if audio_seg_curves is not None and seg_idx < len(audio_seg_curves):
            seg_curve = audio_seg_curves[seg_idx]

        if clip_strategy == "timeline" and meta["timeline"]["times"].size > 0:
            start_t = pick_region_by_timeline_windowed(
                meta["timeline"],
                target_dur=target,
                seg_prof=seg_profiles[seg_idx] if pairing == "smart" else {"pace": 0.5, "burst": 0.5},
                seg_curve=seg_curve,
                no_reuse=no_reuse,
                used_windows=meta["used"],
                jitter=jitter,
                curve_weight=curve_weight,
                align_weight=align_weight
            )
        else:
            start_t = rng.uniform(0.0, max(0.0, meta["dur"] - target)) if meta["dur"] > target else 0.0

        meta["used"].append((start_t, start_t + target))
        chosen.append({
            "path":         pick["path"],
            "motion_score": pick["motion_score"],
            "bucket":       pick["motion_bucket"],
            "start":        float(max(0.0, start_t)),
            "src_dur":      float(meta["dur"])
        })

    return chosen

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
    bpm: float = 0.0,
    pulse_strength: float = 0.06,
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

    src_durs = [c.get("src_dur") or ffprobe_duration(c["path"]) or 0.0 for c in chosen]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", audio_path]
    for c, seg in zip(chosen, segments):
        start       = max(0.0, float(c.get("start", 0.0)))
        read_window = max(0.001, float(seg["duration"])) + 0.5
        cmd += ["-ss", f"{start:.6f}", "-t", f"{read_window:.6f}", "-i", c["path"]]

    rng        = random.Random(1337)
    directions = ["in", "out", "left", "right", "up", "down", "inleft", "inright", "inup", "indown"]

    def pulse_scale_for_bucket(bucket: str) -> float:
        return 0.85 if bucket == "slow" else (1.0 if bucket == "medium" else 1.25)

    beats_per_sec  = (bpm / 60.0) if bpm and bpm > 0 else 0.0
    enable_pulse   = (pulse_strength is not None and pulse_strength > 0 and beats_per_sec > 0)

    trail_frames   = int(max(0, trail_frames))
    trail_decay    = float(max(0.0, min(1.0, trail_decay)))
    trail_opacity  = float(max(0.0, min(1.0, trail_opacity)))
    enable_trails  = (trail_frames > 1 and trail_opacity > 0.0)

    def trail_weights(frames: int, decay: float) -> str:
        w = [decay**k for k in range(frames)]
        return " ".join(f"{x:.6f}" for x in w)

    def zoom_mag_for_bucket(bucket: str) -> float:
        return 0.01 if bucket == "slow" else (0.015 if bucket == "medium" else 0.02)

    filters, labels = [], []
    for i, (seg, ch) in enumerate(zip(segments, chosen), start=1):
        N       = max(1, int(seg_frames[i-1]))
        target  = N / float(fps)
        srcdur  = float(src_durs[i-1])
        start   = float(ch.get("start", 0.0))
        bucket  = str(ch.get("bucket", "medium"))
        style   = rng.choice(directions)
        zoom_amt= zoom_mag_for_bucket(bucket)
        prog    = f"min(1,on/{N})"

        if "out" in style and "in" not in style:
            z_expr = f"1+{zoom_amt}*(1-{prog})"
        else:
            z_expr = f"1+{zoom_amt}*{prog}"

        pan_frac_x = 0.0
        pan_frac_y = 0.0
        if "left"  in style: pan_frac_x = -0.02
        if "right" in style: pan_frac_x =  0.02
        if "up"    in style: pan_frac_y = -0.02
        if "down"  in style: pan_frac_y =  0.02

        x_expr = f"(iw-iw/zoom)/2+({pan_frac_x})*iw*{prog}"
        y_expr = f"(ih-ih/zoom)/2+({pan_frac_y})*ih*{prog}"

        base  = f"[{i}:v]"
        tail  = max(0.0, srcdur - start)
        if tail >= target - 1e-4:
            chain = base + f"trim=duration={target:.6f},setpts=PTS-STARTPTS"
        else:
            pad_dur = max(0.0, target - tail)
            chain = base + f"trim=duration={tail:.6f},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={pad_dur:.6f}"

        chain += f",zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d=1"
        chain += f",scale={width}:{height}:force_original_aspect_ratio=decrease"
        chain += f",pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        chain += f",eq=saturation=1.04:gamma=1.02"

        if enable_pulse:
            amp    = pulse_strength * pulse_scale_for_bucket(bucket)
            chain += f",hue=s='1+{amp:.6f}*sin(2*PI*{beats_per_sec:.6f}*t)'"

        midlab  = f"seg{i}"
        chain  += f"[{midlab}]"
        filters.append(chain)

        if enable_trails:
            w      = trail_weights(trail_frames, trail_decay)
            trails = (
                f"[{midlab}]format=yuv420p,split=2[{midlab}a][{midlab}b];"
                f"[{midlab}b]tmix=frames={trail_frames}:weights='{w}'[{midlab}t];"
                f"[{midlab}a][{midlab}t]blend=all_mode={trail_mode}:all_opacity={trail_opacity:.3f}[v{i}]"
            )
            filters.append(trails)
            labels.append(f"[v{i}]")
        else:
            labels.append(f"[{midlab}]")

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
    bucket_index: Dict[str, List[Dict]],
    timeline_cache: Dict[str, Dict[str, np.ndarray]],
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
        beats_per_bar=args.beats_per_bar
    )

    segments = audio_info["segments"]
    if not segments:
        raise RuntimeError(f"No segments found in {audio_path}.")

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

    print(f"{label}Choosing diverse clip regions...")
    chosen = choose_clips_for_segments(
        segments=segments,
        bucket_index=bucket_index,
        audio_seg_curves=audio_seg_curves,
        clip_strategy=args.clip_strategy,
        vt_step=args.vt_step,
        vt_max_samples=args.vt_max_samples,
        motion_method=args.motion_method,
        no_reuse=args.no_reuse,
        jitter=args.clip_jitter,
        diversity_cooldown=args.diversity_cooldown,
        pairing=args.pairing,
        curve_weight=args.curve_weight,
        align_weight=args.align_weight,
        min_source_gap=args.min_source_gap,
        clip_topk=args.clip_topk,
        timeline_cache=timeline_cache
    )

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
        "clip_strategy":  args.clip_strategy,
        "motion_method":  args.motion_method,
        "curve_method":   args.curve_method,
        "curve_weight":   args.curve_weight,
        "placements": [
            {
                "segment_start":    s["start"],
                "segment_end":      s["end"],
                "segment_duration": s["duration"],
                "audio_features":   s["features"],
                "desired_bucket":   s["bucket"],
                "chosen_clip":      c["path"],
                "chosen_motion_score": c["motion_score"],
                "chosen_bucket":    c["bucket"],
                "clip_start_offset": c["start"]
            }
            for s, c in zip(segments, chosen)
        ],
        "seed": 1337
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"{label}Wrote manifest: {manifest_path}")

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
        bpm=args.bpm,
        pulse_strength=args.pulse_strength,
        trail_frames=args.trail_frames,
        trail_decay=args.trail_decay,
        trail_opacity=args.trail_opacity,
        trail_mode=args.trail_mode,
        cut_lead_frames=args.cut_lead
    )

    print(f"{label}Rendering with ffmpeg...")
    run(cmd)
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

    ap.add_argument("--segment-mode", choices=["auto", "onsets", "bars", "grid"], default="auto")
    ap.add_argument("--bpm",    type=float, default=0.0, help="Required for segment-mode=grid. Also used for pulse.")
    ap.add_argument("--subdiv", type=int,   default=1,   help="For grid mode: beats subdiv (1=beats, 2=half, 4=quarter, etc.)")

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

    ap.add_argument("--f0", action="store_true",
                    help="Enable F0/pitch analysis (adds voiced_pct & f0_median_hz features; slow).")

    ap.add_argument("--clip-strategy",    choices=["timeline", "random"], default="timeline")
    ap.add_argument("--vt-step",          type=float, default=0.25)
    ap.add_argument("--vt-max-samples",   type=int,   default=1200)
    ap.add_argument("--motion-method",    choices=["flow", "diff"], default="diff",
                    help="Motion scoring method. 'diff' is ~20x faster than 'flow' with similar bucketing accuracy.")
    ap.add_argument("--workers",          type=int, default=0,
                    help="Parallel processes for video scoring (0 = all CPU cores).")
    ap.add_argument("--clip-jitter",      type=float, default=0.10)
    ap.add_argument("--no-reuse",         type=float, default=1.25)
    ap.add_argument("--diversity-cooldown", type=int, default=3,
                    help="Minimum distinct sources between reuses (lower bound when --min-source-gap is auto).")
    ap.add_argument("--min-source-gap",   type=int, default=0,
                    help="Min distinct sources that must intervene before a source repeats. 0 = auto (~half the library).")
    ap.add_argument("--clip-topk",        type=int, default=4,
                    help="Randomly pick among this many best-matching eligible sources (higher = more varied order).")
    ap.add_argument("--pairing",          choices=["classic", "smart"], default="smart")

    ap.add_argument("--curve-method", choices=["novelty", "onset", "rms"], default="novelty")
    ap.add_argument("--curve-weight", type=float, default=0.60,
                    help="0..1 weight for curve correlation vs pace/burst match.")
    ap.add_argument("--align-weight", type=float, default=0.30,
                    help="Weight for aligning the clip's motion peak to the segment's musical accent (0 disables).")

    ap.add_argument("--fps",    type=int, default=30)
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--crf",    type=int, default=18)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--manifest", default=None)

    ap.add_argument("--pulse-strength", type=float, default=0.06)
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

    print("Scoring candidate videos...")
    workers = args.workers if args.workers and args.workers > 0 else None
    scored, timeline_cache = score_all_videos(
        args.videos_dir,
        motion_method=args.motion_method,
        clip_strategy=args.clip_strategy,
        vt_step=args.vt_step,
        vt_max_samples=args.vt_max_samples,
        workers=workers
    )
    bucket_index = build_bucket_index(scored)

    failures = 0
    for i, (audio_path, out_path) in enumerate(jobs, start=1):
        label = f"[{i}/{len(jobs)}] " if len(jobs) > 1 else ""
        manifest_path = Path(args.manifest) if (args.manifest and single) else None
        try:
            process_one(audio_path, out_path, args, bucket_index, timeline_cache,
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
