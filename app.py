import argparse, json, math, subprocess, shutil, random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import deque, defaultdict
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
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(p.stdout.strip())
    except Exception:
        return 0.0

def superflux_onset_env(y: np.ndarray, sr: int, hop: int, n_fft: int, n_mels: int = 128, smooth_frames: int = 3) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=1.0)
    S = np.log1p(S)
    D = np.diff(S, axis=1)
    D = np.maximum(D, 0.0)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)
    def band_mask(lo, hi):
        return (mel_freqs >= lo) & (mel_freqs < hi)
    low = D[band_mask(20, 200), :]
    mid = D[band_mask(200, 1500), :]
    high = D[band_mask(1500, 8000), :]
    env = (0.9*np.mean(low, axis=0, keepdims=True) if low.size else 0) + (1.2*np.mean(mid, axis=0, keepdims=True) if mid.size else 0) + (1.1*np.mean(high, axis=0, keepdims=True) if high.size else 0)
    env = env.ravel()
    if smooth_frames > 1:
        env = uniform_filter1d(env, size=smooth_frames)
    return np.pad(env, (1,0), mode="edge")

def detect_onsets_percussive(y: np.ndarray, sr: int, hop: int, n_fft: int, delta: float, pre_max: int, post_max: int, pre_avg: int, post_avg: int, backtrack: bool) -> np.ndarray:
    y_harm, y_perc = librosa.effects.hpss(y, margin=(1.0, 2.0))
    env = superflux_onset_env(y_perc, sr, hop, n_fft)
    onset_frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units="frames", pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg, delta=delta, backtrack=backtrack)
    return onset_frames

def choose_segments(y: np.ndarray, sr: int, hop: int, min_seg: float, mode: str, onset_kwargs: Dict) -> Tuple[List[Tuple[float,float]], float]:
    duration = len(y)/sr
    n_fft = 2048
    onset_frames = detect_onsets_percussive(y, sr, hop, n_fft, **onset_kwargs)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    env_perc = superflux_onset_env(librosa.effects.hpss(y, margin=(1.0,2.0))[1], sr, hop, n_fft)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=env_perc, sr=sr, hop_length=hop, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    def segments_from_onsets(times: np.ndarray) -> List[Tuple[float,float]]:
        if len(times) < 2:
            return [(0.0, duration)]
        times = np.clip(times, 0, duration)
        t = np.concatenate([[0.0], times, [duration]])
        return [(float(t[i]), float(t[i+1])) for i in range(len(t)-1)]
    def segments_from_bars(beat_times: np.ndarray, beats_per_bar: int = 4) -> List[Tuple[float,float]]:
        if len(beat_times) < beats_per_bar:
            return [(0.0, duration)]
        bars = [0.0]
        for i, t in enumerate(beat_times, start=1):
            if i % beats_per_bar == 1:
                bars.append(float(t))
        bars.append(duration)
        bars = sorted(set(bars))
        return [(float(bars[i]), float(bars[i+1])) for i in range(len(bars)-1)]
    if mode == "onsets":
        segs = segments_from_onsets(onset_times)
    elif mode == "bars":
        segs = segments_from_bars(beat_times)
    else:
        segs_on = segments_from_onsets(onset_times)
        med_on = np.median([e-s for s,e in segs_on]) if segs_on else 0
        if len(onset_times) >= 12 and med_on >= 0.25:
            segs = segs_on
        else:
            segs = segments_from_bars(beat_times)
    merged = []
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
    merged = [(max(0.0,s), min(duration,e)) for s,e in merged if e > s + 1e-6]
    return merged, duration

def key_mode_from_chroma(chroma_vec: np.ndarray) -> Tuple[str, float]:
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=float)
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=float)
    c = chroma_vec / (np.sum(chroma_vec) + 1e-9)
    def best_score(profile):
        scores = [np.dot(c, np.roll(profile, k)) for k in range(12)]
        return float(np.max(scores))
    sM = best_score(major_profile)
    sN = best_score(minor_profile)
    total = sM + sN + 1e-9
    if sM >= sN:
        return "major", sM / total
    else:
        return "minor", sN / total

def analyze_audio_segments(audio_path: str, min_seg: float = 0.35, hop: int = 256, percussive_boost: float = 1.0, segment_mode: str = "auto", onset_delta: float = 0.08, onset_pre_max: int = 12, onset_post_max: int = 12, onset_pre_avg: int = 50, onset_post_avg: int = 50, onset_backtrack: bool = True) -> Dict:
    y, sr = librosa.load(audio_path, sr=44100, mono=True)
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
    if percussive_boost > 0:
        y_h, y_p = librosa.effects.hpss(y, margin=(1.0, 2.0))
        y = librosa.util.normalize((1.0 - percussive_boost)*y + percussive_boost*y_p)
    else:
        y_h, y_p = librosa.effects.hpss(y, margin=(1.0, 2.0))
    onset_kwargs = dict(delta=onset_delta, pre_max=onset_pre_max, post_max=onset_post_max, pre_avg=onset_pre_avg, post_avg=onset_post_avg, backtrack=onset_backtrack)
    segments, duration = choose_segments(y, sr, hop, min_seg, segment_mode, onset_kwargs)
    n_fft = 2048
    S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True, window="hann")
    S_mag = np.abs(S_complex)
    Pxx = (S_mag ** 2)
    mel = librosa.feature.melspectrogram(S=Pxx, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    rms = librosa.feature.rms(S=S_mag, hop_length=hop, center=True)[0]
    cen = librosa.feature.spectral_centroid(S=S_mag, sr=sr, hop_length=hop, center=True)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop, center=True)[0]
    onset_env = librosa.onset.onset_strength(S=mel, sr=sr, hop_length=hop, center=True)
    rolloff85 = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, roll_percent=0.85, hop_length=hop)[0]
    rolloff95 = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, roll_percent=0.95, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S_mag, sr=sr, hop_length=hop, center=True)[0]
    flatness = librosa.feature.spectral_flatness(S=S_mag, hop_length=hop, center=True)[0]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    hfc = np.sum((S_mag.T * freqs).T, axis=0) / (np.sum(S_mag, axis=0) + 1e-9)
    mfcc = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=13)
    dmfcc = librosa.feature.delta(mfcc)
    ddmfcc = librosa.feature.delta(mfcc, order=2)
    H = librosa.feature.rms(y=y_h, hop_length=hop, center=True)[0]
    P = librosa.feature.rms(y=y_p, hop_length=hop, center=True)[0]
    hpr = (H + 1e-9) / (P + 1e-9)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chroma_dom = np.max(chroma, axis=0) / (np.sum(chroma, axis=0) + 1e-9)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="frames")
    beat_conf = float(np.mean(onset_env[beat_frames])) if len(beat_frames) else 0.0
    onset_frames_all = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="frames")
    onset_times_all = librosa.frames_to_time(onset_frames_all, sr=sr, hop_length=hop)
    try:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, frame_length=n_fft, hop_length=hop)
        voiced_mask = np.isfinite(f0) & (f0 > 0)
    except Exception:
        f0 = None
        voiced_mask = None
    mel_freqs = librosa.mel_frequencies(n_mels=mel.shape[0], fmin=0, fmax=sr/2)
    sub_mask = mel_freqs < 80
    low_mask = (mel_freqs >= 80) & (mel_freqs < 300)
    mid_mask = (mel_freqs >= 300) & (mel_freqs < 2500)
    high_mask = mel_freqs >= 2500
    mel_norm = mel / (np.maximum(np.sum(mel, axis=0, keepdims=True), 1e-9))
    flux = np.maximum(0.0, np.diff(mel_norm, axis=1))
    novelty = np.sum(flux, axis=0)
    novelty = np.pad(novelty, (1,0), mode="edge")
    def smooth(a):
        return uniform_filter1d(a, size=3)
    rms_s, cen_s, zcr_s, on_s = map(smooth, (rms, cen, zcr, onset_env))
    roll85_s = smooth(rolloff85)
    roll95_s = smooth(rolloff95)
    bw_s = smooth(bandwidth)
    flat_s = smooth(flatness)
    hfc_s = smooth(hfc)
    hpr_s = smooth(hpr)
    chroma_dom_s = smooth(chroma_dom)
    novelty_s = smooth(novelty)
    n = min(*[len(a) for a in (rms_s, cen_s, zcr_s, on_s, roll85_s, roll95_s, bw_s, flat_s, hfc_s, hpr_s, chroma_dom_s, novelty_s)])
    rms_s, cen_s, zcr_s, on_s = (a[:n] for a in (rms_s, cen_s, zcr_s, on_s))
    roll85_s, roll95_s, bw_s, flat_s, hfc_s, hpr_s, chroma_dom_s, novelty_s = (a[:n] for a in (roll85_s, roll95_s, bw_s, flat_s, hfc_s, hpr_s, chroma_dom_s, novelty_s))
    mfcc = mfcc[:, :n]; dmfcc = dmfcc[:, :n]; ddmfcc = ddmfcc[:, :n]
    if tonnetz.shape[1] > n:
        tonnetz = tonnetz[:, :n]
    hop_times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)
    chroma_mean_track = np.mean(chroma[:, :n], axis=1)
    global_mode, global_mode_conf = key_mode_from_chroma(chroma_mean_track)
    def mean_std(a, mask):
        v = a[mask]
        return (float(np.mean(v)) if v.size else 0.0, float(np.std(v)) if v.size else 0.0)
    out = []
    for s, e in segments:
        mask = (hop_times >= s) & (hop_times < e)
        dur = max(1e-9, float(e - s))
        rms_mean, _ = mean_std(rms_s, mask)
        cen_mean, _ = mean_std(cen_s, mask)
        zcr_mean, _ = mean_std(zcr_s, mask)
        on_mean, on_std = mean_std(on_s, mask)
        roll85_mean, roll85_std = mean_std(roll85_s, mask)
        roll95_mean, roll95_std = mean_std(roll95_s, mask)
        bw_mean, bw_std = mean_std(bw_s, mask)
        flat_mean, flat_std = mean_std(flat_s, mask)
        hfc_mean, hfc_std = mean_std(hfc_s, mask)
        hpr_mean, _ = mean_std(hpr_s, mask)
        chroma_dom_mean, _ = mean_std(chroma_dom_s, mask)
        novelty_mean, nov_std = mean_std(novelty_s, mask)
        mfcc_means = [float(np.mean(c[mask])) if np.any(mask) else 0.0 for c in mfcc]
        mfcc_stds = [float(np.std(c[mask])) if np.any(mask) else 0.0 for c in mfcc]
        dmfcc_means = [float(np.mean(c[mask])) if np.any(mask) else 0.0 for c in dmfcc]
        dmfcc_stds = [float(np.std(c[mask])) if np.any(mask) else 0.0 for c in dmfcc]
        ddmfcc_means = [float(np.mean(c[mask])) if np.any(mask) else 0.0 for c in ddmfcc]
        ddmfcc_stds = [float(np.std(c[mask])) if np.any(mask) else 0.0 for c in ddmfcc]
        mel_seg = mel[:, (hop_times >= s) & (hop_times < e)]
        mel_tot = float(np.sum(mel_seg) + 1e-9)
        sub_ratio = float(np.sum(mel_seg[sub_mask, :]) / mel_tot) if mel_seg.size else 0.0
        low_ratio = float(np.sum(mel_seg[low_mask, :]) / mel_tot) if mel_seg.size else 0.0
        mid_ratio = float(np.sum(mel_seg[mid_mask, :]) / mel_tot) if mel_seg.size else 0.0
        high_ratio = float(np.sum(mel_seg[high_mask, :]) / mel_tot) if mel_seg.size else 0.0
        on_in = onset_times_all[(onset_times_all >= s) & (onset_times_all < e)]
        onset_rate = float(len(on_in) / dur)
        if len(on_in) >= 2:
            ioi = np.diff(on_in)
            ioi_std = float(np.std(ioi))
        else:
            ioi_std = 0.0
        if len(beat_frames):
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
            local_beats = (beat_times >= s) & (beat_times < e)
            local_conf = float(np.mean(onset_env[beat_frames][local_beats])) if np.any(local_beats) else beat_conf
        else:
            local_conf = 0.0
        if f0 is not None:
            f0_seg = f0[(hop_times >= s) & (hop_times < e)]
            vm = voiced_mask[(hop_times >= s) & (hop_times < e)] if voiced_mask is not None else None
            voiced_pct = float(np.mean(vm)) if vm is not None and vm.size else 0.0
            f0_med = float(np.median(f0_seg[np.isfinite(f0_seg)])) if f0_seg.size and np.any(np.isfinite(f0_seg)) else 0.0
        else:
            voiced_pct, f0_med = 0.0, 0.0
        chroma_seg = chroma[:, (hop_times >= s) & (hop_times < e)]
        if chroma_seg.size and np.sum(chroma_seg) > 1e-6:
            mode_lbl, mode_conf = key_mode_from_chroma(np.mean(chroma_seg, axis=1))
        else:
            mode_lbl, mode_conf = global_mode, global_mode_conf
        if tonnetz.shape[1] == len(hop_times):
            tn = tonnetz[:, (hop_times >= s) & (hop_times < e)]
            tonnetz_std = float(np.mean(np.std(tn, axis=1))) if tn.size else 0.0
        else:
            tonnetz_std = 0.0
        segment_samples = y[int(s*sr):int(e*sr)]
        peak = float(np.max(np.abs(segment_samples))) if segment_samples.size else 0.0
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
    return {"sr": sr, "duration": duration, "segments": out}

def motion_score_video(path: str, sample_every: int = 10, max_samples: int = 300) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    scores, prev_gray, taken = [], None, 0
    for idx in range(0, length, sample_every):
        if taken >= max_samples:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            scores.append(float(np.mean(diff)))
        prev_gray = gray
        taken += 1
    cap.release()
    return float(np.mean(scores)) if scores else 0.0

def score_all_videos(videos_dir: str) -> List[Dict]:
    exts = (".mp4",".mov",".mkv",".webm")
    paths = [str(p) for p in Path(videos_dir).glob("*") if p.suffix and p.suffix.lower() in exts]
    if not paths:
        raise RuntimeError("No video files found in videos_dir.")
    results = []
    for p in tqdm(paths, desc="Scoring video motion"):
        results.append({"path": p, "motion_score": motion_score_video(p)})
    scores = np.array([r["motion_score"] for r in results])
    q33, q66 = np.quantile(scores, [0.33, 0.66]) if len(scores) else (0.0, 0.0)
    for r in results:
        r["motion_bucket"] = "slow" if r["motion_score"] < q33 else ("medium" if r["motion_score"] < q66 else "fast")
    results.sort(key=lambda d: d["motion_score"])
    return results

def motion_timeline(path: str, step_s: float = 0.25, max_samples: int = 1200) -> Dict[str, np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"times": np.array([]), "scores": np.array([])}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    dur = ffprobe_duration(path)
    if dur <= 0 and fps > 0 and frames > 0:
        dur = frames / fps
    if dur <= 0:
        cap.release()
        return {"times": np.array([]), "scores": np.array([])}
    step = max(0.05, step_s)
    n_steps = min(int(math.ceil(dur / step)), max_samples)
    times = np.linspace(0, dur, num=n_steps, endpoint=False)
    scores = np.zeros_like(times)
    prev_gray = None
    for i, t in enumerate(times):
        ok = False
        if cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t*1000.0)):
            ok, frame = cap.read()
        if not ok:
            frame_idx = int(round(t * (fps if fps > 0 else frames / max(dur,1e-9))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_idx, max(frames-1,0))))
            ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            scores[i] = float(np.mean(diff))
        prev_gray = gray
    cap.release()
    if len(scores) > 3:
        scores = uniform_filter1d(scores, size=3)
    return {"times": times, "scores": scores}

def classify_audio_segment(feat: Dict, rms_q: Tuple[float,float], onset_q: Tuple[float,float]) -> str:
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

def pick_region_by_timeline_windowed(timeline: Dict[str, np.ndarray], target_dur: float, seg_prof: Dict, no_reuse: float, used_windows: List[Tuple[float,float]], jitter: float) -> float:
    times = timeline["times"]; scores = timeline["scores"]
    if times.size == 0 or target_dur <= 0:
        return 0.0
    step = float(times[1]-times[0]) if times.size > 1 else 0.1
    win = max(1, int(round(target_dur / step)))
    if win > scores.size:
        win = scores.size
    if win <= 1:
        w_mean = np.array([np.mean(scores)])
        w_std = np.array([np.std(scores)])
        w_idx = np.array([0])
    else:
        cumsum = np.cumsum(np.insert(scores, 0, 0.0))
        mean = (cumsum[win:] - cumsum[:-win]) / win
        sq = scores*scores
        cumsq = np.cumsum(np.insert(sq, 0, 0.0))
        var = (cumsq[win:] - cumsq[:-win]) / win - mean*mean
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        w_mean, w_std = mean, std
        w_idx = np.arange(len(mean))
    m01, _, _ = _minmax01(w_mean)
    s01, _, _ = _minmax01(w_std)
    best = None
    for i, idx in enumerate(w_idx):
        start = float(times[idx])
        too_close = any((start < (s2 + no_reuse) and (start + target_dur) > (s2 - no_reuse)) for s2, _ in used_windows)
        if too_close:
            continue
        d_p = (float(m01[i]) - seg_prof["pace"])
        d_b = (float(s01[i]) - seg_prof["burst"])
        score = d_p*d_p + d_b*d_b
        if best is None or score < best[0]:
            best = (score, start)
    if best is None:
        dur_est = float(times[-1] if times.size else target_dur*2)
        for _ in range(20):
            start = random.uniform(0.0, max(0.0, dur_est - target_dur))
            too_close = any((start < (s2 + no_reuse) and (start + target_dur) > (s2 - no_reuse)) for s2, _ in used_windows)
            if not too_close:
                best = (0.0, start)
                break
    start = 0.0 if best is None else best[1]
    if jitter > 0:
        start = max(0.0, start + random.uniform(-jitter, jitter))
    return start

def _minmax01(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 1e-9:
        return np.zeros_like(x, dtype=float), lo, hi
    return (x - lo) / (hi - lo), lo, hi

def build_segment_profiles(segments: List[Dict]) -> List[Dict]:
    arr = lambda k: np.array([s["features"].get(k, 0.0) for s in segments], dtype=float)
    rms = arr("rms")
    onset_rt = arr("onset_rate")
    beat_cf = arr("beat_conf")
    roll85 = arr("rolloff85_mean")
    flatn = arr("flatness_mean")
    subr = arr("sub_ratio")
    nov = arr("novelty")
    onset_std = arr("onset_std")
    nov_std = arr("novelty_std")
    rms01, _, _ = _minmax01(rms)
    onset01, _, _ = _minmax01(onset_rt)
    beat01, _, _ = _minmax01(beat_cf)
    bright01, _, _ = _minmax01(roll85)
    noise01, _, _ = _minmax01(flatn)
    bass01, _, _ = _minmax01(subr)
    nov01, _, _ = _minmax01(nov)
    onsetstd01, _, _ = _minmax01(onset_std)
    novstd01, _, _ = _minmax01(nov_std)
    profiles = []
    for i in range(len(segments)):
        pace = float(0.45*rms01[i] + 0.35*onset01[i] + 0.20*beat01[i])
        burst = float(0.5*onsetstd01[i] + 0.5*novstd01[i])
        profiles.append({
            "pace": pace,
            "burst": burst,
            "groove": float(beat01[i]),
            "brightness": float(bright01[i]),
            "noise": float(noise01[i]),
            "bass": float(bass01[i]),
            "novelty": float(nov01[i])
        })
    return profiles

def compute_video_groove_raw(scores: np.ndarray) -> float:
    if scores.size < 8:
        return 0.0
    x = scores - float(np.mean(scores))
    fft = np.fft.rfft(x)
    mags = np.abs(fft)
    if mags.size <= 1:
        return 0.0
    return float(np.max(mags[1:]) / (np.sum(mags[1:]) + 1e-9))

def build_video_metrics(timeline: Dict[str, np.ndarray], motion_score: float) -> Dict:
    times = timeline.get("times", np.array([]))
    scores = timeline.get("scores", np.array([]))
    if times.size == 0 or scores.size == 0:
        return {"pace_raw": float(motion_score), "burst_raw": 0.0, "groove_raw": 0.0}
    return {"pace_raw": float(np.mean(scores)), "burst_raw": float(np.std(scores)), "groove_raw": compute_video_groove_raw(scores)}

def av_match_score_multi(seg_prof: Dict, vm: Dict, reuse_penalty: float, recent_penalty: float) -> float:
    d2 = (vm["pace01"] - seg_prof["pace"])**2
    d2 += (vm["burst01"] - seg_prof["burst"])**2
    if "groove01" in vm:
        d2 += 0.25*(vm["groove01"] - seg_prof.get("groove", 0.0))**2
    return d2 + reuse_penalty + recent_penalty

def choose_clips_for_segments(segments: List[Dict], bucket_index: Dict[str, List[Dict]], clip_strategy: str = "timeline", vt_step: float = 0.25, vt_max_samples: int = 1200, no_reuse: float = 1.25, jitter: float = 0.15, diversity_cooldown: int = 3, pairing: str = "smart") -> List[Dict]:
    video_meta: Dict[str, Dict] = {}
    def ensure_meta(vpath: str, motion_score: float):
        if vpath in video_meta:
            return video_meta[vpath]
        tl = motion_timeline(vpath, step_s=vt_step, max_samples=vt_max_samples) if clip_strategy == "timeline" else {"times": np.array([]), "scores": np.array([])}
        vm = {"dur": ffprobe_duration(vpath), "timeline": tl, "used": [], "motion_score": float(motion_score)}
        vm.update(build_video_metrics(tl, vm["motion_score"]))
        video_meta[vpath] = vm
        return vm
    recent = deque(maxlen=max(1, diversity_cooldown)) if diversity_cooldown > 0 else None
    use_count = defaultdict(int)
    rng = random.Random(1337)
    seg_profiles = build_segment_profiles(segments) if pairing == "smart" else [{} for _ in segments]
    rr_ptr = {"slow":0,"medium":0,"fast":0}
    chosen: List[Dict] = []
    all_cands = []
    for lst in bucket_index.values():
        all_cands.extend(lst)
    seen = set()
    unique_cands = []
    for c in all_cands:
        if c["path"] not in seen:
            seen.add(c["path"])
            unique_cands.append(c)
    for c in unique_cands:
        ensure_meta(c["path"], c["motion_score"])
    pace_raw = np.array([video_meta[c["path"]]["pace_raw"] for c in unique_cands], dtype=float) if unique_cands else np.array([0.0])
    burst_raw = np.array([video_meta[c["path"]]["burst_raw"] for c in unique_cands], dtype=float) if unique_cands else np.array([0.0])
    groove_raw = np.array([video_meta[c["path"]]["groove_raw"] for c in unique_cands], dtype=float) if unique_cands else np.array([0.0])
    p01, _, _ = _minmax01(pace_raw)
    b01, _, _ = _minmax01(burst_raw)
    g01, _, _ = _minmax01(groove_raw)
    for i, c in enumerate(unique_cands):
        vm = video_meta[c["path"]]
        vm["pace01"] = float(p01[i])
        vm["burst01"] = float(b01[i])
        vm["groove01"] = float(g01[i])
    for seg_idx, s in enumerate(segments):
        bucket = s["bucket"]
        cands = bucket_index.get(bucket) or bucket_index.get("medium") or sum(bucket_index.values(), [])
        if not cands:
            raise RuntimeError("No scored videos available to choose from.")
        if pairing == "classic":
            start = rr_ptr[bucket] % len(cands)
            ordered = cands[start:] + cands[:start]
            pick = None
            if recent:
                for cand in ordered:
                    if cand["path"] not in recent:
                        pick = cand
                        break
            if pick is None:
                pick = ordered[0]
            rr_ptr[bucket] += 1
            use_count[pick["path"]] += 1
            if recent:
                recent.append(pick["path"])
        else:
            seg_prof = seg_profiles[seg_idx]
            scores = []
            for cand in cands:
                vm = video_meta[cand["path"]]
                reuse_pen = 0.02 * use_count[cand["path"]]
                recent_pen = 0.03 if (recent and cand["path"] in recent) else 0.0
                score = av_match_score_multi(seg_prof, vm, reuse_pen, recent_pen)
                scores.append((score, cand))
            scores.sort(key=lambda x: (x[0], use_count[x[1]["path"]], x[1]["motion_score"], x[1]["path"]))
            pick = scores[0][1]
            use_count[pick["path"]] += 1
            if recent:
                recent.append(pick["path"])
        meta = video_meta[pick["path"]]
        target = s["duration"]
        if clip_strategy == "timeline" and meta["timeline"]["times"].size > 0:
            start_t = pick_region_by_timeline_windowed(meta["timeline"], target, seg_profiles[seg_idx] if pairing=="smart" else {"pace":0.5,"burst":0.5}, no_reuse, meta["used"], jitter)
        else:
            start_t = rng.uniform(0.0, max(0.0, meta["dur"] - target)) if meta["dur"] > target else 0.0
        meta["used"].append((start_t, start_t + target))
        chosen.append({"path": pick["path"], "motion_score": pick["motion_score"], "bucket": pick["motion_bucket"], "start": float(max(0.0, start_t)), "src_dur": float(meta["dur"])})
    return chosen

def build_ffmpeg_graph(audio_path: str, segments: List[Dict], chosen: List[Dict], fps: int, width: int, height: int, crf: int, preset: str, out_path: str, bpm: float = 0.0, pulse_strength: float = 0.06, trail_frames: int = 6, trail_decay: float = 0.70, trail_opacity: float = 0.35, trail_mode: str = "screen"):
    src_durs = [c.get("src_dur") or ffprobe_duration(c["path"]) or 0.0 for c in chosen]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", audio_path]
    for c in chosen:
        cmd += ["-i", c["path"]]
    rng = random.Random(1337)
    directions = ["in", "out", "left", "right", "up", "down", "inleft", "inright", "inup", "indown"]
    def pulse_scale_for_bucket(bucket: str) -> float:
        return 0.85 if bucket == "slow" else (1.0 if bucket == "medium" else 1.25)
    beats_per_sec = (bpm / 60.0) if bpm and bpm > 0 else 0.0
    enable_pulse = (pulse_strength is not None and pulse_strength > 0 and beats_per_sec > 0)
    trail_frames = int(max(0, trail_frames))
    trail_decay = float(max(0.0, min(1.0, trail_decay)))
    trail_opacity = float(max(0.0, min(1.0, trail_opacity)))
    enable_trails = (trail_frames > 1 and trail_opacity > 0.0)
    def trail_weights(frames: int, decay: float) -> str:
        w = [decay**k for k in range(frames)]
        return " ".join(f"{x:.6f}" for x in w)
    def zoom_mag_for_bucket(bucket: str) -> float:
        return 0.01 if bucket == "slow" else (0.015 if bucket == "medium" else 0.02)
    filters, labels = [], []
    for i, (seg, ch) in enumerate(zip(segments, chosen), start=1):
        target = max(0.001, float(seg["duration"]))
        srcdur = float(src_durs[i-1])
        start = float(ch.get("start", 0.0))
        bucket = str(ch.get("bucket", "medium"))
        N = max(1, int(round(target * fps)))
        style = rng.choice(directions)
        zoom_amt = zoom_mag_for_bucket(bucket)
        prog = f"min(1,on/{N})"
        if "out" in style and "in" not in style:
            z_expr = f"1+{zoom_amt}*(1-{prog})"
        else:
            z_expr = f"1+{zoom_amt}*{prog}"
        pan_frac_x = 0.0
        pan_frac_y = 0.0
        if "left" in style:
            pan_frac_x = -0.02
        if "right" in style:
            pan_frac_x = 0.02
        if "up" in style:
            pan_frac_y = -0.02
        if "down" in style:
            pan_frac_y = 0.02
        x_expr = f"(iw-iw/zoom)/2+({pan_frac_x})*iw*{prog}"
        y_expr = f"(ih-ih/zoom)/2+({pan_frac_y})*ih*{prog}"
        base = f"[{i}:v]"
        chain = base + f"trim=start={max(0.0, start):.6f}"
        tail = max(0.0, srcdur - start)
        if tail >= target - 1e-4:
            chain += f":duration={target:.6f},setpts=PTS-STARTPTS"
        else:
            pad_dur = max(0.0, target - tail)
            chain += f":duration={tail:.6f},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={pad_dur:.6f}"
        chain += f",zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d=1"
        chain += f",scale={width}:{height}:force_original_aspect_ratio=decrease"
        chain += f",pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        chain += f",eq=saturation=1.04:gamma=1.02"
        if enable_pulse:
            amp = pulse_strength * pulse_scale_for_bucket(bucket)
            chain += f",hue=s='1+{amp:.6f}*sin(2*PI*{beats_per_sec:.6f}*t)'"
        midlab = f"seg{i}"
        chain += f"[{midlab}]"
        filters.append(chain)
        if enable_trails:
            w = trail_weights(trail_frames, trail_decay)
            trails = f"[{midlab}]format=yuv420p,split=2[{midlab}a][{midlab}b];[{midlab}b]tmix=frames={trail_frames}:weights='{w}'[{midlab}t];[{midlab}a][{midlab}t]blend=all_mode={trail_mode}:all_opacity={trail_opacity:.3f}[v{i}]"
            filters.append(trails)
            labels.append(f"[v{i}]")
        else:
            labels.append(f"[{midlab}]")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[vout]")
    cmd += ["-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "0:a:0", "-shortest", "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", preset, "-crf", str(crf), "-c:a", "aac", out_path]
    return cmd

def main():
    ap = argparse.ArgumentParser(description="Beat/onset-synced visual editor using ffmpeg (no MoviePy).")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--segment-mode", choices=["auto","onsets","bars"], default="auto")
    ap.add_argument("--min-seg", type=float, default=0.35)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--percussive-boost", type=float, default=1.0)
    ap.add_argument("--onset-delta", type=float, default=0.08)
    ap.add_argument("--onset-pre-max", type=int, default=12)
    ap.add_argument("--onset-post-max", type=int, default=12)
    ap.add_argument("--onset-pre-avg", type=int, default=50)
    ap.add_argument("--onset-post-avg", type=int, default=50)
    ap.add_argument("--onset-backtrack", action="store_true")
    ap.add_argument("--clip-strategy", choices=["timeline","random"], default="timeline")
    ap.add_argument("--vt-step", type=float, default=0.25)
    ap.add_argument("--vt-max-samples", type=int, default=1200)
    ap.add_argument("--clip-jitter", type=float, default=0.15)
    ap.add_argument("--no-reuse", type=float, default=1.25)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--bpm", type=float, default=0.0)
    ap.add_argument("--pulse-strength", type=float, default=0.06)
    ap.add_argument("--trail-frames", type=int, default=6)
    ap.add_argument("--trail-decay", type=float, default=0.70)
    ap.add_argument("--trail-opacity", type=float, default=0.35)
    ap.add_argument("--trail-mode", choices=["screen","lighten","addition","overlay"], default="screen")
    ap.add_argument("--diversity-cooldown", type=int, default=3)
    ap.add_argument("--pairing", choices=["classic","smart"], default="smart")
    args = ap.parse_args()
    has_ffmpeg()
    random.seed(1337); np.random.seed(1337)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("Analyzing audio (drum-sensitive onsets)…")
    audio_info = analyze_audio_segments(args.audio, min_seg=args.min_seg, hop=args.hop, percussive_boost=args.percussive_boost, segment_mode=args.segment_mode, onset_delta=args.onset_delta, onset_pre_max=args.onset_pre_max, onset_post_max=args.onset_post_max, onset_pre_avg=args.onset_pre_avg, onset_post_avg=args.onset_post_avg, onset_backtrack=args.onset_backtrack)
    segments = audio_info["segments"]
    if not segments:
        raise RuntimeError("No segments found.")
    rms_vals = np.array([s["features"]["rms"] for s in segments])
    onset_vals = np.array([s["features"]["onset_avg"] for s in segments])
    def safe_quantiles(x, qs=(0.33,0.66)):
        if len(x) < 3:
            m = float(np.median(x)) if len(x) else 0.0
            return (m, m)
        q = np.quantile(x, qs)
        return float(q[0]), float(q[1])
    rms_q = safe_quantiles(rms_vals)
    onset_q = safe_quantiles(onset_vals)
    for s in segments:
        s["bucket"] = classify_audio_segment(s["features"], rms_q, onset_q)
    print("Scoring candidate videos…")
    scored = score_all_videos(args.videos_dir)
    bucket_index = build_bucket_index(scored)
    print("Choosing diverse clip regions…")
    chosen = choose_clips_for_segments(segments=segments, bucket_index=bucket_index, clip_strategy=args.clip_strategy, vt_step=args.vt_step, vt_max_samples=args.vt_max_samples, no_reuse=args.no_reuse, jitter=args.clip_jitter, diversity_cooldown=args.diversity_cooldown, pairing=args.pairing)
    manifest_path = Path(args.manifest) if args.manifest else out_path.with_suffix(".json")
    meta = {
        "audio_path": args.audio,
        "videos_dir": args.videos_dir,
        "output_path": str(out_path),
        "sr": audio_info["sr"],
        "audio_duration": audio_info["duration"],
        "num_segments": len(segments),
        "placements": [
            {
                "segment_start": s["start"],
                "segment_end": s["end"],
                "segment_duration": s["duration"],
                "audio_features": s["features"],
                "desired_bucket": s["bucket"],
                "chosen_clip": c["path"],
                "chosen_motion_score": c["motion_score"],
                "chosen_bucket": c["bucket"],
                "clip_start_offset": c["start"]
            }
            for s, c in zip(segments, chosen)
        ],
        "seed": 1337
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote manifest: {manifest_path}")
    print("Building ffmpeg graph…")
    cmd = build_ffmpeg_graph(audio_path=args.audio, segments=segments, chosen=chosen, fps=args.fps, width=args.width, height=args.height, crf=args.crf, preset=args.preset, out_path=str(out_path), bpm=args.bpm, pulse_strength=args.pulse_strength, trail_frames=args.trail_frames, trail_decay=args.trail_decay, trail_opacity=args.trail_opacity, trail_mode=args.trail_mode)
    print("Rendering with ffmpeg…")
    run(cmd)
    print(f"Done → {out_path}")

if __name__ == "__main__":
    main()
