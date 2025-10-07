import argparse, json, math, subprocess, shutil, random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import deque, defaultdict

import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d
import cv2
from tqdm import tqdm

# ---------- Shell Helpers ----------
def run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}):\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr[:8000]}")

def has_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found. Install ffmpeg (sudo apt install ffmpeg).")

def ffprobe_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1",
        path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(p.stdout.strip())
    except Exception:
        return 0.0

# ---------- Onset Tools ----------
def superflux_onset_env(y: np.ndarray, sr: int, hop: int, n_fft: int,
                        n_mels: int = 128, smooth_frames: int = 3) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, power=1.0)
    S = np.log1p(S)
    D = np.diff(S, axis=1)
    D = np.maximum(D, 0.0)

    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)
    def band_mask(lo, hi):
        return (mel_freqs >= lo) & (mel_freqs < hi)

    low  = D[band_mask(20, 200), :]
    mid  = D[band_mask(200, 1500), :]
    high = D[band_mask(1500, 8000), :]

    env = (0.9*np.mean(low, axis=0, keepdims=True) if low.size else 0) + \
          (1.2*np.mean(mid, axis=0, keepdims=True) if mid.size else 0) + \
          (1.1*np.mean(high, axis=0, keepdims=True) if high.size else 0)
    env = env.ravel()
    if smooth_frames > 1:
        env = uniform_filter1d(env, size=smooth_frames)
    return np.pad(env, (1,0), mode="edge")

def detect_onsets_percussive(y: np.ndarray, sr: int, hop: int, n_fft: int,
                             delta: float, pre_max: int, post_max: int,
                             pre_avg: int, post_avg: int, backtrack: bool) -> np.ndarray:
    y_harm, y_perc = librosa.effects.hpss(y, margin=(1.0, 2.0))
    env = superflux_onset_env(y_perc, sr, hop, n_fft)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=env,
        sr=sr,
        hop_length=hop,
        units="frames",
        pre_max=pre_max, post_max=post_max,
        pre_avg=pre_avg, post_avg=post_avg,
        delta=delta,
        backtrack=backtrack
    )
    return onset_frames

def choose_segments(y: np.ndarray, sr: int, hop: int, min_seg: float,
                    mode: str, onset_kwargs: Dict) -> Tuple[List[Tuple[float,float]], float]:
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

def analyze_audio_segments(audio_path: str,
                           min_seg: float = 0.35,
                           hop: int = 256,
                           percussive_boost: float = 1.0,
                           segment_mode: str = "auto",
                           onset_delta: float = 0.08,
                           onset_pre_max: int = 12,
                           onset_post_max: int = 12,
                           onset_pre_avg: int = 50,
                           onset_post_avg: int = 50,
                           onset_backtrack: bool = True) -> Dict:
    y, sr = librosa.load(audio_path, sr=44100, mono=True)

    if percussive_boost > 0:
        y_h, y_p = librosa.effects.hpss(y, margin=(1.0, 2.0))
        y = librosa.util.normalize((1.0 - percussive_boost)*y + percussive_boost*y_p)

    onset_kwargs = dict(
        delta=onset_delta,
        pre_max=onset_pre_max,
        post_max=onset_post_max,
        pre_avg=onset_pre_avg,
        post_avg=onset_post_avg,
        backtrack=onset_backtrack
    )
    segments, duration = choose_segments(y, sr, hop, min_seg, segment_mode, onset_kwargs)

    rms = librosa.feature.rms(y=y, hop_length=hop, center=True)[0]
    cen = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop, center=True)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop, center=True)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, center=True)

    rms_s = uniform_filter1d(rms, size=3)
    cen_s = uniform_filter1d(cen, size=3)
    zcr_s = uniform_filter1d(zcr, size=3)
    on_s  = uniform_filter1d(onset_env, size=3)

    n = min(len(rms_s), len(cen_s), len(zcr_s), len(on_s))
    rms_s, cen_s, zcr_s, on_s = (a[:n] for a in (rms_s, cen_s, zcr_s, on_s))
    hop_times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)

    out = []
    for s, e in segments:
        mask = (hop_times >= s) & (hop_times < e)
        def avg(a):
            v = a[mask]; return float(np.mean(v)) if v.size else 0.0
        out.append({
            "start": float(s),
            "end": float(e),
            "duration": float(e - s),
            "features": {
                "rms": avg(rms_s),
                "centroid": avg(cen_s),
                "zcr": avg(zcr_s),
                "onset_avg": avg(on_s)
            }
        })
    return {"sr": sr, "duration": duration, "segments": out}

# ---------- Motion Scoring ----------
def motion_score_video(path: str, sample_every: int = 10, max_samples: int = 300) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    scores, prev_gray, taken = [], None, 0
    for idx in range(0, length, sample_every):
        if taken >= max_samples: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok: break
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

# ---------- Timeline Analysis For Diversity ----------
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
        frame_idx = int(round(t * (fps if fps > 0 else frames / dur)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_idx, frames-1)))
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
    if energy >= rms_q[1] or trans >= onset_q[1]: return "fast"
    if energy <= rms_q[0] and trans <= onset_q[0]: return "slow"
    return "medium"

def build_bucket_index(videos_scored: List[Dict]) -> Dict[str, List[Dict]]:
    idx = {"slow": [], "medium": [], "fast": []}
    for v in videos_scored:
        idx[v["motion_bucket"]].append(v)
    for k in idx:
        idx[k] = sorted(idx[k], key=lambda d: d["motion_score"])
    return idx

# ---------- Region Selection For Diversity ----------
def pick_region_by_timeline(timeline: Dict[str, np.ndarray], target_dur: float, bucket: str,
                            no_reuse: float, used_windows: List[Tuple[float,float]],
                            jitter: float) -> float:
    times = timeline["times"]; scores = timeline["scores"]
    if times.size == 0:
        return 0.0

    if bucket == "fast":
        order = np.argsort(-scores)
    elif bucket == "slow":
        order = np.argsort(scores)
    else:
        lo, hi = np.quantile(scores, [0.33, 0.66])
        mid_mask = (scores >= lo) & (scores <= hi)
        mid_idxs = np.nonzero(mid_mask)[0]
        if mid_idxs.size == 0:
            order = np.argsort(np.abs(scores - (lo+hi)/2.0))
        else:
            order = mid_idxs

    for idx in order[: max(50, int(0.2*len(order)))]:
        start = float(times[idx])
        start = max(0.0, start)
        too_close = any((start < (s2 + no_reuse) and (start + target_dur) > (s2 - no_reuse)) for s2, _ in used_windows)
        if too_close:
            continue
        if jitter > 0:
            start = max(0.0, start + random.uniform(-jitter, jitter))
        return start
    dur_est = float(times[-1] if times.size else target_dur*2)
    for _ in range(20):
        start = random.uniform(0.0, max(0.0, dur_est - target_dur))
        too_close = any((start < (s2 + no_reuse) and (start + target_dur) > (s2 - no_reuse)) for s2, _ in used_windows)
        if not too_close:
            return start
    return 0.0

def choose_clips_for_segments(
    segments: List[Dict],
    bucket_index: Dict[str, List[Dict]],
    clip_strategy: str = "timeline",
    vt_step: float = 0.25,
    vt_max_samples: int = 1200,
    no_reuse: float = 1.25,
    jitter: float = 0.15,
    diversity_cooldown: int = 3,
) -> List[Dict]:
    video_meta: Dict[str, Dict] = {}
    def ensure_meta(vpath: str):
        if vpath in video_meta:
            return video_meta[vpath]
        tl = motion_timeline(vpath, step_s=vt_step, max_samples=vt_max_samples) if clip_strategy == "timeline" else {"times": np.array([]), "scores": np.array([])}
        video_meta[vpath] = {
            "dur": ffprobe_duration(vpath),
            "timeline": tl,
            "used": []
        }
        return video_meta[vpath]
    
    recent = deque(maxlen=max(1, diversity_cooldown)) if diversity_cooldown > 0 else None
    
    use_count = defaultdict(int)

    def pick_with_diversity(candidates: List[Dict], bucket: str, rr_ptr: Dict[str, int]) -> Dict:
        if not candidates:
            raise RuntimeError("No scored videos available to choose from.")

        start = rr_ptr[bucket] % len(candidates)
        ordered = candidates[start:] + candidates[:start]

        if recent is None or len(recent) == 0:
            chosen = ordered[0]
        else:
            chosen = None
            for cand in ordered:
                if cand["path"] not in recent:
                    chosen = cand
                    break

            if chosen is None:
                chosen = min(
                    ordered,
                    key=lambda c: (use_count[c["path"]], ordered.index(c))
                )

        rr_ptr[bucket] += 1
        use_count[chosen["path"]] += 1
        if recent is not None:
            recent.append(chosen["path"])

        return chosen


    ptr = {"slow":0,"medium":0,"fast":0}
    chosen = []
    rng = random.Random(1337) 
    for s in segments:
        bucket = s["bucket"]
        lst = bucket_index.get(bucket) or bucket_index.get("medium") or sum(bucket_index.values(), [])
        if not lst:
            raise RuntimeError("No scored videos available to choose from.")
        pick = pick_with_diversity(lst, bucket, ptr)


        meta = ensure_meta(pick["path"])
        target = s["duration"]
        start = 0.0

        if clip_strategy == "timeline" and meta["timeline"]["times"].size > 0:
            start = pick_region_by_timeline(meta["timeline"], target, bucket, no_reuse, meta["used"], jitter)
        else:
            if meta["dur"] > target:
                start = rng.uniform(0.0, max(0.0, meta["dur"] - target))
            else:
                start = 0.0

        meta["used"].append((start, start + target))
        chosen.append({
            "path": pick["path"],
            "motion_score": pick["motion_score"],
            "bucket": pick["motion_bucket"],
            "start": float(max(0.0, start)),
            "src_dur": float(meta["dur"])
        })
    return chosen

# ---------- FFMPEG Edit Graph ----------
def build_ffmpeg_graph(audio_path: str, segments: List[Dict], chosen: List[Dict],
                       fps: int, width: int, height: int, crf: int, preset: str, out_path: str,
                       bpm: float = 0.0, pulse_strength: float = 0.06,
                       trail_frames: int = 6, trail_decay: float = 0.70,
                       trail_opacity: float = 0.35, trail_mode: str = "screen"):

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
        start  = float(ch.get("start", 0.0))
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
        if "left" in style:  pan_frac_x = -0.02
        if "right" in style: pan_frac_x =  0.02
        if "up" in style:    pan_frac_y = -0.02
        if "down" in style:  pan_frac_y =  0.02

        x_expr = f"(iw-iw/zoom)/2+({pan_frac_x})*iw*{prog}"
        y_expr = f"(ih-ih/zoom)/2+({pan_frac_y})*ih*{prog}"

        base = f"[{i}:v]"

        chain = base + f"trim=start={max(0.0, start):.6f}"
        tail = max(0.0, srcdur - start)
        if tail >= target - 1e-4:
            chain += f":duration={target:.6f},setpts=PTS-STARTPTS"
        else:
            pad_dur = max(0.0, target - tail)
            chain += f":duration={tail:.6f},setpts=PTS-STARTPTS," \
                     f"tpad=stop_mode=clone:stop_duration={pad_dur:.6f}"

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
        "-map", "[vout]",
        "-map", "0:a:0",
        "-shortest",
        "-r", str(fps),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        out_path
    ]
    return cmd

# ---------- Pipeline ----------
def main():
    ap = argparse.ArgumentParser(description="Beat/onset-synced visual editor using ffmpeg (no MoviePy).")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--segment-mode", choices=["auto","onsets","bars"], default="auto",
                    help="auto prefers onsets if stable; 'onsets' is most drum-sensitive.")
    ap.add_argument("--min-seg", type=float, default=0.35, help="Minimum segment duration (s).")
    ap.add_argument("--hop", type=int, default=256, help="Analysis hop length (smaller = finer timing).")
    ap.add_argument("--percussive-boost", type=float, default=1.0, help="0..1 percussive mix weight.")
    ap.add_argument("--onset-delta", type=float, default=0.08, help="Peak-pick delta (lower = more onsets).")
    ap.add_argument("--onset-pre-max", type=int, default=12)
    ap.add_argument("--onset-post-max", type=int, default=12)
    ap.add_argument("--onset-pre-avg", type=int, default=50)
    ap.add_argument("--onset-post-avg", type=int, default=50)
    ap.add_argument("--onset-backtrack", action="store_true", help="Backtrack onsets to nearest preceding minimum.")
    ap.add_argument("--clip-strategy", choices=["timeline","random"], default="timeline",
                    help="timeline = pick start offsets guided by motion timeline per video; random = uniform offset.")
    ap.add_argument("--vt-step", type=float, default=0.25, help="Seconds between motion samples for timeline.")
    ap.add_argument("--vt-max-samples", type=int, default=1200, help="Cap timeline samples per video.")
    ap.add_argument("--clip-jitter", type=float, default=0.15, help="Random jitter (s) around chosen start.")
    ap.add_argument("--no-reuse", type=float, default=1.25, help="Avoid reusing regions within this many seconds.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--bpm", type=float, default=0.0,
                    help="If > 0, use this BPM for beat-synced color pulsing; if 0, pulsing is disabled.")
    ap.add_argument("--pulse-strength", type=float, default=0.06,
                    help="Base saturation pulse amplitude (e.g., 0.06 => ±6%%). Set 0 to disable.")
    ap.add_argument("--trail-frames", type=int, default=6,
                    help="Number of frames to accumulate for motion trails (<=1 disables).")
    ap.add_argument("--trail-decay", type=float, default=0.70,
                    help="Exponential decay per frame for trail weights (0..1).")
    ap.add_argument("--trail-opacity", type=float, default=0.35,
                    help="Blend opacity of the trail overlay (0..1). 0 disables.")
    ap.add_argument("--trail-mode", choices=["screen","lighten","addition","overlay"], default="screen",
                    help="Blend mode used for overlaying the trails.")
    ap.add_argument("--diversity-cooldown", type=int, default=3,
                help="Avoid reusing the same source file within this many subsequent segments. 0 disables.")
    
    args = ap.parse_args()

    has_ffmpeg()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Analyzing audio (drum-sensitive onsets)…")
    audio_info = analyze_audio_segments(
        args.audio,
        min_seg=args.min_seg,
        hop=args.hop,
        percussive_boost=args.percussive_boost,
        segment_mode=args.segment_mode,
        onset_delta=args.onset_delta,
        onset_pre_max=args.onset_pre_max,
        onset_post_max=args.onset_post_max,
        onset_pre_avg=args.onset_pre_avg,
        onset_post_avg=args.onset_post_avg,
        onset_backtrack=args.onset_backtrack
    )
    segments = audio_info["segments"]
    if not segments:
        raise RuntimeError("No segments found.")
    rms_vals   = np.array([s["features"]["rms"]        for s in segments])
    onset_vals = np.array([s["features"]["onset_avg"]  for s in segments])
    rms_q = (float(np.quantile(rms_vals, 0.33)), float(np.quantile(rms_vals, 0.66)))
    onset_q = (float(np.quantile(onset_vals, 0.33)), float(np.quantile(onset_vals, 0.66)))
    for s in segments:
        s["bucket"] = classify_audio_segment(s["features"], rms_q, onset_q)

    print("Scoring candidate videos…")
    scored = score_all_videos(args.videos_dir)
    bucket_index = build_bucket_index(scored)

    print("Choosing diverse clip regions…")
    chosen = choose_clips_for_segments(
        segments=segments,
        bucket_index=bucket_index,
        clip_strategy=args.clip_strategy,
        vt_step=args.vt_step,
        vt_max_samples=args.vt_max_samples,
        no_reuse=args.no_reuse,
        jitter=args.clip_jitter,
        diversity_cooldown=args.diversity_cooldown,
    )

    manifest_path = Path(args.manifest) if args.manifest else out_path.with_suffix(".json")
    meta = {
        "audio_path": args.audio,
        "videos_dir": args.videos_dir,
        "output_path": str(out_path),
        "sr": audio_info["sr"],
        "audio_duration": audio_info["duration"],
        "num_segments": len(segments),
        "rms_q": rms_q,
        "onset_q": onset_q,
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
        ]
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote manifest: {manifest_path}")

    print("Building ffmpeg graph…")
    cmd = build_ffmpeg_graph(
        audio_path=args.audio,
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
        trail_mode=args.trail_mode
    )


    print("Rendering with ffmpeg…")
    run(cmd)
    print(f"Done → {out_path}")

if __name__ == "__main__":
    main()
