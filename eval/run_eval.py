"""Score every clip under every phone-channel condition.

Writes scores.csv with one row per (clip, condition):
  z_win  first 4 s window          (single-window decision)
  z_call median of all 4 s windows (EchoLock call-level decision)
  z_full whole clip, capped at 12 s (model-card style)
"""
import argparse
import csv
import glob
import hashlib
import os
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

import channel
import model as M


def split_of(speaker, dev_share=0.3):
    h = int(hashlib.sha1(speaker.encode()).hexdigest(), 16) % 1000
    return "dev" if h < dev_share * 1000 else "test"


def segment_team(src, out, min_sil=0.6, min_len=1.5):
    """Cut each team recording into sentences at the ~2 s pauses the script asked for."""
    os.makedirs(out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(src, "S*", "*.wav"))):
        wav, sr = sf.read(f, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = resample_poly(wav, 160, sr // 100).astype(np.float32) if sr != 16000 else wav
        fr = 320  # 20 ms
        e = np.array([np.sqrt(np.mean(wav[i:i + fr] ** 2)) for i in range(0, len(wav) - fr, fr)])
        thr = max(np.percentile(e, 20) * 3, 1e-4)
        voiced = e > thr
        segs, start, sil = [], None, 0
        for i, v in enumerate(voiced):
            if v:
                start = i if start is None else start
                sil = 0
            elif start is not None:
                sil += 1
                if sil * 0.02 >= min_sil:
                    segs.append((start, i - sil + 1))
                    start, sil = None, 0
        if start is not None:
            segs.append((start, len(voiced)))
        name = os.path.basename(f)[:-4]
        spk, rec = name.split("_", 1)
        for k, (a, b) in enumerate(segs):
            a, b = max(0, a * fr - 3200), min(len(wav), b * fr + 3200)  # keep 0.2 s of context
            if (b - a) / 16000 < min_len:
                continue
            path = os.path.join(out, f"{name}_{k:02d}.wav")
            sf.write(path, wav[a:b], 16000, subtype="PCM_16")
            rows.append(dict(path=path, label="genuine", source="team", generator=f"human_{rec}", speaker=spk, dur=round((b - a) / 16000, 2)))
    return rows


@torch.no_grad()
def scores(det, wav, dev):
    w = M.windows(wav)
    x = torch.as_tensor(np.stack([np.pad(v, (0, 64000 - len(v))) if len(v) < 64000 else v for v in w]), device=dev)
    x = F.layer_norm(x, x.shape[-1:])
    lg = det(x)
    z = (lg[:, 0] - lg[:, 1]).float().cpu().numpy()
    return float(z[0]), float(np.median(z)), M.score(det, wav[:12 * 16000], dev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/hi/meta.csv")
    ap.add_argument("--team", default=r"D:\raksham_iitd\data_collection\raw")
    ap.add_argument("--out", default="results/scores.csv")
    ap.add_argument("--limit", type=int, default=0, help="score only N clips per class (smoke test)")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.meta, encoding="utf-8"))) if os.path.exists(a.meta) else []
    if a.team and os.path.isdir(a.team):
        rows += segment_team(a.team, "data/team_segments")
    if a.limit:
        keep, cnt = [], {}
        for r in rows:
            k = (r["source"], r["label"])
            if cnt.get(k, 0) < a.limit:
                keep.append(r)
                cnt[k] = cnt.get(k, 0) + 1
        rows = keep
    for r in rows:
        r["split"] = "team" if r["source"] == "team" else split_of(r["speaker"])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = M.load(dev)
    conds = channel.conditions()
    print(f"{len(rows)} clips x {len(conds)} conditions on {dev}: {list(conds)}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source", "generator", "speaker", "split", "dur", "condition", "z_win", "z_call", "z_full"])
        for i, r in enumerate(rows):
            wav, sr = sf.read(r["path"], dtype="float32")
            if len(wav) < 16000:  # under 1 s (e.g. a failed decode): nothing to score
                continue
            for c, fn in conds.items():
                rng = np.random.default_rng(int(hashlib.md5((r["path"] + c).encode()).hexdigest()[:8], 16))
                y = fn(wav, rng)
                w.writerow([r["path"], r["label"], r["source"], r["generator"], r["speaker"], r["split"], r["dur"], c,
                            *(round(v, 4) for v in scores(det, y, dev))])
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    print("done", a.out)
