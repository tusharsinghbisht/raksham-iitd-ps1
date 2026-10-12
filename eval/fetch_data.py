"""Download the Hindi evaluation set.

genuine: Kathbath (AI4Bharat, IndicSUPERB) clean test-known, Hindi only, streamed from the tar.
fake:    IndicSynth (ACL 2025) Hindi, a few parquet shards, sampled evenly per generative model.
Writes 16 kHz mono wavs to data/hi/{genuine,fake}/ and data/hi/meta.csv.
"""
import argparse
import csv
import io
import os
import subprocess
import tarfile
import tempfile
import urllib.request
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

from channel import FFMPEG

KATHBATH = "https://objectstore.e2enetworks.net/indic-superb/kathbath/clean/testkn_audio.tar"
SYNTH = "vdivyasharma/IndicSynth"


def save16k(path, wav, sr):
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g)
    sf.write(path, wav.astype(np.float32), 16000, subtype="PCM_16")
    return len(wav) / 16000


def decode_m4a(data):
    """m4a often keeps its index at the end, so ffmpeg needs a seekable file, not a pipe."""
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
        f.write(data)
    try:
        return subprocess.run([FFMPEG, "-v", "error", "-i", f.name, "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
                              capture_output=True, check=True).stdout
    finally:
        os.remove(f.name)


def kathbath(out, limit, rows):
    os.makedirs(out, exist_ok=True)
    n = 0
    with urllib.request.urlopen(KATHBATH) as r, tarfile.open(fileobj=r, mode="r|") as tar:
        for m in tar:
            name = m.name.replace("\\", "/")
            if not (m.isfile() and "/hindi/" in name.lower() and name.endswith(".m4a")):
                continue
            data = tar.extractfile(m).read()
            pcm = decode_m4a(data)
            wav = np.frombuffer(pcm, np.float32)
            base = os.path.basename(name)[:-4]
            # Kathbath file names look like <utt>-<speaker>-<gender>.m4a
            parts = base.split("-")
            spk = parts[-2] if len(parts) >= 3 else base
            path = os.path.join(out, f"kb_{base}.wav")
            dur = save16k(path, wav, 16000)
            rows.append(dict(path=path, label="genuine", source="kathbath", generator="human", speaker=f"kb{spk}", dur=round(dur, 2)))
            n += 1
            if n % 200 == 0:
                print("kathbath", n, flush=True)
            if n >= limit:
                break
    print("kathbath done:", n)


def indicsynth(out, raw, shards, per_model, rows):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    taken = defaultdict(int)
    quota = per_model // len(shards) + 1
    for s in shards:
        f = hf_hub_download(SYNTH, f"Hindi/train-{s:05d}-of-00107.parquet", repo_type="dataset", local_dir=raw)
        t = pq.read_table(f)
        cols = t.column_names
        gen = t.column("Generative Model").to_pylist()
        spk = t.column("Target Speaker ID").to_pylist()
        print(f"shard {s}: {t.num_rows} rows, models={dict(zip(*np.unique(gen, return_counts=True)))}", flush=True)
        audio = t.column(cols[0]) if "audio" not in cols else t.column("audio")
        by_model = defaultdict(list)
        for i, g in enumerate(gen):
            by_model[g].append(i)
        for g, idx in by_model.items():
            pick = rng.permutation(idx)[:min(quota, per_model - taken[g])]
            for i in pick:
                a = audio[int(i)].as_py()
                wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
                if len(wav) / sr < 2.0:
                    continue
                path = os.path.join(out, f"is_{s:03d}_{int(i):06d}.wav")
                dur = save16k(path, wav, sr)
                rows.append(dict(path=path, label="fake", source="indicsynth", generator=g, speaker=f"is{spk[int(i)]}", dur=round(dur, 2)))
                taken[g] += 1
        del t
    print("indicsynth done:", dict(taken))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hi")
    ap.add_argument("--genuine", type=int, default=1500)
    ap.add_argument("--per-model", type=int, default=500)
    ap.add_argument("--shards", default="0,35,70,106")
    a = ap.parse_args()
    rows = []
    indicsynth(os.path.join(a.out, "fake"), os.path.join(a.out, "_raw"), [int(x) for x in a.shards.split(",")], a.per_model, rows)
    kathbath(os.path.join(a.out, "genuine"), a.genuine, rows)
    with open(os.path.join(a.out, "meta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("rows:", len(rows))
