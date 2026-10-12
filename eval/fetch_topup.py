"""Rebuild data/hi/meta.csv with better speaker coverage.

fake:    IndicSynth Hindi rows sampled at random offsets across the whole split through the
         Hugging Face rows API (shards are sorted by generator and speaker), max N per speaker.
genuine: Kathbath Hindi test-known, whole Hindi section streamed, max N per speaker.
IndicSynth target speakers use Kathbath speaker IDs, so speaker <id> is the same person in both.
"""
import argparse
import csv
import io
import json
import os
import subprocess
import tarfile
import time
import urllib.request
from collections import Counter

import numpy as np
import soundfile as sf

from channel import FFMPEG
from fetch_data import KATHBATH, decode_m4a, save16k

ROWS = "https://datasets-server.huggingface.co/rows?dataset=vdivyasharma/IndicSynth&config=Hindi&split=train&offset={}&length={}"
TOTAL = 205938


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # rate limits and transient errors
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def synth(out, calls, per_call, cap, rows):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1)
    per_spk = Counter()
    for c, off in enumerate(sorted(rng.choice(TOTAL - per_call, calls, replace=False))):
        try:
            batch = json.loads(get(ROWS.format(int(off), per_call)))["rows"]
        except Exception as e:
            print("skip offset", off, e, flush=True)
            continue
        for r in batch:
            row, idx = r["row"], r["row_idx"]
            spk = str(row["Target Speaker ID"])
            if per_spk[spk] >= cap:
                continue
            try:
                wav, sr = sf.read(io.BytesIO(get(row["audio"][0]["src"])), dtype="float32")
            except Exception as e:
                print("skip row", idx, e, flush=True)
                continue
            if len(wav) / sr < 2.0:
                continue
            path = os.path.join(out, f"isr_{idx:06d}.wav")
            dur = save16k(path, wav, sr)
            rows.append(dict(path=path, label="fake", source="indicsynth", generator=row["Generative Model"],
                             speaker=spk, dur=round(dur, 2)))
            per_spk[spk] += 1
        if (c + 1) % 20 == 0:
            print(f"synth calls {c + 1}/{calls}, clips {sum(per_spk.values())}, speakers {len(per_spk)}", flush=True)
    print("synth done:", Counter(r["generator"] for r in rows if r["label"] == "fake"), "speakers", len(per_spk))


def kathbath(out, cap, rows):
    os.makedirs(out, exist_ok=True)
    per_spk = Counter()
    with urllib.request.urlopen(KATHBATH) as r, tarfile.open(fileobj=r, mode="r|") as tar:
        for m in tar:
            name = m.name.replace("\\", "/")
            if not (m.isfile() and "/hindi/" in name.lower() and name.endswith(".m4a")):
                continue
            base = os.path.basename(name)[:-4]
            spk = base.split("-")[-2]
            if per_spk[spk] >= cap:
                continue
            pcm = decode_m4a(tar.extractfile(m).read())
            path = os.path.join(out, f"kb_{base}.wav")
            dur = save16k(path, np.frombuffer(pcm, np.float32), 16000)
            rows.append(dict(path=path, label="genuine", source="kathbath", generator="human", speaker=spk, dur=round(dur, 2)))
            per_spk[spk] += 1
    print("kathbath done:", sum(per_spk.values()), "clips,", len(per_spk), "speakers", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hi")
    ap.add_argument("--calls", type=int, default=160)
    ap.add_argument("--per-call", type=int, default=6)
    ap.add_argument("--cap", type=int, default=12, help="max clips per speaker per class")
    a = ap.parse_args()
    rows = []
    synth(os.path.join(a.out, "fake_r"), a.calls, a.per_call, a.cap, rows)
    kathbath(os.path.join(a.out, "genuine"), a.cap * 2, rows)
    with open(os.path.join(a.out, "meta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("rows:", len(rows))
