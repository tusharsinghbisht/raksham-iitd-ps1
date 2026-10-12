"""Append Kathbath Hindi genuine clips from another split (default: clean test-unknown) to a meta csv."""
import csv, os, sys, subprocess, tarfile, urllib.request
from collections import Counter
import numpy as np
from channel import FFMPEG
from fetch_data import decode_m4a, save16k

url = sys.argv[1] if len(sys.argv) > 1 else "https://objectstore.e2enetworks.net/indic-superb/kathbath/clean/testunk_audio.tar"
meta, out, cap = "data/hi/meta.csv", "data/hi/genuine_extra", 24
os.makedirs(out, exist_ok=True)
rows, per = [], Counter()
with urllib.request.urlopen(url) as r, tarfile.open(fileobj=r, mode="r|") as tar:
    for m in tar:
        n = m.name.replace("\\", "/")
        if not (m.isfile() and "/hindi/" in n.lower() and n.endswith(".m4a")):
            continue
        base = os.path.basename(n)[:-4]
        spk = base.split("-")[-2]
        if per[spk] >= cap:
            continue
        pcm = decode_m4a(tar.extractfile(m).read())
        path = os.path.join(out, f"kb_{base}.wav")
        dur = save16k(path, np.frombuffer(pcm, np.float32), 16000)
        rows.append(dict(path=path, label="genuine", source="kathbath", generator="human", speaker=spk, dur=round(dur, 2)))
        per[spk] += 1
with open(meta, "a", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=list(rows[0])).writerows(rows)
print("kathbath extra done:", len(rows), "clips,", len(per), "speakers")
