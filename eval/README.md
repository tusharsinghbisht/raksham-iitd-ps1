# Detector evaluation

Measures [`nii-yamagishilab/wav2vec-small-anti-deepfake`](https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake) on phone-quality Hindi and on our own voices. Results: [`RESULTS.md`](RESULTS.md). Model details: [`../docs/model_architecture.md`](../docs/model_architecture.md).

| Data | Role | Licence |
|---|---|---|
| Kathbath Hindi, clean test-known + test-unknown (AI4Bharat, IndicSUPERB) | real speech | CC BY 4.0 |
| IndicSynth Hindi (Sharma et al., ACL 2025): XTTS-v2 and FreeVC clones | cloned speech | CC BY-NC 4.0 |
| Team recordings, [`../data_collection/raw`](../data_collection) | real speech, false-alarm check | CC BY-NC 4.0, see data README |

The model card lists its training corpora. FLEURS (which includes Hindi) and MLAAD are on that list, so we don't test on them. Kathbath and IndicSynth are not. Downloaded audio is not committed; the scripts fetch it.

## Files

| File | What it does |
|---|---|
| `model.py` | wav2vec 2.0 Base + linear head in plain PyTorch, with fairseq's parameter names so the published checkpoint loads strictly. `score()` returns z = fake logit − real logit |
| `channel.py` | Phone-channel simulation: 8 kHz band-limit, G.711 µ-law, AMR-NB 12.2k and Opus 12k (via ffmpeg), packet loss with simple concealment, additive noise |
| `fetch_data.py` | Shared helpers (m4a decoding, resampling) and a first-pass sampler |
| `fetch_topup.py` | Builds `data/hi/meta.csv`: IndicSynth rows at random offsets (the shards are sorted by tool and speaker), Kathbath test-known, capped per speaker |
| `fetch_kb_extra.py` | Appends Kathbath test-unknown Hindi speakers |
| `run_eval.py` | Cuts team recordings into sentences, applies every channel to every clip, writes `results/scores.csv` |
| `metrics.py` | EER with speaker-clustered bootstrap CIs, τ at 1% dev false alarms, test rates at τ (Wilson CIs), same-person check |
| `make_appendix.py` | Regenerates the submission's appendix page from `results/` |

## Run

Python 3.11, CUDA optional (CPU works, about 15× slower). From this folder, in PowerShell:

```
uv venv -p 3.11 .venv
uv pip install -p .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -p .venv\Scripts\python.exe numpy scipy soundfile safetensors huggingface_hub pyarrow pandas imageio-ffmpeg pypdf
$env:HF_HOME = "$PWD\.hf"

.venv\Scripts\python fetch_topup.py      # ~800 IndicSynth clips via the HF rows API + Kathbath stream (3.6 GB, not stored)
.venv\Scripts\python fetch_kb_extra.py   # + Kathbath test-unknown (2 GB stream)
.venv\Scripts\python run_eval.py         # ~20 min on an RTX 3050
.venv\Scripts\python metrics.py
```

`run_eval.py --limit 20` is a quick smoke test.

## Method

- **Channels:** clean 16 kHz, 8 kHz narrowband, G.711, G.711 + 3% packet loss + 20 dB noise, AMR-NB 12.2k, Opus 12k VoIP.
- **Scores per clip:** first 4 s window, median over 4 s windows with a 2 s hop (the live pipeline's call-level rule), and the whole clip up to 12 s.
- **Split:** by speaker hash, 30% dev / 70% test. IndicSynth clones Kathbath speakers and shares their IDs, so a person's real and cloned clips always land on the same side.
- **Threshold:** τ = 99th percentile of dev real-speech scores (1% false alarms), frozen, then applied to test and team clips.

## Limitations

- Real and cloned clips come from different corpora (Kathbath: phone-app recordings, AAC; IndicSynth: synthesised). Part of the separation may come from that. The same-person check gives similar numbers, which helps but doesn't rule it out.
- Hindi IndicSynth has only two cloning tools (no VITS).
- Two team speakers is a sanity check, not a statistic.
