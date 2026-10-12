# EchoLock detector: model architecture

EchoLock's Detect layer uses **Wav2Vec2-Small-AntiDeepfake** (`nii-yamagishilab/wav2vec-small-anti-deepfake`), released by the Yamagishi Lab at NII Japan. It is a wav2vec 2.0 Base speech encoder, post-trained by NII to tell real speech from synthetic speech, with a one-layer classifier on top.

Our loader is `eval/model.py`. It rebuilds the network in plain PyTorch with the same parameter names as the original fairseq code, so the published weights load with a strict check: every weight must match, or loading fails.

---

## 1. Summary

| Item | Value |
|---|---|
| Input | 16 kHz mono waveform, any length. Normalised to zero mean and unit variance per clip |
| Backbone | wav2vec 2.0 Base: 7-layer CNN feature encoder + 12-layer Transformer |
| Head | Mean over time, then Linear 768 → 2 (fake, real) |
| Output | Two logits. EchoLock uses **z = logit_fake − logit_real**; P(fake) = sigmoid(z) |
| Parameters | **94.37 M** (CNN 4.20 M, Transformer and projection 90.17 M, head 1,538) |
| Frame rate | One feature vector per **20 ms** (stride 320 samples); each sees 25 ms of audio |
| Latency, one 4 s window (measured by us) | **22 ms p50 / 25 ms p95** on an RTX 3050 Laptop GPU · **299 ms p50 / 354 ms p95** on a Ryzen 7 4800H CPU |
| Licence | CC BY-NC-SA 4.0: research and education only |

Latency was measured with PyTorch 2.6 (fp32, batch 1) over 50 GPU and 30 CPU runs, while a download was running in the background. EchoLock scores one window every 2 s, so even on CPU the model uses about 15% of real time.

---

## 2. Data flow, layer by layer

Shapes are for one 4 s window (64,000 samples). B is batch, T is time.

```
waveform [B, 64000]
  │  layer_norm over the whole clip          (zero mean, unit variance)
  ▼
CNN feature encoder (7 × Conv1d, 512 channels)
  │  conv1  k=10 s=5  → GroupNorm → GELU     [B, 512, 12799]
  │  conv2  k=3  s=2  → GELU                 [B, 512,  6399]
  │  conv3  k=3  s=2  → GELU                 [B, 512,  3199]
  │  conv4  k=3  s=2  → GELU                 [B, 512,  1599]
  │  conv5  k=3  s=2  → GELU                 [B, 512,   799]
  │  conv6  k=2  s=2  → GELU                 [B, 512,   399]
  │  conv7  k=2  s=2  → GELU                 [B, 512,   199]
  ▼  transpose                               [B, 199, 512]
LayerNorm(512) → Linear 512→768              [B, 199, 768]
  ▼
Positional convolution (added to the input)
  │  Conv1d 768→768, k=128, 16 groups, weight-normalised, GELU
  ▼  + residual → LayerNorm(768)
Transformer encoder, 12 layers (post-LN)
  │  each layer:  x = LN(x + MultiHeadAttention(x))     12 heads × 64 dims
  │               x = LN(x + Linear(GELU(Linear(x))))   768 → 3072 → 768
  ▼                                          [B, 199, 768]
Mean over time                               [B, 768]
  ▼
Linear 768 → 2                               [B, 2] = (fake logit, real logit)
```

### 2.1 Input normalisation
The waveform is layer-normalised over its full length, so loudness and DC offset don't change the score. This matters for calls: agent-side gain control and different handsets change level a lot.

### 2.2 CNN feature encoder (4.2 M parameters)
Seven 1-D convolutions turn raw samples into one 512-dim vector per 20 ms. The total stride is 5·2⁶ = 320 samples and the receptive field is 400 samples (25 ms), much like a classic spectrogram frame. Unlike a fixed spectrogram, the filters are learned, so they can pick up phase and fine waveform detail. That detail is where vocoders and voice converters leave traces. Only the first layer has a normalisation (GroupNorm with one group per channel). This is fairseq's "default" mode for Base models.

### 2.3 Feature projection
A LayerNorm and a linear layer lift the 512-dim CNN features to the Transformer width of 768.

### 2.4 Convolutional positional embedding
wav2vec 2.0 has no fixed position table. Instead, a wide grouped convolution (kernel 128 frames ≈ 2.56 s, 16 groups) mixes each frame with its neighbours and adds the result back. This gives relative position information and works for any clip length. Its weights are stored in weight-normalised form (`weight_g` × `weight_v` / ‖`weight_v`‖, norm over dimensions 0 and 1). Because the kernel is even, one output frame is dropped so the length stays T.

### 2.5 Transformer encoder (85 M parameters)
Twelve identical blocks with 12-head self-attention (64 dims per head) and a 3072-wide feed-forward layer. They use post-LayerNorm, as wav2vec 2.0 Base does: the norm comes after each residual add. Attention lets every 20 ms frame compare itself with the whole window. This is how the model notices artifacts that only show over time: too-regular pitch, smoothed breaths, or a voice timbre that doesn't fit its prosody.

### 2.6 Pooling and classifier (1,538 parameters)
Adaptive average pooling takes the mean of the 199 frame vectors, and a single linear layer maps it to two logits. Almost all the detection ability lives in the backbone. The head is only a linear read-out.

### 2.7 Weights in the checkpoint that we don't use
The checkpoint still contains wav2vec 2.0 pre-training modules: the quantiser, `project_q`, `final_proj` and `mask_emb`. They are only used for the self-supervised pre-training objective. Our loader drops them explicitly and fails if anything else is missing or unexpected.

---

## 3. How it was trained

| Stage | What happened | Data |
|---|---|---|
| 1. Self-supervised pre-training (Meta, `facebook/wav2vec2-base`) | Masked contrastive learning: predict quantised versions of masked 20 ms frames. No labels | LibriSpeech 960 h, English read speech |
| 2. Post-training for deepfake detection (NII AntiDeepfake) | **The whole network**, CNN, Transformer and head, trained with cross-entropy on real vs fake. AdamW, learning rate 1e-7, dynamic batches of up to 100 s of audio | **74k h**: 56k h real + 18k h fake, 100+ languages, from 29 corpora (ASVspoof 2019/2021/5, MLAAD, CodecFake, FLEURS(-R), MLS, VoxCeleb2, SpoofCeleb and others) |
| Augmentation | RawBoost (1)+(2): linear and non-linear convolutive noise plus impulsive signal-dependent noise. These mimic channel and device distortion | applied during post-training |

Two things matter for EchoLock:
- **The base was pre-trained only on English.** The large multilingual post-training set is what makes it usable on Hindi. That is why we measure Hindi directly, rather than assume it works.
- **Test-data hygiene.** FLEURS (which includes Hindi) and MLAAD are in the training set, so we never test on them. IndicSynth and Kathbath are not, which is why `eval/` uses those.

---

## 4. Accuracy

### Published (NII model card, clean audio)

| Test set | EER | Note |
|---|---|---|
| In-the-Wild | **4.24%** | real vs deepfake speech of public figures |
| ADD 2023 | 13.02% | Chinese |
| Deepfake-Eval-2024 | **33.33%** | newer, harder, in-the-wild deepfakes |
| Deepfake-Eval-2024 after fine-tuning on its train set | 17.72% (4 s input) · 12.22% (30 s) | longer input helps |

EER (equal error rate) is the operating point where the false-alarm rate equals the miss rate. The jump from 4% to 33% on newer fakes is why EchoLock treats the score as **evidence**, and makes app confirmation the safety guarantee. The longer-input result also supports our call-level median over several windows.

### Our measurement: Hindi, phone channels (`eval/RESULTS.md`)

Kathbath real speech vs IndicSynth clones (XTTS-v2, FreeVC). Speaker-disjoint test set of 960 clips from 79 speakers. Call-level score.

| Channel | EER [95% CI] | τ (1% dev false alarms) | Detection @τ | False alarms @τ |
|---|---|---|---|---|
| Clean 16 kHz | 1.6% [0.8–2.7] | 1.13 | 95.8% | 0.7% |
| G.711 8 kHz | 3.4% [2.1–5.0] | −2.27 | 96.6% | 3.9% |
| Opus 12 kbps | 5.9% [3.5–8.5] | −1.57 | 93.9% | 5.3% |
| AMR-NB 12.2 kbps | 8.5% [5.4–11.8] | 0.58 | 83.8% | 3.5% |
| G.711 + 3% packet loss + 20 dB noise | 21.7% [17.0–25.7] | 0.43 | 26.0% | 0.7% |

What this tells us:
- Codecs alone keep errors under 9%. Packet loss plus noise is what breaks it.
- τ shifts by more than 3 units between codecs, so it **must be set per codec**. The contact centre knows the codec from call signalling.
- XTTS-v2 clones are harder to catch than FreeVC (e.g. 75.8% vs 90.2% on AMR-NB).
- See the caveats in `eval/RESULTS.md`: the real and cloned clips come from different corpora.

---

## 5. How EchoLock uses the model

| Step | Setting | Why |
|---|---|---|
| Audio source | the bank's own call leg (SIPREC fork), resampled to 16 kHz | we never need the customer's phone (see the deck, slide 4) |
| Window | **4 s**, hop **2 s** | matches the model card's 4 s result; 2 s refresh keeps latency low |
| Per-window score | z = logit_fake − logit_real | z is better spread out than a probability; P(fake) = sigmoid(z) |
| Call-level score | **median of the last 3 valid windows** | one noisy window can't trigger or clear an alert |
| Decision | elevate when median ≥ **τ**, then **latch** for the session | τ is set on our dev split for a 1% false-alarm rate, then frozen |
| UNKNOWN | too little speech, clipping, silence or an unsupported codec → no score | missing evidence never counts as "genuine" |
| Output to policy engine | {risk: elevated / normal / unknown, z, windows used, model revision} | the policy engine, not the model, decides what happens to the request |

---

## 6. Alternatives we considered

| Model | Size | Trade-off |
|---|---|---|
| **Wav2Vec2-Small-AntiDeepfake** (chosen) | 94 M | runs on CPU in real time, small enough for a 48-hour build and a modest pilot server |
| MMS-300M / MMS-1B-AntiDeepfake | 300 M / 1 B | MMS is pre-trained on 1,400+ languages, a likely gain on Indic speech. MMS-1B gets 1.82% / 27.70% EER on In-the-Wild / Deepfake-Eval-2024, but needs a GPU per few concurrent calls |
| AASIST / AASIST-L | 0.3 M / 85 k | tiny spectro-temporal graph model, our fallback if the SSL model can't be deployed. Generalises worse to unseen cloners |

Upgrade path: keep the window and median logic and the policy engine, swap the scorer for MMS-300M, and re-set τ on the same dev split.

---

## 7. Known limitations

- **Unseen cloners:** accuracy drops on generators and channels missing from training (33% EER on Deepfake-Eval-2024).
- **Phone channels:** G.711, AMR-NB, Opus and packet loss push genuine voices towards "fake". Our team smoke test moved mean z from −6.0 (clean) to −2.8 (G.711 + 3% loss + 20 dB noise). τ must be set on phone-quality audio, which `eval/` does.
- **Legitimate synthetic voices:** customers using TTS for accessibility score as fake by design. The policy routes them to a quick review with no penalty.
- **Licence:** CC BY-NC-SA 4.0 covers the hackathon and research only. Production needs a licence from NII or a model trained on licensed data.

## Sources
- NII model card: `research/source_snapshots/antideepfake_small_card.txt`
- Post-training paper: "Post-training for Deepfake Speech Detection", arXiv:2506.21090
- wav2vec 2.0: Baevski et al., NeurIPS 2020
- Our loader and measurements: `eval/model.py`, `eval/README.md`
