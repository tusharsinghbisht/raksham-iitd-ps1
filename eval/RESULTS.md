# Detector results: Hindi, phone channels (2026-10-12)

Model: wav2vec-small-anti-deepfake. Test split: 434 real clips (21 speakers, Kathbath) + 526 cloned clips (58 speakers, IndicSynth XTTS-v2 + FreeVC). Dev split, used only to set tau: 408 real (17 speakers) + 242 cloned (26 speakers). Speaker-disjoint. Team = 117 sentence clips from 2 teammates (S02, S03), English/Hindi/Hinglish.

| Condition | EER whole clip | EER call-level (median 4 s) [95% CI] | tau (dev, 1% FA) | Test detection @tau | Test false alarm @tau | Team false alarm @tau |
|---|---|---|---|---|---|---|
| clean_16k | 2.1% | 1.6% [0.8%-2.7%] | 1.13 | 95.8% (526) | 0.7% (434) | 1.7% (117) |
| narrowband_8k | 1.9% | 3.6% [2.1%-5.1%] | -1.25 | 95.8% (526) | 2.1% (434) | 8.5% (117) |
| g711 | 1.9% | 3.4% [2.1%-5.0%] | -2.27 | 96.6% (526) | 3.9% (434) | 16.2% (117) |
| g711_loss3_snr20 | 17.9% | 21.7% [17.0%-25.7%] | 0.43 | 26.0% (526) | 0.7% (434) | 7.7% (117) |
| amr_nb_12k2 | 5.7% | 8.5% [5.4%-11.8%] | 0.58 | 83.8% (526) | 3.5% (434) | 4.3% (117) |
| opus_12k_voip | 3.4% | 5.9% [3.5%-8.5%] | -1.57 | 93.9% (526) | 5.3% (434) | 17.1% (117) |

Same-person check (real speaker X vs clone of X), 14 speakers, all splits, call-level EER:
- clean_16k: 2.3% (312 real, 130 cloned)
- narrowband_8k: 3.1% (312 real, 130 cloned)
- g711: 2.3% (312 real, 130 cloned)
- g711_loss3_snr20: 23.1% (312 real, 130 cloned)
- amr_nb_12k2: 9.8% (312 real, 130 cloned)
- opus_12k_voip: 5.4% (312 real, 130 cloned)

Detection by generator @tau (test): 
- clean_16k: freevc24 99.7%, xtts_v2 90.9%
- narrowband_8k: freevc24 100.0%, xtts_v2 90.5%
- g711: freevc24 99.7%, xtts_v2 92.6%
- g711_loss3_snr20: freevc24 27.8%, xtts_v2 23.8%
- amr_nb_12k2: freevc24 90.2%, xtts_v2 75.8%
- opus_12k_voip: freevc24 99.7%, xtts_v2 86.6%

## Caveats
- Real and cloned clips come from different corpora (Kathbath: phone-app recorded, AAC; IndicSynth: synthesised WAV). Part of the separation may be corpus difference. The same-person check (same speakers on both sides) gives similar numbers, which helps but does not remove this.
- 118 Kathbath clips failed to decode (m4a index at end of file, fixed in the fetch scripts since) and were dropped; this is unrelated to content.
- Team false alarms at the Hindi-calibrated tau are higher (2-17%): a different microphone, room and language mix. tau must be re-set on the bank's own genuine calls in shadow mode.
- Hindi IndicSynth has no VITS clones; only 2 cloning tools are covered.
