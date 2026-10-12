# Team voice recordings

Two team members recorded the script in [`recording_script.md`](recording_script.md) on their own phones (voice recorder app, 44.1 kHz mono WAV, quiet room, 20–30 cm from the mouth).

| Speaker | Who | Files | Total |
|---|---|---|---|
| S02 | Reedhan Garg | `raw/S02/S02_{REF-EN,REF-HI,EN,HI,HG,FREE}.wav` | 6.1 min |
| S03 | Shobhit Pandey | `raw/S03/S03_{REF-EN,REF-HI,EN,HI,HG,FREE}.wav` | 5.3 min |

Recording types: `REF-EN` / `REF-HI` are ~45 s read passages, `EN` / `HI` / `HG` are 12 bank-call sentences each in English, Hindi and Hinglish, and `FREE` is unscripted speech. Sentences are separated by ~2 s pauses; `eval/run_eval.py` cuts them into 117 clips automatically.

They were used as **genuine** speech to measure false alarms on a real phone microphone (results in [`eval/RESULTS.md`](../eval/RESULTS.md)). No clones of these voices are included.

## Consent and terms

Both speakers consented to recording for the EchoLock project and, on 12 Oct 2026, to publishing these files here.

- Use them for research on detecting voice cloning and spoofing, for example as genuine test speech.
- **Do not** use them to clone, imitate or impersonate either speaker, or to train voice-cloning or TTS models.
- Licence: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), with the restriction above.

All numbers spoken in the recordings are made up. No real phone numbers, account details or addresses are in them.
