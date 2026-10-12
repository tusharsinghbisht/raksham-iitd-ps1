"""Phone-channel simulation. Every condition takes and returns 16 kHz mono float32."""
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    FFMPEG = "ffmpeg"


def to8k(x):
    return resample_poly(x, 1, 2).astype(np.float32)


def to16k(x):
    return resample_poly(x, 2, 1).astype(np.float32)


def mulaw(x, mu=255):
    """G.711 mu-law: compand, quantise to 8 bits, expand."""
    x = np.clip(x, -1, 1)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * mu) / mu * 2 - 1
    return (np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu).astype(np.float32)


def peak(x, level=0.5):
    m = np.abs(x).max()
    return (x / m * level).astype(np.float32) if m > 0 else x


def ffmpeg_codec(x8k, args, ext):
    """Encode and decode 8 kHz audio with ffmpeg; returns 8 kHz float32."""
    with tempfile.TemporaryDirectory() as d:
        src, enc, dst = (os.path.join(d, n) for n in ("in.wav", "enc" + ext, "out.wav"))
        sf.write(src, x8k, 8000, subtype="PCM_16")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", src, *args, enc], check=True)
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", enc, "-ar", "8000", "-ac", "1", dst], check=True)
        y, _ = sf.read(dst, dtype="float32")
    return y


def has_encoder(name):
    out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return f" {name} " in out


def packet_loss(x8k, rate, rng, frame=160):
    """Drop 20 ms frames at random; a lost frame repeats the last good one (simple PLC)."""
    y, last = x8k.copy(), np.zeros(frame, np.float32)
    for i in range(0, len(y) - frame + 1, frame):
        if rng.random() < rate:
            y[i:i + frame] = last * 0.5
        else:
            last = y[i:i + frame].copy()
    return y


def noise(x, snr_db, rng):
    p = np.mean(x ** 2) + 1e-12
    return (x + rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(x))).astype(np.float32)


def conditions():
    c = {
        "clean_16k": lambda x, r: x,
        "narrowband_8k": lambda x, r: to16k(to8k(x)),
        "g711": lambda x, r: to16k(mulaw(to8k(peak(x)))),
        "g711_loss3_snr20": lambda x, r: to16k(mulaw(packet_loss(noise(to8k(peak(x)), 20, r), 0.03, r))),
    }
    if has_encoder("libopencore_amrnb"):
        c["amr_nb_12k2"] = lambda x, r: to16k(ffmpeg_codec(to8k(peak(x)), ["-c:a", "libopencore_amrnb", "-b:a", "12.2k"], ".amr"))
    if has_encoder("libopus"):
        c["opus_12k_voip"] = lambda x, r: to16k(ffmpeg_codec(to8k(peak(x)), ["-c:a", "libopus", "-b:a", "12k", "-application", "voip"], ".ogg"))
    return c
