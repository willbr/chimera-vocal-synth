"""Word timings for a recording: whisper_words.py <in.wav> <out.json>

Run through `uv run --with faster-whisper==1.2.1`. Uses the speech judge's model (small.en, greedy)
and its model cache, but nothing else of the judge's.
"""

import json
import os
import sys
import wave

import numpy as np
from faster_whisper import WhisperModel


def load16k(path):
    """Mono float32 at 16 kHz from a 16-bit PCM WAV, as tools/asr/speech_wer.py loads it. Passing
    faster-whisper a path breaks: its PyAV call uses an argument newer PyAV no longer accepts."""
    with wave.open(path) as w:
        ch, rate = w.getnchannels(), w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    x = x.reshape(-1, ch).mean(axis=1)
    n = int(round(len(x) * 16000 / rate))
    return (np.fft.irfft(np.fft.rfft(x)[: n // 2 + 1], n) * (n / len(x))).astype(np.float32)


jobs = [sys.argv[1:3]]

root = os.path.expanduser("~/.cache/chimera-asr/models")
os.makedirs(root, exist_ok=True)
model = WhisperModel("small.en", device="cpu", compute_type="int8", download_root=root)
for wav, out in jobs:
    segments, _ = model.transcribe(
        load16k(wav),
        language="en",
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=True,
    )
    words = [{"word": w.word.strip(), "start": w.start, "end": w.end} for s in segments for w in s.words]
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"words": words}, f)
