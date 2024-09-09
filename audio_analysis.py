from pathlib import Path

import numpy as np
from pydub import AudioSegment
import soundfile as sf


### Also loads m4a
def load_mp3file(path):
    """MP3 to numpy array"""
    a = AudioSegment.from_file(path)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
        y = y.mean(axis=1)
    return a.frame_rate, y


def load_audio(path, stereo=False):
    try:
        ext = path.suffix
    except:
        ext = Path(path).suffix

    if ext[1:].upper() in sf.available_formats():
        wav, fr = sf.read(path)
    elif ext[1:].lower() in ['mp3', 'm4a']:
        fr, wav = load_mp3file(str(path))
    else:
        raise Exception(f"File extension '{ext}' not recognized\n{path}")

    if stereo:
        return fr, wav

    if len(wav.shape)>1:
        return fr, wav.mean(axis=1)
    else:
        return fr, wav


def frame_generator(wav, fr, window=2048, hop=256):
    wav = np.array(wav, dtype=np.float32) / (2.**15 + 1)
    steps = int(wav.size/hop) + 2
    for i in range(steps):
        frame = np.zeros(window, dtype=float)
        wav_start = int(i*hop-window/2)
        wav_end   = int(i*hop+window/2)
        frame_start = max(-wav_start, 0)
        frame_end   = min(window, window - (wav_end - wav.size))
        frame[frame_start:frame_end] = wav[max(0, wav_start):min(wav_end, wav.size)]
        yield frame


def signal_energy(fr, wav, hop=256, window=2048):
    if wav.dtype != np.dtype(np.float64):
        wav = wav.astype(np.float64)
    frames = list(frame_generator(wav, fr, window=window, hop=hop))
    amp = np.array([np.sum(np.square(frame)) for frame in frames]) * (window / fr)
    tm = np.arange(amp.size) / float(amp.size) * wav.size / fr
    return tm, amp


