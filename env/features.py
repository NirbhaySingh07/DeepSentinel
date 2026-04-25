"""
VoiceGuard – DSP Feature Extraction (numpy-only, no torch)
Extracts AudioStats for the OpenEnv environment observations.
"""

from __future__ import annotations
import numpy as np

SAMPLE_RATE = 16_000
N_MFCC = 13
MAX_DURATION_S = 4
MAX_LEN = SAMPLE_RATE * MAX_DURATION_S
HOP_LENGTH = 160
WIN_LENGTH = 400
N_FFT = 400
N_MELS = 80


def pad_or_truncate(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]
    else:
        audio = np.concatenate([audio, np.zeros(MAX_LEN - len(audio), dtype=np.float32)])
    return audio.astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    fmin, fmax = 0.0, sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hz = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        for j in range(lo, mid):
            fb[i, j] = (j - lo) / max(mid - lo, 1)
        for j in range(mid, hi):
            fb[i, j] = (hi - j) / max(hi - mid, 1)
    return fb


def _dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    basis = np.zeros((n_mfcc, n_mels), dtype=np.float32)
    for k in range(n_mfcc):
        for n in range(n_mels):
            basis[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
    return basis


_FB = _mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)
_DCT = _dct_matrix(N_MFCC, N_MELS)


def _compute_mfcc(audio: np.ndarray) -> np.ndarray:
    frames = []
    for i in range(0, len(audio) - N_FFT, HOP_LENGTH):
        frame = audio[i:i + N_FFT] * np.hanning(N_FFT)
        frames.append(frame)
    if not frames:
        frames = [audio[:N_FFT] * np.hanning(N_FFT)]
    spectra = np.abs(np.fft.rfft(np.array(frames), n=N_FFT)) ** 2
    mel = np.dot(spectra, _FB.T) + 1e-10
    log_mel = np.log(mel)
    mfcc = np.dot(log_mel, _DCT.T)
    return mfcc  # (T, N_MFCC)


def extract_audio_stats(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    audio = pad_or_truncate(audio)
    duration_sec = len(audio) / sr
    rms_energy = float(np.sqrt(np.mean(audio ** 2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)
    frames = [audio[i:i + N_FFT] for i in range(0, len(audio) - N_FFT, HOP_LENGTH)]
    if not frames:
        frames = [audio[:N_FFT]]
    spectra = np.abs(np.fft.rfft(np.array(frames), n=N_FFT)) ** 2
    mean_spectrum = spectra.mean(axis=0)
    total_power = mean_spectrum.sum() + 1e-8
    spectral_centroid = float(np.sum(freqs * mean_spectrum) / total_power)
    cumsum = np.cumsum(mean_spectrum)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * total_power)
    spectral_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    mfcc = _compute_mfcc(audio)
    mfcc_mean = mfcc.mean(axis=0).tolist()
    mfcc_std = mfcc.std(axis=0).tolist()

    pitch_mean, pitch_std = _estimate_pitch(audio)

    return dict(
        duration_sec=duration_sec, sample_rate=sr,
        rms_energy=rms_energy, zero_crossing_rate=zcr,
        spectral_centroid=spectral_centroid, spectral_rolloff=spectral_rolloff,
        mfcc_mean=mfcc_mean, mfcc_std=mfcc_std,
        pitch_mean=pitch_mean, pitch_std=pitch_std,
    )


def _estimate_pitch(audio: np.ndarray, frame_len: int = 2048):
    min_lag = int(SAMPLE_RATE / 400)
    max_lag = int(SAMPLE_RATE / 50)
    pitches = []
    for start in range(0, len(audio) - frame_len, frame_len // 2):
        frame = audio[start:start + frame_len]
        corr = np.correlate(frame, frame, mode="full")[frame_len - 1:]
        seg = corr[min_lag:max_lag]
        if seg.size == 0:
            continue
        peak_lag = np.argmax(seg) + min_lag
        if corr[0] > 1e-6:
            pitches.append(SAMPLE_RATE / peak_lag)
    if not pitches:
        return None, None
    arr = np.array(pitches)
    return float(arr.mean()), float(arr.std())


def apply_corruption(audio: np.ndarray, corruption_type: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = audio.copy()
    if corruption_type == "noise":
        a = np.clip(a + rng.normal(0, 0.005, size=a.shape).astype(np.float32), -1.0, 1.0)
    elif corruption_type == "compression":
        a = (a * 32).astype(np.int8).astype(np.float32) / 32
        a = np.convolve(a, np.ones(4) / 4, mode="same").astype(np.float32)
    elif corruption_type == "speed":
        factor = rng.choice([0.9, 1.1])
        indices = np.round(np.arange(0, len(a), factor)).astype(int)
        indices = indices[indices < len(a)]
        a = a[indices]
    elif corruption_type == "bandlimit":
        fft = np.fft.rfft(a)
        fft[len(fft) // 2:] = 0.0
        a = np.fft.irfft(fft, n=len(a)).astype(np.float32)
    return a
