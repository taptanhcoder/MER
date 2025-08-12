import numpy as np

def frame(data, window_length, hop_length):
    num_samples = data.shape[0]
    if num_samples < window_length:
        pad = window_length - num_samples
        data = np.pad(data, (0, pad), mode="constant")
        num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def periodic_hann(window_length):
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length)))

def stft_magnitude(signal, fft_length, hop_length=None, window_length=None):
    frames = frame(signal, window_length, hop_length)
    window = periodic_hann(window_length)
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))

_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

def hertz_to_mel(frequencies_hertz):
    return _MEL_HIGH_FREQUENCY_Q * np.log(1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))

def spectrogram_to_mel_matrix(num_mel_bins=64,
                              num_spectrogram_bins=257,
                              audio_sample_rate=16000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=7500.0):
    nyquist_hz = audio_sample_rate / 2.0
    if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz must be >= 0")
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz >= upper_edge_hertz")
    if upper_edge_hertz > nyquist_hz:
        raise ValueError("upper_edge_hertz > Nyquist")
    spectrogram_bins_hz  = np.linspace(0.0, nyquist_hz, num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hz)
    band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                                 hertz_to_mel(upper_edge_hertz),
                                 num_mel_bins + 2)
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i+3]
        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    mel_weights_matrix[0, :] = 0.0  
    return mel_weights_matrix

def log_mel_spectrogram(data,
                        audio_sample_rate=16000,
                        log_offset=1e-2,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        num_mel_bins=64,
                        lower_edge_hertz=125.0,
                        upper_edge_hertz=7500.0,
                        **kwargs):
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples    = int(round(audio_sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log2(window_length_samples)))

    if data.shape[0] < window_length_samples:
        pad = window_length_samples - data.shape[0]
        data = np.pad(data, (0, pad), mode="constant")

    spectrogram = stft_magnitude(
        data,
        fft_length=fft_length,
        hop_length=hop_length_samples,
        window_length=window_length_samples
    )
    mel_mat = spectrogram_to_mel_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=spectrogram.shape[1],
        audio_sample_rate=audio_sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
    )
    mel_spectrogram = np.dot(spectrogram, mel_mat)
    return np.log(mel_spectrogram + log_offset)
