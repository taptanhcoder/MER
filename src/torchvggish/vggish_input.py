import torch
import numpy as np
import resampy
import soundfile as sf

from . import mel_features
from . import vggish_params

def waveform_to_examples(data,
                         sample_rate,
                         return_tensor=True,
                         device=None):
    """
    Convert waveform -> VGGish input patches.
    Output: (num_examples, 1, NUM_FRAMES, NUM_BANDS) if return_tensor=True.
    """

    if data.ndim > 1:
        data = np.mean(data, axis=1)


    if sample_rate != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE,
        log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_MEL_BINS,
        lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ
    )  


    features_sr = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS  
    win_len = int(round(vggish_params.EXAMPLE_WINDOW_SECONDS * features_sr))  
    hop_len = int(round(vggish_params.EXAMPLE_HOP_SECONDS * features_sr))    

    if log_mel.shape[0] < win_len:
        pad = win_len - log_mel.shape[0]
        log_mel = np.pad(log_mel, ((0, pad), (0, 0)), mode="constant")

    log_mel_examples = mel_features.frame(
        log_mel,
        window_length=win_len,
        hop_length=hop_len
    )  

    if not return_tensor:
        return log_mel_examples

    t = torch.from_numpy(log_mel_examples).float().unsqueeze(1)  
    if device is not None:
        t = t.to(device)
    return t

def wavfile_to_examples(wav_file, return_tensor=True, device=None):
    """
    Read audio (int/float) -> VGGish patches tensor (N,1,96,64).
    """
    wav_data, sr = sf.read(wav_file, always_2d=False)
    if np.issubdtype(wav_data.dtype, np.integer):
        info = np.iinfo(wav_data.dtype)
        denom = max(abs(info.min), info.max)
        samples = wav_data.astype(np.float32) / float(denom)
    else:
        samples = wav_data.astype(np.float32)
        maxabs = np.max(np.abs(samples)) if samples.size > 0 else 1.0
        if maxabs > 1.0:
            samples = samples / maxabs

    return waveform_to_examples(samples, sr, return_tensor=return_tensor, device=device)
