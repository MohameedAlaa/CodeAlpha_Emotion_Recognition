import librosa
import numpy as np
from src.config import SAMPLE_RATE, MAX_FRAMES, N_MFCC

def extract_mfcc(file_path: str = None, audio_data: np.ndarray = None) -> np.ndarray:
    """
    Extracts MFCC features from an audio file or an audio array.
    """
    if audio_data is not None:
        audio = audio_data
    elif file_path is not None:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    else:
        raise ValueError("Either file_path or audio_data must be provided.")
        
    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
        
    # Pad or truncate
    if len(audio) > MAX_FRAMES:
        audio = audio[:MAX_FRAMES]
    else:
        padding = MAX_FRAMES - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')
        
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc = mfcc.T
    
    return mfcc

def add_noise(audio: np.ndarray, noise_factor: float) -> np.ndarray:
    """Adds random Gaussian noise to the audio."""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Applies time stretching to the audio."""
    return librosa.effects.time_stretch(y=audio, rate=rate)

def pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Applies pitch shifting to the audio."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def extract_mel_spectrogram(file_path: str) -> np.ndarray:
    """
    Extracts log Mel-Spectrogram features from an audio file for 2D CNN input.
    
    This function explicitly:
    1. Loads the audio and converts it to Mono.
    2. Resamples it to `SAMPLE_RATE` (16kHz).
    3. Normalizes the waveform.
    4. Pads or truncates the audio to a fixed length.
    5. Extracts Mel-Spectrogram features and converts to log scale (dB).
    6. Reshapes to (T, N_MELS, 1).

    Args:
        file_path (str): The absolute or relative path to the .wav file.

    Returns:
        np.ndarray: The extracted Mel-Spectrogram of shape (T, N_MELS, 1).
    """
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
        
    # Pad or truncate
    if len(audio) > MAX_FRAMES:
        audio = audio[:MAX_FRAMES]
    else:
        padding = MAX_FRAMES - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')
        
    # Extract Mel-Spectrogram
    from src.config import N_MELS
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Shape of log_mel_spectrogram is (N_MELS, T). 
    # Transpose to (T, N_MELS) and add a channel dimension to make it (T, N_MELS, 1)
    log_mel_spectrogram = log_mel_spectrogram.T
    log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=-1)
    
    return log_mel_spectrogram
