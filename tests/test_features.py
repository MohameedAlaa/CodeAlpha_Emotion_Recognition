import numpy as np
import pytest
from src.features import extract_mfcc, extract_mel_spectrogram
from src.config import MAX_FRAMES, N_MFCC, N_MELS
import soundfile as sf
import tempfile
import os

def test_extract_mfcc_shape():
    # Create dummy 48kHz stereo sine wave
    sr = 48000
    duration = 1.0 # 1 second
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t)
    stereo_audio = np.vstack((audio, audio)).T
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_path = f.name
        
    try:
        sf.write(tmp_path, stereo_audio, sr)
        
        mfcc = extract_mfcc(tmp_path)
        
        # We expect mfcc to be (MAX_FRAMES / hop_length, N_MFCC)
        # librosa default hop length is 512
        expected_frames = int(np.ceil(MAX_FRAMES / 512))
        
        assert mfcc.shape[1] == N_MFCC
        assert mfcc.shape[0] == expected_frames or mfcc.shape[0] == expected_frames + 1
    finally:
        os.remove(tmp_path)

def test_extract_mel_shape():
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_path = f.name
        
    try:
        sf.write(tmp_path, audio, sr)
        
        mel = extract_mel_spectrogram(tmp_path)
        
        expected_frames = int(np.ceil(MAX_FRAMES / 512))
        
        # (T, N_MELS, 1)
        assert mel.shape[1] == N_MELS
        assert mel.shape[2] == 1
        assert mel.shape[0] == expected_frames or mel.shape[0] == expected_frames + 1
    finally:
        os.remove(tmp_path)

def test_add_noise():
    from src.features import add_noise
    audio = np.zeros(16000)
    noisy_audio = add_noise(audio, 0.05)
    assert noisy_audio.shape == audio.shape
    assert np.any(noisy_audio != 0)

def test_time_stretch():
    from src.features import time_stretch
    audio = np.ones(16000)
    stretched = time_stretch(audio, 1.5) # Faster
    assert len(stretched) < len(audio)

def test_pitch_shift():
    from src.features import pitch_shift
    audio = np.ones(16000)
    shifted = pitch_shift(audio, 16000, 2)
    assert len(shifted) == len(audio)
