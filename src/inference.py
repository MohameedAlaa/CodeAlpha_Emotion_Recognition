import numpy as np
import tensorflow as tf
import pickle
from typing import Tuple, Dict

from src.config import MODEL_PATH, ENCODER_PATH
from src.features import extract_mfcc

# Global variables for lazy loading
_MODEL = None
_ENCODER = None

def load_artifacts():
    """Loads the model and encoder lazily."""
    global _MODEL, _ENCODER
    if _MODEL is None:
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
    if _ENCODER is None:
        with open(ENCODER_PATH, 'rb') as f:
            _ENCODER = pickle.load(f)

def predict(audio_path: str) -> Tuple[str, Dict[str, float]]:
    """
    Predicts the emotion of a given audio file.
    
    Args:
        audio_path (str): Path to the WAV file.
        
    Returns:
        Tuple[str, Dict[str, float]]: The predicted emotion string and a dictionary of probabilities.
    """
    load_artifacts()
    
    # 1. Exact same preprocessing as training
    mfcc = extract_mfcc(file_path=audio_path)
    
    # Expand dims to match batch shape: (1, T, N_MFCC)
    X = np.expand_dims(mfcc, axis=0)
    
    # 2. Predict
    probs = _MODEL.predict(X)[0]
    
    # 3. Decode
    predicted_idx = np.argmax(probs)
    predicted_emotion = _ENCODER.inverse_transform([predicted_idx])[0]
    
    prob_dict = {
        _ENCODER.inverse_transform([i])[0]: float(probs[i])
        for i in range(len(probs))
    }
    
    return predicted_emotion, prob_dict
