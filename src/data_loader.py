import os
from typing import Tuple, List
import numpy as np
from src.config import DATA_DIR, EMOTION_MAP, TRAIN_ACTORS, VAL_ACTORS, TEST_ACTORS

def get_actor_from_path(file_path: str) -> str:
    """Extracts actor folder name from file path."""
    # file_path example: .../Actor_01/03-01-01-01-01-01-01.wav
    return os.path.basename(os.path.dirname(file_path))

def get_emotion_from_filename(filename: str) -> str:
    """Extracts the emotion string from the RAVDESS filename."""
    # RAVDESS format: modality-vocal_channel-emotion-emotional_intensity-statement-repetition-actor.wav
    parts = filename.replace('.wav', '').split('-')
    if len(parts) == 7:
        emotion_code = parts[2]
        return EMOTION_MAP.get(emotion_code, 'unknown')
    return 'unknown'

def verify_splits(train_paths: List[str], val_paths: List[str], test_paths: List[str]):
    """Programmatically verifies that no actor overlaps across splits."""
    train_actors = {get_actor_from_path(p) for p in train_paths}
    val_actors = {get_actor_from_path(p) for p in val_paths}
    test_actors = {get_actor_from_path(p) for p in test_paths}
    
    # Check intersections
    assert len(train_actors.intersection(val_actors)) == 0, "Leakage between Train and Val"
    assert len(train_actors.intersection(test_actors)) == 0, "Leakage between Train and Test"
    assert len(val_actors.intersection(test_actors)) == 0, "Leakage between Val and Test"
    print("Dataset splits verified successfully: No actor overlap detected.")

def load_dataset_paths() -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:
    """
    Crawls the RAVDESS dataset, ignoring the duplicate folder, and splits
    file paths and labels into Train, Validation, and Test sets based on Actor ID.
    
    Returns:
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
    """
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    # We only care about canonical Actor_XX folders
    actor_folders = [f for f in os.listdir(DATA_DIR) if f.startswith('Actor_') and os.path.isdir(os.path.join(DATA_DIR, f))]
    
    for actor in actor_folders:
        actor_path = os.path.join(DATA_DIR, actor)
        for f in os.listdir(actor_path):
            if f.endswith('.wav'):
                file_path = os.path.join(actor_path, f)
                emotion = get_emotion_from_filename(f)
                
                if emotion == 'unknown':
                    continue
                    
                if actor in TRAIN_ACTORS:
                    train_paths.append(file_path)
                    train_labels.append(emotion)
                elif actor in VAL_ACTORS:
                    val_paths.append(file_path)
                    val_labels.append(emotion)
                elif actor in TEST_ACTORS:
                    test_paths.append(file_path)
                    test_labels.append(emotion)
                    
    verify_splits(train_paths, val_paths, test_paths)
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
