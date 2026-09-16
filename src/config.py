from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "RAVDESS"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset splits (by Actor ID to prevent leakage)
TRAIN_ACTORS = {f"Actor_{i:02d}" for i in range(1, 19)}  # 1 to 18
VAL_ACTORS = {f"Actor_{i:02d}" for i in range(19, 22)}   # 19 to 21
TEST_ACTORS = {f"Actor_{i:02d}" for i in range(22, 25)}  # 22 to 24

# Emotion Mapping
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

CLASSES = list(EMOTION_MAP.values())
NUM_CLASSES = len(CLASSES)

# Audio Parameters
SAMPLE_RATE = 16000
MAX_DURATION = 4.0  # seconds
MAX_FRAMES = int(SAMPLE_RATE * MAX_DURATION)
N_MFCC = 40
N_MELS = 128

# Model Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Data Augmentation Parameters
AUG_TIME_RATES = [0.9, 1.1]
AUG_PITCH_STEPS = [-1, 1]
AUG_NOISE_FACTOR = 0.005

# Files
MODEL_PATH_BASELINE = MODELS_DIR / "ser_cnn_model.keras"
MODEL_PATH = MODELS_DIR / "ser_cnn_exp3_model.keras"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
