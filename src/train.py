import os
import numpy as np
import pickle
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.config import MODEL_PATH, ENCODER_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE, SAMPLE_RATE
from src.config import AUG_TIME_RATES, AUG_PITCH_STEPS, AUG_NOISE_FACTOR
from src.data_loader import load_dataset_paths
from src.features import extract_mfcc, add_noise, time_stretch, pitch_shift
from src.model import build_cnn_model

class AudioDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras with on-the-fly augmentation."""
    def __init__(self, audio_data, labels, batch_size=32, augment=False):
        self.audio_data = audio_data
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.audio_data))
        
    def __len__(self):
        return int(np.ceil(len(self.audio_data) / float(self.batch_size)))
        
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_audio = [self.audio_data[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]
        
        X = []
        for audio in batch_audio:
            if self.augment:
                # Randomly choose one augmentation or no augmentation
                aug_type = np.random.choice(['none', 'noise', 'stretch', 'pitch'])
                
                if aug_type == 'noise':
                    audio = add_noise(audio, AUG_NOISE_FACTOR)
                elif aug_type == 'stretch':
                    rate = np.random.choice(AUG_TIME_RATES)
                    audio = time_stretch(audio, rate)
                elif aug_type == 'pitch':
                    steps = np.random.choice(AUG_PITCH_STEPS)
                    audio = pitch_shift(audio, SAMPLE_RATE, steps)
                    
            mfcc = extract_mfcc(audio_data=audio)
            X.append(mfcc)
            
        return np.array(X), np.array(batch_labels)
        
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def load_audio_waveforms(paths):
    """Loads all audio files into RAM."""
    audio_data = []
    for path in paths:
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            audio_data.append(audio)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            audio_data.append(np.zeros(SAMPLE_RATE)) # Dummy to preserve length
    return audio_data

def train_model():
    print("Loading dataset splits...")
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = load_dataset_paths()
    
    print(f"Train samples: {len(train_paths)}")
    print(f"Val samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    # Encode labels
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_labels)
    y_val = encoder.transform(val_labels)
    
    # Save encoder
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(encoder, f)
        
    num_classes = len(encoder.classes_)
    
    # Compute class weights for the 'neutral' class imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class Weights computed: {class_weight_dict}")
    
    print("Loading audio waveforms into RAM for fast generation...")
    train_audio = load_audio_waveforms(train_paths)
    val_audio = load_audio_waveforms(val_paths)
    
    train_gen = AudioDataGenerator(train_audio, y_train, batch_size=BATCH_SIZE, augment=True)
    val_gen = AudioDataGenerator(val_audio, y_val, batch_size=BATCH_SIZE, augment=False)
    
    # Get one batch to determine input shape
    X_sample, _ = train_gen[0]
    input_shape = (X_sample.shape[1], X_sample.shape[2])
    print(f"Input shape: {input_shape}")
    
    model = build_cnn_model(input_shape, num_classes)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    print("Starting training (Experiment 3)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    print(f"Training completed. Best model saved to {MODEL_PATH}")
    return history

if __name__ == "__main__":
    train_model()
