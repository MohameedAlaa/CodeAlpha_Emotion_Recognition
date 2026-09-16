import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import MODEL_PATH, ENCODER_PATH, RESULTS_DIR, BATCH_SIZE
from src.data_loader import load_dataset_paths
from src.train import load_audio_waveforms, AudioDataGenerator

def evaluate_model():
    print("Loading test set...")
    _, _, (test_paths, test_labels) = load_dataset_paths()
    
    print(f"Test samples: {len(test_paths)}")
    
    # Load encoder
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
        
    y_test = encoder.transform(test_labels)
    
    print("Loading test audio waveforms...")
    test_audio = load_audio_waveforms(test_paths)
    
    # We use augment=False and batch_size=len(test_audio) to get everything in one batch
    # or just iterate through the generator to get predictions.
    # It's easier to just call __getitem__(0) if batch_size is large enough
    test_gen = AudioDataGenerator(test_audio, y_test, batch_size=len(test_audio), augment=False)
    X_test, y_test_gen = test_gen[0]
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test_gen, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification Report
    target_names = encoder.classes_
    report = classification_report(y_test_gen, y_pred, target_names=target_names)
    print("\nClassification Report:\n", report)
    
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)
        f.write(f"\nTest Loss: {loss:.4f}\nTest Accuracy: {accuracy:.4f}")
        
    # Confusion Matrix
    cm = confusion_matrix(y_test_gen, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")

if __name__ == "__main__":
    evaluate_model()
