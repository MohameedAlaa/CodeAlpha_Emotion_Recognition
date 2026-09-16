import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Helper to add cells
def add_md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def add_code(text):
    nb.cells.append(nbf.v4.new_code_cell(text))

add_md("# CodeAlpha Task 2: Speech Emotion Recognition")

add_md("## Objective\nTo build a Speech Emotion Recognition (SER) system that accurately classifies 8 emotions using the RAVDESS dataset. This notebook documents the data preprocessing, model selection, and experiments.")

add_md("## 1. Environment & Reproducibility")
add_code("""import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath('..'))
from src import config

print(f"Python executable: {sys.executable}")
print(f"Dataset path: {config.DATA_DIR}")""")

add_md("## 2. Dataset Overview")
add_code("""from src.data_loader import load_dataset_paths
(train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = load_dataset_paths()
print(f"Train samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")
print(f"Test samples: {len(test_paths)}")
""")

add_md("## 3. Emotion Distribution")
add_code("""# Visualize emotion distribution
import collections
# We count instances from the labels
counts = collections.Counter(train_labels)
emotions = list(counts.keys())
plt.figure(figsize=(10, 5))
sns.barplot(x=emotions, y=list(counts.values()))
plt.title("Emotion Class Distribution (Training Set)")
plt.ylabel("Count")
plt.show()""")

add_md("## 4. Audio Preprocessing")
add_code("""import librosa
import librosa.display

# Load a sample file for demonstration
sample_file = train_paths[0]
y, sr = librosa.load(sample_file, sr=config.SAMPLE_RATE, mono=True)

plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform of Speech")
plt.show()""")

add_md("## 5. MFCC Feature Extraction")
add_code("""from src.features import extract_mfcc

# Extract features
mfcc = extract_mfcc(sample_file)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc.T, x_axis='time', sr=config.SAMPLE_RATE)
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC representation')
plt.tight_layout()
plt.show()
print(f"MFCC shape: {mfcc.shape}")""")

add_md("## 6. Actor-Level Train/Validation/Test Split")
add_code("""print("To prevent data leakage, the dataset is strictly split by actors:")
print(f"Train: Actors {config.TRAIN_ACTORS}")
print(f"Validation: Actors {config.VAL_ACTORS}")
print(f"Test: Actors {config.TEST_ACTORS}")""")

add_md("## 7. Baseline CNN")
add_code("""# The baseline model was an un-regularized CNN:
# Conv1D 64 -> Conv1D 128 -> Conv1D 256
# Dense 128
print("Baseline CNN Architecture: High capacity, no explicit regularization (L2 or strong Dropout).")""")

add_md("## 8. Baseline Training Results")
add_code("""# From initial runs:
print("Baseline Results:")
print("Test Accuracy: 0.5056")
print("Validation peaked early (~Epoch 7) at 0.5556, while train accuracy climbed to >0.80.")
print("This indicated strong overfitting.")""")

add_md("## 9. Experiment 1: Reduced CNN Capacity + Regularization")
add_code("""from src.model import build_cnn_model
model = build_cnn_model(input_shape=(126, 40), num_classes=8)
model.summary()""")

add_md("## 10. Experiment 1 Training Results")
add_code("""# The model was trained using the pipeline
print("Training completed using early stopping.")
print("The learning rate was reduced on plateaus.")""")

add_md("## 11. Baseline vs Experiment 1 Comparison")
add_code("""import pandas as pd
data = {
    'Model': ['Baseline', 'Experiment 1 (Reduced)'],
    'Test Accuracy': [0.5056, 0.3944],
    'Macro Precision': [0.466, 0.380],
    'Macro Recall': [0.473, 0.370],
    'Macro F1': [0.445, 0.310]
}
df = pd.DataFrame(data)
display(df)
print("Observation: The heavily regularized model generalized worse to the unseen test actors, suggesting the reduced capacity might have underfitted the complexities of speech emotion representation.")""")

add_md("## 12. Final Model Evaluation")
add_code("""from tensorflow.keras.models import load_model
from src.train import load_and_extract_features
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# We will use the model artifact saved by run_pipeline.py
model_path = os.path.join("..", "models", "ser_cnn_model.keras")
model = load_model(model_path)
with open(os.path.join("..", "models", "label_encoder.pkl"), 'rb') as f:
    le = pickle.load(f)

# Evaluate on test set
print("Extracting test features...")
X_test, y_test_str = load_and_extract_features(test_paths, test_labels)
y_test = le.transform(y_test_str)

print("Evaluating on Test Set...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)""")

add_md("## 13. Confusion Matrix")
add_code("""cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()""")

add_md("## 14. Classification Report")
add_code("""print(classification_report(y_test, y_pred, target_names=le.classes_))""")

add_md("## 15. Example Predictions")
add_code("""# Show a few predictions vs true labels
for i in range(5):
    true_label = le.inverse_transform([y_test[i]])[0]
    pred_label = le.inverse_transform([y_pred[i]])[0]
    print(f"Sample {i}: True='{true_label}', Predicted='{pred_label}'")""")

add_md("## 16. Conclusion for Baseline & Exp 1")
add_md("""- The Baseline model severely overfit the small dataset.
- Experiment 1 attempted to fix this by severely constraining the network (16->32->64 filters) and adding dropout/L2.
- The outcome was a drop in test performance, implying the network lacked the capacity to learn generalizable features for unseen actors.
- **Future Work**: The 1D CNN over MFCC features treats frequency bands as independent channels. We recommend adopting a 2D CNN (Experiment 3) or Mel-Spectrogram features, alongside Data Augmentation, to properly solve the generalization gap without crippling the model capacity.""")

add_md("## 17. Experiment 2: 2D CNN with Mel-Spectrogram")
add_code("""# In Experiment 2, we changed the feature extraction to output Mel-Spectrograms instead of MFCCs.
# We also changed the architecture to a 2D CNN to take advantage of the 2D nature of Mel-Spectrograms (Time x Frequency).
# The new input shape is (T, N_MELS, 1).
# We used a moderate capacity network (32 -> 64 -> 128) with Dropout and L2 regularization.""")

add_md("## 18. Experiment 2 Final Model Evaluation")
add_code("""# Evaluate on test set (Experiment 2 Model)
print("Extracting test features for Experiment 2...")
# Note: we re-run extraction because the underlying function extract_mel_spectrogram is now the default
X_test_exp2, y_test_str_exp2 = load_and_extract_features(test_paths, test_labels)
y_test_exp2 = le.transform(y_test_str_exp2)

print("Evaluating on Test Set...")
loss_exp2, accuracy_exp2 = model.evaluate(X_test_exp2, y_test_exp2, verbose=0)
print(f"Experiment 2 Test Loss: {loss_exp2:.4f}")
print(f"Experiment 2 Test Accuracy: {accuracy_exp2:.4f}")
y_pred_probs_exp2 = model.predict(X_test_exp2)
y_pred_exp2 = np.argmax(y_pred_probs_exp2, axis=1)""")

add_md("## 19. Experiment 2 Confusion Matrix")
add_code("""cm_exp2 = confusion_matrix(y_test_exp2, y_pred_exp2)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_exp2, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Experiment 2 Confusion Matrix')
plt.show()""")

add_md("## 20. Experiment 2 Classification Report")
add_code("""print(classification_report(y_test_exp2, y_pred_exp2, target_names=le.classes_))""")

add_md("## 21. Comparison: Baseline vs Exp 1 vs Exp 2")
add_code("""import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(y_test_exp2, y_pred_exp2, average='macro')
data_comp = {
    'Model': ['Baseline', 'Experiment 1 (Reduced)', 'Experiment 2 (2D CNN + Mel)'],
    'Test Accuracy': [0.5056, 0.3944, accuracy_exp2],
    'Macro Precision': [0.466, 0.380, precision],
    'Macro Recall': [0.473, 0.370, recall],
    'Macro F1': [0.445, 0.310, f1]
}
df_comp = pd.DataFrame(data_comp)
display(df_comp)
""")

add_md("## 22. Experiment 3: Baseline 1D CNN with Data Augmentation")
add_code("""# Overfitting and Generalization Problem:
# - Our Baseline 1D CNN over MFCCs overfit the training actors (Test Acc: 0.5056, Train > 0.80).
# - Experiment 1 proved that artificially limiting model capacity (fewer filters, huge dropout) hurts generalization further (Test Acc: 0.3944).
# - Experiment 2 proved that switching to Mel-Spectrograms + 2D CNN severely overfits due to high parameter capacity on a tiny dataset (Test Acc: 0.2333).
#
# To solve this, we returned to the Baseline 1D CNN architecture over MFCCs and applied Data Augmentation.
# Data Augmentation artificially expands the training set by applying realistic audio transformations (noise injection, time stretching, pitch shifting).
# This forces the network to learn invariant emotion representations instead of memorizing specific actor traits.
#
# Augmentations applied on-the-fly during training (only on training actors):
# - Additive Gaussian Noise (factor=0.005)
# - Time Stretching (rates: 0.9, 1.1)
# - Pitch Shifting (steps: -1, +1)
""")

add_md("## 23. Train Experiment 3 (Interactive)")
add_code("""# Run this cell to train Experiment 3 from within the notebook.
# The logic is imported from `src.train` to keep the architecture clean.
# Training will display live epoch-by-epoch progress, accuracy, loss, and early stopping.

# Uncomment to train:
# from src.train import train_model
# history_exp3 = train_model()
""")

add_md("## 24. Training History Plots")
add_code("""# If you trained the model using the cell above, you can plot its history.
# Uncomment to plot:
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history_exp3.history['accuracy'], label='Train Accuracy')
# plt.plot(history_exp3.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Experiment 3: Accuracy vs. Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(history_exp3.history['loss'], label='Train Loss')
# plt.plot(history_exp3.history['val_loss'], label='Validation Loss')
# plt.title('Experiment 3: Loss vs. Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.tight_layout()
# plt.show()
""")

add_md("## 25. Experiment 3 Final Model Evaluation")
add_code("""# Evaluate on test set (Experiment 3 Model)
print("Extracting test features for Experiment 3 (MFCCs)...")
# Note: we re-run extraction because the underlying function extract_mfcc is now the default again
X_test_exp3, y_test_str_exp3 = load_and_extract_features(test_paths, test_labels) if 'load_and_extract_features' in globals() else (None, None)

if X_test_exp3 is None:
    # Use load_audio_waveforms as defined in train.py for Experiment 3
    from src.train import load_audio_waveforms, AudioDataGenerator
    test_audio = load_audio_waveforms(test_paths)
    test_gen = AudioDataGenerator(test_audio, le.transform(test_labels), batch_size=len(test_audio), augment=False)
    X_test_exp3, y_test_exp3 = test_gen[0]
else:
    y_test_exp3 = le.transform(y_test_str_exp3)

model_exp3 = load_model(os.path.join("..", "models", "ser_cnn_exp3_model.keras"))

print("Evaluating on Test Set...")
loss_exp3, accuracy_exp3 = model_exp3.evaluate(X_test_exp3, y_test_exp3, verbose=0)
print(f"Experiment 3 Test Loss: {loss_exp3:.4f}")
print(f"Experiment 3 Test Accuracy: {accuracy_exp3:.4f}")
y_pred_probs_exp3 = model_exp3.predict(X_test_exp3)
y_pred_exp3 = np.argmax(y_pred_probs_exp3, axis=1)""")

add_md("## 26. Experiment 3 Confusion Matrix")
add_code("""cm_exp3 = confusion_matrix(y_test_exp3, y_pred_exp3)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_exp3, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Experiment 3 Confusion Matrix')
plt.show()""")

add_md("## 27. Experiment 3 Classification Report")
add_code("""print(classification_report(y_test_exp3, y_pred_exp3, target_names=le.classes_))""")

add_md("## 28. Comparison: Baseline vs Exp 1 vs Exp 2 vs Exp 3")
add_code("""precision_exp3, recall_exp3, f1_exp3, _ = precision_recall_fscore_support(y_test_exp3, y_pred_exp3, average='macro')
data_comp = {
    'Model': ['Baseline (1D CNN)', 'Exp 1 (Reduced)', 'Exp 2 (2D CNN + Mel)', 'Exp 3 (1D CNN + Augmentation)'],
    'Test Accuracy': [0.5056, 0.3944, accuracy_exp2, accuracy_exp3],
    'Macro Precision': [0.466, 0.380, precision, precision_exp3],
    'Macro Recall': [0.473, 0.370, recall, recall_exp3],
    'Macro F1': [0.445, 0.310, f1, f1_exp3]
}
df_comp = pd.DataFrame(data_comp)
display(df_comp)
""")

add_md("## 29. Conclusion")
add_md("""- Data Augmentation effectively tests whether artificially generating varied training samples helps the baseline model generalize better.
- Reviewing the table above reveals the exact empirical value of pitch shifting, time stretching, and noise injection for SER over the RAVDESS dataset.
- (See exact final evaluation metrics in the executed cells).""")

# Save to file
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/CodeAlpha_Task2_Emotion_Recognition.ipynb", "w", encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook generated.")
