# Speech Emotion Recognition (SER)

This project implements a robust Convolutional Neural Network (CNN) to classify emotions from speech using the RAVDESS dataset. 

## Project Objective
To build a Speech Emotion Recognition (SER) system that accurately classifies 8 emotions using the RAVDESS dataset. This project documents the full machine learning pipeline: data preprocessing, exploratory data analysis, class imbalance handling, modeling, and controlled experimentation. A key requirement is strictly avoiding speaker leakage to ensure robust generalization to unseen actors.

## Dataset and Splits

The dataset is the **RAVDESS** dataset.
It contains 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

**Note: The RAVDESS dataset is excluded from GitHub via `.gitignore` due to size constraints.**

The dataset is split cleanly by **Actor ID** to prevent the model from memorizing voice characteristics (data leakage).
*   **Train Actors:** `Actor_01` through `Actor_18`
*   **Validation Actors:** `Actor_19` through `Actor_21`
*   **Test Actors:** `Actor_22` through `Actor_24`

## Preprocessing & Architecture
The project strictly separates responsibilities to ensure data integrity, prevent leakage, and guarantee that inference utilizes the exact same preprocessing pipeline as training.

- **Preprocessing:** Audio files are normalized, padded/truncated to 4 seconds, and converted to mono at a 16kHz sampling rate.
- **MFCC:** We extract 40 Mel-Frequency Cepstral Coefficients (MFCCs) as features for our primary pipeline.
- **CNN:** A high-capacity 1D Convolutional Neural Network (CNN) processes the sequential MFCC features.
- **Data Augmentation:** On-the-fly additive Gaussian noise, time stretching, and pitch shifting are applied.

## Experiments and Results

1. **Baseline Model (1D CNN + MFCC)**
   - High-capacity 1D CNN over MFCC features.
   - Result: Test Accuracy = **0.5056**, Macro F1 = **0.445**. The model severely overfit the training actors but failed to generalize well to unseen test actors.

2. **Experiment 1 (Reduced Capacity + Regularization)**
   - Drastically reduced 1D CNN filters and added Dropout/L2 regularization.
   - Result: Test Accuracy = **0.3944**, Macro F1 = **0.310**. The model underfit, confirming that we shouldn't artificially cripple capacity to solve generalization issues.

3. **Experiment 2 (2D CNN + Mel-Spectrogram)**
   - Switched from MFCCs to 128-band Log-Mel-Spectrograms, leveraging a 2D CNN.
   - Result: Test Accuracy = **0.2333**. The performance dropped significantly due to severe overfitting of the parameter-heavy 2D CNN on this small dataset.

4. **Experiment 3 (1D CNN + Data Augmentation)**
   - Baseline 1D CNN over MFCCs with on-the-fly Data Augmentation exclusively applied to the training set.
   - Result:
     - **Best Validation Accuracy = 0.7056**
     - **Test Accuracy = 0.5944**
     - **Macro F1 = 0.59**
     - **Weighted F1 = 0.59**
   - This experiment successfully bridged the generalization gap, significantly raising test accuracy making it our most robust architecture.

## How to Run (Setup Instructions)

1.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Tests:**
    Verify data loader and feature extraction logic:
    ```bash
    python -m pytest tests/
    ```

3.  **Notebook Usage:**
    The final executed notebook is available at `notebooks/CodeAlpha_Task2_Emotion_Recognition.ipynb`. You can open it and explore the entire ML pipeline, visualization, and inference examples.

4.  **Streamlit Usage:**
    Run the real-time inference web app (Uses final Exp 3 model):
    ```bash
    streamlit run app.py
    ```
