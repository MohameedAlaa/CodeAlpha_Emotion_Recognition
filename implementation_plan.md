# Implementation Plan: Restructuring the ML Notebook

The goal is to deeply refactor `notebooks/CodeAlpha_Task2_Emotion_Recognition.ipynb` to clearly demonstrate the complete ML pipeline, explicitly show training configurations, and replace all mocked data with real artifacts, all while preserving the integrity of the project's saved artifacts.

## Proposed Changes

### 1. Refactor `create_notebook_final.py`
Instead of manually editing the notebook, I will rewrite `create_notebook_final.py` to systematically generate the required notebook structure.

The generated notebook will include these exact sections:
1. **Project Objective**
2. **Imports / Reproducibility**
3. **Dataset Loading** (using `src.data_loader`)
4. **Dataset Inspection & EDA**
5. **Emotion Distribution**
6. **Audio Preprocessing & Feature Extraction**
7. **Actor-Level Train/Validation/Test Split** (with explicit proof of 01-18, 19-21, 22-24 split)
8. **Speaker Leakage Explanation**
9. **Class Imbalance Handling** (showing class weights)
10. **Baseline Model Architecture** (reading configuration from `src.config`)
11. **Baseline Training Configuration & Results** (documenting 0.5056 acc)
12. **Experiment 1 (Regularization)** (documenting 0.3944 acc)
13. **Experiment 2 (2D CNN)** (documenting 0.2333 acc)
14. **Experiment 3 (Data Augmentation)**
15. **Experiment 3 Training Configuration** (Explicitly printing Batch Size, Epochs, LR, Optimizer, Augmentation settings from `src.config`)
16. **Experiment 3 Training** (Cell with `RUN_TRAINING = False`, failing clearly if history is missing)
17. **Experiment 3 Training History** (Plotting real artifact)
18. **Final Test Evaluation** (Evaluating `ser_cnn_exp3_model.keras` to yield 0.5944)
19. **Classification Report, Confusion Matrix, Error Analysis**
20. **Conclusion**

### 2. Refactor `src/train.py`
I will update `train_model` in `src/train.py` to automatically save the training history to `results/history_exp3.pkl` via `pickle`.

### 3. Notebook Execution Constraint
The notebook will be structured like this for Experiment 3 training:
```python
import os, pickle

RUN_TRAINING = False
history_path = "../results/history_exp3.pkl"

if RUN_TRAINING:
    from src.train import train_model
    history = train_model()
else:
    if not os.path.exists(history_path):
        raise FileNotFoundError("Real training history artifact not found. Please set RUN_TRAINING = True to generate it.")
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
```

## User Review Required

> [!WARNING]
> **Missing Training History Artifact**
> The original training history object for Experiment 3 was never saved to disk. 
> Since you requested that the notebook fallback `RUN_TRAINING = False` must FAIL clearly if the history artifact doesn't exist, running `nbconvert` on the fresh notebook will currently result in a `FileNotFoundError` (and a failed validation). 

## Open Questions

> [!IMPORTANT]
> How would you like me to resolve the missing `history_exp3.pkl` artifact so that `nbconvert` can complete with **ZERO execution errors**?
> 
> **Option A:** I can run a quick background script to train a model for the full 50 epochs to generate a real `history_exp3.pkl`, but **discard** the resulting model to preserve your exact `ser_cnn_exp3_model.keras` (0.5944 accuracy). 
> **Option B:** I can set `RUN_TRAINING = True` in the notebook, but configure it to save the history and model to a temporary path, leaving your `ser_cnn_exp3_model.keras` untouched.
> **Option C:** I can catch the exception in the notebook and print a Markdown error instead of raising a python exception, allowing `nbconvert` to pass without the artifact.

Please let me know which option you prefer before I proceed!

## Verification Plan
1. Generate the notebook via the updated `create_notebook_final.py`.
2. Apply the chosen solution for the missing history artifact.
3. Run `pytest tests/`.
4. Run `jupyter nbconvert --execute --to notebook --inplace notebooks/CodeAlpha_Task2_Emotion_Recognition.ipynb`.
5. Verify zero mock data and exact preservation of 0.5944 accuracy.
