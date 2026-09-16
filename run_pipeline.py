from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    print("=== Pipeline Started ===")
    print("--- Training ---")
    train_model()
    print("--- Evaluation ---")
    evaluate_model()
    print("=== Pipeline Completed ===")
