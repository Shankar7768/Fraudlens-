
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from training.xgboost_model import prepare_data
from xgboost import XGBClassifier

MODEL_PATH = "models/xgboost_fraud_model.json"

print("Loading test data...")
_, _, X_test, _, _, y_test = prepare_data()

print("\nLoading existing trained model...")
model = XGBClassifier()
model.load_model(MODEL_PATH)

probabilities = model.predict_proba(X_test)[:, 1]

print("\nTEST SET THRESHOLD COMPARISON")
print("-" * 65)
print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Fraud flagged':<15}")

for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    predictions = (probabilities >= threshold).astype(int)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    flagged = int(predictions.sum())

    print(
        f"{threshold:<12.2f}{precision:<12.3f}"
        f"{recall:<12.3f}{f1:<12.3f}{flagged:<15}"
    )

print("\nNo model or dataset files were modified.")
