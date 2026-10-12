
from training.xgboost_model import prepare_data
from xgboost import XGBClassifier
from training.generate_llm_features import rule_based_features, FEATURES
import numpy as np

MODEL_PATH = "models/xgboost_fraud_model.json"

suspicious_job = """
Title: Remote Customer Support Executive – Work From Home
Company Profile: Bright Future Global Careers offers international remote opportunities.
Location: Remote India
Description: We urgently need customer support executives. Earn ₹95,000 per week working two hours per day. No experience required. Start immediately without an interview. Complete registration within 30 minutes.
Requirements: Submit Aadhaar, PAN, and bank account details.
Benefits: Flexible hours, high weekly earnings, guaranteed payments.
Application Process: Pay a refundable registration and training fee of ₹2,500 before joining. Send personal and banking details through WhatsApp.
"""

print("Loading the existing model...")
model = XGBClassifier()
model.load_model(MODEL_PATH)

print("\nChecking the suspicious job's warning indicators:")
flags = rule_based_features(suspicious_job)
for feature in FEATURES:
    print(f"{feature}: {flags[feature]}")

print(f"\nTotal flagged: {sum(flags.values())}/10")

print("\nPreparing the held-out test set...")
_, _, X_test, _, _, y_test = prepare_data()

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.30).astype(int)

print(f"Test samples: {len(y_test)}")
print(f"Test fraud cases: {int(np.sum(y_test))}")
print(f"Predicted fraud cases: {int(np.sum(predictions))}")

print("\nSuspicious sample's warning vector:")
print([flags[name] for name in FEATURES])

print("\nDiagnostic complete. No files modified.")
