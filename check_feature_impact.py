
import numpy as np
from xgboost import XGBClassifier
from training.roberta_embeddings import RoBERTaEmbedder
from training.generate_llm_features import rule_based_features, FEATURES

MODEL_PATH = "models/xgboost_fraud_model.json"

job = """
Title: Remote Customer Support Executive – Work From Home
Company Profile: Bright Future Global Careers offers international remote opportunities.
Location: Remote India
Description: We urgently need customer support executives. Earn ₹95,000 per week working two hours per day. No experience required. Start immediately without an interview. Complete registration within 30 minutes.
Requirements: Submit Aadhaar, PAN, and bank account details.
Benefits: Flexible hours, high weekly earnings, guaranteed payments.
Application Process: Pay a refundable registration and training fee of ₹2,500 before joining. Send personal and banking details through WhatsApp.
"""

print("Loading model and RoBERTa...")
model = XGBClassifier()
model.load_model(MODEL_PATH)
embedder = RoBERTaEmbedder()

embedding = embedder.encode([job]).cpu().numpy()[0]
flags = rule_based_features(job)
risk = np.array([flags[name] for name in FEATURES], dtype=np.float32)

full_vector = np.concatenate([embedding, risk]).reshape(1, -1)
no_flags_vector = np.concatenate(
    [embedding, np.zeros(len(FEATURES), dtype=np.float32)]
).reshape(1, -1)

full_prob = model.predict_proba(full_vector)[0][1]
no_flags_prob = model.predict_proba(no_flags_vector)[0][1]

print("\nDIAGNOSTIC RESULTS")
print("Active warning indicators:", int(risk.sum()), "/", len(FEATURES))
print(f"With warning indicators:    {full_prob:.2%} fraud probability")
print(f"With all indicators zero:   {no_flags_prob:.2%} fraud probability")
print(f"Probability difference:     {(full_prob - no_flags_prob):+.2%}")
print("\nNo model or dataset files were modified.")
