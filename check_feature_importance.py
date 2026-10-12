
from xgboost import XGBClassifier
from training.generate_llm_features import FEATURES

MODEL_PATH = "models/xgboost_fraud_model.json"

model = XGBClassifier()
model.load_model(MODEL_PATH)

importance = model.feature_importances_

print("\nFEATURE IMPORTANCE")
print("-" * 45)

# The first 768 features are RoBERTa embedding dimensions.
# The final 10 features are the risk indicators.
roberta_importance = importance[:768]
risk_importance = importance[768:778]

print(f"Total RoBERTa importance: {roberta_importance.sum():.6f}")
print(f"Total risk-feature importance: {risk_importance.sum():.6f}")

print("\nIndividual risk indicators:")
for name, value in sorted(
    zip(FEATURES, risk_importance),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{name:35s} {value:.6f}")

print("\nNo model or dataset files were modified.")
