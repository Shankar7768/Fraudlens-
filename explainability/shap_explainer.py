import os
import sys
from pathlib import Path
from typing import Dict, Any, Union, List
import numpy as np
import shap
from xgboost import XGBClassifier

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from training.xgboost_model import FEATURES as RISK_FEATURES, prepare_data
except ImportError:
    from xgboost_model import FEATURES as RISK_FEATURES, prepare_data


MODEL_PATH = "models/xgboost_fraud_model.json"

NUM_ROBERTA_FEATURES = 768
NUM_RISK_FEATURES = 10
TOTAL_FEATURES = 778
ZERO_THRESHOLD = 1e-4

ROBERTA_FEATURE_NAMES = [f"roberta_embedding_{i}" for i in range(NUM_ROBERTA_FEATURES)]
ALL_FEATURE_NAMES = ROBERTA_FEATURE_NAMES + RISK_FEATURES


def get_impact_direction(shap_value: float, epsilon: float = ZERO_THRESHOLD) -> str:
    """
    Determine the contribution direction based on the actual SHAP value:
      - Positive SHAP (> epsilon): 'Increases Fraud Risk'
      - Negative SHAP (< -epsilon): 'Decreases Fraud Risk'
      - Near-zero SHAP (|SHAP| <= epsilon): 'Minimal Contribution'
    """
    if shap_value > epsilon:
        return "Increases Fraud Risk"
    elif shap_value < -epsilon:
        return "Decreases Fraud Risk"
    else:
        return "Minimal Contribution"


class FraudSHAPExplainer:
    """
    SHAP TreeExplainer for FraudLens XGBoost predictions.
    Explains individual predictions by computing Shapley values across:
      - 768 RoBERTa text semantic embedding features
      - 10 explicit domain risk indicator features
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        zero_threshold: float = ZERO_THRESHOLD
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                f"Please train the model first using training/train_xgboost.py."
            )

        self.model_path = model_path
        self.zero_threshold = zero_threshold
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        self.explainer = shap.TreeExplainer(self.model)

        self.roberta_feature_names = ROBERTA_FEATURE_NAMES
        self.risk_feature_names = RISK_FEATURES
        self.all_feature_names = ALL_FEATURE_NAMES

    def explain(
        self,
        feature_vector: Union[np.ndarray, List[float]],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Compute SHAP explanation for a single 778-dimensional job feature vector.

        Parameters
        ----------
        feature_vector : np.ndarray or list
            Feature vector of shape (778,) or (1, 778).
        top_k : int
            Number of top contributing features to include in summary.

        Returns
        -------
        dict
            Structured explanation including prediction, probability, base value,
            top overall features, explicit 10 risk features, and RoBERTa aggregate impact.
        """
        x = np.asarray(feature_vector, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != TOTAL_FEATURES:
            raise ValueError(
                f"Expected {TOTAL_FEATURES} features (768 RoBERTa + 10 Risk Features), "
                f"but got {x.shape[1]} features."
            )

        # Model Inference
        predicted_class = int(self.model.predict(x)[0])
        probabilities = self.model.predict_proba(x)[0]
        fraud_probability = float(probabilities[1])
        genuine_probability = float(probabilities[0])

        # SHAP calculation
        shap_result = self.explainer(x)
        shap_values = shap_result.values[0]
        base_value = float(shap_result.base_values[0])
        net_margin = float(base_value + shap_values.sum())

        # Compile feature-level records
        feature_records = []
        for i, name in enumerate(self.all_feature_names):
            is_risk_feature = i >= NUM_ROBERTA_FEATURES
            val = float(x[0, i])
            shap_val = float(shap_values[i])
            direction = get_impact_direction(shap_val, self.zero_threshold)
            feature_records.append({
                "feature_index": i,
                "feature_name": name,
                "feature_type": "risk_feature" if is_risk_feature else "roberta_embedding",
                "feature_value": val,
                "shap_value": shap_val,
                "impact_direction": direction
            })

        # Rank all features by absolute SHAP contribution
        sorted_by_impact = sorted(
            feature_records,
            key=lambda item: abs(item["shap_value"]),
            reverse=True
        )

        # Top features pushing towards fraud vs genuine
        top_fraud_drivers = [
            item for item in sorted_by_impact
            if item["shap_value"] > self.zero_threshold
        ][:top_k]
        top_genuine_drivers = [
            item for item in sorted_by_impact
            if item["shap_value"] < -self.zero_threshold
        ][:top_k]

        # Explicit 10 Named Risk Features Breakdown
        risk_features_breakdown = feature_records[NUM_ROBERTA_FEATURES:]

        # RoBERTa Embeddings Aggregate Impact
        roberta_records = feature_records[:NUM_ROBERTA_FEATURES]
        roberta_shap_vals = [r["shap_value"] for r in roberta_records]
        roberta_pos_impact = sum(v for v in roberta_shap_vals if v > 0)
        roberta_neg_impact = sum(v for v in roberta_shap_vals if v < 0)
        roberta_net_impact = sum(roberta_shap_vals)
        roberta_abs_impact = sum(abs(v) for v in roberta_shap_vals)

        # Risk Features Aggregate Impact
        risk_shap_vals = [r["shap_value"] for r in risk_features_breakdown]
        risk_pos_impact = sum(v for v in risk_shap_vals if v > 0)
        risk_neg_impact = sum(v for v in risk_shap_vals if v < 0)
        risk_net_impact = sum(risk_shap_vals)
        risk_abs_impact = sum(abs(v) for v in risk_shap_vals)

        return {
            "prediction": {
                "predicted_label": "FRAUDULENT" if predicted_class == 1 else "GENUINE",
                "predicted_class": predicted_class,
                "fraud_probability": fraud_probability,
                "genuine_probability": genuine_probability,
                "base_value": base_value,
                "net_margin": net_margin
            },
            "top_overall_features": sorted_by_impact[:top_k],
            "top_fraud_drivers": top_fraud_drivers,
            "top_genuine_drivers": top_genuine_drivers,
            "named_risk_features": risk_features_breakdown,
            "group_attribution": {
                "roberta_embeddings": {
                    "total_features": NUM_ROBERTA_FEATURES,
                    "net_shap_impact": float(roberta_net_impact),
                    "positive_shap_impact": float(roberta_pos_impact),
                    "negative_shap_impact": float(roberta_neg_impact),
                    "total_absolute_shap": float(roberta_abs_impact)
                },
                "named_risk_features": {
                    "total_features": NUM_RISK_FEATURES,
                    "net_shap_impact": float(risk_net_impact),
                    "positive_shap_impact": float(risk_pos_impact),
                    "negative_shap_impact": float(risk_neg_impact),
                    "total_absolute_shap": float(risk_abs_impact)
                }
            }
        }

    def explain_job(
        self,
        roberta_embedding: Union[np.ndarray, List[float]],
        llm_risk_features: Union[np.ndarray, List[float], Dict[str, int]],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Convenience method to explain a job given separate RoBERTa embedding
        and LLM risk features (either list/array or dictionary).
        """
        emb = np.asarray(roberta_embedding, dtype=np.float32).flatten()
        if len(emb) != NUM_ROBERTA_FEATURES:
            raise ValueError(f"Expected {NUM_ROBERTA_FEATURES} RoBERTa embedding dimensions, got {len(emb)}")

        if isinstance(llm_risk_features, dict):
            risk_vec = np.array(
                [int(llm_risk_features.get(f, 0)) for f in self.risk_feature_names],
                dtype=np.float32
            )
        else:
            risk_vec = np.asarray(llm_risk_features, dtype=np.float32).flatten()

        if len(risk_vec) != NUM_RISK_FEATURES:
            raise ValueError(f"Expected {NUM_RISK_FEATURES} risk features, got {len(risk_vec)}")

        combined = np.concatenate([emb, risk_vec])
        return self.explain(combined, top_k=top_k)

    def print_explanation(self, explanation: Dict[str, Any], top_k: int = 10):
        """
        Pretty-print the SHAP explanation report to the console.
        """
        pred = explanation["prediction"]
        groups = explanation["group_attribution"]

        print("=" * 80)
        print("                   FRAUDLENS SHAP EXPLANATION REPORT")
        print("=" * 80)
        print(f"Prediction:         {pred['predicted_label']} (Class: {pred['predicted_class']})")
        print(f"Fraud Probability:  {pred['fraud_probability']:.2%}")
        print(f"Base Value (prior): {pred['base_value']:.4f}")
        print(f"Net Margin Score:   {pred['net_margin']:.4f}")

        print("\n" + "-" * 80)
        print("FEATURE GROUP ATTRIBUTION SUMMARY")
        print("-" * 80)
        rob_grp = groups["roberta_embeddings"]
        risk_grp = groups["named_risk_features"]
        print(f"1. 768 RoBERTa Embeddings -> Net Impact: {rob_grp['net_shap_impact']:+.4f} | Total Magnitude: {rob_grp['total_absolute_shap']:.4f}")
        print(f"2. 10 Named Risk Features -> Net Impact: {risk_grp['net_shap_impact']:+.4f} | Total Magnitude: {risk_grp['total_absolute_shap']:.4f}")

        print("\n" + "-" * 80)
        print(f"TOP {top_k} MOST INFLUENTIAL FEATURES (OVERALL)")
        print("-" * 80)
        print(f"{'Feature Name':<32} | {'Type':<18} | {'Value':<8} | {'SHAP Value':<10}")
        print("-" * 80)
        for item in explanation["top_overall_features"][:top_k]:
            val_str = f"{item['feature_value']:.0f}" if item["feature_type"] == "risk_feature" else f"{item['feature_value']:.3f}"
            print(
                f"{item['feature_name']:<32} | "
                f"{item['feature_type']:<18} | "
                f"{val_str:<8} | "
                f"{item['shap_value']:+.4f}"
            )

        print("\n" + "-" * 80)
        print("10 NAMED RISK FEATURES BREAKDOWN")
        print("-" * 80)
        print(f"{'Risk Indicator':<32} | {'Flag Value':<10} | {'SHAP Impact':<12} | Direction")
        print("-" * 80)
        for item in explanation["named_risk_features"]:
            print(
                f"{item['feature_name']:<32} | "
                f"{int(item['feature_value']):<10} | "
                f"{item['shap_value']:+.4f}       | "
                f"{item['impact_direction']}"
            )
        print("=" * 80)


def test_explainer():
    """
    Test the SHAP explainer on a sample job posting from the test dataset.
    """
    print("Initializing FraudSHAPExplainer...")
    explainer = FraudSHAPExplainer()

    print("Loading test dataset features...")
    _, _, X_test, _, _, y_test = prepare_data()

    # Select a sample with high fraud probability for a meaningful explanation demo
    sample_index = 115
    test_sample = X_test[sample_index]
    actual_label = int(y_test[sample_index])

    print(f"\nExplaining Job Posting (Test Set Index: {sample_index}, Ground Truth: {'FRAUDULENT' if actual_label == 1 else 'GENUINE'})")

    explanation = explainer.explain(test_sample, top_k=10)
    explainer.print_explanation(explanation, top_k=10)


if __name__ == "__main__":
    test_explainer()
