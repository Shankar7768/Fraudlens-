
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from training.roberta_embeddings import RoBERTaEmbedder
    from training.generate_llm_features import (
        rule_based_features,
        FEATURES as RISK_FEATURES
    )
    from explainability.shap_explainer import FraudSHAPExplainer
except ImportError:
    from roberta_embeddings import RoBERTaEmbedder
    from generate_llm_features import (
        rule_based_features,
        FEATURES as RISK_FEATURES
    )
    from shap_explainer import FraudSHAPExplainer


MODEL_PATH = "models/xgboost_fraud_model.json"
NUM_ROBERTA_FEATURES = 768
NUM_RISK_FEATURES = 10
TOTAL_FEATURES = 778
FRAUD_THRESHOLD = 0.30


class FraudPredictor:
    """Extract text and risk features, predict fraud, and explain results."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                "Train it using training/train_xgboost.py first."
            )

        print("Initializing FraudPredictor...")
        self.model_path = model_path
        self.embedder = RoBERTaEmbedder()
        self.explainer = FraudSHAPExplainer(model_path=model_path)
        self.model = self.explainer.model
        self.risk_features_list = list(RISK_FEATURES)

        if len(self.risk_features_list) != NUM_RISK_FEATURES:
            raise ValueError(
                f"Expected {NUM_RISK_FEATURES} risk features, "
                f"but found {len(self.risk_features_list)}."
            )

    def extract_features(self, job_text: str) -> Dict[str, Any]:
        """Create a validated 778-dimensional feature vector."""

        if not isinstance(job_text, str) or not job_text.strip():
            raise ValueError("Job posting text cannot be empty.")

        cleaned_text = job_text.strip()

        embedding_tensor = self.embedder.encode([cleaned_text])
        roberta_embedding = embedding_tensor.cpu().numpy()[0]

        if roberta_embedding.shape != (NUM_ROBERTA_FEATURES,):
            raise ValueError(
                f"Expected a 768-dimensional embedding, "
                f"got {roberta_embedding.shape}."
            )

        risk_features_dict = rule_based_features(cleaned_text)

        risk_vector = np.array(
            [
                int(risk_features_dict.get(feature, 0))
                for feature in self.risk_features_list
            ],
            dtype=np.float32
        )

        if risk_vector.shape != (NUM_RISK_FEATURES,):
            raise ValueError(
                f"Expected 10 risk values, got {risk_vector.shape}."
            )

        combined_vector = np.concatenate(
            [roberta_embedding, risk_vector]
        ).astype(np.float32).reshape(1, -1)

        if combined_vector.shape != (1, TOTAL_FEATURES):
            raise ValueError(
                f"Expected feature shape (1, {TOTAL_FEATURES}), "
                f"got {combined_vector.shape}."
            )

        if not np.isfinite(combined_vector).all():
            raise ValueError("Features contain NaN or infinite values.")

        return {
            "combined_vector": combined_vector,
            "roberta_embedding": roberta_embedding,
            "risk_features_dict": risk_features_dict,
            "risk_vector": risk_vector
        }

    def predict(
        self,
        job_text: str,
        top_k_explanations: int = 10,
        include_shap: bool = True
    ) -> Dict[str, Any]:
        """Predict fraud probability and return risk indicators."""

        features_data = self.extract_features(job_text)
        x = features_data["combined_vector"]

        # Diagnostic: show exactly what the rules detected.
        risk_flags = features_data["risk_features_dict"]

        print("\nRisk feature diagnostic:")
        for name in self.risk_features_list:
            print(f"  {name}: {risk_flags.get(name, 0)}")

        print(
            f"Active risk flags: "
            f"{int(sum(int(risk_flags.get(name, 0)) for name in self.risk_features_list))}"
            f" / {NUM_RISK_FEATURES}"
        )

        probabilities = self.model.predict_proba(x)[0]
        fraud_prob = float(probabilities[1])
        genuine_prob = float(probabilities[0])

        
        # Final decision: combine the trained model with strong risk indicators.
        active_flags = {
            name
            for name in self.risk_features_list
            if int(risk_flags.get(name, 0)) == 1
        }

        strong_flags = {
            "payment_request",
            "unrealistic_salary",
            "personal_information_request",
            "suspicious_contact",
            "suspicious_url",
            "too_good_to_be_true",
        }

        strong_flag_count = len(active_flags & strong_flags)

        # The model can independently classify a posting as fraudulent.
        model_says_fraud = fraud_prob >= FRAUD_THRESHOLD

        # Rule override: require multiple strong signals and several
        # total warnings, rather than relying on warning count alone.
        rule_says_fraud = (
            len(active_flags) >= 4
            and strong_flag_count >= 2
        )

        predicted_class = int(model_says_fraud or rule_says_fraud)

        if predicted_class == 1:
            risk_level = (
                "HIGH RISK"
                if fraud_prob >= 0.70 or rule_says_fraud
                else "MEDIUM RISK"
            )
        elif fraud_prob >= 0.15:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"


        result = {
            "prediction": {
                "label": (
                    "FRAUDULENT"
                    if predicted_class == 1
                    else "GENUINE"
                ),
                "class": predicted_class,
                "fraud_probability": fraud_prob,
                "genuine_probability": genuine_prob,
                "risk_level": risk_level
            },
            "risk_indicators": risk_flags,
            "feature_summary": {
                "roberta_dimensions": NUM_ROBERTA_FEATURES,
                "risk_features_count": NUM_RISK_FEATURES,
                "total_features": TOTAL_FEATURES,
                "active_risk_flags": sum(
                    int(risk_flags.get(name, 0))
                    for name in self.risk_features_list
                )
            }
        }

        if include_shap:
            result["explanation"] = self.explainer.explain(
                x,
                top_k=top_k_explanations
            )

        return result


_PREDICTOR_INSTANCE: Optional[FraudPredictor] = None


def get_predictor(model_path: str = MODEL_PATH) -> FraudPredictor:
    """Return the shared predictor instance."""

    global _PREDICTOR_INSTANCE

    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = FraudPredictor(model_path=model_path)

    return _PREDICTOR_INSTANCE


def predict_job(job_text: str, top_k: int = 10) -> Dict[str, Any]:
    """Convenience function for prediction."""

    return get_predictor().predict(
        job_text,
        top_k_explanations=top_k
    )


def print_prediction_report(result: Dict[str, Any]):
    """Print the prediction and SHAP explanation."""

    pred = result["prediction"]
    summary = result["feature_summary"]
    risk_flags = result["risk_indicators"]

    print("=" * 80)
    print("FRAUDLENS JOB PREDICTION REPORT")
    print("=" * 80)
    print(f"Prediction:           {pred['label']}")
    print(f"Risk assessment:      {pred['risk_level']}")
    print(f"Fraud probability:    {pred['fraud_probability']:.2%}")
    print(f"Genuine probability:  {pred['genuine_probability']:.2%}")
    print(
        f"Active risk flags:    "
        f"{summary['active_risk_flags']} / "
        f"{summary['risk_features_count']}"
    )

    print("\nDETECTED RISK INDICATORS")
    print("-" * 60)

    for name in RISK_FEATURES:
        value = int(risk_flags.get(name, 0))
        print(f"{name:<35} {'FLAGGED' if value else 'Clear'}")

    if "explanation" in result:
        explanation = result["explanation"]
        groups = explanation["group_attribution"]

        print("\nSHAP SUMMARY")
        print("-" * 60)

        rob = groups["roberta_embeddings"]
        risk = groups["named_risk_features"]

        print(
            f"RoBERTa net SHAP impact: "
            f"{rob['net_shap_impact']:+.4f}"
        )
        print(
            f"Risk features net SHAP impact: "
            f"{risk['net_shap_impact']:+.4f}"
        )

        print("\nTOP CONTRIBUTING FEATURES")
        print("-" * 80)

        for item in explanation["top_overall_features"]:
            print(
                f"{item['feature_name']:<32} | "
                f"{item['feature_type']:<18} | "
                f"SHAP: {item['shap_value']:+.4f} | "
                f"{item['impact_direction']}"
            )

    print("=" * 80)


def test_predict_pipeline():
    """Test one suspicious and one legitimate sample."""

    predictor = get_predictor()

    suspicious_job = """
    Title: Urgent Hiring - Work From Home Data Assistant
    Company Profile: Fast growing global venture.
    Description: We are hiring immediately for remote data assistant.
    Earn $5,000 per week working 2 hours a day. No experience required.
    Freshers can apply. Guaranteed daily payments.
    Requirements: Must have smartphone or PC. Freshers welcome.
    No qualifications required.
    Benefits: High payout, work from anywhere.
    Contact: Contact our hiring manager on Telegram
    @remote_hiring_manager only. Registration fee of $50 required
    to process paperwork and ID documents. Send bank account details
    for direct deposit.
    """

    genuine_job = """
    Title: Senior Python Backend Software Engineer
    Company Profile: DataCloud Technologies is an established enterprise
    cloud computing and data infrastructure company with over 500 employees
    across North America and Europe. We provide scalable database and
    analytics solutions to Fortune 500 companies.
    Description: We are seeking an experienced Senior Software Engineer
    to join our distributed backend engineering team. You will architect
    and maintain high-throughput distributed microservices in Python,
    FastAPI, PostgreSQL, and Redis.
    Requirements: 5+ years of experience with Python software engineering,
    relational databases, SQL, API design, microservices, Docker,
    Kubernetes, and cloud platforms (GCP or AWS). Bachelor's degree in
    Computer Science or equivalent experience.
    Benefits: Competitive salary ($140,000 - $175,000 per year),
    retirement matching, health insurance, flexible PTO, remote work
    stipend, and an annual learning budget.
    """

    for label, job_text in [
        ("SUSPICIOUS JOB", suspicious_job),
        ("GENUINE JOB", genuine_job)
    ]:
        print(f"\n>>> TESTING: {label} <<<")
        result = predictor.predict(job_text)
        print_prediction_report(result)


if __name__ == "__main__":
    test_predict_pipeline()
