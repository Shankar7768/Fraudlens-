import os
import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np

# Ensure project root is in sys.path for direct script and module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from training.roberta_embeddings import RoBERTaEmbedder
    from training.generate_llm_features import rule_based_features, FEATURES as RISK_FEATURES
    from explainability.shap_explainer import FraudSHAPExplainer
except ImportError:
    from roberta_embeddings import RoBERTaEmbedder
    from generate_llm_features import rule_based_features, FEATURES as RISK_FEATURES
    from shap_explainer import FraudSHAPExplainer


MODEL_PATH = "models/xgboost_fraud_model.json"
NUM_ROBERTA_FEATURES = 768
NUM_RISK_FEATURES = 10
TOTAL_FEATURES = 778


class FraudPredictor:
    """
    End-to-end inference engine for FraudLens:
      1. Receives raw job posting text.
      2. Generates 768-dimensional RoBERTa semantic text embeddings.
      3. Extracts 10 deterministic domain risk indicator features.
      4. Combines into a 778-dimensional feature representation.
      5. Runs XGBoost fraud classification to produce label & probability.
      6. Executes SHAP explainability to attribute prediction drivers.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                f"Please train the model first using training/train_xgboost.py."
            )

        print("Initializing FraudPredictor inference components...")
        self.model_path = model_path
        self.embedder = RoBERTaEmbedder()
        self.explainer = FraudSHAPExplainer(model_path=model_path)
        self.model = self.explainer.model
        self.risk_features_list = RISK_FEATURES

    def extract_features(self, job_text: str) -> Dict[str, Any]:
        """
        Transform raw job posting text into the unified 778-dimensional feature vector.
        """
        cleaned_text = job_text.strip()
        if not cleaned_text:
            raise ValueError("Job posting text cannot be empty.")

        # 1. 768-dimensional RoBERTa embeddings
        embedding_tensor = self.embedder.encode([cleaned_text])
        roberta_embedding = embedding_tensor.cpu().numpy()[0]  # shape: (768,)

        # 2. 10 deterministic domain risk features
        risk_features_dict = rule_based_features(cleaned_text)
        risk_vector = np.array(
            [int(risk_features_dict.get(feat, 0)) for feat in self.risk_features_list],
            dtype=np.float32
        )  # shape: (10,)

        # 3. Concatenate into 778-dimensional feature vector
        combined_vector = np.concatenate([roberta_embedding, risk_vector]).reshape(1, -1)

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
        """
        Run fraud prediction and SHAP explanation on a new job posting text.

        Parameters
        ----------
        job_text : str
            Full text content of the job posting.
        top_k_explanations : int
            Number of top SHAP feature drivers to retrieve.
        include_shap : bool
            Whether to compute SHAP explanations.

        Returns
        -------
        dict
            Prediction verdict, fraud probability, risk features, and SHAP report.
        """
        features_data = self.extract_features(job_text)
        x = features_data["combined_vector"]

        # Classification Inference
        predicted_class = int(self.model.predict(x)[0])
        probabilities = self.model.predict_proba(x)[0]
        fraud_prob = float(probabilities[1])
        genuine_prob = float(probabilities[0])

        # Categorize risk level
        if fraud_prob >= 0.70:
            risk_level = "HIGH RISK"
        elif fraud_prob >= 0.30:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"

        result = {
            "prediction": {
                "label": "FRAUDULENT" if predicted_class == 1 else "GENUINE",
                "class": predicted_class,
                "fraud_probability": fraud_prob,
                "genuine_probability": genuine_prob,
                "risk_level": risk_level
            },
            "risk_indicators": features_data["risk_features_dict"],
            "feature_summary": {
                "roberta_dimensions": NUM_ROBERTA_FEATURES,
                "risk_features_count": NUM_RISK_FEATURES,
                "total_features": TOTAL_FEATURES,
                "active_risk_flags": int(sum(features_data["risk_features_dict"].values()))
            }
        }

        # SHAP Explainability
        if include_shap:
            shap_explanation = self.explainer.explain(x, top_k=top_k_explanations)
            result["explanation"] = shap_explanation

        return result


_PREDICTOR_INSTANCE: Optional[FraudPredictor] = None


def get_predictor(model_path: str = MODEL_PATH) -> FraudPredictor:
    """
    Get or initialize the singleton FraudPredictor instance.
    """
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = FraudPredictor(model_path=model_path)
    return _PREDICTOR_INSTANCE


def predict_job(job_text: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Convenience functional API to predict fraud and explain a job posting.
    """
    predictor = get_predictor()
    return predictor.predict(job_text, top_k_explanations=top_k)


def print_prediction_report(result: Dict[str, Any]):
    """
    Display a formatted terminal prediction report.
    """
    pred = result["prediction"]
    summary = result["feature_summary"]
    risk_flags = result["risk_indicators"]

    print("=" * 80)
    print("                     FRAUDLENS JOB PREDICTION REPORT")
    print("=" * 80)
    print(f"Prediction Verdict:     {pred['label']} (Class {pred['class']})")
    print(f"Risk Assessment:        {pred['risk_level']}")
    print(f"Fraud Probability:      {pred['fraud_probability']:.2%}")
    print(f"Genuine Probability:    {pred['genuine_probability']:.2%}")
    print(f"Active Risk Flags:      {summary['active_risk_flags']} of {summary['risk_features_count']}")

    print("\n" + "-" * 80)
    print("DETECTED DOMAIN RISK INDICATORS")
    print("-" * 80)
    print(f"{'Risk Indicator':<35} | {'Detected Flag'}")
    print("-" * 80)
    for k, v in risk_flags.items():
        flag_str = "YES (FLAGGED)" if v == 1 else "NO (Clear)"
        print(f"{k:<35} | {flag_str}")

    if "explanation" in result:
        expl = result["explanation"]
        groups = expl["group_attribution"]
        print("\n" + "-" * 80)
        print("SHAP EXPLAINABILITY SUMMARY")
        print("-" * 80)
        rob_grp = groups["roberta_embeddings"]
        risk_grp = groups["named_risk_features"]
        print(f"Prior Base Value (log-odds):  {expl['prediction']['base_value']:.4f}")
        print(f"Net Margin Score (log-odds):  {expl['prediction']['net_margin']:.4f}")
        print(f"RoBERTa Text Semantic Impact: {rob_grp['net_shap_impact']:+.4f} (Magnitude: {rob_grp['total_absolute_shap']:.4f})")
        print(f"10 Risk Indicators Impact:    {risk_grp['net_shap_impact']:+.4f} (Magnitude: {risk_grp['total_absolute_shap']:.4f})")

        print("\n" + "-" * 80)
        print("TOP CONTRIBUTING FEATURES TO PREDICTION")
        print("-" * 80)
        print(f"{'Feature Name':<32} | {'Type':<18} | {'Value':<8} | {'SHAP Impact':<12} | Direction")
        print("-" * 80)
        for item in expl["top_overall_features"]:
            val_str = f"{item['feature_value']:.0f}" if item["feature_type"] == "risk_feature" else f"{item['feature_value']:.3f}"
            print(
                f"{item['feature_name']:<32} | "
                f"{item['feature_type']:<18} | "
                f"{val_str:<8} | "
                f"{item['shap_value']:+.4f}      | "
                f"{item['impact_direction']}"
            )
    print("=" * 80)


def test_predict_pipeline():
    """
    Test prediction and explainability on both suspicious and legitimate sample job postings.
    """
    print("\n" + "#" * 80)
    print("RUNNING FRAUDLENS PREDICTION PIPELINE TESTS")
    print("#" * 80 + "\n")

    predictor = get_predictor()

    # Test Sample 1: Highly suspicious job posting
    suspicious_job = """
    Title: Urgent Hiring - Work From Home Data Assistant
    Company Profile: Fast growing global venture.
    Description: We are hiring immediately for remote data assistant. Earn $5,000 per week working 2 hours a day. No experience required. Freshers can apply. Guaranteed daily payments.
    Requirements: Must have smartphone or PC. Freshers welcome. No qualifications required.
    Benefits: High payout, work from anywhere.
    Contact: Contact our hiring manager on Telegram @remote_hiring_manager only. Registration fee of $50 required to process paperwork and ID documents. Send bank account details for direct deposit.
    """

    print("\n>>> TESTING SAMPLE 1: SUSPICIOUS JOB POSTING <<<")
    result1 = predictor.predict(suspicious_job)
    print_prediction_report(result1)

    # Test Sample 2: Genuine job posting
    genuine_job = """
    Title: Senior Python Backend Software Engineer
    Company Profile: DataCloud Technologies is an established enterprise cloud computing and data infrastructure company with over 500 employees across North America and Europe. We provide scalable database and analytics solutions to Fortune 500 companies.
    Description: We are seeking an experienced Senior Software Engineer to join our distributed backend engineering team. You will architect and maintain high-throughput distributed microservices in Python, FastAPI, PostgreSQL, and Redis.
    Requirements: 5+ years of experience with Python software engineering, experience with relational databases and SQL, strong background in API design, microservices, Docker, Kubernetes, and cloud platforms (GCP or AWS). Bachelor's degree in Computer Science or equivalent experience.
    Benefits: Competitive salary ($140,000 - $175,000 per year), 401(k) matching, comprehensive health, dental, and vision insurance, flexible PTO, remote work stipend, annual learning budget.
    """

    print("\n>>> TESTING SAMPLE 2: GENUINE JOB POSTING <<<")
    result2 = predictor.predict(genuine_job)
    print_prediction_report(result2)


if __name__ == "__main__":
    test_predict_pipeline()
