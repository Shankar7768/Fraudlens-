"""
FraudLens FastAPI Application
REST API server for explainable fake job detection.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path for direct script and module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.predict import get_predictor, predict_job, FraudPredictor
except ImportError:
    from predict import get_predictor, predict_job, FraudPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to initialize and warm up model on startup.
    """
    try:
        get_predictor()
    except Exception as exc:
        print(f"Warning: Predictor warmup deferred: {exc}")
    yield


app = FastAPI(
    title="FraudLens API",
    description="Explainable Fake Job Detection API using RoBERTa, LLM Risk Features, XGBoost, and SHAP.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    job_text: str = Field(
        ...,
        description="Full text of the job posting to analyze for fraudulent patterns",
        example="Urgent hiring for remote data assistant. Earn $5,000 per week. No experience required. Telegram @recruiter."
    )
    top_k: Optional[int] = Field(
        default=10,
        description="Number of top SHAP contributing features to return"
    )


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Classification label: 'FRAUDULENT' or 'GENUINE'")
    fraud_probability: float = Field(..., description="Estimated probability that the job is fraudulent (0.0 - 1.0)")
    genuine_probability: float = Field(..., description="Estimated probability that the job is genuine (0.0 - 1.0)")
    risk_level: str = Field(..., description="Categorical risk assessment: 'LOW RISK', 'MEDIUM RISK', or 'HIGH RISK'")
    active_risk_flags: int = Field(..., description="Count of triggered domain risk indicators")
    top_shap_features: List[Dict[str, Any]] = Field(
        ...,
        description="Top contributing features ranked by absolute SHAP impact with impact direction"
    )
    risk_indicators: Optional[Dict[str, int]] = Field(
        default=None,
        description="Detailed dictionary of all 10 domain risk indicators and their trigger status (0 or 1)"
    )
    explanation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Comprehensive SHAP explanation report including group attribution and prior base value"
    )


@app.get("/", summary="Health and Status Check")
def read_root():
    """
    Root endpoint verifying that the FraudLens API is active and running.
    """
    return {"message": "FraudLens API is running"}


@app.post("/predict", response_model=PredictResponse, summary="Analyze Job Posting")
def predict(request: PredictRequest):
    """
    Analyze a job posting text for fraud risk using the FraudLens inference pipeline:
      1. Extract 768-dimensional RoBERTa semantic embeddings
      2. Extract 10 domain risk indicator features
      3. Classify with trained XGBoost model
      4. Compute SHAP explanations for top contributing features
    """
    cleaned_text = request.job_text.strip()
    if not cleaned_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="job_text must not be empty or whitespace only."
        )

    try:
        predictor = get_predictor()
        top_k = request.top_k if request.top_k and request.top_k > 0 else 10
        raw_result = predictor.predict(cleaned_text, top_k_explanations=top_k, include_shap=True)

        prediction_info = raw_result.get("prediction", {})
        summary_info = raw_result.get("feature_summary", {})
        explanation_info = raw_result.get("explanation", {})

        top_shap_features = explanation_info.get("top_overall_features", [])

        return PredictResponse(
            prediction=prediction_info.get("label", "UNKNOWN"),
            fraud_probability=prediction_info.get("fraud_probability", 0.0),
            genuine_probability=prediction_info.get("genuine_probability", 0.0),
            risk_level=prediction_info.get("risk_level", "UNKNOWN"),
            active_risk_flags=summary_info.get("active_risk_flags", 0),
            top_shap_features=top_shap_features,
            risk_indicators=raw_result.get("risk_indicators"),
            explanation=explanation_info,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction pipeline error: {str(exc)}"
        )
