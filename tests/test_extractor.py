"""Unit tests for feature extraction and explainable flags."""
from src.extraction.heuristic import extract_heuristic_features
from src.extraction.llm_extractor import LLMFeatureExtractor
from src.extraction.schemas import ExplainableFlags


def test_heuristic_extraction_on_fraudulent_job():
    """Tests that heuristic extractor flags obvious fraud signals."""
    scam_job = {
        "job_id": 999,
        "title": "URGENT DATA ENTRY - EARN $5000/WEEK",
        "company_profile": "",
        "description": "Earn $5000/week! No experience needed. Contact hiring manager via Telegram @scam_recruiter. Background check fee of $100 required via gift card.",
        "requirements": "Anyone can apply. No degree needed.",
        "benefits": "Free Apple package, luxury car.",
        "has_company_logo": 0,
    }

    flags = extract_heuristic_features(scam_job)
    assert isinstance(flags, ExplainableFlags)
    assert flags.overall_fraud_risk_score >= 0.40
    assert flags.risk_level in ("HIGH", "CRITICAL")
    assert flags.recommended_action == "REJECT_AS_SCAM"
    assert len(flags.key_evidence_snippets) > 0


def test_heuristic_extraction_on_legitimate_job():
    """Tests that heuristic extractor marks legitimate job as low risk."""
    real_job = {
        "job_id": 100,
        "title": "Senior Backend Engineer",
        "company_profile": "Enterprise SaaS company established in 2012.",
        "description": "We are seeking a Backend Engineer with 5+ years of experience in distributed systems and database optimization.",
        "requirements": "BS/MS in Computer Science. Strong proficiency in Go, Python, and SQL.",
        "benefits": "Competitive salary, 401(k), healthcare, PTO.",
        "has_company_logo": 1,
    }

    flags = extract_heuristic_features(real_job)
    assert isinstance(flags, ExplainableFlags)
    assert flags.overall_fraud_risk_score < 0.30
    assert flags.risk_level == "LOW"
    assert flags.recommended_action == "APPROVE"


def test_llm_extractor_batch_processing_heuristic_mode():
    """Tests LLMFeatureExtractor in explicit heuristic fallback mode."""
    extractor = LLMFeatureExtractor(api_key="", fallback_to_heuristic=True)
    jobs = [
        {
            "job_id": 1,
            "title": "Senior Python Engineer",
            "company_profile": "Reputable tech company with 10 years of business.",
            "description": "Develop high-throughput REST APIs in Python and FastAPI. Minimum 3 years experience.",
            "requirements": "Proficiency in Python and PostgreSQL.",
            "has_company_logo": 1,
        },
        {
            "job_id": 2,
            "title": "Urgent Cash Check Mule Needed",
            "company_profile": "",
            "description": "URGENT start immediately cash check to your personal bank account. Telegram @mule",
            "has_company_logo": 0,
        },
    ]

    enriched = extractor.process_batch(jobs, show_progress=False)
    assert len(enriched) == 2
    assert enriched[0].job_id == 1
    assert enriched[0].llm_features.risk_level == "LOW"
    assert enriched[0].extraction_method == "heuristic_fallback"
    assert enriched[1].job_id == 2
    assert enriched[1].llm_features.risk_level in ("HIGH", "CRITICAL")

