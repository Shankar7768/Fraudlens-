"""Unit tests for Pydantic schemas in Fraud Lens."""
import pytest
from pydantic import ValidationError

from src.extraction.schemas import EnrichedJobRecord, ExplainableFlags


def test_valid_explainable_flags():
    """Tests creating a valid ExplainableFlags instance."""
    flags = ExplainableFlags(
        urgency_pressure_score=0.8,
        urgency_flags=["immediate_hire_without_interview"],
        financial_anomaly_score=0.9,
        financial_red_flags=["fake_check_reimbursement"],
        unrealistic_promises_score=0.7,
        promise_flags=["excessive_salary_for_entry_level"],
        suspicious_contact_score=0.85,
        contact_red_flags=["telegram_or_whatsapp_interview"],
        vague_requirements_score=0.6,
        requirement_flags=["zero_experience_necessary"],
        company_legitimacy_score=0.75,
        company_red_flags=["missing_company_profile"],
        overall_fraud_risk_score=0.82,
        risk_level="CRITICAL",
        explanation_summary="High probability scam with check reimbursement and Telegram contact.",
        key_evidence_snippets=["Earn $5000/week", "Contact on Telegram @HiringMgr"],
        recommended_action="REJECT_AS_SCAM",
    )
    assert flags.overall_fraud_risk_score == 0.82
    assert flags.risk_level == "CRITICAL"
    assert "REJECT_AS_SCAM" == flags.recommended_action
    assert len(flags.key_evidence_snippets) == 2


def test_invalid_score_bounds():
    """Tests that scores out of [0.0, 1.0] raise ValidationError."""
    with pytest.raises(ValidationError):
        ExplainableFlags(
            urgency_pressure_score=1.5,  # Invalid: > 1.0
            financial_anomaly_score=0.0,
            unrealistic_promises_score=0.0,
            suspicious_contact_score=0.0,
            vague_requirements_score=0.0,
            company_legitimacy_score=0.0,
            overall_fraud_risk_score=0.5,
            risk_level="MEDIUM",
            explanation_summary="Test",
            recommended_action="APPROVE",
        )


def test_enriched_job_record_serialization():
    """Tests EnrichedJobRecord serialization and deserialization."""
    flags = ExplainableFlags(
        urgency_pressure_score=0.1,
        financial_anomaly_score=0.0,
        unrealistic_promises_score=0.0,
        suspicious_contact_score=0.0,
        vague_requirements_score=0.1,
        company_legitimacy_score=0.0,
        overall_fraud_risk_score=0.05,
        risk_level="LOW",
        explanation_summary="Normal posting.",
        key_evidence_snippets=[],
        recommended_action="APPROVE",
    )
    record = EnrichedJobRecord(
        job_id=101,
        title="Software Engineer",
        description="Write clean code in Python.",
        fraudulent=0,
        llm_features=flags,
        extraction_method="heuristic_fallback",
    )

    data_dict = record.model_dump()
    assert data_dict["job_id"] == 101
    assert data_dict["llm_features"]["risk_level"] == "LOW"
    assert data_dict["extraction_method"] == "heuristic_fallback"
