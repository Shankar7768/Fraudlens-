"""Rule-based heuristic extractor for offline analysis, baselines, and fallback."""
import re
from typing import Any, Dict, List, Tuple
from src.extraction.schemas import ExplainableFlags


def _check_patterns(text: str, patterns: List[Tuple[str, str]]) -> Tuple[float, List[str], List[str]]:
    """Checks a text string against regex pattern tuples (regex, flag_name).

    Returns:
        (score, matched_flags, evidence_snippets)
    """
    matched_flags: List[str] = []
    evidence: List[str] = []

    for pattern, flag in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            matched_flags.append(flag)
            snippet = match.group(0)
            if snippet not in evidence:
                evidence.append(snippet)

    # Score calculation proportional to matches, capped at 1.0
    score = min(1.0, len(matched_flags) * 0.50)
    return score, matched_flags, evidence


def extract_heuristic_features(job: Dict[str, Any]) -> ExplainableFlags:
    """Extracts explainable fraud flags using heuristic rules and regex patterns.

    Args:
        job: Job posting dictionary.

    Returns:
        ExplainableFlags populated with heuristic indicators.
    """
    full_text = " ".join([
        str(job.get("title") or ""),
        str(job.get("company_profile") or ""),
        str(job.get("description") or ""),
        str(job.get("requirements") or ""),
        str(job.get("benefits") or ""),
    ])

    all_evidence: List[str] = []

    # 1. Urgency Patterns
    urgency_patterns = [
        (r"\b(urgent|immediate hire|start immediately|hiring today|act now|hurry|apply fast)\b", "high_urgency_call_to_action"),
        (r"\b(limited slots|limited vacancies|only a few spots)\b", "artificial_scarcity_pressure"),
        (r"\b(no interview (needed|required)|instant hire|hired on the spot)\b", "no_interview_hiring"),
    ]
    urgency_score, urgency_flags, u_ev = _check_patterns(full_text, urgency_patterns)
    all_evidence.extend(u_ev)

    # 2. Financial Patterns
    financial_patterns = [
        (r"\b(cash check|wire transfer|western union|moneygram|money order|send a check|sent a check)\b", "suspicious_money_transfer"),
        (r"\b(bitcoin|crypto|usdt|ethereum|wallet address|gift card)\b", "cryptocurrency_or_gift_card_payment"),
        (r"\b(background check fee|application fee|training fee|purchase .* kit|buy .* equipment|fee of \$\d+)\b", "upfront_fee_or_equipment_scheme"),
        (r"\b(personal bank account|receive funds into your account|forward payment)\b", "money_mule_or_personal_account_use"),
    ]
    fin_score, fin_flags, f_ev = _check_patterns(full_text, financial_patterns)
    all_evidence.extend(f_ev)

    # 3. Unrealistic Promise Patterns
    promise_patterns = [
        (r"\b(\$?\s*5000\s*(/|per)\s*week|\$?\s*[2-9]\d{3,}\s*(/|per)\s*week|earn\s+\$?\d{4,}\s*(daily|weekly)|earn\s+up\s+to\s+\$?\d{4,})\b", "excessive_compensation_promise"),
        (r"\b(luxury (company )?car|free apple (package|equipment)|guaranteed (bonuses|wealth))\b", "unrealistic_luxury_perk"),
        (r"\b(get rich|guaranteed income|earn from home with no effort)\b", "get_rich_quick_claims"),
    ]
    promise_score, promise_flags, p_ev = _check_patterns(full_text, promise_patterns)
    all_evidence.extend(p_ev)

    # 4. Suspicious Contact Patterns
    contact_patterns = [
        (r"(telegram\s*(@|:|\b)|whatsapp\s*(@|:|\b|\+)|@\w*recruiter\b)", "informal_messaging_app_interview"),
        (r"\b([a-zA-Z0-9_.+-]+@(gmail|yahoo|hotmail|outlook)\.com)\b", "free_webmail_recruiter_address"),
        (r"\b(bit\.ly|tinyurl\.com|t\.me)\b", "shortened_unverified_url"),
    ]
    contact_score, contact_flags, c_ev = _check_patterns(full_text, contact_patterns)
    all_evidence.extend(c_ev)

    # 5. Vague Requirements Patterns
    req_patterns = [
        (r"\b(no (degree|skills|experience) (required|needed)|anyone can apply|no background needed)\b", "zero_qualification_barrier"),
        (r"\b(simple (data entry|copy paste|typing work))\b", "generic_low_effort_hook"),
    ]
    req_score, req_flags, r_ev = _check_patterns(full_text, req_patterns)
    all_evidence.extend(r_ev)

    # 6. Company Legitimacy Checks
    company_profile = str(job.get("company_profile") or "").strip()
    company_flags: List[str] = []
    company_score = 0.0

    if not company_profile or len(company_profile) < 20:
        company_score += 0.4
        company_flags.append("missing_or_sparse_company_profile")
    if job.get("has_company_logo") == 0:
        company_score += 0.2
        company_flags.append("missing_company_logo")

    # Aggregate Overall Fraud Risk Score
    weights = [0.15, 0.35, 0.20, 0.15, 0.10, 0.05]
    scores = [urgency_score, fin_score, promise_score, contact_score, req_score, company_score]
    base_score = sum(w * s for w, s in zip(weights, scores))

    # Multi-category compounding: severe scams trigger red flags across multiple distinct areas
    active_fraud_dimensions = sum(1 for s in [urgency_score, fin_score, promise_score, contact_score] if s > 0)
    compound_boost = 0.10 * max(0, active_fraud_dimensions - 1) if active_fraud_dimensions >= 2 else 0.0

    overall_score = round(min(1.0, max(0.0, base_score + compound_boost)), 3)

    # Determine Risk Tier
    if overall_score >= 0.65 or fin_score >= 0.70:
        risk_level = "CRITICAL"
        action = "REJECT_AS_SCAM"
    elif overall_score >= 0.40 or (fin_score > 0 and contact_score > 0):
        risk_level = "HIGH"
        action = "REJECT_AS_SCAM"
    elif overall_score >= 0.20:
        risk_level = "MEDIUM"
        action = "FLAG_FOR_MANUAL_REVIEW"
    else:
        risk_level = "LOW"
        action = "APPROVE"

    # Generate Explanation Summary
    if risk_level in ("HIGH", "CRITICAL"):
        explanation = (
            f"Heuristic analysis detected significant high-risk scam indicators (Risk Score: {overall_score:.2f}). "
            f"Key triggers include: {', '.join(urgency_flags + fin_flags + promise_flags + contact_flags) or 'Multiple subtle anomalies'}."
        )
    elif risk_level == "MEDIUM":
        explanation = (
            f"Heuristic analysis detected moderate risk flags (Risk Score: {overall_score:.2f}) such as "
            f"{', '.join(req_flags + company_flags) or 'incomplete company disclosures'}. Manual review recommended."
        )
    else:
        explanation = (
            f"Posting displays standard recruitment characteristics with low risk score ({overall_score:.2f}). "
            "No severe financial or contact anomalies detected."
        )

    return ExplainableFlags(
        urgency_pressure_score=round(urgency_score, 2),
        urgency_flags=urgency_flags,
        financial_anomaly_score=round(fin_score, 2),
        financial_red_flags=fin_flags,
        unrealistic_promises_score=round(promise_score, 2),
        promise_flags=promise_flags,
        suspicious_contact_score=round(contact_score, 2),
        contact_red_flags=contact_flags,
        vague_requirements_score=round(req_score, 2),
        requirement_flags=req_flags,
        company_legitimacy_score=round(company_score, 2),
        company_red_flags=company_flags,
        overall_fraud_risk_score=overall_score,
        risk_level=risk_level,
        explanation_summary=explanation,
        key_evidence_snippets=all_evidence[:10],
        recommended_action=action,
    )
