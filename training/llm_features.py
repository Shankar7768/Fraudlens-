import json


FEATURES = [
    "payment_request",
    "unrealistic_salary",
    "urgency",
    "personal_information_request",
    "suspicious_contact",
    "missing_company_information",
    "vague_job_description",
    "no_experience_required",
    "suspicious_url",
    "too_good_to_be_true"
]


def create_feature_prompt(job_text):
    feature_list = "\n".join(
        f"- {feature}" for feature in FEATURES
    )

    return f"""
Analyze the following job posting for potential fraud indicators.

Return ONLY valid JSON.
Each feature must have a value of 0 or 1.

Features:
{feature_list}

Job Posting:
{job_text}

JSON format:
{{
    "payment_request": 0,
    "unrealistic_salary": 0,
    "urgency": 0,
    "personal_information_request": 0,
    "suspicious_contact": 0,
    "missing_company_information": 0,
    "vague_job_description": 0,
    "no_experience_required": 0,
    "suspicious_url": 0,
    "too_good_to_be_true": 0
}}
"""


def validate_features(response_text):
    try:
        features = json.loads(response_text)

        return {
            feature: int(features.get(feature, 0))
            for feature in FEATURES
        }

    except (json.JSONDecodeError, ValueError, TypeError):
        return {feature: 0 for feature in FEATURES}