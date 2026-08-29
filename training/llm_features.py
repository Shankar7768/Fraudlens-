import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)


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

Return JSON in exactly this format:
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


def extract_llm_features(job_text):
    prompt = create_feature_prompt(job_text)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    response_text = response.text.strip()

    # Remove Markdown code fences if Gemini adds them
    if response_text.startswith("```"):
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        response_text = response_text.strip()

    features = json.loads(response_text)

    return {
        feature: int(features.get(feature, 0))
        for feature in FEATURES
    }


if __name__ == "__main__":
    sample_job = """
    We are hiring immediately. Earn $10,000 per month working from home.
    No experience required. Send your bank details and pay a registration fee
    to start immediately.
    """

    features = extract_llm_features(sample_job)

    print("LLM feature extraction successful.")
    print(json.dumps(features, indent=2))