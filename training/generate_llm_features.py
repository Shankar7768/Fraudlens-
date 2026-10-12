
import json
import os
import re
import sys

try:
    from preprocess import preprocess_data
except ImportError:
    from training.preprocess import preprocess_data


OUTPUT_PATH = "data/features/llm_features.json"

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
    "too_good_to_be_true",
]


def load_existing_results():
    if not os.path.exists(OUTPUT_PATH):
        return []

    with open(OUTPUT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_results(results):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)



def rule_based_features(job_text):
    """Extract ten deterministic job-posting warning indicators."""

    text = str(job_text or "").lower()
    features = {}

    # 1. Payment requests associated with recruitment or onboarding
    features["payment_request"] = int(
        bool(re.search(
            r"(?:(?:pay|send|transfer|submit).{0,50}"
            r"(?:registration|training|application|processing)"
            r".{0,20}(?:fee|deposit|payment|money))|"
            r"(?:(?:registration|training|application|processing)"
            r".{0,30}(?:fee|deposit).{0,60}"
            r"(?:before joining|before starting|to secure the job|"
            r"to get hired|prior to joining))|"
            r"refundable.{0,30}(?:fee|deposit)|"
            r"pay.{0,40}(?:to get hired|to secure the job)|"
            r"wire transfer|send money to the recruiter",
            text,
            re.IGNORECASE,
        ))
    )

    # 2. Potentially unusual daily or weekly salary.
    # This is a warning indicator, not proof that a salary is fraudulent.
    amount = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?"
    currency = r"(?:₹|rs\.?|inr|\$)\s*"

    salary_pattern = (
        currency + amount + r"\s*(?:per|/)\s*(?:day|week)\b"
        + r"|"
        + amount + r"\s*(?:rupees?|inr|dollars?)?\s*"
        + r"(?:per|/)\s*(?:day|week)\b"
    )

    features["unrealistic_salary"] = int(
        bool(re.search(salary_pattern, text, re.IGNORECASE))
    )

    # 3. Urgency, especially when combined with immediate action
    features["urgency"] = int(
        bool(re.search(
            r"urgent|immediately|hiring now|apply now|act now|"
            r"start immediately|limited time|last chance|"
            r"within \d+ minutes?|today only",
            text,
            re.IGNORECASE,
        ))
    )

    # 4. Sensitive information requests.
    # Avoid flagging passport eligibility or ordinary document requirements alone.
    sensitive_request = (
        r"(?:send|share|submit|upload|provide|message|"
        r"whatsapp|telegram).{0,60}"
        r"(?:bank account details|bank details|"
        r"credit card details|online banking credentials|"
        r"aadhaar|aadhar|pan card|social security number|"
        r"identity document|id proof|passport number)"
        r"|"
        r"(?:bank account details|bank details|"
        r"credit card details|online banking credentials|"
        r"aadhaar|aadhar|pan card|social security number|"
        r"identity document|id proof|passport number)"
        r".{0,60}(?:send|share|submit|upload|provide)"
    )

    features["personal_information_request"] = int(
        bool(re.search(sensitive_request, text, re.IGNORECASE))
    )

    # 5. Contact methods that warrant additional verification.
    # Mentioning WhatsApp or Telegram alone is not enough.
    features["suspicious_contact"] = int(
        bool(re.search(
            r"(?:contact|message|send|apply|submit).{0,50}"
            r"(?:telegram|whatsapp|gmail\.com|yahoo\.com|hotmail\.com)"
            r"|"
            r"(?:telegram|whatsapp).{0,50}"
            r"(?:fee|deposit|bank details|aadhaar|pan card|"
            r"payment|personal information)"
            r"|"
            r"message me privately|whatsapp.{0,20}only",
            text,
            re.IGNORECASE,
        ))
    )

    # 6. Missing company information.
    # This works with the labelled text format used by the current project.
    company_profile = ""

    if "company profile:" in text:
        company_profile = (
            text.split("company profile:", 1)[1]
            .split("description:", 1)[0]
            .strip()
        )

    features["missing_company_information"] = int(
        len(company_profile) < 20
        or company_profile in {"nan", "none", "not provided", "n/a"}
    )

    # 7. Vague or very short job description
    description = ""

    if "description:" in text:
        description = (
            text.split("description:", 1)[1]
            .split("requirements:", 1)[0]
            .strip()
        )

    features["vague_job_description"] = int(
        len(description) < 100
        or description in {"nan", "none", "not provided", "n/a"}
    )

    # 8. No experience required.
    # This alone should not be treated as proof of fraud.
    features["no_experience_required"] = int(
        bool(re.search(
            r"no experience(?: is)? required|without experience|"
            r"experience not required|freshers welcome|"
            r"freshers can apply|no prior experience",
            text,
            re.IGNORECASE,
        ))
    )

    # 9. Suspicious links: do not flag every ordinary HTTP URL.
    features["suspicious_url"] = int(
        bool(re.search(
            r"https?://(?:www\.)?"
            r"(?:bit\.ly|tinyurl\.com|t\.co|"
            r"cutt\.ly|is\.gd|rb\.gy|shorturl\.at)/"
            r"|"
            r"(?:click here|download this file).{0,50}"
            r"(?:verify your account|claim your payment|"
            r"enter your password|provide bank details)",
            text,
            re.IGNORECASE,
        ))
    )

    # 10. Exaggerated earnings and guaranteed-income claims
    features["too_good_to_be_true"] = int(
        bool(re.search(
            r"earn.{0,100}(?:₹|rs\.?|inr|\$)\s*"
            r"\d[\d,]*(?:\.\d{1,2})?.{0,80}"
            r"(?:daily|per day|/day|weekly|per week|/week|"
            r"monthly|per month|/month)|"
            r"guaranteed income|guaranteed earnings|"
            r"easy money|work.{0,40}few hours.{0,40}earn|"
            r"no experience.{0,40}high salary",
            text,
            re.IGNORECASE,
        ))
    )

    return features


def run_diagnostic_test():
    """Test rules without modifying the saved dataset."""

    test_job = """
    Title: Urgent Remote Data Assistant
    Company Profile: Not provided.
    Description: Earn $5,000 per week working two hours a day.
    No experience required. Guaranteed daily payments.
    Pay a registration fee immediately.
    Contact the recruiter on Telegram.
    Send bank account details.
    """

    print("FRAUDLENS RISK-FEATURE DIAGNOSTIC")
    print("-" * 45)

    features = rule_based_features(test_job)

    for name in FEATURES:
        print(f"{name}: {features[name]}")

    print("-" * 45)
    print(f"Warning checks flagged: {sum(features.values())}/10")
    print("Diagnostic complete. Saved dataset was not modified.")


def generate_llm_features():
    """Generate and save risk features for database jobs."""

    df, _ = preprocess_data()
    results = load_existing_results()

    processed_ids = {item["job_id"] for item in results}
    remaining = df[~df["job_id"].isin(processed_ids)]

    print("Total jobs:", len(df))
    print("Existing records:", len(results))
    print("Remaining jobs:", len(remaining))

    for count, (_, row) in enumerate(remaining.iterrows(), start=1):
        features = rule_based_features(row["job_text"])

        results.append({
            "job_id": int(row["job_id"]),
            "features": features,
            "source": "rule_based_fallback",
        })

        if count % 100 == 0 or count == len(remaining):
            save_results(results)
            print(
                f"Processed {count}/{len(remaining)} "
                f"| Total saved: {len(results)}"
            )

    save_results(results)

    print("\nFeature generation completed.")
    print("Total feature records:", len(results))
    print("No CSV files are used or created by this script.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_diagnostic_test()
    else:
        generate_llm_features()
