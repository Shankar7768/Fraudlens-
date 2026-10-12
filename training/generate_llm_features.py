
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

    
    # 1. Payment or money request
    features["payment_request"] = int(
        bool(re.search(
            r"registration.{0,30}fee|"
            r"training.{0,30}fee|"
            r"application.{0,30}fee|"
            r"processing.{0,30}fee|"
            r"refundable.{0,30}(?:fee|deposit)|"
            r"pay.{0,80}(?:fee|deposit|money|amount)|"
            r"(?:fee|deposit).{0,80}(?:before|prior to|"
            r"to secure|to receive|to get|to start)|"
            r"payment required|"
            r"wire transfer|send money",
            text,
            re.IGNORECASE,
        ))
    )


    # 2. Potentially unusual daily or weekly salary
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

    # 3. Urgency
    features["urgency"] = int(
        bool(re.search(
            r"urgent|immediately|hiring now|apply now|act now|"
            r"start immediately|limited time|last chance",
            text,
        ))
    )

    # 4. Sensitive personal information
    features["personal_information_request"] = int(
        bool(re.search(
            r"bank details|bank account|credit card|social security|"
            r"passport|aadhar|aadhaar|pan card|identity document|"
            r"send.{0,40}personal information|upload.{0,30}id proof",
            text,
        ))
    )

    # 5. Suspicious or difficult-to-verify contact
    features["suspicious_contact"] = int(
        bool(re.search(
            r"telegram|whatsapp.{0,20}only|"
            r"contact.{0,30}gmail|contact.{0,30}yahoo|"
            r"contact.{0,30}hotmail|message me privately",
            text,
        ))
    )

    # 6. Missing company information
    company_profile = ""

    if "company profile:" in text:
        company_profile = (
            text.split("company profile:", 1)[1]
            .split("description:", 1)[0]
            .strip()
        )

    features["missing_company_information"] = int(
        len(company_profile) < 20
    )

    # 7. Vague job description
    description = ""

    if "description:" in text:
        description = (
            text.split("description:", 1)[1]
            .split("requirements:", 1)[0]
            .strip()
        )

    features["vague_job_description"] = int(
        len(description) < 100
    )

    # 8. No experience required
    features["no_experience_required"] = int(
        bool(re.search(
            r"no experience|without experience|"
            r"experience not required|freshers welcome|"
            r"freshers can apply",
            text,
        ))
    )

    # 9. Suspicious URL
    features["suspicious_url"] = int(
        bool(re.search(
            r"http://|bit\.ly|tinyurl|t\.co/|"
            r"click here.{0,40}link|download.{0,40}link",
            text,
        ))
    )

    # 10. Exaggerated earnings claims
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
