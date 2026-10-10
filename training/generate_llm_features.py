import json
import os
import re

try:
    from preprocess import preprocess_data
    from llm_features import extract_llm_features
except ImportError:
    from training.preprocess import preprocess_data
    from training.llm_features import extract_llm_features


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
    "too_good_to_be_true"
]


def load_existing_results():
    if not os.path.exists(OUTPUT_PATH):
        return []

    with open(OUTPUT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_results(results):
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def rule_based_features(job_text):
    """
    Deterministic fallback for jobs that cannot be sent to Gemini
    because of API quota limitations.
    """

    text = job_text.lower()

    features = {}

    # Payment or money request
    features["payment_request"] = int(
        bool(re.search(
            r"registration fee|processing fee|payment required|pay.*fee|"
            r"pay.*money|deposit|wire transfer|send money",
            text
        ))
    )

        
    # Unrealistic salary
    features["unrealistic_salary"] = int(
        bool(re.search(
            r"(?:₹|rs\.?|inr|\$)\s*"
            r"(?:10,000|15,000|20,000|30,000|50,000)"
            r"\s*(?:per|/)\s*(?:day|week)|"
            r"(?:10,000|15000|20000|30000|50000)"
            r"\s*(?:per|/)\s*day",
            text
        ))
    )


    # Urgency
    features["urgency"] = int(
        bool(re.search(
            r"urgent|immediately|hiring now|apply now|act now|"
            r"start immediately|limited time",
            text
        ))
    )

    # Personal information request
    features["personal_information_request"] = int(
        bool(re.search(
            r"bank details|bank account|credit card|social security|"
            r"passport|aadhar|aadhaar|pan card|identity document|"
            r"send.*personal information",
            text
        ))
    )

    # Suspicious contact
    features["suspicious_contact"] = int(
        bool(re.search(
            r"telegram|whatsapp.*only|contact.*gmail|contact.*yahoo|"
            r"contact.*hotmail",
            text
        ))
    )

    # Missing company information
    company_profile = ""
    if "company profile:" in text:
        company_profile = text.split(
            "company profile:", 1
        )[1].split(
            "description:", 1
        )[0].strip()

    features["missing_company_information"] = int(
        len(company_profile) < 20
    )

    # Vague job description
    description = ""
    if "description:" in text:
        description = text.split(
            "description:", 1
        )[1].split(
            "requirements:", 1
        )[0].strip()

    features["vague_job_description"] = int(
        len(description) < 100
    )

    # No experience required
    features["no_experience_required"] = int(
        bool(re.search(
            r"no experience|without experience|"
            r"experience not required|freshers welcome|"
            r"freshers can apply",
            text
        ))
    )

    # Suspicious URL
    features["suspicious_url"] = int(
        bool(re.search(
            r"http://|bit\.ly|tinyurl|t\.co/|"
            r"click here.*link|download.*link",
            text
        ))
    )

       
    # Too good to be true
    features["too_good_to_be_true"] = int(
        bool(re.search(
            r"earn.*(?:₹|rs\.?|inr|\$)?\s*\d[\d,]*.*"
            r"(?:daily|per day|/day|weekly|per week|/week|"
            r"monthly|per month|/month)|"
            r"guaranteed income|guaranteed earnings|"
            r"easy money|work.*few hours.*earn|"
            r"no experience.*high salary",
            text
        ))
    )


    return features


def generate_llm_features():
    df, _ = preprocess_data()

    results = load_existing_results()

    processed_ids = {
        item["job_id"]
        for item in results
    }

    remaining = df[
        ~df["job_id"].isin(processed_ids)
    ]

    print("Total jobs:", len(df))
    print("Existing records:", len(results))
    print("Remaining jobs:", len(remaining))

    for count, (_, row) in enumerate(
        remaining.iterrows(),
        start=1
    ):
        features = rule_based_features(
            row["job_text"]
        )

        results.append({
            "job_id": int(row["job_id"]),
            "features": features,
            "source": "rule_based_fallback"
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
    print("Gemini records already preserved: 21")
    print("Remaining records generated using deterministic rules.")


if __name__ == "__main__":
    generate_llm_features()