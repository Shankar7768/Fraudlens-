import json
import os

from database import load_job_postings
from preprocess import preprocess_data
from llm_features import extract_llm_features


OUTPUT_PATH = "data/features/llm_features_test.json"


def generate_test_features():
    df, _ = preprocess_data()

    # Test with only 5 job postings first
    test_df = df.head(5)

    results = []

    for index, row in test_df.iterrows():
        print(f"Processing test job {index + 1}/5...")

        features = extract_llm_features(row["job_text"])

        results.append({
            "job_id": int(row["job_id"]),
            "features": features
        })

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nLLM test feature generation completed.")
    print("Jobs processed:", len(results))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    generate_test_features()