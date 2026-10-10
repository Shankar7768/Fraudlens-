
import json
import os

try:
    from preprocess import preprocess_data
    from generate_llm_features import rule_based_features
except ImportError:
    from training.preprocess import preprocess_data
    from training.generate_llm_features import rule_based_features


OUTPUT_PATH = "data/features/llm_features.json"


def rebuild():
    print("Loading job postings from SQLite database...")

    df, _ = preprocess_data()
    df = df.sort_values("job_id").reset_index(drop=True)

    results = []

    print("Generating rule-based features for all jobs...")

    for _, row in df.iterrows():
        job_text = str(row["job_text"])

        results.append({
            "job_id": int(row["job_id"]),
            "features": rule_based_features(job_text),
            "source": "rule_based_current"
        })

    # Write to a temporary file first for safety.
    temp_path = OUTPUT_PATH + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    os.replace(temp_path, OUTPUT_PATH)

    print("\nFeature rebuilding completed!")
    print(f"Total records: {len(results)}")
    print(f"First job ID: {results[0]['job_id']}")
    print(f"Last job ID: {results[-1]['job_id']}")
    print("Gemini API calls made: 0")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    rebuild()
