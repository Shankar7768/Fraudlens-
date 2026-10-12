
import json
import os

from training.preprocess import preprocess_data
from training.generate_llm_features import FEATURES, rule_based_features

OUTPUT_PATH = "data/features/llm_features.json"
BACKUP_PATH = "data/features/llm_features_backup_20261012.json"


def main():
    if not os.path.exists(BACKUP_PATH):
        raise FileNotFoundError(
            f"Backup not found: {BACKUP_PATH}. Nothing was changed."
        )

    print("Loading jobs from SQLite...")
    df, _ = preprocess_data()

    print("Rebuilding risk features with current rules...")
    results = []

    for _, row in df.sort_values("job_id").iterrows():
        features = rule_based_features(row["job_text"])

        results.append({
            "job_id": int(row["job_id"]),
            "features": {name: int(features[name]) for name in FEATURES},
            "source": "rule_based_current",
        })

    ids = [item["job_id"] for item in results]

    if len(results) != 17880 or len(set(ids)) != len(ids):
        raise ValueError(
            f"Validation failed: records={len(results)}, unique IDs={len(set(ids))}. "
            "Original feature file was not overwritten."
        )

    temp_path = OUTPUT_PATH + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    # Verify the temporary JSON before replacing the current file.
    with open(temp_path, "r", encoding="utf-8") as file:
        check = json.load(file)

    if len(check) != 17880 or [x["job_id"] for x in check] != sorted(ids):
        os.remove(temp_path)
        raise ValueError("Temporary file validation failed. Original file preserved.")

    os.replace(temp_path, OUTPUT_PATH)

    print("\nFeature rebuild completed successfully.")
    print("Total records:", len(results))
    print("Unique job IDs:", len(set(ids)))
    print("Feature count per job:", len(FEATURES))
    print("Backup preserved:", BACKUP_PATH)
    print("\nFeature totals:")
    for name in FEATURES:
        print(f"  {name}: {sum(x['features'][name] for x in results)}")


if __name__ == "__main__":
    main()
