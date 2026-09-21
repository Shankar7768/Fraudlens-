import json
import os
import time

from preprocess import preprocess_data
from llm_features import extract_llm_features


OUTPUT_PATH = "data/features/llm_features.json"
DELAY_SECONDS = 1


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
    print("Already processed:", len(results))
    print("Remaining:", len(remaining))

    for count, (_, row) in enumerate(
        remaining.iterrows(),
        start=1
    ):

        max_retries = 5

        try:
            for attempt in range(max_retries):
                try:
                    features = extract_llm_features(
                        row["job_text"]
                    )
                    break

                except Exception as error:
                    if "503" in str(error) or "UNAVAILABLE" in str(error):
                        wait_time = 10 * (attempt + 1)

                        print(
                            f"Gemini temporarily unavailable. "
                            f"Retrying in {wait_time} seconds..."
                        )

                        time.sleep(wait_time)

                        if attempt == max_retries - 1:
                            raise
                    else:
                        raise

            results.append({
                "job_id": int(row["job_id"]),
                "features": features
            })

            save_results(results)

            print(
                f"Processed {count}/{len(remaining)} "
                f"| Total saved: {len(results)}"
            )

            time.sleep(DELAY_SECONDS)

        except Exception as error:
            print("\nError occurred:")
            print(error)
            print("\nProgress has been saved.")
            print("Run the same script again to resume.")
            break


if __name__ == "__main__":
    generate_llm_features()