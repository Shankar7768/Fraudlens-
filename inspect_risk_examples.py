
import json
import sqlite3
import pandas as pd

FEATURE_PATH = "data/features/llm_features.json"
DB_PATH = "data/fraudlens.db"

with open(FEATURE_PATH, "r", encoding="utf-8") as file:
    records = json.load(file)

features_df = pd.DataFrame([
    {"job_id": record["job_id"], **record["features"]}
    for record in records
])

with sqlite3.connect(DB_PATH) as connection:
    jobs_df = pd.read_sql_query(
        """
        SELECT job_id, title, company_profile, description,
               requirements, fraudulent
        FROM job_postings
        """,
        connection
    )

df = features_df.merge(
    jobs_df,
    on="job_id",
    validate="one_to_one"
)

for feature in [
    "suspicious_url",
    "personal_information_request",
    "payment_request"
]:
    print(f"\n{'=' * 65}")
    print(f"EXAMPLES: {feature}")
    print("=" * 65)

    examples = df[df[feature] == 1].head(5)

    if examples.empty:
        print("No matching examples.")
        continue

    for _, row in examples.iterrows():
        print(f"\nJob ID: {row['job_id']}")
        print(f"Label: {'FRAUD' if row['fraudulent'] == 1 else 'GENUINE'}")
        print(f"Title: {row['title']}")
        print(f"Company profile: {str(row['company_profile'])[:350]}")
        print(f"Description: {str(row['description'])[:500]}")
        print(f"Requirements: {str(row['requirements'])[:300]}")

print("\nInspection only. No files were modified.")
