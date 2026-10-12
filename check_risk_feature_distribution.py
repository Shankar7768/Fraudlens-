
import json
import pandas as pd

FEATURE_PATH = "data/features/llm_features.json"
DB_PATH = "data/fraudlens.db"

with open(FEATURE_PATH, "r", encoding="utf-8") as file:
    records = json.load(file)

features_df = pd.DataFrame([
    {"job_id": record["job_id"], **record["features"]}
    for record in records
])

# Load the actual labels from the project's SQLite database.
import sqlite3

with sqlite3.connect(DB_PATH) as connection:
    labels_df = pd.read_sql_query(
        "SELECT job_id, fraudulent FROM job_postings ORDER BY job_id",
        connection
    )

df = features_df.merge(labels_df, on="job_id", validate="one_to_one")

print("\nRISK INDICATOR DISTRIBUTION BY LABEL")
print("0 = Genuine, 1 = Fraudulent\n")

for column in features_df.columns:
    if column == "job_id":
        continue

    genuine_rate = df.loc[df["fraudulent"] == 0, column].mean() * 100
    fraud_rate = df.loc[df["fraudulent"] == 1, column].mean() * 100

    print(
        f"{column:35s} "
        f"Genuine: {genuine_rate:6.2f}% | "
        f"Fraudulent: {fraud_rate:6.2f}%"
    )

print("\nNo model or dataset files were modified.")
