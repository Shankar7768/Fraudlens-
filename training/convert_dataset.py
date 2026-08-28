import sqlite3
import pandas as pd

CSV_PATH = "data/raw/fake_job_postings.csv"
DB_PATH = "data/fraudlens.db"

# Read the original dataset
df = pd.read_csv(CSV_PATH)

# Create SQLite database
connection = sqlite3.connect(DB_PATH)

# Store dataset in SQLite
df.to_sql("job_postings", connection, if_exists="replace", index=False)

connection.close()

print("Dataset successfully converted to SQLite.")
print(f"Rows inserted: {len(df)}")
print(f"Database created: {DB_PATH}")