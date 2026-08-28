import sqlite3
import pandas as pd

DB_PATH = "data/fraudlens.db"


def load_job_postings():
    connection = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        "SELECT * FROM job_postings",
        connection
    )

    connection.close()

    return df


if __name__ == "__main__":
    df = load_job_postings()

    print("Database loaded successfully.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Column names:")
    print(df.columns.tolist())