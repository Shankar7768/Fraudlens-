import sqlite3
import pandas as pd

connection = sqlite3.connect("data/fraudlens.db")

query = """
SELECT
    fraudulent,
    COUNT(*) AS total_jobs,
    SUM(
        CASE
            WHEN salary_range IS NOT NULL
             AND TRIM(salary_range) != ''
            THEN 1
            ELSE 0
        END
    ) AS jobs_with_salary
FROM job_postings
GROUP BY fraudulent
"""

data = pd.read_sql_query(query, connection)
print(data.to_string(index=False))

connection.close()
