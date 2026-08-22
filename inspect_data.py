import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

csv_filename = "fake_job_postings.csv"

print("=" * 80)
print("             FRAUDLENS DATASET INSPECTION REPORT")
print("=" * 80)

# Load dataset
df = pd.read_csv(csv_filename)
print(f"[+] Dataset successfully loaded: '{csv_filename}'")
print(f"[+] Total Records: {len(df):,}")
print(f"[+] Total Features: {df.shape[1]}")
print()

# 1. First 5 rows
print("=" * 80)
print("1. FIRST 5 ROWS (df.head())")
print("=" * 80)
print(df.head())
print()

# 2. Info / Data types
print("=" * 80)
print("2. COLUMN NAMES & DATA TYPES (df.info())")
print("=" * 80)
df.info()
print()

# 3. Numerical summary statistics
print("=" * 80)
print("3. SUMMARY STATISTICS FOR NUMERICAL COLUMNS (df.describe())")
print("=" * 80)
print(df.describe())
print()

# 4. Missing values
print("=" * 80)
print("4. MISSING / NULL VALUES PER COLUMN")
print("=" * 80)
null_counts = df.isnull().sum()
null_percentages = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': null_counts,
    'Missing (%)': null_percentages.round(2)
})
print(missing_df.sort_values(by='Missing Count', ascending=False))
print()

# 5. Target Class Distribution
print("=" * 80)
print("5. TARGET CLASS DISTRIBUTION ('fraudulent')")
print("=" * 80)
counts = df['fraudulent'].value_counts()
pcts = df['fraudulent'].value_counts(normalize=True) * 100
target_df = pd.DataFrame({
    'Class': ['Genuine (0)', 'Fraudulent / Fake (1)'],
    'Count': [counts.get(0, 0), counts.get(1, 0)],
    'Percentage (%)': [pcts.get(0, 0).round(2), pcts.get(1, 0).round(2)]
})
print(target_df.to_string(index=False))
print("=" * 80)
