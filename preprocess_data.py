import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

print("=" * 80)
print("             FRAUDLENS DATA PREPROCESSING & VECTORIZATION")
print("=" * 80)

# -------------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------------
csv_filename = "fake_job_postings.csv"
df = pd.read_csv(csv_filename)
print(f"[+] Loaded raw dataset: {df.shape[0]:,} rows, {df.shape[1]} columns\n")

# -------------------------------------------------------------
# 2. Handle Missing Text Values
# -------------------------------------------------------------
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
print("[+] Step 1: Handling missing text values...")
for col in text_cols:
    df[col] = df[col].fillna('')
print(f"    Cleaned text columns: {text_cols}\n")

# -------------------------------------------------------------
# 3. Combine Relevant Text Columns into 'combined_text'
# -------------------------------------------------------------
print("[+] Step 2: Combining relevant text fields into 'combined_text'...")
df['combined_text'] = (
    df['title'] + " " +
    df['company_profile'] + " " +
    df['description'] + " " +
    df['requirements'] + " " +
    df['benefits']
).str.strip()

print(f"    Sample combined text preview (first 120 chars):")
print(f"    \"{df['combined_text'].iloc[0][:120]}...\"\n")

# -------------------------------------------------------------
# 4. Prepare Features (X) and Target (y)
# -------------------------------------------------------------
print("[+] Step 3: Preparing Target (y) and Features (X)...")
X_raw_text = df['combined_text']
y = df['fraudulent']

# -------------------------------------------------------------
# 5. Train / Test Split (Stratified to maintain 4.8% fraud ratio)
# -------------------------------------------------------------
print("[+] Step 4: Splitting into Train (80%) and Test (20%) sets (Stratified)...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_raw_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"    Train size: {len(X_train_text):,} samples | Genuine: {(y_train == 0).sum():,}, Fake: {(y_train == 1).sum():,} ({y_train.mean()*100:.2f}%)")
print(f"    Test size:  {len(X_test_text):,} samples  | Genuine: {(y_test == 0).sum():,}, Fake: {(y_test == 1).sum():,} ({y_test.mean()*100:.2f}%)\n")

# -------------------------------------------------------------
# 6. TF-IDF Vectorization (Fitted on Train only to prevent data leakage)
# -------------------------------------------------------------
print("[+] Step 5: Applying TF-IDF Vectorization (max_features=10,000, n-grams=(1,2))...")
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

print(f"    X_train_tfidf Shape: {X_train_tfidf.shape} (Sparse Matrix)")
print(f"    X_test_tfidf  Shape: {X_test_tfidf.shape} (Sparse Matrix)\n")

# -------------------------------------------------------------
# 7. Summary & Sample Extracted Features
# -------------------------------------------------------------
print("=" * 80)
print("                   PREPROCESSING SUMMARY")
print("=" * 80)
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"[+] Total Vocabulary Size: {len(feature_names):,} n-grams")
print(f"[+] Sample Top Features:   {list(feature_names[500:515])}")
print("[+] Ready for ML Model Training (Random Forest, XGBoost, Logistic Regression, etc.)!")
print("=" * 80)
