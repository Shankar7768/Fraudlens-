import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("=" * 80)
print("       FRAUDLENS: LOGISTIC REGRESSION TRAINING & EVALUATION")
print("=" * 80)

# 1. Load Data
csv_filename = "fake_job_postings.csv"
df = pd.read_csv(csv_filename)

# 2. Fill Missing Values & Combine Text
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_cols:
    df[col] = df[col].fillna('')

df['combined_text'] = (
    df['title'] + " " +
    df['company_profile'] + " " +
    df['description'] + " " +
    df['requirements'] + " " +
    df['benefits']
).str.strip()

X = df['combined_text']
y = df['fraudulent']

# 3. Stratified Train-Test Split (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. TF-IDF Vectorization
print("[+] Vectorizing text with TF-IDF (10,000 features, n-grams=(1,2))...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train Logistic Regression Model
print("[+] Training Logistic Regression classifier...")
model = LogisticRegression(
    class_weight='balanced',  # Handles the 4.84% class imbalance
    max_iter=1000,
    random_state=42,
    C=1.0
)
model.fit(X_train_tfidf, y_train)

# 6. Evaluation on Test Set
y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1)
rec = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 80)
print("                      MODEL PERFORMANCE SUMMARY")
print("=" * 80)
print(f"Accuracy:                {acc * 100:.2f}%")
print(f"Precision (Fake / 1):    {prec * 100:.2f}%")
print(f"Recall    (Fake / 1):    {rec * 100:.2f}%")
print(f"F1-Score  (Fake / 1):    {f1 * 100:.2f}%")
print(f"ROC-AUC Score:           {roc_auc * 100:.2f}%")

print("\n--- Confusion Matrix ---")
print(f"True Negatives  (Genuine correctly identified): {cm[0][0]:,}")
print(f"False Positives (Genuine flagged as Fake):       {cm[0][1]:,}")
print(f"False Negatives (Fake missed by model):          {cm[1][0]:,}")
print(f"True Positives  (Fake correctly caught):         {cm[1][1]:,}")

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Genuine (0)', 'Fake / Fraudulent (1)'], digits=4))

# 7. Feature Explainability (Top words indicating Fake vs Genuine)
feature_names = np.array(vectorizer.get_feature_names_out())
coefs = model.coef_[0]

top_fake_idx = np.argsort(coefs)[-15:][::-1]
top_genuine_idx = np.argsort(coefs)[:15]

print("=" * 80)
print("            TOP LINGUISTIC SIGNALS LEARNED BY THE MODEL")
print("=" * 80)
print(" Top Features Correlated with FRAUDULENT Postings:")
for idx in top_fake_idx:
    print(f"   • {feature_names[idx]:<25} (Weight: +{coefs[idx]:.4f})")

print("\n Top Features Correlated with GENUINE Postings:")
for idx in top_genuine_idx:
    print(f"   • {feature_names[idx]:<25} (Weight: {coefs[idx]:.4f})")
print("=" * 80)
