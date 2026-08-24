# FraudLens: Explainable Fake Job Detection Using LLM-Assisted Feature Engineering

FraudLens is an end-to-end machine learning project designed to detect fake and fraudulent job postings on online job boards using advanced NLP, feature engineering, and explainable AI.

---

##  Dataset
* **Dataset Name**: EMSCAD (Employment Scam Aegean Dataset) / Real or Fake Job Postings
* **Total Records**: 17,880 postings (17,014 Genuine, 866 Fraudulent)
* **Target Variable**: `fraudulent` (0 = Genuine, 1 = Fake / Scam)

---

##  Project Structure
```text
FraudLens/
├── inspect_data.py               # Dataset EDA, missing value & class distribution analysis
├── preprocess_data.py            # Text cleaning, n-gram TF-IDF vectorization & stratified splits
├── train_logistic_regression.py  # Baseline model training, evaluation metrics & feature weights
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## ⚡ Baseline Model Performance (Logistic Regression + TF-IDF)
* **Accuracy**: 97.87%
* **Precision (Fraudulent)**: 72.35%
* **Recall (Fraudulent)**: 90.75%
* **F1-Score (Fraudulent)**: 80.51%
* **ROC-AUC**: 98.83%

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Inspect Dataset
```bash
python inspect_data.py
```

### 3. Preprocess & Vectorize Text
```bash
python preprocess_data.py
```

### 4. Train & Evaluate Baseline Classifier
```bash
python train_logistic_regression.py
```
