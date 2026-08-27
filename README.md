# FraudLens: Explainable Fake Job Detection Using LLM-Assisted Feature Engineering

## 📌 Architecture Overview

FraudLens is designed around a multi-stage explainable fraud detection pipeline:

```
[Raw Job Posting]
       │
       ▼
[1. LLM-Assisted Feature Engineering] (Google Gemini / Domain Red-Flag Extraction)
       │
       ├──────────────────────────────┐
       ▼                              ▼
[2. RoBERTa Embeddings]       [Structured Risk Features]
(Contextual Text Representation) (Quantitative & Categorical Risk Scores)
       │                              │
       └──────────────┬───────────────┘
                      ▼
            [3. XGBoost Classifier]
            (Gradient-Boosted Decision Trees)
                      │
                      ▼
            [4. SHAP Explainability]
       (Feature Importance & Attribution)
                      │
                      ▼
[FastAPI Backend & Interactive Dashboard]
```

---

## 📁 Project Structure

```text
fraudlens/
├── backend/            # FastAPI REST API endpoints & server logic
├── data/               # Non-CSV formatted data storage (JSON Lines / SQLite)
│   ├── raw/            # Ingested raw job postings
│   └── processed/      # LLM-enriched & preprocessed dataset
├── explainability/     # SHAP explainability & feature attribution modules
├── frontend/           # Interactive user interface (to be implemented)
├── models/             # Serialized model artifacts, embeddings & weights
├── training/           # Training scripts & feature extraction pipelines
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```
