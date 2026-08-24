# FraudLens: Explainable Fake Job Detection Using LLM-Assisted Feature Engineering

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Format](https://img.shields.io/badge/format-JSONLines%20(.jsonl)-orange.svg)](https://jsonlines.org/)
[![Architecture](https://img.shields.io/badge/Model-RoBERTa%20%2B%20XGBoost%20%2B%20SHAP-purple.svg)]()

**FraudLens** is an explainable fake job detection framework combining **LLM-assisted feature engineering**, **RoBERTa contextual text embeddings**, **XGBoost gradient-boosted decision trees**, and **SHAP (SHapley Additive exPlanations)**.

---

## 🌟 Key Features

- **Standard Benchmark Dataset (EMSCAD / Kaggle)**: Seamlessly loads and standardizes all 17,880 postings from the EMSCAD (Employment Scam Aegean Dataset) mirror on Hugging Face (`victor/real-or-fake-fake-jobposting-prediction`) or local Kaggle CSVs.
- **Robust Preprocessing & 70/15/15 Splitting**: Performs HTML entity decoding, whitespace normalization, text deduplication, and automated stratified splitting into **70% Training (`train.jsonl`)**, **15% Validation (`val.jsonl`)**, and **15% Testing (`test.jsonl`)**.
- **Strict JSON Lines Architecture**: Saves all partitions strictly as newline-delimited JSON Lines (`.jsonl`), avoiding CSV quote-escaping corruptions across multi-paragraph job descriptions.
- **LLM-Assisted Feature Engineering**: Uses Google Gemini (`gemini-3.6-flash` / `gemini-3.7-flash`) with structured schema generation to extract 6 risk dimensions:
  1. ⚡ **Urgency & Pressure Tactics**: High-pressure urgency, immediate hiring without interviews, artificial scarcity.
  2. 💳 **Financial & Payment Anomalies**: Upfront fee demands, fake check cashing/reshipping, gift card payments, crypto schemes.
  3. 💎 **Unrealistic Promises**: Disproportionate compensation for entry-level roles, get-rich-quick guarantees.
  4. 📱 **Suspicious Contact Channels**: Telegram/WhatsApp interview funnels, free personal webmail addresses (`@gmail.com`).
  5. ❓ **Vague Requirements**: Non-existent skill criteria, generic copy-paste responsibilities.
  6. 🏢 **Company Legitimacy Signals**: Sparse/missing corporate profiles, missing logos, identity impersonation clues.
- **Hybrid Modeling Architecture**: Designed for RoBERTa contextual text representation + XGBoost classification + SHAP feature importance attribution.
- **Heuristic Baseline & Offline Mode**: Built-in regex and heuristic extractors for zero-cost execution, benchmarking, and offline test environments.

---

## 📁 Project Structure

```text
Fraudlens/
├── config/
│   ├── __init__.py
│   └── settings.py               # Centralized configuration (paths, models, environment)
├── data/
│   ├── raw/                      # Ingested raw dataset (dataset.jsonl)
│   └── processed/                # Enriched dataset (dataset_with_features.jsonl)
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── ingest_dataset.py     # Fetches dataset & strictly saves as JSON Lines
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── schemas.py            # Pydantic structured output models
│   │   ├── prompts.py            # Domain-specific LLM fraud investigator prompts
│   │   ├── heuristic.py          # Fast rule-based heuristic extractor
│   │   └── llm_extractor.py      # Google GenAI LLM feature extractor
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── run_pipeline.py       # End-to-end orchestration CLI
│   └── utils/
│       ├── __init__.py
│       ├── io_helpers.py         # Streaming JSONL reader and writer
│       └── logger.py             # Formatted logging utility
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py         # JSONL streaming and ingestion tests
│   ├── test_schemas.py           # Pydantic schema validation tests
│   └── test_extractor.py         # LLM and heuristic extraction tests
├── .env.example                  # Sample environment configuration
├── .gitignore                    # Git ignore rules
├── README.md                     # Documentation
└── requirements.txt              # Project dependencies
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/Fraudlens.git
cd Fraudlens

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the sample environment file and add your Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
LLM_MODEL=gemini-3.7-flash
```

---

## 🛠️ Usage

### Step 1: Ingest Dataset into JSON Lines

Fetches the Real or Fake Job Postings dataset from Hugging Face and saves it strictly as `data/raw/dataset.jsonl`:

```bash
# Ingest full dataset
python src/ingestion/ingest_dataset.py

# Ingest a limited sample for testing
python src/ingestion/ingest_dataset.py --limit 100

# Use built-in synthetic benchmark dataset (no network required)
python src/ingestion/ingest_dataset.py --synthetic
```

### Step 2: Extract Explainable Features with LLM

Run the complete pipeline to extract explainable risk flags and write the enriched dataset to `data/processed/dataset_with_features.jsonl`:

```bash
# Run with Gemini LLM
python src/pipeline/run_pipeline.py --limit 50

# Run with heuristic rule-based mode (zero API cost / offline)
python src/pipeline/run_pipeline.py --use-heuristic --limit 100
```

---

## 📊 Sample Enriched Output (JSONL)

Each line in `dataset_with_features.jsonl` contains the original job metadata along with structured LLM features:

```json
{
  "job_id": 2,
  "title": "URGENT Data Entry Assistant - Earn $5000/Week",
  "salary_range": "200000-250000",
  "fraudulent": 1,
  "llm_features": {
    "urgency_pressure_score": 0.95,
    "urgency_flags": ["high_urgency_call_to_action", "no_interview_hiring"],
    "financial_anomaly_score": 0.90,
    "financial_red_flags": ["fake_check_reimbursement", "equipment_purchase_scheme"],
    "unrealistic_promises_score": 0.95,
    "promise_flags": ["excessive_compensation_promise", "unrealistic_luxury_perk"],
    "suspicious_contact_score": 1.0,
    "contact_red_flags": ["informal_messaging_app_interview"],
    "vague_requirements_score": 0.85,
    "requirement_flags": ["zero_qualification_barrier"],
    "company_legitimacy_score": 0.80,
    "company_red_flags": ["missing_company_logo", "sparse_profile"],
    "overall_fraud_risk_score": 0.91,
    "risk_level": "CRITICAL",
    "explanation_summary": "Posting exhibits critical indicators of advance-fee / check cashing recruitment fraud, offering $5000/wk for data entry and routing candidates to Telegram.",
    "key_evidence_snippets": [
      "Earn up to $5000 per week",
      "message our hiring manager directly on Telegram @HiringMgr99",
      "You will be sent a check to purchase home office equipment"
    ],
    "recommended_action": "REJECT_AS_SCAM"
  },
  "extraction_method": "llm"
}
```

---

## 🧪 Running Tests

Run the comprehensive unit test suite:

```bash
pytest tests/ -v
```

---

## 📜 License

Distributed under the MIT License.
