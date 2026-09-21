import json
import os
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from database import load_job_postings
except ImportError:
    from training.database import load_job_postings


ROBERTA_EMBEDDINGS_PATH = "data/features/roberta_embeddings.npy"
LLM_FEATURES_PATH = "data/features/llm_features.json"

FEATURES = [
    "payment_request",
    "unrealistic_salary",
    "urgency",
    "personal_information_request",
    "suspicious_contact",
    "missing_company_information",
    "vague_job_description",
    "no_experience_required",
    "suspicious_url",
    "too_good_to_be_true"
]


def load_roberta_embeddings(file_path=ROBERTA_EMBEDDINGS_PATH):
    """
    Load the precomputed 768-dimensional RoBERTa embeddings from a .npy file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"RoBERTa embeddings file not found: {file_path}")

    embeddings = np.load(file_path)
    return embeddings


def load_llm_features(file_path=LLM_FEATURES_PATH):
    """
    Load LLM risk features from a JSON file and construct a 10-feature
    NumPy matrix ordered strictly by job_id.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LLM features file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    # Ensure records are sorted by job_id in ascending order
    sorted_records = sorted(records, key=lambda item: int(item["job_id"]))

    matrix = np.array(
        [
            [int(item["features"].get(feature, 0)) for feature in FEATURES]
            for item in sorted_records
        ],
        dtype=np.float32
    )

    return matrix


def prepare_data(
    roberta_path=ROBERTA_EMBEDDINGS_PATH,
    llm_path=LLM_FEATURES_PATH
):
    """
    Load RoBERTa embeddings, LLM risk features, and database labels,
    combine them into a 778-dimensional feature matrix, and perform
    a 70% / 15% / 15% stratified train/validation/test split.
    """
    # 1. Load RoBERTa embeddings (768-dim)
    roberta_features = load_roberta_embeddings(roberta_path)

    # 2. Load 10-feature LLM risk matrix in job_id order (10-dim)
    llm_features = load_llm_features(llm_path)

    if roberta_features.shape[0] != llm_features.shape[0]:
        raise ValueError(
            f"Mismatch in sample count: RoBERTa has {roberta_features.shape[0]} "
            f"rows, but LLM features have {llm_features.shape[0]} rows."
        )

    # 3. Combine RoBERTa + LLM features into 778-dimensional matrix
    X = np.concatenate([roberta_features, llm_features], axis=1)

    # 4. Load ground truth labels from the SQLite database
    df = load_job_postings()
    df = df.sort_values("job_id").reset_index(drop=True)

    if len(df) != X.shape[0]:
        raise ValueError(
            f"Mismatch between database rows ({len(df)}) and features ({X.shape[0]})."
        )

    y = df["fraudulent"].astype(int).to_numpy()

    # 5. First split: 70% training, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # 6. Second split: temporary 30% into 15% validation + 15% testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("Feature preparation completed.")
    print("RoBERTa features:", roberta_features.shape[1])
    print("LLM features:", llm_features.shape[1])
    print("Combined features:", X.shape[1])
    print("Total samples:", X.shape[0])

    print("\nDataset split:")
    print(f"Training:   {len(X_train)} samples ({y_train.sum()} fraudulent, {y_train.mean():.2%})")
    print(f"Validation: {len(X_val)} samples ({y_val.sum()} fraudulent, {y_val.mean():.2%})")
    print(f"Testing:    {len(X_test)} samples ({y_test.sum()} fraudulent, {y_test.mean():.2%})")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


if __name__ == "__main__":
    prepare_data()