import numpy as np
from sklearn.model_selection import train_test_split
from database import load_job_postings


def prepare_data(roberta_embeddings, llm_features):
    df = load_job_postings()

    # Convert RoBERTa embeddings to NumPy
    roberta_features = np.asarray(roberta_embeddings)

    # Convert the 10 LLM features to NumPy
    llm_features = np.asarray(llm_features)

    # Combine RoBERTa + LLM features
    X = np.concatenate(
        [roberta_features, llm_features],
        axis=1
    )

    # Target
    y = df["fraudulent"].astype(int).to_numpy()

    # First split: 70% training, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Second split: temporary 30% into 15% validation + 15% testing
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

    print("\nDataset split:")
    print("Training:", len(X_train))
    print("Validation:", len(X_val))
    print("Testing:", len(X_test))

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )