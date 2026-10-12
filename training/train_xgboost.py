
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from xgboost import XGBClassifier

try:
    from xgboost_model import prepare_data
except ImportError:
    from training.xgboost_model import prepare_data


MODEL_OUTPUT_PATH = "models/xgboost_fraud_model.json"

RISK_FEATURES = [
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

ROBERTA_FEATURE_COUNT = 768


def train_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
):
    """Train an XGBoost classifier with class-imbalance weighting."""

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    print("Initializing XGBoost Classifier...")
    print(
        f"  - Training samples: {len(y_train)} "
        f"(Genuine: {neg_count}, Fraudulent: {pos_count})"
    )
    print(f"  - scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=random_state,
        n_jobs=-1
    )

    eval_set = (
        [(X_val, y_val)]
        if X_val is not None and y_val is not None
        else None
    )

    print("\nTraining XGBoost model...")
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False
    )

    print("Training complete.")
    return model


def evaluate_model(model, X, y, dataset_name="Evaluation"):
    """Evaluate model performance at a fraud threshold of 0.30."""

    y_prob = model.predict_proba(X)[:, 1]
    fraud_threshold = 0.30
    y_pred = (y_prob >= fraud_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(
            precision_score(y, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob))
    }

    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    print(f"\n{'=' * 15} {dataset_name} Metrics {'=' * 15}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0, 0]:5d} | FP: {cm[0, 1]:5d}")
    print(f"  FN: {cm[1, 0]:5d} | TP: {cm[1, 1]:5d}")

    return metrics


def print_feature_importance(model):
    """
    Print XGBoost split-gain importance for the 10 risk features
    and the 15 most important features overall.
    """

    importance = model.feature_importances_

    if len(importance) != ROBERTA_FEATURE_COUNT + len(RISK_FEATURES):
        raise ValueError(
            f"Expected 778 features, but model has {len(importance)}."
        )

    print("\n" + "=" * 55)
    print("RISK FEATURE IMPORTANCE")
    print("=" * 55)

    risk_importance = importance[
        ROBERTA_FEATURE_COUNT:
        ROBERTA_FEATURE_COUNT + len(RISK_FEATURES)
    ]

    for name, score in sorted(
        zip(RISK_FEATURES, risk_importance),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{name:35s}: {score:.8f}")

    print("\n" + "=" * 55)
    print("TOP 15 FEATURES OVERALL")
    print("=" * 55)

    top_indices = np.argsort(importance)[::-1][:15]

    for index in top_indices:
        if index < ROBERTA_FEATURE_COUNT:
            feature_name = f"roberta_embedding_{index}"
        else:
            risk_index = index - ROBERTA_FEATURE_COUNT
            feature_name = RISK_FEATURES[risk_index]

        print(f"{feature_name:35s}: {importance[index]:.8f}")


def save_trained_model(model, output_path=MODEL_OUTPUT_PATH):
    """Save the trained XGBoost model."""

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model.save_model(output_path)
    print(f"\nModel saved to: {output_path}")


def run_training_pipeline():
    """Train, evaluate, report feature importance, and save the model."""

    print("Step 1: Preparing features from SQLite, RoBERTa, and risk rules...")

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = prepare_data()

    print("\nStep 2: Training XGBoost...")
    model = train_model(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val
    )

    print("\nStep 3: Evaluating model...")
    val_metrics = evaluate_model(
        model, X_val, y_val, "Validation Set"
    )
    test_metrics = evaluate_model(
        model, X_test, y_test, "Test Set"
    )

    print("\nStep 4: Reporting feature importance...")
    print_feature_importance(model)

    print("\nStep 5: Saving model...")
    save_trained_model(model, MODEL_OUTPUT_PATH)

    print("\n" + "=" * 55)
    print("FRAUDLENS XGBOOST PIPELINE SUMMARY")
    print("=" * 55)
    print(
        f"Validation -> F1: {val_metrics['f1']:.4f} | "
        f"ROC-AUC: {val_metrics['roc_auc']:.4f} | "
        f"Recall: {val_metrics['recall']:.4f}"
    )
    print(
        f"Test Set   -> F1: {test_metrics['f1']:.4f} | "
        f"ROC-AUC: {test_metrics['roc_auc']:.4f} | "
        f"Recall: {test_metrics['recall']:.4f}"
    )
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print("=" * 55)

    return model, val_metrics, test_metrics


if __name__ == "__main__":
    run_training_pipeline()
