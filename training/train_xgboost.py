import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

try:
    from xgboost_model import prepare_data
except ImportError:
    from training.xgboost_model import prepare_data


MODEL_OUTPUT_PATH = "models/xgboost_fraud_model.json"


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
    """
    Train an XGBoost Classifier with class-imbalance weighting.
    """
    # Calculate scale_pos_weight for handling fraud class imbalance
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    print("Initializing XGBoost Classifier...")
    print(f"  - Training samples: {len(y_train)} (Genuine: {neg_count}, Fraudulent: {pos_count})")
    print(f"  - Calculated scale_pos_weight: {scale_pos_weight:.2f}")

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

    eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

    print("\nTraining XGBoost model on training set...")
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False
    )
    print("Training complete.")

    return model


def evaluate_model(model, X, y, dataset_name="Evaluation"):
    """
    Evaluate the model on a given dataset and return performance metrics.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob))
    }

    cm = confusion_matrix(y, y_pred)

    print(f"\n{'=' * 20} {dataset_name} Metrics {'=' * 20}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0, 0]:5d} | FP: {cm[0, 1]:5d}")
    print(f"  FN: {cm[1, 0]:5d} | TP: {cm[1, 1]:5d}")

    return metrics


def save_trained_model(model, output_path=MODEL_OUTPUT_PATH):
    """
    Save the trained XGBoost model to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path)
    print(f"\nTrained model successfully saved to: {output_path}")


def run_training_pipeline():
    """
    End-to-end training and evaluation pipeline for FraudLens XGBoost classifier.
    """
    print("Step 1: Loading and preparing features from SQLite, RoBERTa, and LLM...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

    print("\nStep 2: Training XGBoost model...")
    model = train_model(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val
    )

    print("\nStep 3: Evaluating model performance...")
    val_metrics = evaluate_model(model, X_val, y_val, dataset_name="Validation Set")
    test_metrics = evaluate_model(model, X_test, y_test, dataset_name="Test Set")

    print("\nStep 4: Saving model artifact...")
    save_trained_model(model, MODEL_OUTPUT_PATH)

    print("\n" + "=" * 55)
    print("FRAUDLENS XGBOOST PIPELINE SUMMARY")
    print("=" * 55)
    print(f"Validation -> F1: {val_metrics['f1']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f} | Recall: {val_metrics['recall']:.4f}")
    print(f"Test Set   -> F1: {test_metrics['f1']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f} | Recall: {test_metrics['recall']:.4f}")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print("=" * 55)

    return model, val_metrics, test_metrics


if __name__ == "__main__":
    run_training_pipeline()
