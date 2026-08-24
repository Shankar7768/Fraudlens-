"""End-to-end pipeline script for Fraud Lens.

Orchestrates:
1. Ingesting raw fake jobs dataset into JSON Lines format.
2. Extracting explainable LLM-assisted features from unstructured fields.
3. Saving enriched dataset to processed JSON Lines file.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.extraction.llm_extractor import LLMFeatureExtractor
from src.ingestion.ingest_dataset import ingest_fake_jobs_dataset
from src.utils.io_helpers import read_jsonl, write_jsonl
from src.utils.logger import get_logger

logger = get_logger("fraudlens.pipeline")


def run_pipeline(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
    use_heuristic: bool = False,
    force_ingest: bool = False,
    synthetic_ingest: bool = False,
    model_name: Optional[str] = None,
) -> Path:
    """Executes the full Fraud Lens data ingestion and feature extraction pipeline.

    Args:
        input_path: Path to raw dataset.jsonl (ingested if missing or force_ingest=True).
        output_path: Target path for enriched dataset_with_features.jsonl.
        limit: Optional maximum number of records to process.
        use_heuristic: If True, forces heuristic fallback extractor.
        force_ingest: If True, re-runs ingestion even if raw file exists.
        synthetic_ingest: If True, uses built-in synthetic benchmark records.
        model_name: Custom LLM model name.

    Returns:
        Path to the saved enriched JSONL file.
    """
    settings = get_settings()
    raw_path = Path(input_path) if input_path else settings.RAW_DATA_PATH
    processed_path = Path(output_path) if output_path else settings.PROCESSED_DATA_PATH

    # Step 1: Ensure Raw Dataset is Ingested
    if force_ingest or not raw_path.exists():
        logger.info(f"Raw dataset not found at {raw_path} or force_ingest requested. Starting native JSONL ingestion...")
        res = ingest_fake_jobs_dataset(
            output_path=raw_path,
            limit=limit,
            use_synthetic=synthetic_ingest,
        )
        raw_path = res["all"] if isinstance(res, dict) else Path(res)
    else:
        logger.info(f"Using existing raw dataset at: {raw_path}")

    # Step 2: Load Raw Records
    logger.info(f"Loading raw records from {raw_path}...")
    records = read_jsonl(raw_path, limit=limit)
    logger.info(f"Loaded {len(records)} records for feature extraction.")

    if not records:
        raise ValueError(f"No records found in {raw_path}")

    # Step 3: Initialize Feature Extractor
    api_key = None if use_heuristic else settings.GEMINI_API_KEY
    extractor = LLMFeatureExtractor(
        api_key=api_key,
        model_name=model_name,
        fallback_to_heuristic=True,
    )

    # Step 4: Extract Features
    logger.info("Extracting explainable flags and fraud signals...")
    enriched_records = extractor.process_batch(records, show_progress=True)

    # Convert Pydantic records to dicts for JSON Lines export
    serialized_records = [record.model_dump() for record in enriched_records]

    # Step 5: Save strictly to JSON Lines
    count = write_jsonl(serialized_records, processed_path, mode="w")
    logger.info(f"Successfully wrote {count} enriched records strictly to JSON Lines: {processed_path}")

    # Step 6: Print Pipeline Summary Metrics
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    action_counts = {"APPROVE": 0, "FLAG_FOR_MANUAL_REVIEW": 0, "REJECT_AS_SCAM": 0}

    for rec in enriched_records:
        r_level = rec.llm_features.risk_level
        action = rec.llm_features.recommended_action
        risk_counts[r_level] = risk_counts.get(r_level, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1

    print("\n" + "=" * 60)
    print("           FRAUD LENS PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total Processed Postings : {len(enriched_records)}")
    print(f"Extraction Method        : {'Heuristic (Rule-based)' if (use_heuristic or not api_key) else f'LLM ({extractor.model_name})'}")
    print(f"Processed JSONL Output   : {processed_path}")
    print("-" * 60)
    print("Risk Level Breakdown:")
    for level, cnt in risk_counts.items():
        pct = (cnt / len(enriched_records) * 100) if enriched_records else 0
        print(f"  - {level:<10}: {cnt:>4} ({pct:>5.1f}%)")
    print("-" * 60)
    print("Action Recommendations:")
    for act, cnt in action_counts.items():
        print(f"  - {act:<24}: {cnt:>4}")
    print("=" * 60 + "\n")

    return processed_path


def main() -> None:
    """CLI entry point for running the complete Fraud Lens pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the Fraud Lens End-to-End Ingestion and LLM Feature Extraction Pipeline."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Path to raw dataset.jsonl file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path for processed dataset_with_features.jsonl output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of records to process (useful for quick testing/demo).",
    )
    parser.add_argument(
        "--use-heuristic",
        action="store_true",
        help="Use rule-based heuristic extractor instead of Gemini LLM.",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Force re-downloading/ingesting the dataset into raw JSONL before feature extraction.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use built-in synthetic benchmark samples during ingestion.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Custom Gemini model name (default: gemini-3.7-flash).",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            input_path=Path(args.input_path) if args.input_path else None,
            output_path=Path(args.output_path) if args.output_path else None,
            limit=args.limit,
            use_heuristic=args.use_heuristic,
            force_ingest=args.force_ingest,
            synthetic_ingest=args.synthetic,
            model_name=args.model,
        )
    except Exception as err:
        logger.error(f"Pipeline execution failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
