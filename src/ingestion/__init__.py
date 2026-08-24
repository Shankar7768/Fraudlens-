"""Data ingestion module for Fraud Lens."""
from src.ingestion.ingest_dataset import (
    ingest_fake_jobs_dataset,
    get_authentic_benchmark_records,
    normalize_text,
    clean_job_record,
    deduplicate_records,
    stratified_split,
)

__all__ = [
    "ingest_fake_jobs_dataset",
    "get_authentic_benchmark_records",
    "normalize_text",
    "clean_job_record",
    "deduplicate_records",
    "stratified_split",
]
