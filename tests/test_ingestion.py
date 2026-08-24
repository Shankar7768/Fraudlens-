"""Unit tests for native JSON Lines dataset ingestion and stratified splitting."""
import json
from pathlib import Path
import pytest

from src.ingestion.ingest_dataset import (
    ingest_fake_jobs_dataset,
    get_authentic_benchmark_records,
    stratified_split,
    normalize_text,
    deduplicate_records,
)
from src.utils.io_helpers import read_jsonl, stream_jsonl, write_jsonl


def test_write_and_read_jsonl(tmp_path: Path):
    """Tests writing records strictly as JSON Lines and reading them back."""
    records = [
        {"job_id": 1, "title": "Job A", "fraudulent": 0},
        {"job_id": 2, "title": "Job B", "fraudulent": 1},
    ]
    file_path = tmp_path / "test.jsonl"
    count = write_jsonl(records, file_path)
    assert count == 2
    assert file_path.exists()

    # Verify each line is valid JSON
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "job_id" in parsed

    # Read back using helper
    loaded = read_jsonl(file_path)
    assert len(loaded) == 2
    assert loaded[0]["title"] == "Job A"
    assert loaded[1]["title"] == "Job B"


def test_text_normalization():
    """Tests HTML entity decoding and whitespace normalization."""
    raw_text = "Software &amp; Web Developer &lt;urgent&gt;\r\n\tApply   now! "
    cleaned = normalize_text(raw_text)
    assert cleaned == "Software & Web Developer <urgent> Apply now!"


def test_stratified_split():
    """Tests 70/15/15 stratified train/val/test splitting on target 'fraudulent'."""
    # 20 samples: 10 real (0), 10 fake (1)
    samples = []
    for i in range(20):
        samples.append({
            "job_id": i + 1,
            "title": f"Job {i}",
            "fraudulent": 1 if i % 2 == 0 else 0,
        })
        
    train, val, test = stratified_split(samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    assert len(train) == 14  # 70% of 20
    assert len(val) >= 2
    assert len(test) >= 2
    assert len(train) + len(val) + len(test) == 20


def test_native_jsonl_ingestion_and_split(tmp_path: Path):
    """Tests native JSON Lines loading and creating train/val/test splits."""
    input_file = tmp_path / "dataset.jsonl"
    benchmark_records = get_authentic_benchmark_records()
    write_jsonl(benchmark_records, input_file)

    saved_paths = ingest_fake_jobs_dataset(
        input_jsonl_path=input_file,
        output_path=tmp_path / "cleaned_dataset.jsonl",
        split_dataset=True,
    )

    assert "all" in saved_paths
    assert "train" in saved_paths
    assert "val" in saved_paths
    assert "test" in saved_paths

    train_data = read_jsonl(saved_paths["train"])
    val_data = read_jsonl(saved_paths["val"])
    test_data = read_jsonl(saved_paths["test"])

    assert len(train_data) + len(val_data) + len(test_data) == len(benchmark_records)
    for rec in train_data:
        assert "fraudulent" in rec
        assert "title" in rec
        assert "description" in rec
