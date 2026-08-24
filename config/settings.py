"""Settings and environment management for Fraud Lens."""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    """Project-wide settings and parameters."""

    # Project Root
    BASE_DIR: Path = BASE_DIR

    # Gemini & LLM Configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.6-flash")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))
    LLM_RATE_LIMIT_DELAY: float = float(os.getenv("LLM_RATE_LIMIT_DELAY", "0.5"))

    # Data Ingestion Configuration
    HF_DATASET_NAME: str = os.getenv("HF_DATASET_NAME", "victor/real-or-fake-fake-jobposting-prediction")
    RAW_DATA_PATH: Path = BASE_DIR / os.getenv("RAW_DATA_PATH", "data/raw/dataset.jsonl")
    TRAIN_DATA_PATH: Path = BASE_DIR / os.getenv("TRAIN_DATA_PATH", "data/raw/train.jsonl")
    VAL_DATA_PATH: Path = BASE_DIR / os.getenv("VAL_DATA_PATH", "data/raw/val.jsonl")
    TEST_DATA_PATH: Path = BASE_DIR / os.getenv("TEST_DATA_PATH", "data/raw/test.jsonl")
    PROCESSED_DATA_PATH: Path = BASE_DIR / os.getenv("PROCESSED_DATA_PATH", "data/processed/dataset_with_features.jsonl")

    # Dataset Split Ratios (PPT Slide 8: 70% Train, 15% Val, 15% Test)
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Hybrid Model Architecture Settings (PPT Slide 6 & 7: RoBERTa + XGBoost)
    ROBERTA_MODEL_NAME: str = os.getenv("ROBERTA_MODEL_NAME", "roberta-base")

    # Processing Defaults
    DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "10"))
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure raw and processed directories exist."""
        cls.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Factory function to get application settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings
