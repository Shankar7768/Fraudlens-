"""Feature extraction module for Fraud Lens."""
from src.extraction.schemas import ExplainableFlags, EnrichedJobRecord
from src.extraction.prompts import SYSTEM_PROMPT, build_analysis_prompt
from src.extraction.heuristic import extract_heuristic_features
from src.extraction.llm_extractor import LLMFeatureExtractor

__all__ = [
    "ExplainableFlags",
    "EnrichedJobRecord",
    "SYSTEM_PROMPT",
    "build_analysis_prompt",
    "extract_heuristic_features",
    "LLMFeatureExtractor",
]
