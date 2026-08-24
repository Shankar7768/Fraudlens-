"""LLM-assisted feature extractor using Google GenAI SDK with structured outputs."""
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings, get_settings
from src.extraction.heuristic import extract_heuristic_features
from src.extraction.prompts import SYSTEM_PROMPT, build_analysis_prompt
from src.extraction.schemas import EnrichedJobRecord, ExplainableFlags
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMFeatureExtractor:
    """Extracts explainable fraud features from unstructured job postings using Gemini LLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        fallback_to_heuristic: bool = True,
        rate_limit_delay: Optional[float] = None,
    ) -> None:
        """Initializes LLM feature extractor.

        Args:
            api_key: Optional Gemini API Key.
            model_name: Gemini model name (default: gemini-3.7-flash).
            fallback_to_heuristic: Fallback to heuristic rules if LLM is unavailable or fails.
            rate_limit_delay: Sleep time between consecutive API calls.
        """
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model_name = model_name or settings.LLM_MODEL
        self.fallback_to_heuristic = fallback_to_heuristic
        self.rate_limit_delay = rate_limit_delay if rate_limit_delay is not None else settings.LLM_RATE_LIMIT_DELAY
        self._client = None

        if self.api_key and self.api_key.strip():
            try:
                from google import genai  # Official Google GenAI SDK

                self._client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Google GenAI Client with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Google GenAI Client: {e}")
        else:
            logger.info("No GEMINI_API_KEY provided; will operate in heuristic mode.")

    def extract_features(self, job: Dict[str, Any]) -> Tuple[ExplainableFlags, str]:
        """Extracts explainable features for a single job posting.

        Args:
            job: Standardized job dictionary.

        Returns:
            Tuple of (ExplainableFlags object, extraction_method_str).
        """
        if self._client is None:
            if self.fallback_to_heuristic:
                return extract_heuristic_features(job), "heuristic_fallback"
            raise ValueError("LLM client is not configured and fallback_to_heuristic is False.")

        prompt = build_analysis_prompt(job)

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=ExplainableFlags,
            )

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            # Check for parsed object or parse response text
            if hasattr(response, "parsed") and response.parsed is not None:
                if isinstance(response.parsed, ExplainableFlags):
                    flags = response.parsed
                else:
                    flags = ExplainableFlags.model_validate(response.parsed)
            else:
                flags = ExplainableFlags.model_validate_json(response.text)

            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

            return flags, "llm"

        except Exception as e:
            logger.warning(f"LLM extraction failed for Job ID {job.get('job_id')}: {e}")
            if self.fallback_to_heuristic:
                logger.info(f"Using heuristic fallback for Job ID {job.get('job_id')}")
                return extract_heuristic_features(job), "heuristic_fallback"
            raise

    def enrich_record(self, job: Dict[str, Any]) -> EnrichedJobRecord:
        """Enriches a job dictionary with explainable LLM features.

        Args:
            job: Standardized job dictionary.

        Returns:
            EnrichedJobRecord instance.
        """
        flags, method = self.extract_features(job)

        return EnrichedJobRecord(
            job_id=int(job.get("job_id", 0)),
            title=str(job.get("title", "")),
            location=str(job.get("location", "")),
            department=str(job.get("department", "")),
            salary_range=str(job.get("salary_range", "")),
            company_profile=str(job.get("company_profile", "")),
            description=str(job.get("description", "")),
            requirements=str(job.get("requirements", "")),
            benefits=str(job.get("benefits", "")),
            telecommuting=int(job.get("telecommuting", 0)),
            has_company_logo=int(job.get("has_company_logo", 0)),
            has_questions=int(job.get("has_questions", 0)),
            employment_type=str(job.get("employment_type", "")),
            required_experience=str(job.get("required_experience", "")),
            required_education=str(job.get("required_education", "")),
            industry=str(job.get("industry", "")),
            function=str(job.get("function", "")),
            fraudulent=int(job.get("fraudulent", 0)),
            llm_features=flags,
            extraction_method=method,
        )

    def process_batch(
        self,
        jobs: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> List[EnrichedJobRecord]:
        """Processes a batch of job dictionaries and returns enriched records.

        Args:
            jobs: List of standardized job dictionaries.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            List of EnrichedJobRecord instances.
        """
        results: List[EnrichedJobRecord] = []
        iterator = jobs
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(jobs, desc="Extracting Explainable Features", unit="job")

        for job in iterator:
            enriched = self.enrich_record(job)
            results.append(enriched)

        return results
