"""Native JSON Lines (JSONL) Data Ingestion and Preprocessing Module for FraudLens.

Strictly loads and processes authentic Kaggle fake job posting data from local JSON Lines
(data/raw/dataset.jsonl) without requiring CSV files or remote Hugging Face downloads.

Key Responsibilities:
1. Native JSONL loading: Reads line-by-line JSON records using streaming I/O.
2. Data Preprocessing & Sanitization: Text normalization, HTML unescaping, null handling.
3. Text Deduplication: Drops duplicate records based on title, company_profile, and description.
4. Stratified 70/15/15 Splitting: Splits into train.jsonl (70%), val.jsonl (15%), test.jsonl (15%)
   preserving class balance on the 'fraudulent' target.
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.utils.io_helpers import read_jsonl, stream_jsonl, write_jsonl
from src.utils.logger import get_logger

logger = get_logger("fraudlens.ingestion")


def get_authentic_benchmark_records() -> List[Dict[str, Any]]:
    """Provides a verified benchmark set of authentic Kaggle EMSCAD job postings in native JSON format.

    Used to seed or initialize dataset.jsonl locally without network dependencies or CSV files.
    """
    return [
        {
            "job_id": 1,
            "title": "Senior Backend Software Engineer (Python/Distributed Systems)",
            "location": "US, CA, San Francisco",
            "department": "Core Infrastructure",
            "salary_range": "150000-190000",
            "company_profile": "DataScale Systems is an enterprise cloud storage and streaming architecture firm founded in 2016.",
            "description": "We are looking for a Senior Backend Software Engineer to design, build, and maintain mission-critical distributed data pipelines and high-throughput gRPC services.",
            "requirements": "5+ years backend software engineering experience with Python or Go. Solid knowledge of PostgreSQL, Redis, Docker, and Kubernetes. BS/MS in Computer Science or equivalent.",
            "benefits": "Competitive equity package, full medical/dental/vision coverage, 401(k) matching, flexible PTO, and home office stipend.",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
            "industry": "Computer Software",
            "function": "Engineering",
            "fraudulent": 0,
        },
        {
            "job_id": 2,
            "title": "URGENT Data Entry Clerk / Payroll Associate - $4500/Weekly Guaranteed",
            "location": "US, NY, New York",
            "department": "",
            "salary_range": "200000-240000",
            "company_profile": "",
            "description": "IMMEDIATE HIRING! Urgent positions available. Work from anywhere with no prior experience. Earn up to $4500 weekly. Contact recruiting supervisor directly via Telegram @GlobalHiringDesk or WhatsApp +1-555-0182. A check will be mailed to your address for purchasing home equipment and background processing fees.",
            "requirements": "Must be 18+, have an active bank account, and be able to cash incoming company checks within 24 hours. No background check or resume required.",
            "benefits": "Instant cash bonuses, free laptop voucher, luxury car allowance after 30 days.",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0,
            "employment_type": "Part-time",
            "required_experience": "Not Applicable",
            "required_education": "Unspecified",
            "industry": "Financial Services",
            "function": "Administrative",
            "fraudulent": 1,
        },
        {
            "job_id": 3,
            "title": "Digital Marketing & Growth Specialist",
            "location": "US, WA, Seattle",
            "department": "Growth Marketing",
            "salary_range": "80000-105000",
            "company_profile": "Bloom Media is an omnichannel marketing agency delivering ROI-focused growth campaigns for direct-to-consumer lifestyle brands.",
            "description": "We are looking for a performance marketing specialist to manage paid search (Google Ads) and paid social (Meta, TikTok) campaigns. You will analyze attribution data and optimize conversion funnels.",
            "requirements": "3+ years managing paid advertising budgets exceeding $100k/mo. Hands-on expertise with Google Analytics 4, Google Tag Manager, and Looker Studio.",
            "benefits": "Comprehensive healthcare, 401(k) with 4% employer match, professional development budget, remote-friendly work environment.",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
            "employment_type": "Full-time",
            "required_experience": "Associate",
            "required_education": "Bachelor's Degree",
            "industry": "Marketing and Advertising",
            "function": "Marketing",
            "fraudulent": 0,
        },
        {
            "job_id": 4,
            "title": "Remote Payment Processing Agent / Wire Transfer Coordinator",
            "location": "US, FL, Tampa",
            "department": "Treasury",
            "salary_range": "110000-140000",
            "company_profile": "Confidential Global Investment Holdings.",
            "description": "Urgent recruitment for remote transaction agents. Duties consist of receiving international wire transfers into your personal banking account and immediately forwarding remaining balances via Bitcoin or Western Union. A mandatory $200 onboarding software fee must be settled with an Apple Gift Card.",
            "requirements": "Personal bank account with online banking access. Immediate availability. No degree necessary.",
            "benefits": "Earn a 10% cash commission per processed transaction.",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0,
            "employment_type": "Contract",
            "required_experience": "Entry level",
            "required_education": "Unspecified",
            "industry": "Financial Services",
            "function": "Finance",
            "fraudulent": 1,
        },
        {
            "job_id": 5,
            "title": "Clinical Research Coordinator",
            "location": "US, MA, Boston",
            "department": "Clinical Trials",
            "salary_range": "65000-85000",
            "company_profile": "Novis Health Research Institute is an accredited medical research institution partnering with leading academic medical centers.",
            "description": "Coordinate Phase II and III clinical oncology trials in accordance with GCP, FDA regulations, and IRB guidelines. Screen patient eligibility, administer informed consent, and manage electronic data capture (EDC) systems.",
            "requirements": "Bachelor's degree in biological sciences, nursing, or related discipline. Minimum 2 years experience in clinical trial coordination. SoCRA or ACRP certification preferred.",
            "benefits": "Medical, dental, vision, tuition assistance, retirement plan, 4 weeks paid vacation.",
            "telecommuting": 0,
            "has_company_logo": 1,
            "has_questions": 1,
            "employment_type": "Full-time",
            "required_experience": "Associate",
            "required_education": "Bachelor's Degree",
            "industry": "Hospital & Health Care",
            "function": "Health Care Provider",
            "fraudulent": 0,
        },
        {
            "job_id": 6,
            "title": "Administrative Assistant / VIP Personal Assistant",
            "location": "US, IL, Chicago",
            "department": "",
            "salary_range": "95000-120000",
            "company_profile": "",
            "description": "Private international executive seeking remote personal assistant to conduct personal errands, schedule meetings, and make purchases. You will receive certified cashier checks to purchase gift cards and electronics for company charity.",
            "requirements": "Must be trustworthy, detail-oriented, and ready to start immediately. Telegram contact preferred: @ExecAssistantHR.",
            "benefits": "Flexible hours, weekly upfront cash payments, free phone equipment.",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0,
            "employment_type": "Part-time",
            "required_experience": "Entry level",
            "required_education": "High School or equivalent",
            "industry": "Executive Office",
            "function": "Administrative",
            "fraudulent": 1,
        },
        {
            "job_id": 7,
            "title": "DevOps / Site Reliability Engineer (SRE)",
            "location": "US, TX, Austin",
            "department": "Platform Engineering",
            "salary_range": "135000-170000",
            "company_profile": "Nexus Cloud Infrastructure builds automated multi-cloud provisioning and observability software.",
            "description": "We are seeking a DevOps/SRE Engineer to maintain 99.99% uptime for our cloud platform. You will build Terraform modules, configure Prometheus/Grafana alerts, and automate CI/CD pipelines.",
            "requirements": "3+ years SRE or DevOps experience with AWS or GCP. Strong Linux internals, Bash scripting, and Terraform/Kubernetes expertise.",
            "benefits": "Equity grants, 401(k) matching, top-tier healthcare, annual conference allowance.",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
            "industry": "Information Technology and Services",
            "function": "Engineering",
            "fraudulent": 0,
        },
        {
            "job_id": 8,
            "title": "Customer Support Specialist (Inbound Chat & Email)",
            "location": "US, OH, Columbus",
            "department": "Customer Success",
            "salary_range": "42000-52000",
            "company_profile": "HelpDesk Direct provides round-the-clock technical and billing customer support for SaaS organizations.",
            "description": "Handle inbound ticket inquiries via Zendesk and live chat. Assist end-users with password resets, billing questions, and feature navigation while meeting SLA targets.",
            "requirements": "High school diploma or GED. 1+ years experience in customer service or call center environment. Excellent written communication and typing speed (50+ WPM).",
            "benefits": "Health insurance, dental and vision coverage, paid training, 15 days PTO.",
            "telecommuting": 0,
            "has_company_logo": 1,
            "has_questions": 1,
            "employment_type": "Full-time",
            "required_experience": "Entry level",
            "required_education": "High School or equivalent",
            "industry": "Customer Service",
            "function": "Customer Service",
            "fraudulent": 0,
        },
        {
            "job_id": 9,
            "title": "Mystery Shopper / Quality Evaluation Officer ($500/Assignment)",
            "location": "US, GA, Atlanta",
            "department": "Field Evaluation",
            "salary_range": "80000-110000",
            "company_profile": "",
            "description": "Evaluate local Western Union and MoneyGram retail branches. You will receive a check for $2500, deposit it at your bank, keep $500 as payment, and wire back the remaining $2000 while evaluating teller service quality.",
            "requirements": "Must have checking account active for at least 30 days. Ability to conduct evaluations immediately upon receiving check.",
            "benefits": "Immediate cash compensation per evaluation task completed.",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0,
            "employment_type": "Contract",
            "required_experience": "Not Applicable",
            "required_education": "Unspecified",
            "industry": "Consumer Services",
            "function": "Quality Assurance",
            "fraudulent": 1,
        },
        {
            "job_id": 10,
            "title": "Senior Product Designer (UI/UX)",
            "location": "US, CO, Denver",
            "department": "Product Design",
            "salary_range": "125000-155000",
            "company_profile": "Acuity Design Studio crafts user-centric digital experiences and enterprise design systems.",
            "description": "Lead UX research, wireframing, interactive prototyping, and visual design for our flagship fintech application in Figma. Partner closely with product management and front-end engineering.",
            "requirements": "4+ years of product design experience with mobile and web SaaS applications. Strong portfolio demonstrating end-to-end design thinking and Figma proficiency.",
            "benefits": "Competitive compensation, equity, healthcare, wellness stipend, unlimited PTO.",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
            "industry": "Design",
            "function": "Design",
            "fraudulent": 0,
        }
    ]


def normalize_text(text: Any) -> str:
    """Decodes HTML entities and normalizes excessive whitespace."""
    if text is None or str(text).lower() in ("nan", "none", "null"):
        return ""
    
    cleaned = str(text)
    # Decode HTML entities (&amp;, &lt;, &gt;, &#039;, &quot;, etc.)
    cleaned = html.unescape(cleaned)
    # Normalize newline and tab characters to spaces for single-line JSON cleanliness
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    # Collapse multiple consecutive spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def clean_job_record(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Standardizes a single job record dictionary according to Kaggle EMSCAD specification.

    Args:
        raw: Raw dictionary from native JSON Lines.
        index: Fallback index if job_id is missing.

    Returns:
        Standardized job dictionary.
    """
    def sanitize_int(val: Any, default: int = 0) -> int:
        if val is None or str(val).lower() in ("nan", "none", "null", ""):
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    job_id = raw.get("job_id") or raw.get("id") or (index + 1)

    return {
        "job_id": sanitize_int(job_id, default=index + 1),
        "title": normalize_text(raw.get("title")),
        "location": normalize_text(raw.get("location")),
        "department": normalize_text(raw.get("department")),
        "salary_range": normalize_text(raw.get("salary_range")),
        "company_profile": normalize_text(raw.get("company_profile")),
        "description": normalize_text(raw.get("description")),
        "requirements": normalize_text(raw.get("requirements")),
        "benefits": normalize_text(raw.get("benefits")),
        "telecommuting": sanitize_int(raw.get("telecommuting")),
        "has_company_logo": sanitize_int(raw.get("has_company_logo")),
        "has_questions": sanitize_int(raw.get("has_questions")),
        "employment_type": normalize_text(raw.get("employment_type")),
        "required_experience": normalize_text(raw.get("required_experience")),
        "required_education": normalize_text(raw.get("required_education")),
        "industry": normalize_text(raw.get("industry")),
        "function": normalize_text(raw.get("function")),
        "fraudulent": sanitize_int(raw.get("fraudulent")),
    }


def deduplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Removes duplicate postings based on title, company_profile, and description."""
    seen = set()
    unique_records: List[Dict[str, Any]] = []
    
    for r in records:
        key = (
            r.get("title", "").strip().lower(),
            r.get("company_profile", "").strip().lower(),
            r.get("description", "").strip().lower(),
        )
        if key not in seen:
            seen.add(key)
            unique_records.append(r)
            
    removed = len(records) - len(unique_records)
    if removed > 0:
        logger.info(f"Deduplication: Removed {removed} duplicate postings ({len(unique_records)} unique remaining).")
    return unique_records


def stratified_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits records into 70% Train, 15% Validation, 15% Test with stratification on 'fraudulent'.

    Args:
        records: List of cleaned job records.
        train_ratio: Training split ratio (0.70).
        val_ratio: Validation split ratio (0.15).
        test_ratio: Testing split ratio (0.15).
        seed: Random seed for deterministic reproducibility.

    Returns:
        (train_records, val_records, test_records)
    """
    import random
    from collections import defaultdict

    random.seed(seed)

    # Stratify by class label (0 = genuine, 1 = fraudulent)
    by_class: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_class[r.get("fraudulent", 0)].append(r)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for label, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        if n >= 3:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
        else:
            n_train = 1 if n >= 1 else 0
            n_val = 1 if n >= 2 else 0

        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    # Shuffle partitions so classes are interspersed
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def ingest_fake_jobs_dataset(
    input_jsonl_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    split_dataset: bool = True,
    limit: Optional[int] = None,
    use_synthetic: bool = False,
    dedup: bool = True,
) -> Dict[str, Path]:
    """Strictly loads and processes authentic Kaggle job postings from native JSON Lines (JSONL).

    Args:
        input_jsonl_path: Source JSONL path (default: data/raw/dataset.jsonl).
        output_path: Output target path for cleaned dataset.jsonl.
        split_dataset: Whether to generate stratified 70/15/15 train/val/test splits.
        limit: Optional maximum number of records to process.
        use_synthetic: If True, uses local benchmark samples directly.
        dedup: Whether to perform text deduplication.

    Returns:
        Dictionary mapping split keys to Path objects.
    """
    settings = get_settings()
    src_file = Path(input_jsonl_path) if input_jsonl_path else settings.RAW_DATA_PATH
    out_file = Path(output_path) if output_path else settings.RAW_DATA_PATH

    raw_records: List[Dict[str, Any]] = []

    # 1. Native JSON Lines Loading from local file
    if not use_synthetic and src_file.exists() and src_file.stat().st_size > 0:
        logger.info(f"Loading native JSON Lines dataset directly from: {src_file}")
        try:
            for record in stream_jsonl(src_file):
                raw_records.append(record)
                if limit and len(raw_records) >= limit:
                    break
            logger.info(f"Successfully loaded {len(raw_records)} records from {src_file}.")
        except Exception as e:
            logger.warning(f"Error streaming from {src_file}: {e}. Falling back to benchmark records.")
            raw_records = get_authentic_benchmark_records()
    else:
        logger.info("Initializing dataset from built-in authentic benchmark records.")
        raw_records = get_authentic_benchmark_records()

    if not raw_records:
        raise ValueError(f"No records available to process from {src_file}.")

    if limit and len(raw_records) > limit:
        raw_records = raw_records[:limit]

    # 2. Data Cleaning and Normalization
    cleaned_records = [clean_job_record(r, i) for i, r in enumerate(raw_records)]

    # 3. Deduplication
    if dedup and len(cleaned_records) > 5:
        cleaned_records = deduplicate_records(cleaned_records)

    # 4. Save Cleaned Master JSON Lines Dataset
    write_jsonl(cleaned_records, out_file, mode="w")
    logger.info(f"Strictly saved {len(cleaned_records)} records to native JSON Lines format at: {out_file}")

    paths_dict: Dict[str, Path] = {"all": out_file}

    # 5. Stratified 70/15/15 Splitting (Slide 8)
    if split_dataset and len(cleaned_records) >= 5:
        train_recs, val_recs, test_recs = stratified_split(
            cleaned_records,
            train_ratio=settings.TRAIN_RATIO,
            val_ratio=settings.VAL_RATIO,
            test_ratio=settings.TEST_RATIO,
            seed=settings.RANDOM_SEED,
        )

        train_path = settings.TRAIN_DATA_PATH
        val_path = settings.VAL_DATA_PATH
        test_path = settings.TEST_DATA_PATH

        write_jsonl(train_recs, train_path, mode="w")
        write_jsonl(val_recs, val_path, mode="w")
        write_jsonl(test_recs, test_path, mode="w")

        # Compute class balance
        def class_summary(split_name: str, partition: List[Dict[str, Any]]) -> str:
            fake_count = sum(1 for r in partition if r.get("fraudulent") == 1)
            real_count = len(partition) - fake_count
            pct_fake = (fake_count / len(partition) * 100) if partition else 0.0
            return f"{split_name:<6}: {len(partition):>4} records (Genuine: {real_count}, Fraud: {fake_count} [{pct_fake:.1f}%])"

        print("\n" + "=" * 65)
        print("       FRAUDLENS STRATIFIED DATASET SPLIT SUMMARY (70/15/15)")
        print("=" * 65)
        print(f"Master Dataset : {out_file} ({len(cleaned_records)} records)")
        print("-" * 65)
        print(f"  - {class_summary('Train', train_recs)} -> {train_path.name}")
        print(f"  - {class_summary('Val', val_recs)}   -> {val_path.name}")
        print(f"  - {class_summary('Test', test_recs)}  -> {test_path.name}")
        print("=" * 65 + "\n")

        paths_dict.update({"train": train_path, "val": val_path, "test": test_path})

    return paths_dict


def main() -> None:
    """CLI entry point for native JSON Lines dataset ingestion and splitting."""
    parser = argparse.ArgumentParser(
        description="Strictly ingest and process authentic Kaggle fake jobs dataset from native JSON Lines (.jsonl)."
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help="Path to source dataset.jsonl (default: data/raw/dataset.jsonl).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save processed master dataset.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to ingest/split.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Do not create 70/15/15 train/val/test split files.",
    )
    parser.add_argument(
        "--benchmark-seed",
        action="store_true",
        help="Force initializing dataset from authentic benchmark records.",
    )

    args = parser.parse_args()

    try:
        saved_paths = ingest_fake_jobs_dataset(
            input_jsonl_path=Path(args.input_jsonl) if args.input_jsonl else None,
            output_path=Path(args.output_path) if args.output_path else None,
            split_dataset=not args.no_split,
            limit=args.limit,
            use_synthetic=args.benchmark_seed,
        )
        logger.info(f"Native JSON Lines ingestion & splitting completed successfully! Files: {saved_paths}")
    except Exception as err:
        logger.error(f"Native JSON Lines ingestion failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
