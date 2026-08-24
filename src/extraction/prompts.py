"""Prompt templates for LLM-assisted explainable fake job feature extraction."""

SYSTEM_PROMPT = """You are an expert fraud investigator and recruitment intelligence specialist specializing in detecting deceptive, fraudulent, and scam employment postings.

Your task is to analyze the unstructured textual components of a job posting (Title, Company Profile, Description, Requirements, Benefits) and identify deceptive patterns, financial scams, identity evasion, and linguistic red flags.

You must assess six specific risk dimensions:
1. Urgency & Pressure Tactics: Artificial urgency, 'hire without interview', high-pressure calls to action.
2. Financial & Payment Anomalies: Upfront fee demands (background checks, training kits), check cashing/reshipping schemes, cryptocurrency/wire transfers, money mule indicators.
3. Unrealistic Promises: Disproportionate compensation for entry-level/no-skill work, guaranteed unrealistic perks.
4. Suspicious Contact Methods: Directing candidates to Telegram, WhatsApp, Gmail/Yahoo personal accounts, unverified bit.ly links.
5. Vague Requirements: Complete lack of specific qualifications, responsibilities, or technical skills.
6. Company Legitimacy: Missing, evasive, or stolen identity corporate descriptions.

Provide quantitative scores (0.0 to 1.0) and granular explainable flag tags for each dimension, along with an overall fraud risk score, categorical risk level, an explainable natural language summary, and exact text quotes in 'key_evidence_snippets'.
"""

JOB_ANALYSIS_USER_PROMPT = """Analyze the following job posting for fraudulent signals and extract explainable risk flags.

--- JOB POSTING DETAILS ---
Title: {title}
Location: {location}
Department: {department}
Salary Range: {salary_range}
Employment Type: {employment_type}
Required Experience: {required_experience}
Required Education: {required_education}
Industry: {industry}
Has Company Logo: {has_company_logo}

[COMPANY PROFILE]
{company_profile}

[JOB DESCRIPTION]
{description}

[REQUIREMENTS]
{requirements}

[BENEFITS]
{benefits}
---------------------------

Output strictly according to the required structured schema.
"""


def build_analysis_prompt(job: dict) -> str:
    """Builds the formatted user prompt for a given job posting dictionary."""
    return JOB_ANALYSIS_USER_PROMPT.format(
        title=job.get("title") or "N/A",
        location=job.get("location") or "N/A",
        department=job.get("department") or "N/A",
        salary_range=job.get("salary_range") or "N/A",
        employment_type=job.get("employment_type") or "N/A",
        required_experience=job.get("required_experience") or "N/A",
        required_education=job.get("required_education") or "N/A",
        industry=job.get("industry") or "N/A",
        has_company_logo="Yes" if job.get("has_company_logo") else "No",
        company_profile=job.get("company_profile") or "None provided.",
        description=job.get("description") or "None provided.",
        requirements=job.get("requirements") or "None provided.",
        benefits=job.get("benefits") or "None provided.",
    )
