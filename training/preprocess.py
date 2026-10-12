import pandas as pd
try:
    from database import load_job_postings
except ImportError:
    from training.database import load_job_postings


TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]


def preprocess_data():
    df = load_job_postings()
    df = df.sort_values("job_id").reset_index(drop=True)

    # Replace missing values in text columns
    for column in TEXT_COLUMNS:
        df[column] = df[column].fillna("").astype(str)

    # Combine important job-posting text, including salary
    df["salary_range"] = df["salary_range"].fillna("").astype(str)

    df["job_text"] = (
        "Title: " + df["title"] +
        "\nSalary Range: " + df["salary_range"] +
        "\nCompany Profile: " + df["company_profile"] +
        "\nDescription: " + df["description"] +
        "\nRequirements: " + df["requirements"] +
        "\nBenefits: " + df["benefits"]
    )

    # Target variable
    y = df["fraudulent"].astype(int)

    return df, y


if __name__ == "__main__":
    df, y = preprocess_data()

    print("Preprocessing completed successfully.")
    print("Rows:", len(df))
    print("Job text created:", "job_text" in df.columns)
    print("Target column:", "fraudulent")
    print("Fraudulent jobs:", int(y.sum()))
    print("Genuine jobs:", int((y == 0).sum()))