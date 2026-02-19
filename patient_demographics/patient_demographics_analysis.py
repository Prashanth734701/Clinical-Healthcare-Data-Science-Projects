"""
Patient Demographics Analysis
==============================
Analyzes synthetic patient demographic data to surface age distributions,
gender balance, ethnicity breakdowns, comorbidity profiles, and the
relationship between demographics and health outcomes.
"""

import numpy as np
import pandas as pd
from scipy import stats


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_PATIENTS = 500

ETHNICITIES = ["White", "Black or African American", "Hispanic or Latino", "Asian", "Other"]
ETHNICITY_PROBS = [0.60, 0.13, 0.18, 0.06, 0.03]

COMORBIDITIES = ["Hypertension", "Diabetes", "Obesity", "Heart Disease", "Asthma"]
COMORBIDITY_PROBS = [0.45, 0.30, 0.35, 0.20, 0.15]

INSURANCE_TYPES = ["Private", "Medicare", "Medicaid", "Uninsured"]
INSURANCE_PROBS = [0.50, 0.25, 0.18, 0.07]


def generate_patient_data():
    """Generate a synthetic patient demographics dataset."""
    ages = np.random.normal(loc=55, scale=15, size=N_PATIENTS).clip(18, 90).astype(int)
    genders = np.random.choice(["Male", "Female"], size=N_PATIENTS, p=[0.48, 0.52])
    ethnicities = np.random.choice(ETHNICITIES, size=N_PATIENTS, p=ETHNICITY_PROBS)
    insurance = np.random.choice(INSURANCE_TYPES, size=N_PATIENTS, p=INSURANCE_PROBS)

    comorbidity_matrix = np.random.binomial(1, COMORBIDITY_PROBS, size=(N_PATIENTS, len(COMORBIDITIES)))
    comorbidity_df = pd.DataFrame(comorbidity_matrix, columns=COMORBIDITIES)
    comorbidity_count = comorbidity_df.sum(axis=1)

    # Readmission risk: driven by age, comorbidity count, and insurance type
    base_risk = (
        0.02 * (ages / 10)
        + 0.05 * comorbidity_count
        + np.where(insurance == "Uninsured", 0.15, 0)
        + np.random.normal(0, 0.05, N_PATIENTS)
    )
    readmission_prob = 1 / (1 + np.exp(-base_risk + 1.5))  # sigmoid
    readmitted = np.random.binomial(1, readmission_prob.clip(0.01, 0.99))

    base_df = pd.DataFrame(
        {
            "patient_id": [f"PT{str(i).zfill(4)}" for i in range(1, N_PATIENTS + 1)],
            "age": ages,
            "gender": genders,
            "ethnicity": ethnicities,
            "insurance": insurance,
            "comorbidity_count": comorbidity_count.values,
            "readmitted_30d": readmitted,
        }
    )
    return pd.concat([base_df, comorbidity_df], axis=1)


def age_group_distribution(df):
    """Bin patients into clinical age groups and return counts."""
    bins = [17, 29, 44, 59, 74, 90]
    labels = ["18–29", "30–44", "45–59", "60–74", "75–90"]
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    return df.groupby("age_group", observed=True).size().reset_index(name="count")


def gender_breakdown(df):
    """Return counts and percentages for each gender."""
    counts = df["gender"].value_counts().reset_index()
    counts.columns = ["gender", "count"]
    counts["percentage"] = (counts["count"] / len(df) * 100).round(1)
    return counts


def ethnicity_breakdown(df):
    """Return counts and percentages for each ethnicity."""
    counts = df["ethnicity"].value_counts().reset_index()
    counts.columns = ["ethnicity", "count"]
    counts["percentage"] = (counts["count"] / len(df) * 100).round(1)
    return counts


def comorbidity_prevalence(df):
    """Return prevalence rates for each comorbidity."""
    prevalence = df[COMORBIDITIES].mean().reset_index()
    prevalence.columns = ["comorbidity", "prevalence"]
    prevalence["prevalence_pct"] = (prevalence["prevalence"] * 100).round(1)
    return prevalence.sort_values("prevalence_pct", ascending=False)


def readmission_by_age_group(df):
    """Return 30-day readmission rates by age group."""
    bins = [17, 44, 64, 90]
    labels = ["18–44", "45–64", "65–90"]
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    return (
        df.groupby("age_group", observed=True)["readmitted_30d"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "n", "sum": "readmissions", "mean": "readmission_rate"})
        .round(3)
    )


def anova_readmission_by_ethnicity(df):
    """
    One-way ANOVA testing whether mean readmission rate differs by ethnicity.
    Returns (F_statistic, p_value).
    """
    groups = [group["readmitted_30d"].values for _, group in df.groupby("ethnicity")]
    f_stat, p_val = stats.f_oneway(*groups)
    return round(float(f_stat), 4), round(float(p_val), 6)


def main():
    print("=" * 60)
    print("Patient Demographics Analysis")
    print("=" * 60)

    df = generate_patient_data()
    print(f"\nDataset: {len(df)} patients")
    print(df[["patient_id", "age", "gender", "ethnicity", "insurance", "comorbidity_count", "readmitted_30d"]].head())

    print("\n--- Age Group Distribution ---")
    print(age_group_distribution(df).to_string(index=False))

    print("\n--- Gender Breakdown ---")
    print(gender_breakdown(df).to_string(index=False))

    print("\n--- Ethnicity Breakdown ---")
    print(ethnicity_breakdown(df).to_string(index=False))

    print("\n--- Comorbidity Prevalence ---")
    print(comorbidity_prevalence(df).to_string(index=False))

    print("\n--- 30-Day Readmission Rate by Age Group ---")
    print(readmission_by_age_group(df).to_string())

    print("\n--- ANOVA: Readmission Rate by Ethnicity ---")
    f_stat, p_val = anova_readmission_by_ethnicity(df)
    print(f"  F-statistic : {f_stat}")
    print(f"  p-value     : {p_val}")
    sig = "significant" if p_val < 0.05 else "not significant"
    print(f"  Result      : Ethnicity effect is {sig} (α = 0.05)")

    print("\nAnalysis complete.")
    return df


if __name__ == "__main__":
    main()
