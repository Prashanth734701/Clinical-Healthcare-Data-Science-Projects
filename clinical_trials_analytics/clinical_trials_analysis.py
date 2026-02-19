"""
Clinical Trials Analytics
==========================
Analyzes synthetic clinical trial data to evaluate treatment efficacy,
patient enrollment trends, adverse events, and trial outcome statistics.
"""

import numpy as np
import pandas as pd
from scipy import stats


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_PATIENTS = 300
TRIAL_DURATION_WEEKS = 24


def generate_trial_data():
    """Generate synthetic clinical trial dataset."""
    patient_ids = [f"PT{str(i).zfill(4)}" for i in range(1, N_PATIENTS + 1)]
    arms = np.random.choice(["Treatment", "Control"], size=N_PATIENTS, p=[0.5, 0.5])
    ages = np.random.normal(loc=52, scale=12, size=N_PATIENTS).clip(18, 85).astype(int)
    genders = np.random.choice(["Male", "Female"], size=N_PATIENTS)
    enrollment_week = np.random.randint(1, 13, size=N_PATIENTS)

    baseline_scores = np.where(
        arms == "Treatment",
        np.random.normal(65, 10, N_PATIENTS),
        np.random.normal(64, 10, N_PATIENTS),
    ).clip(0, 100)

    # Treatment arm shows ~15 point improvement; control ~5 points
    improvement = np.where(
        arms == "Treatment",
        np.random.normal(15, 5, N_PATIENTS),
        np.random.normal(5, 5, N_PATIENTS),
    )
    endpoint_scores = (baseline_scores + improvement).clip(0, 100)

    adverse_event_prob = np.where(arms == "Treatment", 0.25, 0.15)
    adverse_events = np.random.binomial(1, adverse_event_prob)

    dropout_prob = np.where(arms == "Treatment", 0.10, 0.08)
    dropout = np.random.binomial(1, dropout_prob)

    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "arm": arms,
            "age": ages,
            "gender": genders,
            "enrollment_week": enrollment_week,
            "baseline_score": baseline_scores.round(1),
            "endpoint_score": endpoint_scores.round(1),
            "adverse_event": adverse_events,
            "dropout": dropout,
        }
    )


def compute_summary_stats(df):
    """Return per-arm summary statistics."""
    return (
        df.groupby("arm")
        .agg(
            n_patients=("patient_id", "count"),
            mean_age=("age", "mean"),
            mean_baseline=("baseline_score", "mean"),
            mean_endpoint=("endpoint_score", "mean"),
            adverse_event_rate=("adverse_event", "mean"),
            dropout_rate=("dropout", "mean"),
        )
        .round(2)
    )


def test_efficacy(df):
    """
    Perform an independent-samples t-test comparing endpoint scores
    between the Treatment and Control arms.
    Returns (t_statistic, p_value).
    """
    treatment_scores = df.loc[df["arm"] == "Treatment", "endpoint_score"]
    control_scores = df.loc[df["arm"] == "Control", "endpoint_score"]
    t_stat, p_value = stats.ttest_ind(treatment_scores, control_scores)
    return round(float(t_stat), 4), round(float(p_value), 6)


def enrollment_by_week(df):
    """Return weekly cumulative enrollment counts per arm."""
    weekly = (
        df.groupby(["enrollment_week", "arm"])
        .size()
        .reset_index(name="new_enrollments")
    )
    weekly = weekly.sort_values(["arm", "enrollment_week"])
    weekly["cumulative_enrollment"] = weekly.groupby("arm")["new_enrollments"].cumsum()
    return weekly


def adverse_event_chi_square(df):
    """
    Chi-square test for independence between trial arm and adverse events.
    Returns (chi2_statistic, p_value).
    """
    contingency = pd.crosstab(df["arm"], df["adverse_event"])
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    return round(float(chi2), 4), round(float(p), 6)


def main():
    print("=" * 60)
    print("Clinical Trials Analytics")
    print("=" * 60)

    df = generate_trial_data()
    print(f"\nDataset: {len(df)} patients across {df['arm'].nunique()} trial arms")
    print(df.head())

    print("\n--- Per-Arm Summary Statistics ---")
    summary = compute_summary_stats(df)
    print(summary.to_string())

    print("\n--- Efficacy: Independent t-Test (endpoint scores) ---")
    t_stat, p_val = test_efficacy(df)
    print(f"  t-statistic : {t_stat}")
    print(f"  p-value     : {p_val}")
    sig = "statistically significant" if p_val < 0.05 else "not statistically significant"
    print(f"  Result      : Difference is {sig} (α = 0.05)")

    print("\n--- Adverse Events: Chi-Square Test ---")
    chi2, p_ae = adverse_event_chi_square(df)
    print(f"  chi2        : {chi2}")
    print(f"  p-value     : {p_ae}")
    ae_sig = "significant" if p_ae < 0.05 else "not significant"
    print(f"  Result      : AE difference is {ae_sig} (α = 0.05)")

    print("\n--- Weekly Enrollment (first 4 weeks) ---")
    weekly = enrollment_by_week(df)
    print(weekly[weekly["enrollment_week"] <= 4].to_string(index=False))

    print("\nAnalysis complete.")
    return df, summary


if __name__ == "__main__":
    main()
