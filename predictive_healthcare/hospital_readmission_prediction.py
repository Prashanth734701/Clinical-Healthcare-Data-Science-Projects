"""
Predictive Healthcare Systems – Hospital Readmission Prediction
================================================================
Trains and evaluates machine-learning models to predict 30-day hospital
readmission risk from synthetic patient features (demographics,
comorbidities, prior utilization, lab values).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N_PATIENTS = 1000


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_patient_data(n=N_PATIENTS):
    """
    Generate a synthetic patient dataset suitable for readmission prediction.
    Features include demographics, comorbidities, prior utilisation, and labs.
    """
    ages = np.random.normal(58, 14, n).clip(18, 90).astype(int)
    genders = np.random.choice(["Male", "Female"], n)
    insurance = np.random.choice(
        ["Private", "Medicare", "Medicaid", "Uninsured"], n, p=[0.50, 0.25, 0.18, 0.07]
    )

    hypertension = np.random.binomial(1, 0.45, n)
    diabetes = np.random.binomial(1, 0.30, n)
    heart_disease = np.random.binomial(1, 0.20, n)
    obesity = np.random.binomial(1, 0.35, n)
    comorbidity_count = hypertension + diabetes + heart_disease + obesity

    prior_admissions = np.random.poisson(1.2, n)
    los_days = np.random.gamma(shape=2, scale=2, size=n).clip(1, 30).astype(int)
    er_visits = np.random.poisson(0.8, n)

    hba1c = np.where(diabetes, np.random.normal(7.5, 1.2, n), np.random.normal(5.5, 0.5, n))
    sodium = np.random.normal(139, 3, n).clip(125, 150)
    creatinine = np.random.gamma(shape=2, scale=0.6, size=n).clip(0.5, 10)

    # Readmission probability model
    log_odds = (
        -2.5
        + 0.03 * (ages - 55)
        + 0.4 * hypertension
        + 0.6 * diabetes
        + 0.7 * heart_disease
        + 0.3 * obesity
        + 0.5 * prior_admissions
        + 0.1 * los_days
        + 0.3 * er_visits
        + 0.2 * (hba1c - 5.5)
        + 0.3 * (creatinine - 1.0)
        - 0.1 * (sodium - 139)
        + np.where(insurance == "Uninsured", 0.5, 0)
        + np.random.normal(0, 0.3, n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    readmitted = np.random.binomial(1, prob.clip(0.01, 0.99))

    df = pd.DataFrame(
        {
            "age": ages,
            "gender": genders,
            "insurance": insurance,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "heart_disease": heart_disease,
            "obesity": obesity,
            "comorbidity_count": comorbidity_count,
            "prior_admissions": prior_admissions,
            "los_days": los_days,
            "er_visits_prior_year": er_visits,
            "hba1c": hba1c.round(1),
            "sodium": sodium.round(1),
            "creatinine": creatinine.round(2),
            "readmitted_30d": readmitted,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def prepare_features(df):
    """Encode categorical features and return X (features) and y (target)."""
    df = df.copy()
    le_gender = LabelEncoder()
    le_insurance = LabelEncoder()
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["insurance_enc"] = le_insurance.fit_transform(df["insurance"])

    feature_cols = [
        "age", "gender_enc", "insurance_enc",
        "hypertension", "diabetes", "heart_disease", "obesity",
        "comorbidity_count", "prior_admissions", "los_days",
        "er_visits_prior_year", "hba1c", "sodium", "creatinine",
    ]
    X = df[feature_cols].values
    y = df["readmitted_30d"].values
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------

def build_pipelines():
    """Return a dict of named sklearn pipelines."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_SEED)),
        ]),
    }


def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    """Fit pipeline and return a dict of evaluation metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "brier_score": round(brier_score_loss(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "y_prob": y_prob,
    }


def cross_validate_models(pipelines, X, y, cv=5):
    """
    Run stratified k-fold CV and return mean ± std AUC for each model.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    results = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv_splitter)
        results[name] = {"mean_auc": round(scores.mean(), 4), "std_auc": round(scores.std(), 4)}
    return results


def feature_importance(pipeline, feature_names):
    """
    Extract feature importances from a fitted Random Forest or Gradient
    Boosting model.  Returns a sorted DataFrame.
    """
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return fi_df.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Predictive Healthcare – Hospital Readmission Prediction")
    print("=" * 65)

    df = generate_patient_data()
    print(f"\nDataset: {len(df)} patients  |  Readmission rate: {df['readmitted_30d'].mean()*100:.1f}%")
    print(df.head())

    X, y, feature_cols = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

    pipelines = build_pipelines()

    print("\n--- 5-Fold Cross-Validation AUC ---")
    cv_results = cross_validate_models(pipelines, X_train, y_train)
    for name, res in cv_results.items():
        print(f"  {name:<25} AUC = {res['mean_auc']:.4f} ± {res['std_auc']:.4f}")

    print("\n--- Hold-Out Test Set Evaluation ---")
    for name, pipe in pipelines.items():
        metrics = evaluate_model(pipe, X_train, X_test, y_train, y_test)
        print(f"\n  Model: {name}")
        print(f"    ROC-AUC      : {metrics['roc_auc']}")
        print(f"    Brier score  : {metrics['brier_score']}")
        print(f"    Confusion matrix:\n{metrics['confusion_matrix']}")

    print("\n--- Feature Importances (Gradient Boosting) ---")
    gb_pipe = pipelines["Gradient Boosting"]
    gb_pipe.fit(X_train, y_train)
    fi = feature_importance(gb_pipe, feature_cols)
    if fi is not None:
        print(fi.head(10).to_string(index=False))

    print("\nAnalysis complete.")
    return df, pipelines


if __name__ == "__main__":
    main()
