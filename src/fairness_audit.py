"""FairAI Bangladesh: a small, reproducible group-fairness audit demo.

Trains a baseline classifier on synthetic job-screening data, then compares
selection rates and error rates across groups and writes charts plus a
Responsible AI report to the reports/ folder.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RANDOM_STATE = 42
MIN_GROUP_SIZE = 30  # warn when a group is too small to interpret

PALETTE = ["#2F6F9F", "#E08A2E", "#4C9A6A", "#8E5EA2"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_demo_data(n: int = 1200) -> pd.DataFrame:
    """Generate synthetic data for learning; not real hiring data."""
    rng = np.random.default_rng(RANDOM_STATE)
    gender = rng.choice(["Women", "Men"], size=n, p=[0.48, 0.52])
    test_score = np.clip(rng.normal(68, 14, n), 0, 100)
    experience_years = np.clip(rng.poisson(3, n), 0, 15)
    education = rng.choice(["HSC", "Bachelor", "Masters"], size=n, p=[0.32, 0.52, 0.16])

    education_bonus = pd.Series(education).map({"HSC": 0, "Bachelor": 5, "Masters": 8}).to_numpy()

    # Proxy feature: career gaps are more common for women in this synthetic data,
    # so a model can pick up group information without ever seeing the group column.
    career_gap_years = np.where(gender == "Women", rng.poisson(1.3, n), rng.poisson(0.3, n))

    # Historical labels penalise career gaps, which encodes a biased past decision.
    logit = (
        (test_score - 65) / 12
        + experience_years * 0.13
        + education_bonus * 0.07
        - career_gap_years * 0.45
        + 0.2
    )
    probability = 1 / (1 + np.exp(-logit))
    qualified = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "gender_group": gender,
            "test_score": test_score.round(1),
            "experience_years": experience_years,
            "education": education,
            "career_gap_years": career_gap_years,
            "qualified": qualified,
        }
    )


def dataset_overview(data: pd.DataFrame) -> dict:
    """Group sizes, class balance per group, and missing values."""
    by_group = (
        data.groupby("gender_group")["qualified"]
        .agg(n="size", positive_rate="mean")
        .reset_index()
        .rename(columns={"gender_group": "group"})
    )
    missing = data.isna().sum()
    return {"by_group": by_group, "missing": missing[missing > 0]}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def calculate_group_metrics(y_true, y_pred, groups) -> pd.DataFrame:
    """Descriptive metrics for each group."""
    frame = pd.DataFrame(
        {"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "group": np.asarray(groups)}
    )
    rows = []
    for group, part in frame.groupby("group"):
        tn, fp, fn, tp = confusion_matrix(part["y_true"], part["y_pred"], labels=[0, 1]).ravel()
        rows.append(
            {
                "group": str(group),
                "n": len(part),
                "selection_rate": part["y_pred"].mean(),
                "accuracy": accuracy_score(part["y_true"], part["y_pred"]),
                "true_positive_rate": safe_divide(tp, tp + fn),
                "false_positive_rate": safe_divide(fp, fp + tn),
                "false_negative_rate": safe_divide(fn, fn + tp),
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def summarize_fairness(metrics: pd.DataFrame) -> dict:
    rates = metrics["selection_rate"].to_numpy()
    max_rate, min_rate = float(np.max(rates)), float(np.min(rates))
    tpr = metrics["true_positive_rate"].to_numpy()
    return {
        "demographic_parity_difference": max_rate - min_rate,
        "disparate_impact_ratio": min_rate / max_rate if max_rate > 0 else np.nan,
        "equal_opportunity_difference": float(np.nanmax(tpr) - np.nanmin(tpr)),
        "small_groups": metrics.loc[metrics["n"] < MIN_GROUP_SIZE, "group"].tolist(),
    }


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def plot_selection_rate(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(metrics["group"], metrics["selection_rate"], color=PALETTE[: len(metrics)], width=0.55)
    for bar, value in zip(bars, metrics["selection_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.1%}", ha="center", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted positive rate")
    ax.set_title("Selection Rate by Group", loc="left", fontweight="bold")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_group_performance(metrics: pd.DataFrame, path: Path) -> None:
    names = ["accuracy", "true_positive_rate", "false_positive_rate"]
    labels = ["Accuracy", "True positive rate", "False positive rate"]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (name, label) in enumerate(zip(names, labels)):
        ax.bar(x + (i - 1) * width, metrics[name], width, label=label, color=PALETTE[i])
    ax.set_xticks(x, metrics["group"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Model Performance by Group", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def fmt(value: float) -> str:
    return f"{value:.3f}" if np.isfinite(value) else "undefined"


def write_report(metrics: pd.DataFrame, overview: dict, data: pd.DataFrame, summary: dict) -> None:
    missing = overview["missing"]
    missing_line = (
        "- Missing values: none"
        if missing.empty
        else "- Missing values: " + ", ".join(f"{k} ({v})" for k, v in missing.items())
    )
    warning = (
        [f"- **Warning:** very small groups (n < {MIN_GROUP_SIZE}): {', '.join(summary['small_groups'])}."]
        if summary["small_groups"]
        else []
    )

    lines = [
        "# FairAI Bangladesh: Responsible AI Audit Report",
        "",
        "## Scope",
        "",
        "This report audits a baseline classifier on **synthetic demonstration data**. "
        "It is not an assessment of real hiring practices or a legal compliance opinion.",
        "",
        "## Dataset overview",
        "",
        f"- Rows: {len(data)}",
        "- Features: test score, experience years, education, career gap years (a proxy correlated with group)",
        "- Group column: `gender_group`",
        f"- Overall positive target rate: {data['qualified'].mean():.3f}",
        missing_line,
        *warning,
        "",
        overview["by_group"].round(3).to_markdown(index=False),
        "",
        "## Group metrics (test set)",
        "",
        metrics.round(3).to_markdown(index=False),
        "",
        "![Selection rate by group](selection_rate_by_group.png)",
        "",
        "![Model performance by group](performance_by_group.png)",
        "",
        "## Fairness summary",
        "",
        f"- Demographic parity difference (max selection rate - min): **{fmt(summary['demographic_parity_difference'])}**",
        f"- Disparate impact ratio (min selection rate / max): **{fmt(summary['disparate_impact_ratio'])}**",
        f"- Equal opportunity difference (max TPR - min TPR): **{fmt(summary['equal_opportunity_difference'])}**",
        "",
        "These values are descriptive signals, not automatic pass/fail judgments. "
        "The acceptable level of difference depends on context, measurement quality, "
        "sample size, law, and stakeholder input.",
        "",
        "## How to read the metrics",
        "",
        "- **Selection rate:** share of a group predicted positive.",
        "- **Demographic parity difference:** gap between the highest and lowest selection rates. 0 means equal rates.",
        "- **Disparate impact ratio:** lowest selection rate divided by the highest. Closer to 1 means more similar rates. "
        "A common screening heuristic is 0.8 (the \"four-fifths rule\"), but it is a rule of thumb, not proof of fairness or unfairness.",
        "- **Equal opportunity difference:** gap in true positive rates, i.e. how often qualified people in each group are correctly selected.",
        "",
        "## Possible mitigation steps",
        "",
        "- Review how the labels were created and whether they encode past bias.",
        "- Check whether features act as proxies for group membership.",
        "- Try re-weighting or re-sampling training data, or group-aware threshold tuning, and re-measure.",
        "- Compare several metrics together; improving one can worsen another.",
        "- Keep a human review step and document decisions in a model card.",
        "",
        "## Interpretation checklist",
        "",
        "- Check group sample sizes and data quality.",
        "- Compare false-positive, false-negative, and true-positive rates as well as selection rates.",
        "- Investigate how labels were created and whether features are valid for the intended use.",
        "- Consult affected stakeholders and qualified legal and ethics experts before deployment.",
        "- Do not use this synthetic demo to screen real applicants.",
        "",
        "## Limitations",
        "",
        "- Synthetic data may not reflect real populations or social processes.",
        "- The group column is not a model input, but a correlated proxy feature (career gap years) can still carry group information.",
        "- Only a binary group field is represented; identities and intersections are more complex.",
        "- The audit examines one train/test split and one baseline model, with no confidence intervals.",
        "- Statistical parity alone does not establish fairness.",
        "",
    ]
    (REPORTS / "responsible_ai_report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_model() -> Pipeline:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric_pipe, ["test_score", "experience_years", "career_gap_years"]),
            ("categorical", categorical_pipe, ["education"]),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    data = make_demo_data()
    data.to_csv(REPORTS / "demo_dataset.csv", index=False)
    overview = dataset_overview(data)

    features = data[["test_score", "experience_years", "education", "career_gap_years"]]
    X_train, X_test, y_train, y_test, _, groups_test = train_test_split(
        features,
        data["qualified"],
        data["gender_group"],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=data["qualified"],
    )

    model = build_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = calculate_group_metrics(y_test, predictions, groups_test)
    metrics.to_csv(REPORTS / "group_metrics.csv", index=False)
    summary = summarize_fairness(metrics)

    plot_selection_rate(metrics, REPORTS / "selection_rate_by_group.png")
    plot_group_performance(metrics, REPORTS / "performance_by_group.png")
    write_report(metrics, overview, data, summary)

    print("\nFairAI Bangladesh audit complete.\n")
    print("Dataset by group:")
    print(overview["by_group"].round(3).to_string(index=False))
    print("\nGroup metrics (test set):")
    print(metrics.round(3).to_string(index=False))
    print(f"\nDemographic parity difference: {fmt(summary['demographic_parity_difference'])}")
    print(f"Disparate impact ratio:        {fmt(summary['disparate_impact_ratio'])}")
    print(f"Equal opportunity difference:  {fmt(summary['equal_opportunity_difference'])}")
    print(f"\nOutputs saved to: {REPORTS}")


if __name__ == "__main__":
    main()
