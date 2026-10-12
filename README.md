# FairAI Bangladesh: AI Bias & Fairness Checker

A beginner-friendly Responsible AI project that audits a binary classification model for group-level differences, using synthetic Bangladesh-inspired job screening data.

> **Important:** This is an educational audit tool, not a legal compliance assessment or proof that a model is fair or unfair. Metrics are context-dependent. The demo uses synthetic data and must not be used to make real hiring decisions.

## Features

- **Dataset bias analysis:** group sizes, class distribution per group, missing values
- **Group-wise comparison:** selection rate, accuracy, true/false positive rates
- **Fairness metrics:** demographic parity difference, disparate impact ratio, equal opportunity difference
- **Charts:** selection rate and performance by group
- **Responsible AI report:** findings, metric explanations, mitigation ideas, limitations

## Tech stack

Python 3.10+ · Pandas · NumPy · scikit-learn · Matplotlib

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python src/fairness_audit.py
```

Outputs are saved to `reports/`:

| File | Description |
|---|---|
| `demo_dataset.csv` | Synthetic dataset used in the audit |
| `group_metrics.csv` | Per-group metrics on the test set |
| `selection_rate_by_group.png` | Selection rate chart |
| `performance_by_group.png` | Accuracy, TPR and FPR by group |
| `responsible_ai_report.md` | Full Markdown report |

## Results

These charts are generated automatically by GitHub Actions each time the code changes.

![Selection rate by group](reports/selection_rate_by_group.png)

![Model performance by group](reports/performance_by_group.png)

Full report: [responsible_ai_report.md](reports/responsible_ai_report.md)

## Dataset

The script generates a reproducible synthetic dataset with test score, experience, education, career gap years, a group column (`gender_group`), and a target (`qualified`). The group column is **not** given to the model, but career gap years is correlated with group and the historical labels penalise it. This shows how bias can leak in through a proxy feature. It does **not** represent real Bangladesh labor-market data.

## Metrics

- **Selection rate:** share of each group predicted positive.
- **Demographic parity difference:** highest minus lowest group selection rate.
- **Disparate impact ratio:** lowest selection rate divided by the highest (undefined if no group is selected). The 0.8 "four-fifths" value is a screening heuristic only.
- **Equal opportunity difference:** highest minus lowest true positive rate across groups.
- **Accuracy, TPR, FPR, FNR by group:** to see whether errors fall unevenly.

No single metric captures every kind of unfairness, and different metrics can conflict.

## Project structure

```text
fairai-bangladesh/
├── .github/workflows/run-audit.yml
├── src/
│   └── fairness_audit.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Next improvements

1. Add a Streamlit dashboard.
2. Support CSV upload and column selection.
3. Add intersectional analysis with minimum group-size warnings.
4. Add bootstrap confidence intervals for each metric.
5. Add tests and a model-card template.
6. Add a mitigation experiment (re-weighting or threshold tuning) with before/after comparison.

## Responsible use

- Document how groups and labels were defined.
- Check sample sizes and data quality before interpreting differences.
- Investigate plausible causes; do not remove features or change thresholds based on one metric alone.
- Protected attributes are not a complete representation of identity.
