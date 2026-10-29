# Credit Default PD Model: Scorecard Development, Validation and Monitoring

A 12-month probability-of-default (PD) model for unsecured consumer instalment loans, built on the Lending Club accepted loans file (2007 to 2018 Q4, 2.26 million loans). The champion model is a weight-of-evidence (WoE) logistic scorecard. The model is developed on vintage-based samples, validated out of time, and documented to model risk governance expectations.

## Headline results

| Sample | Issue period | Loans | Actual 12-month bad rate | Gini | KS | Brier |
|---|---|---|---|---|---|---|
| Validation | Jul 2014 to Dec 2015 | 498,802 | 3.58% | 0.432 | 0.321 | 0.0337 |
| Out-of-time | Jan 2016 to Sep 2017 | 427,382 | 6.71% | 0.384 | 0.279 | 0.0621 |

Without the lender's grade, out-of-time Gini is 0.302. Grade is kept in the champion model and documented as a dependency.

Tree challengers (XGBoost, LightGBM, CatBoost and their ensemble) reach Gini 0.381 to 0.386 out of time. None beats the scorecard by the 0.02 margin required for replacement.

**Calibration and drift are the main limitations.** Predicted PD is about half the observed default rate out of time, and monitoring shows the gap widening through 2016 and 2017. The retraining review trigger has fired. See the validation report, sections 8 and 11.

_The challenger figures come from a 5-trial tuning run. A 50-trial refresh is scheduled._

## Scope

- Unsecured consumer instalment loans from a US peer-to-peer platform.
- 12-month PD on final-outcome loans only, with a maturity rule so that every loan has a full outcome window.
- Logistic regression scorecard (WoE bins, points scaling, reason codes) as champion. Tree challengers (XGBoost, LightGBM, CatBoost) and their calibrated ensemble, with the champion-challenger rule applied.
- Probability of default only. LGD, EAD and expected loss are out of scope.

## Repository structure

```
src/        pipeline modules: data preparation, samples, features, scorecard, metrics
notebooks/  stage notebooks 01 to 04, run in order
tests/      unit tests for metrics, feature selection and the scorecard
docs/       validation report, assumptions, data quality log, feature decisions
reports/    generated tables and figures
```

## How to run

Requires Python 3.11 or later.

```bash
pip install -r requirements.txt
# place accepted_2007_to_2018Q4.csv in data/raw/ (see data/raw/README.md)
python -m src.data_prep          # builds the labelled dataset and the data quality funnel
python -m src.label_sensitivity  # 12-month label under 90, 120 and 150 day charge-off lags
python -m src.make_slim          # optional: slim working file for the notebooks
python -m pytest -q              # unit tests
```

Then run the notebooks in order, using Run All from the top of each.

## Documentation

- [Validation report](docs/validation_report.md): data, methodology, results, findings and limitations
- [Assumptions](docs/assumptions.md): every assumption, its value and what it affects
- [Data quality log](docs/data_quality_log.md): row counts after each rule and the lag decision
- [Feature decisions](docs/feature_decisions.md): keep or drop decision for each candidate feature, with evidence
