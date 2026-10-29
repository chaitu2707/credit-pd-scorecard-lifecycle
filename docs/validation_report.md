# Validation report: 12-month PD scorecard

Status: **5-trial run.** Challenger tuning uses 5 Optuna trials per model. A 50-trial run is scheduled and will refresh the challenger figures in this report. The champion and its verdict are not expected to change, and sections marked "to refresh" will be updated when it completes.

## 1. Purpose and scope

This report documents the development and out-of-time validation of a 12-month probability-of-default model for unsecured consumer instalment loans. It covers data, the label, sample design, feature selection, the scorecard, challenger models, validation results, monitoring and limitations.

Out of scope: loss given default, exposure at default, lifetime PD, macroeconomic overlays.

## 2. Data

Source: Lending Club accepted loans, 2007 to 2018 Q4, 2,260,701 loans. Accepted loans only, so rejected applicants are absent.

| Step | Rows remaining | Rows removed |
|---|---|---|
| Raw file | 2,260,701 | n/a |
| Final outcome only | 1,348,099 | 912,602 |
| Issued on or before 30 Sep 2017 | 1,257,680 | 90,419 |
| Charged-off loans with a last payment date | 1,256,067 | 1,613 |

Modelling samples total 1,213,615 loans. The 42,452 loans issued in 2007 to 2011 are excluded, because underwriting changed materially in those years.

## 3. Target

Bad (1): charged off or defaulted, with the estimated default within 12 months of issue. Good (0): otherwise. Estimated default date = last payment + 120 days.

Over the life of the loans, 251,544 were charged off, and 55,777 (22%) defaulted within 12 months. The 12-month bad rate is 4.44%.

Sensitivity to the lag:

| Lag | Bad loans | Bad rate | Loans changing label |
|---|---|---|---|
| 90 days | 65,436 | 5.21% | 9,659 |
| 120 days (base) | 55,777 | 4.44% | 0 |
| 150 days | 45,635 | 3.63% | 10,142 |

The 120-day base gives the higher PD, the more conservative choice for provisioning.

## 4. Samples

| Sample | Issue period | Loans | Bad rate |
|---|---|---|---|
| Training | Jan 2012 to Jun 2014 | 287,431 | 2.75% |
| Validation | Jul 2014 to Dec 2015 | 498,802 | 3.58% |
| Out-of-time | Jan 2016 to Sep 2017 | 427,382 | 6.71% |

The term mix is stable across samples, so the risk increase is not a term effect. Within each term, the bad rate rises from training to out-of-time (36 months 2.44% to 5.79%; 60 months 3.67% to 10.06%).

## 5. Feature selection

Features were screened on training data using Information Value, with stability checked on out-of-time (PSI and missingness). Out-of-time performance was not used for selection.

- Every candidate is in the weak IV band on the 12-month label. The strongest is FICO (0.082). This reflects the label and the accepted population.
- Eleven features were kept (ten borrower features and lender grade). `loan_to_income` was dropped as a near-duplicate of `installment_to_income` (correlation 0.946). After this, every VIF is below 1.25.
- `revol_util` was dropped from the scorecard: weak IV (0.012) and the largest PSI of any feature (0.145). Its median fell steadily from 61 in 2012 to 46 in 2017. It is kept in monitoring.
- `mths_since_last_delinq` was dropped: 55% missing in training, IV 0.004.
- The derogatory counts have near-zero IV, and borrowers with a derogatory record default at a similar or slightly lower rate. See section 10.

The grade mix is stable across samples, so the rise in defaults happens within each grade.

## 6. Scorecard

Bins use training quartiles with confirmed cut-offs. Categories below 1% of training loans merge into "other", except grade and term, which keep all levels. Monotonicity holds for every numeric feature.

Logistic regression on WoE values. Regularisation C = 0.01, chosen on validation log loss. All coefficients are negative.

Points scaling: base score 600 at 50:1 good-to-bad odds, PDO 20. Points reproduce the model PD to a maximum difference of 3.6 × 10⁻¹⁶.

Recalibration: the intercept is shifted on validation so that the mean PD equals the validation bad rate. The shift is +0.31 on the logit scale. Out-of-time is not used.

## 7. Results

| Model | Sample | Gini | KS | AUC | Brier | Mean PD | Bad rate |
|---|---|---|---|---|---|---|---|
| Scorecard with grade (champion, recalibrated) | Train | 0.372 | 0.276 | 0.686 | 0.0265 | 3.70% | 2.75% |
| Scorecard with grade (champion, recalibrated) | Validation | 0.432 | 0.321 | 0.716 | 0.0337 | 3.58% | 3.58% |
| Scorecard with grade (champion, recalibrated) | Out-of-time | 0.384 | 0.279 | 0.692 | 0.0621 | 3.54% | 6.71% |
| Scorecard without grade | Out-of-time | 0.302 | 0.217 | 0.651 | 0.0627 | 3.45% | 6.71% |
| XGBoost (calibrated) | Out-of-time | 0.383 | 0.278 | 0.692 | 0.0619 | 3.56% | 6.71% |
| LightGBM (calibrated) | Out-of-time | 0.381 | 0.275 | 0.690 | 0.0619 | 3.56% | 6.71% |
| CatBoost (calibrated) | Out-of-time | 0.382 | 0.278 | 0.691 | 0.0620 | 3.50% | 6.71% |
| Ensemble (calibrated) | Out-of-time | 0.386 | 0.280 | 0.693 | 0.0619 | 3.54% | 6.71% |
| Lending Club grade alone | Out-of-time | 0.377 | 0.272 | 0.689 | 0.0631 | 2.69% | 6.71% |

Gini and KS use predicted PD as the score, so the same ranking measure applies to every row.

**Champion-challenger verdict.** The rule is to replace the champion only if the out-of-time Gini gain is at least 0.02 and the calibration gap is no larger. The best challenger by validation Gini is the ensemble. Its out-of-time gain over the champion is 0.002. The rule is not met, so the scorecard remains the champion.

**Grade.** Grade alone reaches out-of-time Gini 0.377. The scorecard with grade reaches 0.384, so borrower features add about 0.007 on top of grade. Without grade, the scorecard reaches 0.302. Most of the predictive power is therefore the lender's own rating. The model is documented as relying on that rating being available at application.

**Challengers.** Tree models overfit: LightGBM reaches Gini 0.46 on training but 0.38 out of time. The scorecard falls by only about 0.04 from training to out-of-time, and the trees add nothing out of time. The simpler model is preferred.

## 8. Calibration and cut-off economics

**Calibration.** Every model predicts about 3.5% out of time, against an observed 6.7%. Recalibration on validation does not correct this, because the drift happens after the validation period. This applies to every model, not only the scorecard.

**Cut-off table** (champion, out-of-time, LGD 60%):

| Max PD | Approval rate | Bad rate among approved | Defaults declined | Expected loss (shown) | Realised loss (approx.) |
|---|---|---|---|---|---|
| 3% | 50.8% | 3.37% | 74.5% | 1.12% | 2.02% |
| 4% | 68.0% | 4.24% | 57.1% | 1.35% | 2.54% |
| 5% | 79.5% | 4.89% | 42.0% | 1.54% | 2.94% |
| 6% | 86.9% | 5.39% | 30.3% | 1.69% | 3.23% |
| 8% | 94.9% | 6.07% | 14.3% | 1.90% | 3.64% |

Realised loss = bad rate among approved × LGD, assuming equal exposure per loan. Expected loss uses predicted PD and is understated by about 1.8 to 1.9 times out of time. The approval ranking remains valid. Loss levels should not be used for provisioning until recalibrated to current defaults.

LGD is an assumption. The dashboard lets users set it from 20% to 90%. Approval rates and bad rates do not depend on LGD.

## 9. Explainability

Mean absolute SHAP on out-of-time for the LightGBM challenger: grade 0.360, annual income 0.145, DTI 0.124, recent enquiries 0.121, purpose 0.094, credit history length 0.092, FICO 0.082, installment-to-income 0.075, home ownership 0.051, term 0.026, mortgage accounts 0.023.

Purpose has material influence in both models but is excluded from applicant reason codes. Reason codes exclude purpose and grade, so applicants are never told that the loan purpose or the lender's rating drove a decision.

## 10. Derogatory fields

| Field | Training loans with non-zero value | Bad rate if non-zero | Bad rate if zero |
|---|---|---|---|
| `pub_rec` | 12.6% | 2.54% | 2.77% |
| `delinq_2yrs` | 17.5% | 2.71% | 2.75% |
| `has_delinquency` | 28.2% | 2.63% | 2.79% |
| `collections_12_mths_ex_med` | 0.65% | 2.79% | 2.75% |

This runs against business expectation. The likely cause is that Lending Club's acceptance process screens out applicants with derogatory histories before they appear in the file. This is a hypothesis, not a tested result, and it is reported as a limitation.

## 11. Monitoring

Twenty-one monthly out-of-time batches, January 2016 to September 2017, scored by the frozen champion.

| Measure | Result |
|---|---|
| Score PSI | Green in every month (maximum 0.065) |
| Calibration (actual to predicted) | Amber in 3 months, red in 18 |
| Batch Gini | 0.32 to 0.48, against development Gini 0.43 |
| Overall | Red in 18 months, amber in 3 |
| Retraining review triggered | From May 2016 onwards (17 batches) |

The actual-to-predicted ratio rises steadily from 1.30 in January 2016 to 2.98 in September 2017. The input and score distributions stay stable, so the cause is a change in outcomes for similar borrowers, which is the drift that feature PSI cannot see.

The retraining review trigger has fired and should be acted on. The required decision is in section 12.

## 12. Findings and recommendations

| # | Finding | Severity | Recommendation |
|---|---|---|---|
| 1 | Predicted PD is about half the observed default rate out of time, and the gap widens through 2016 and 2017. | High | Recalibrate on recent data or redevelop before using PD for provisioning or capital. |
| 2 | Retraining review trigger fired from May 2016 and remains active. | High | Complete the review; record the decision in the model card. |
| 3 | Expected loss in the cut-off table understates realised loss by about 1.8 to 1.9 times. | High | Use realised loss rates, or recalibrated PD, for any loss-based decision. |
| 4 | Lender grade provides most of the discrimination. | Medium | Approve grade as an input, or report the model without grade as the bank's own. |
| 5 | Tree challengers add no out-of-time Gini over the scorecard. | Low | Keep the scorecard as champion. Refresh with 50-trial results. |
| 6 | Derogatory fields show no positive link to default in this sample. | Medium | Investigate the acceptance screen; confirm the field definitions with the data owner. |
| 7 | Default timing is estimated from the last payment date. | Medium | Obtain charge-off dates, if available, to replace the 120-day estimate. |

## 13. Limitations

1. Accepted loans only; no reject inference.
2. Estimated default timing (section 3).
3. Calibration drift (section 8).
4. Dependence on lender grade (section 7).
5. No macroeconomic variables.
6. Individual features are weak on this label, so ranking power is modest.
7. Results apply to unsecured personal instalment loans only.

## 14. Governance

Development, validation and monitoring follow the expectations of SR 11-7 on model development, independent validation and ongoing monitoring. Assumptions are in `docs/assumptions.md`, feature decisions in `docs/feature_decisions.md`, data rules in `docs/data_quality_log.md`, and the model card in `docs/model_card.md`. No class weighting or resampling was applied.
