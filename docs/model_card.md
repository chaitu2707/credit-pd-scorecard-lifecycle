# Model card: 12-month PD scorecard for unsecured consumer loans

| Item | Value |
|---|---|
| Model | Logistic regression scorecard on weight-of-evidence bins, scaled to points |
| Champion version | Includes lender-assigned risk grade. Without-grade version reported for comparison |
| Target | Charged off or defaulted within 12 months of issue (estimated default date = last payment + 120 days) |
| Population | Accepted US unsecured consumer instalment loans, issued 2012 to 2017 |
| Development | Training sample, Jan 2012 to Jun 2014 (287,431 loans) |
| Tuning and recalibration | Validation sample, Jul 2014 to Dec 2015 (498,802 loans) |
| Out-of-time test | Jan 2016 to Sep 2017 (427,382 loans), not used for any fitting decision |
| Owner | _to be completed_ |
| Approval | _to be completed_ |
| Review frequency | Monthly monitoring; full review annually or on a red-trigger |

## Intended use

Ranking unsecured personal loan applicants by 12-month risk, and providing reason codes for adverse action notices. Suitable for application screening and as one input to cut-off policy.

## Not intended for

- Provisioning or capital, until recalibrated to current default levels. Predicted PD out of time is about half the observed rate.
- Secured lending, credit cards, mortgages or business lending.
- Populations other than accepted applicants. Rejected applicants are not in the training data.
- Lifetime PD or loss given default.

## Performance (out-of-time, 5-trial run)

| Measure | Value |
|---|---|
| Gini | 0.384 |
| KS | 0.279 |
| AUC | 0.692 |
| Brier score | 0.062 |
| Mean predicted PD | 3.5% |
| Observed 12-month bad rate | 6.7% |

Validation Gini is 0.432. Without grade, out-of-time Gini is 0.302. The tree challengers (XGBoost, LightGBM, CatBoost and their ensemble) reach 0.381 to 0.386 out of time, so they do not justify replacing the scorecard.

## Key drivers

Lender-assigned grade is the largest driver (mean absolute SHAP 0.36), followed by annual income, debt-to-income ratio, recent enquiries, purpose, credit history length and FICO. Grade is excluded from reason codes, and so is purpose.

## Limitations

1. **Calibration drift.** Observed defaults rose through 2016 and 2017 while predicted PD stayed near 3.5%. The gap widened from 1.3 times to 3.0 times over the period.
2. **Dependence on grade.** Grade alone reaches Gini 0.377 out of time. The champion's advantage over the without-grade model comes almost entirely from grade.
3. **Estimated default timing.** The default date is inferred from the last payment date. Results change by about 1.6 percentage points in bad rate across a 90 to 150 day range.
4. **Accepted loans only.** Rejected applicants are absent, so no reject inference is performed.
5. **Derogatory history shows no positive link to default** in this sample. The likely cause is selection in the accepted population, which is not verified.
6. **No macroeconomic variables.**

## Monitoring plan

Monthly out-of-time batches are scored by the frozen champion and checked for score PSI, calibration (actual to predicted), batch Gini and feature PSI, with traffic lights. Two consecutive red batches trigger a retraining review. In the test period, 18 of 21 batches are red, and the review trigger fires from May 2016. The score and input distributions stay stable throughout, so the trigger reflects the outcome drift, not a change in borrowers.

## Decisions required before use

1. Recalibrate on recent data, or redevelop the model on recent vintages, before using predicted PD for provisioning.
2. Approve the use of lender-assigned grade as an input, given that it provides most of the discrimination.
3. Complete the owner and approval fields above.
