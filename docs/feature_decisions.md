# Feature decisions

Stage 3 record. Each decision gives the evidence from `notebooks/03_feature_selection.ipynb`.
Evidence basis: training sample for IV and correlation; training against out-of-time for PSI
and missingness. Out-of-time performance was not used to select features.

Screening rule: IV below 0.02 is dropped unless a business reason is documented here.
Correlation rule: from any pair with |r| above 0.7, drop the one with lower IV.
Stability rule: PSI above 0.25 requires review; missingness change above 5 points is flagged.

## Kept for the scorecard (10)

| Feature | IV | PSI train vs OOT | Missing change | Decision | Reason |
|---|---|---|---|---|---|
| fico | 0.0821 | 0.012 | 0.000 | Keep | Highest IV; stable |
| inq_last_6mths | 0.0636 | 0.055 | 0.000 | Keep, monitor | Below 0.10 PSI; watch recent enquiries |
| purpose | 0.0597 | n/a (categorical) | 0.000 | Keep | Categorical; stable mix |
| annual_inc | 0.0551 | 0.014 | 0.000 | Keep | Stable |
| installment_to_income | 0.0481 | 0.049 | +0.000 | Keep | Payment burden; replaces loan_to_income |
| credit_history_months | 0.0373 | 0.008 | 0.000 | Keep | Stable |
| term_months | 0.0363 | 0.000 | 0.000 | Keep | Risk rises with term; no drift |
| dti | 0.0314 | 0.038 | +0.000 | Keep | Core capacity driver |
| home_ownership | 0.0298 | n/a (categorical) | 0.000 | Keep | Categorical; stable |
| mort_acc | 0.0273 | 0.031 | -0.026 | Keep, note | Missing only in 2012 (training); documented in report |

## Dropped: duplicate

| Feature | IV | PSI | Decision | Reason |
|---|---|---|---|---|
| loan_to_income | 0.0415 | 0.030 | Drop | Correlation 0.946 with installment_to_income, lower IV; VIF 45.0 before drop (installment_to_income 39.5 before drop; recheck after drop in 04 section 8) |

## Dropped: moderate or weak IV, not used in the scorecard

| Feature | IV | PSI | Decision | Reason |
|---|---|---|---|---|
| revol_util | 0.0116 | 0.145 | Drop from scorecard; monitor | Weak IV; largest PSI; steady downward trend (median 61 to 46), consistent with borrower behaviour. Kept in monitoring. |
| emp_years | 0.0146 | 0.024 | Drop | Below 0.02; missingness up 2.0 points |
| verification_status | 0.0119 | n/a (categorical) | Drop | Below 0.02 |
| total_acc | 0.0119 | 0.007 | Drop | Below 0.02 |
| revol_bal | 0.0078 | 0.033 | Drop | Below 0.02 |
| open_acc | 0.0060 | 0.022 | Drop | Below 0.02 |
| loan_amnt | 0.0021 | 0.021 | Drop | Below 0.02 |

## Dropped: derogatory counts (near-zero IV, business-standard drivers)

These are standard credit drivers, so the drop is shown in the data rather than assumed.
The non-zero share in the training sample is recorded below from the notebook.

| Feature | IV | PSI | Decision | Reason |
|---|---|---|---|---|
| delinq_2yrs | 0.0009 | 0.006 | Drop | IV below 0.001. 17.5% of training loans non-zero; bad rate 2.71% if non-zero vs 2.75% if zero (no positive relationship) |
| pub_rec | 0.0008 | 0.041 | Drop | IV below 0.001. 12.6% non-zero; bad rate 2.54% if non-zero vs 2.77% if zero (direction opposite to expectation) |
| has_delinquency | 0.0007 | 0.000 | Drop | IV below 0.001. 28.2% non-zero; bad rate 2.63% if non-zero vs 2.79% if zero |
| collections_12_mths_ex_med | 0.0000 | 0.000 | Drop | 0.65% non-zero (1,864 loans); bad rate 2.79% if non-zero vs 2.75% if zero; too few cases to use |

## Dropped: missingness

| Feature | IV | PSI | Missing change | Decision | Reason |
|---|---|---|---|---|---|
| mths_since_last_delinq | 0.0042 | 0.006 | -0.067 | Drop | 55% missing in training; missingness moved 6.7 points out of time; IV 0.004 |

## Finding for the validation report: derogatory fields carry no signal in this sample

For `pub_rec` and `delinq_2yrs`, borrowers with a derogatory record default slightly
less often than borrowers without one. This runs against business expectation. Likely
explanations, to be tested rather than assumed: Lending Club's acceptance screen
already removes applicants with derogatory history (selection bias, since only accepted
loans are in the file), and the 12-month window captures few defaults. The validation
report must state this finding and the limitation together.

## Checks still open

- IV benchmark against Lending Club grade, and IV against the lifetime label (Stage 3 extension).
- VIF after `loan_to_income` is removed, in `04_woe_scorecard` section 8.
