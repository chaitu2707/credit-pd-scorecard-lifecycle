# Assumptions

Single source of truth for every assumption in the project. The values live in
`src/config.py`. The README and validation report link here rather than repeating them.

| Assumption | Value | Rationale | Affects |
|---|---|---|---|
| Default definition | Charged Off or Default within 12 months of issue = bad (1); Fully Paid, or charged off after 12 months = good (0) | 12-month PD definition | Target |
| Excluded statuses | Current, In Grace Period, Late (16-30), Late (31-120) | Outcome not final | Sample |
| Charge-off timing | Default date = last payment date + 120 days (base case). Sensitivity: 90 and 150 days | Default date is not in the file. 120 days is the base case: it gives the higher, more conservative PD, which is the safer error for provisioning. Sensitivity: 90 days gives 5.21% bad rate, 120 days 4.44%, 150 days 3.63%. About 10,000 loans change label across 90 to 150 days | 12-month label, all samples |
| Horizon | 12 months from issue | Standard PD horizon | Target, maturity |
| Maturity cut-off | Issue date on or before 2017-09-30 | Full 12-month window to 2018-12-31, with a 3-month buffer | Sample |
| Training sample | Issued 2012-01-01 to 2014-06-30 | Post-2012 underwriting; development data | Bins, WoE, models |
| Validation sample | Issued 2014-07-01 to 2015-12-31 | Hyperparameter tuning and challenger selection | Tuning |
| Out-of-time sample | Issued 2016-01-01 to 2017-09-30 | Unseen data for final test | Validation |
| Hyperparameter tuning | Optuna TPE, log loss on validation, 50 trials | Tuning never touches OOT | Challengers |
| LGD | 60% default, dashboard range 20% to 90%, report tests 50/60/70% | Placeholder until recovery analysis | Expected loss, cut-off table |
| Missing values | Own bin for scorecard; native NaN for trees | Missingness can be informative | Binning, models |
| Bin and cap cut-offs | Fitted on training sample only | Avoids leakage into validation | WoE, validation |
| Class imbalance | No resampling; unweighted log loss | Keeps probabilities calibrated | All models |
| Monitoring batches | Monthly issue cohorts, evaluated after a 12-month reporting lag | Mirrors production timing | Monitoring |
| PSI thresholds | Green below 0.10; amber 0.10 to 0.25; red above 0.25 | Common industry convention | Traffic lights |
| Source data limitation | Accepted loans only (no reject inference) | Public file excludes rejected applications | Scope, limitations |
