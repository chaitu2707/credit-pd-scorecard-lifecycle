"""
End-to-end pipeline for Stages 4 to 7. Writes every table and model artefact that the
notebooks, reports and dashboard read.

Run from the repo root:
    python -m src.pipeline
Set PIPELINE_TRIALS to a small number (for example 5) for a quick test run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import models as md
from . import explain as ex
from . import monitoring as mo
from . import scorecard as sc
from . import validation as va
from .feature_selection import add_derived_features
from .metrics import brier_score, gini, ks_statistic
from .splits import assign_sample
from sklearn.metrics import roc_auc_score

SAMPLES = ["train", "validation", "oot"]


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add sample labels and derived features if missing, and drop excluded loans."""
    out = df.copy()
    if "sample" not in out.columns:
        out = assign_sample(out)
    if "credit_history_months" not in out.columns:
        out = add_derived_features(out)
    return out[out["sample"] != "excluded"].reset_index(drop=True)


def load_working(path=None) -> pd.DataFrame:
    if path is None:
        path = config.SLIM_PATH if config.SLIM_PATH.exists() else config.PROCESSED_PATH
    return prepare_frame(pd.read_parquet(path))


def build_spec(train: pd.DataFrame, features) -> dict:
    """Cut-offs are the confirmed settings. Categorical levels are kept from training only."""
    spec = {}
    for f in features:
        if f in config.CATEGORICAL_SCORECARD:
            if f in config.RARE_MERGE_EXEMPT:
                keep = sorted(train[f].fillna("missing").astype(str).unique())
            else:
                shares = train[f].fillna("missing").astype(str).value_counts(normalize=True)
                keep = sorted(shares[shares >= config.RARE_CATEGORY_SHARE].index)
            spec[f] = {"type": "categorical", "keep": [str(k) for k in keep]}
        else:
            spec[f] = {"type": "numeric", "cuts": [float(c) for c in config.SCORECARD_CUTS[f]]}
    return spec


def fit_champion(train, valid, oot, features) -> dict:
    spec = build_spec(train, features)
    btr, orders = sc.assign_bins(train, spec)
    bva, _ = sc.assign_bins(valid, spec)
    boo, _ = sc.assign_bins(oot, spec)
    y_tr, y_va = train["target"].values, valid["target"].values

    fit = sc.fit_scorecard(btr, y_tr, bva, y_va)
    pd_raw = {"train": sc.pd_from_model(fit, btr),
              "validation": sc.pd_from_model(fit, bva),
              "oot": sc.pd_from_model(fit, boo)}
    shift = va.intercept_shift(pd_raw["validation"], y_va)
    pd_cal = {k: va.apply_shift(v, shift) for k, v in pd_raw.items()}
    score = {"train": sc.score(fit, btr), "validation": sc.score(fit, bva), "oot": sc.score(fit, boo)}
    return {"features": list(features), "spec": spec, "fit": fit, "bins": {"train": btr, "validation": bva, "oot": boo},
            "pd_raw": pd_raw, "pd_cal": pd_cal, "shift": shift, "score": score}


def scorecard_json(ch: dict) -> dict:
    fit = ch["fit"]
    _, base = sc.points_table(fit)
    return {
        "features": ch["features"],
        "spec": ch["spec"],
        "maps": {f: {str(k): float(v) for k, v in m.items()} for f, m in fit["maps"].items()},
        "coef": {f: float(c) for f, c in fit["coef"].items()},
        "intercept": float(fit["intercept"]),
        "C": float(fit["C"]),
        "recalibration_shift": float(ch["shift"]),
        "factor": float(sc.FACTOR),
        "offset": float(sc.OFFSET),
        "base_points": float(base),
    }


def metric_row(model: str, sample: str, y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "model": model,
        "sample": sample,
        "gini": gini(y, p),
        "ks": ks_statistic(y, p),
        "auc": float(roc_auc_score(y, p)),
        "brier": brier_score(y, p),
        "mean_pd": float(p.mean()),
        "bad_rate": float(y.mean()),
    }


def run(df: pd.DataFrame, out_dir=None, trials=None) -> dict:
    df = prepare_frame(df)
    out_dir = Path(out_dir) if out_dir else config.ROOT
    trials = config.TUNE_TRIALS if trials is None else trials
    tables = out_dir / "reports" / "tables"
    models_dir = out_dir / "models"
    tables.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train, valid, oot = (df[df["sample"] == s].reset_index(drop=True) for s in SAMPLES)
    y = {"train": train["target"].values, "validation": valid["target"].values, "oot": oot["target"].values}
    splits = {"train": train, "validation": valid, "oot": oot}
    summary = {"trials": trials}

    # ---- sample design
    samples = pd.DataFrame([{
        "sample": s, "loans": len(d), "bad_loans": int(d["target"].sum()),
        "bad_rate": float(d["target"].mean()),
        "issue_from": str(d["issue_date"].min().date()), "issue_to": str(d["issue_date"].max().date()),
        "share_60_month": float((d["term"].astype(str).str.contains("60")).mean()),
    } for s, d in splits.items()])
    samples.to_csv(tables / "sample_summary.csv", index=False)

    # ---- Stage 4: champion scorecard (with grade) and the same model without grade
    feats = list(config.SCORECARD_FEATURES)
    feats_nog = [f for f in feats if f != "grade"]
    champ = fit_champion(train, valid, oot, feats)
    nograde = fit_champion(train, valid, oot, feats_nog)
    (models_dir / "scorecard.json").write_text(json.dumps(scorecard_json(champ), indent=1))
    (models_dir / "scorecard_no_grade.json").write_text(json.dumps(scorecard_json(nograde), indent=1))

    pts, _ = sc.points_table(champ["fit"])
    pts.to_csv(tables / "points_table.csv", index=False)
    pd.DataFrame({"feature": list(champ["fit"]["coef"].keys()),
                  "coefficient": list(champ["fit"]["coef"].values())}).to_csv(tables / "scorecard_coefficients.csv", index=False)
    summary.update({"scorecard_C": champ["fit"]["C"], "recalibration_shift": champ["shift"]})

    # ---- Stage 5: challengers on the same features
    cat_cols = [f for f in config.CATEGORICAL_SCORECARD if f in feats]
    cat_levels = {f: sorted(train[f].astype(object).where(train[f].notna(), "missing").astype(str).unique())
                  for f in cat_cols}
    frames = {
        "tree": {s: md.make_tree_frame(d, feats, cat_levels) for s, d in splits.items()},
        "catboost": {s: md.make_tree_frame(d, feats, cat_levels, as_str=True) for s, d in splits.items()},
    }
    tuned = {
        "xgboost": lambda: md.tune_xgboost(frames["tree"]["train"], y["train"], frames["tree"]["validation"], y["validation"], n_trials=trials),
        "lightgbm": lambda: md.tune_lightgbm(frames["tree"]["train"], y["train"], frames["tree"]["validation"], y["validation"], n_trials=trials),
        "catboost": lambda: md.tune_catboost(frames["catboost"]["train"], y["train"], frames["catboost"]["validation"], y["validation"], cat_cols, n_trials=trials),
    }
    raw, fitted, tune_log = {}, {}, {}
    for name, tune in tuned.items():
        study = tune()
        key = "catboost" if name == "catboost" else "tree"
        model = md.fit_final(name, study, frames[key]["train"], y["train"], cat_cols)
        fitted[name] = model
        raw[name] = {s: md.predict_tree(name, model, frames[key][s]) for s in SAMPLES}
        tune_log[name] = {"best_params": study.best_trial.params,
                          "best_iteration": study.best_trial.user_attrs["best_iteration"],
                          "best_validation_logloss": float(study.best_value)}
    (models_dir / "tuning.json").write_text(json.dumps(tune_log, indent=1, default=str))

    cal = {n: md.fit_platt(raw[n]["validation"], y["validation"]) for n in raw}
    cal_p = {n: {s: md.apply_platt(cal[n], raw[n][s]) for s in SAMPLES} for n in raw}
    cal_p["ensemble"] = {s: md.ensemble_average([cal_p[n][s] for n in raw]) for s in SAMPLES}
    (models_dir / "calibration.json").write_text(json.dumps(cal, indent=1))
    fitted["xgboost"].save_model(str(models_dir / "xgboost.json"))
    fitted["lightgbm"].save_model(str(models_dir / "lightgbm.txt"))
    fitted["catboost"].save_model(str(models_dir / "catboost.cbm"))

    # ---- Stage 6: comparison, deciles, cut-offs, benchmark, explanations
    grade_pd = {s: va.grade_benchmark_pd(train["grade"], y["train"], d["grade"]) for s, d in splits.items()}
    candidates = {
        "scorecard_with_grade_raw": champ["pd_raw"],
        "scorecard_with_grade_recalibrated": champ["pd_cal"],
        "scorecard_no_grade_recalibrated": nograde["pd_cal"],
        "xgboost_calibrated": cal_p["xgboost"],
        "lightgbm_calibrated": cal_p["lightgbm"],
        "catboost_calibrated": cal_p["catboost"],
        "ensemble_calibrated": cal_p["ensemble"],
        "grade_benchmark": grade_pd,
    }
    rows = [metric_row(m, s, y[s], p[s]) for m, p in candidates.items() for s in SAMPLES]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(tables / "model_comparison.csv", index=False)

    va.decile_table(y["oot"], champ["pd_cal"]["oot"]).to_csv(tables / "decile_oot_champion.csv", index=False)
    va.decile_table(y["oot"], cal_p["ensemble"]["oot"]).to_csv(tables / "decile_oot_ensemble.csv", index=False)

    cutoffs = va.cutoff_table(champ["pd_cal"]["oot"], y["oot"], oot["loan_amnt"].values,
                              config.CUTOFF_GRID, config.LGD_GRID)
    cutoffs.to_csv(tables / "cutoff_table.csv", index=False)

    shap_imp = ex.shap_importance(fitted["lightgbm"], frames["tree"]["oot"])
    shap_imp.to_csv(tables / "shap_importance_lightgbm.csv", index=False)

    # challenger verdict: chosen on validation, judged on out-of-time
    chall = {"xgboost": cal_p["xgboost"], "lightgbm": cal_p["lightgbm"],
             "catboost": cal_p["catboost"], "ensemble": cal_p["ensemble"]}
    val_gini = {n: gini(y["validation"], p["validation"]) for n, p in chall.items()}
    best = max(val_gini, key=val_gini.get)
    gain = gini(y["oot"], chall[best]["oot"]) - gini(y["oot"], champ["pd_cal"]["oot"])
    gap_champ = float(champ["pd_cal"]["oot"].mean() - y["oot"].mean())
    gap_best = float(chall[best]["oot"].mean() - y["oot"].mean())
    replace = bool(gain >= config.CHAMPION_GAIN_MIN and abs(gap_best) <= abs(gap_champ))
    verdict = {
        "champion": "scorecard_with_grade_recalibrated",
        "best_challenger_by_validation_gini": best,
        "validation_gini": {k: float(v) for k, v in val_gini.items()},
        "oot_gini_gain_over_champion": float(gain),
        "oot_calibration_gap_champion": gap_champ,
        "oot_calibration_gap_challenger": gap_best,
        "replace_champion": replace,
        "rule": f"replace only if out-of-time Gini gain >= {config.CHAMPION_GAIN_MIN} and calibration gap is no larger",
    }
    (models_dir / "champion_challenger.json").write_text(json.dumps(verdict, indent=1))

    oot_pred = pd.DataFrame({
        "issue_month": oot["issue_date"].dt.to_period("M").astype(str).values,
        "target": y["oot"],
        "loan_amnt": oot["loan_amnt"].values,
        "grade": oot["grade"].astype(str).values,
        "pd_champion": champ["pd_cal"]["oot"],
        "pd_ensemble": cal_p["ensemble"]["oot"],
        "pd_grade_benchmark": grade_pd["oot"],
    })
    oot_pred.to_parquet(tables / "oot_predictions.parquet", index=False)

    # ---- Stage 7: monitoring of monthly out-of-time batches under the frozen champion
    dev_gini = gini(y["validation"], champ["pd_cal"]["validation"])
    batches, feat_psi = mo.monitor(
        oot, champ["pd_cal"]["oot"], champ["score"]["oot"], champ["score"]["train"],
        train[[f for f in feats if f not in config.CATEGORICAL_SCORECARD]],
        [f for f in feats if f not in config.CATEGORICAL_SCORECARD],
        dev_gini,
    )
    batches.to_csv(tables / "monitoring_batches.csv", index=False)
    feat_psi.to_csv(tables / "monitoring_features.csv", index=False)

    summary.update({
        "champion_gini_oot": gini(y["oot"], champ["pd_cal"]["oot"]),
        "champion_gini_validation": dev_gini,
        "challenger_replace": replace,
        "retrain_review_batches": int(batches["retrain_review"].sum()),
        "red_batches": int((batches["overall"] == "red").sum()),
        "batches_total": int(len(batches)),
    })
    (models_dir / "run_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return {"comparison": comparison, "verdict": verdict, "batches": batches, "summary": summary}


def main() -> None:
    df = load_working()
    result = run(df)
    print(result["comparison"].round(4).to_string(index=False))
    print("\nChampion-challenger:", json.dumps(result["verdict"], indent=1, default=float))


if __name__ == "__main__":
    main()
