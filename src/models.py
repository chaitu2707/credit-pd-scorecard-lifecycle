"""
Challenger models: XGBoost, LightGBM and CatBoost, tuned with Optuna on the
validation sample, then calibrated on validation and combined by simple average.
The out-of-time sample is never used for tuning or calibration.
"""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from . import config

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------- data preparation for trees ----------

def make_tree_frame(df: pd.DataFrame, features, cat_levels: dict, as_str: bool = False) -> pd.DataFrame:
    """
    Feature frame for the tree models. Categorical features use fixed levels taken
    from training, so every sample is encoded identically. Missing values stay NaN
    for the numeric features and become "missing" for the categorical ones.
    """
    X = df[features].copy()
    for f in features:
        if f in cat_levels:
            s = df[f].astype(object).where(df[f].notna(), "missing").astype(str)
            X[f] = s if as_str else pd.Categorical(s, categories=cat_levels[f])
    return X


# ---------- tuning ----------

def _run_study(objective, n_trials: int, seed: int):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


def tune_xgboost(Xtr, ytr, Xva, yva, n_trials: int = 50, seed: int = config.RANDOM_STATE):
    from xgboost import XGBClassifier

    def objective(trial):
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
            "eval_metric": "logloss",
            "early_stopping_rounds": 100,
            "enable_categorical": True,
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return log_loss(yva, model.predict_proba(Xva)[:, 1])

    return _run_study(objective, n_trials, seed)


def tune_lightgbm(Xtr, ytr, Xva, yva, n_trials: int = 50, seed: int = config.RANDOM_STATE):
    import lightgbm as lgb

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "seed": seed,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 50.0, log=True),
        }
        dtr = lgb.Dataset(Xtr, ytr)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        booster = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        p = booster.predict(Xva, num_iteration=booster.best_iteration)
        return log_loss(yva, p)

    return _run_study(objective, n_trials, seed)


def tune_catboost(Xtr, ytr, Xva, yva, cat_cols, n_trials: int = 50, seed: int = config.RANDOM_STATE):
    from catboost import CatBoostClassifier

    def objective(trial):
        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            depth=trial.suggest_int("depth", 3, 7),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            loss_function="Logloss",
            eval_metric="Logloss",
            early_stopping_rounds=100,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            cat_features=cat_cols,
        )
        model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        trial.set_user_attr("best_iteration", int(model.get_best_iteration()))
        return log_loss(yva, model.predict_proba(Xva)[:, 1])

    return _run_study(objective, n_trials, seed)


# ---------- final fit and prediction ----------

def fit_final(name: str, study, X, y, cat_cols, seed: int = config.RANDOM_STATE):
    """Refit on training data with the tuned settings and the best iteration count."""
    best = study.best_trial
    n_iter = best.user_attrs["best_iteration"] + 1
    p = dict(best.params)
    if name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(**p, n_estimators=n_iter, enable_categorical=True,
                              tree_method="hist", random_state=seed, n_jobs=-1)
        model.fit(X, y)
        return model
    if name == "lightgbm":
        import lightgbm as lgb
        params = {"objective": "binary", "verbosity": -1, "seed": seed, "bagging_freq": 1, **p}
        return lgb.train(params, lgb.Dataset(X, y), num_boost_round=n_iter)
    if name == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**p, iterations=n_iter, loss_function="Logloss",
                                   random_seed=seed, verbose=False, allow_writing_files=False,
                                   cat_features=cat_cols)
        model.fit(X, y)
        return model
    raise ValueError(name)


def predict_tree(name: str, model, X) -> np.ndarray:
    if name == "lightgbm":
        return np.asarray(model.predict(X), dtype=float)
    return model.predict_proba(X)[:, 1]


# ---------- calibration and ensemble ----------

def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def fit_platt(p_raw_validation, y_validation) -> dict:
    """Platt scaling on the validation sample: maps raw probabilities to observed rates."""
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(_logit(p_raw_validation).reshape(-1, 1), np.asarray(y_validation))
    return {"coef": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}


def apply_platt(cal: dict, p_raw) -> np.ndarray:
    z = cal["intercept"] + cal["coef"] * _logit(p_raw)
    return 1 / (1 + np.exp(-z))


def ensemble_average(prob_list) -> np.ndarray:
    """Simple average of calibrated probabilities. Each member is already on the PD scale."""
    return np.mean(np.vstack(prob_list), axis=0)
