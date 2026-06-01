import pandas as pd
import pickle, numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error

df = pd.read_csv("youth_cutline_ml.csv")

FEATURES = [
    "announcement_year", "phase_num", "season",
    "district_code", "supply_target_code",
    "applicant_rank_code", "review_stage_code",
    "housing_type_num",
    "district_mean", "complex_mean_loo",
    "avg_competition_rate", "avg_supply_count",
]
CAT_FEATURES = [
    "district_code", "supply_target_code",
    "applicant_rank_code", "review_stage_code", "season",
]

X     = df[FEATURES].fillna(-1)
y_cls = df["score_class"]
y_reg = df["cutline_score"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kf  = KFold(n_splits=5, shuffle=True, random_state=42)

# ── CatBoost 분류 ────────────────────────────────────────
def obj_cat_cls(trial):
    p = {
        "iterations":        trial.suggest_int("iterations", 100, 300),
        "depth":             trial.suggest_int("depth", 4, 6),
        "learning_rate":     trial.suggest_float("learning_rate", 0.05, 0.3),
        "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1, 10),
    }
    scores = []
    for tr_idx, te_idx in skf.split(X, y_cls):
        m = CatBoostClassifier(**p, cat_features=CAT_FEATURES,
                            verbose=0, random_seed=42)
        m.fit(X.iloc[tr_idx], y_cls.iloc[tr_idx])
        scores.append(accuracy_score(y_cls.iloc[te_idx], m.predict(X.iloc[te_idx])))
    return np.mean(scores)

print("CatBoost 분류 튜닝 중")
study_cat_cls = optuna.create_study(direction="maximize")
study_cat_cls.optimize(obj_cat_cls, n_trials=30, show_progress_bar=True)

cat_cls = CatBoostClassifier(**study_cat_cls.best_params,
                            cat_features=CAT_FEATURES, verbose=0, random_seed=42)
cat_cls.fit(X, y_cls)

# CV 정확도
cv_scores = []
for tr_idx, te_idx in skf.split(X, y_cls):
    m = CatBoostClassifier(**study_cat_cls.best_params,
                        cat_features=CAT_FEATURES, verbose=0, random_seed=42)
    m.fit(X.iloc[tr_idx], y_cls.iloc[tr_idx])
    cv_scores.append(accuracy_score(y_cls.iloc[te_idx], m.predict(X.iloc[te_idx])))
print(f"CatBoost 분류 CV 정확도: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# ── CatBoost 회귀 ────────────────────────────────────────
def obj_cat_reg(trial):
    p = {
        "iterations":        trial.suggest_int("iterations", 100, 300),
        "depth":             trial.suggest_int("depth", 4, 6),
        "learning_rate":     trial.suggest_float("learning_rate", 0.05, 0.3),
        "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1, 10),
    }
    scores = []
    for tr_idx, te_idx in kf.split(X, y_reg):
        m = CatBoostRegressor(**p, cat_features=CAT_FEATURES,
                            verbose=0, random_seed=42)
        m.fit(X.iloc[tr_idx], y_reg.iloc[tr_idx])
        scores.append(mean_absolute_error(y_reg.iloc[te_idx], m.predict(X.iloc[te_idx])))
    return np.mean(scores)

print("\nCatBoost 회귀 튜닝 중 (100 trials)...")
study_cat_reg = optuna.create_study(direction="minimize")
study_cat_reg.optimize(obj_cat_reg, n_trials=30, show_progress_bar=True)

cat_reg = CatBoostRegressor(**study_cat_reg.best_params,
                            cat_features=CAT_FEATURES, verbose=0, random_seed=42)
cat_reg.fit(X, y_reg)

mae_scores = []
for tr_idx, te_idx in kf.split(X, y_reg):
    m = CatBoostRegressor(**study_cat_reg.best_params,
                        cat_features=CAT_FEATURES, verbose=0, random_seed=42)
    m.fit(X.iloc[tr_idx], y_reg.iloc[tr_idx])
    mae_scores.append(mean_absolute_error(y_reg.iloc[te_idx], m.predict(X.iloc[te_idx])))
print(f"CatBoost 회귀 CV MAE: {np.mean(mae_scores):.4f} (±{np.std(mae_scores):.4f})점")

# ── 피처 중요도 ─────────────────────────────────────────
print("\n=== 피처 중요도 (CatBoost 분류) ===")
for feat, imp in sorted(
    zip(FEATURES, cat_cls.feature_importances_),
    key=lambda x: -x[1]
):
    print(f"  {feat}: {imp:.1f}")

# ── 저장 ────────────────────────────────────────────────
with open("model_cls.pkl", "wb") as f: pickle.dump(cat_cls, f)
with open("model_reg.pkl", "wb") as f: pickle.dump(cat_reg, f)
print("\n저장 완료: model_cls.pkl / model_reg.pkl")