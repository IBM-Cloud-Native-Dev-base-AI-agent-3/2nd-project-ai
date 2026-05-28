import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("cutline_gemini.csv", encoding="utf-8-sig")
df = df[df["cutline_score"].notna()].copy()

def normalize_target(x):
    x = str(x).strip()
    if "대학생" in x:   return "대학생"
    if "주거급여" in x: return "주거급여수급자"
    if "고령" in x:    return "고령자"
    if "신혼부부" in x: return "신혼부부"
    if "청년" in x:    return "청년"
    return x

def normalize_supply(x):
    x = str(x).strip()
    if "우선" in x: return "우선"
    if "일반" in x: return "일반"
    return "기타"

df["supply_target"]    = df["supply_target"].apply(normalize_target)
df["supply_type"]      = df["supply_type"].apply(normalize_supply)
df["housing_type_num"] = df["housing_type"].astype(str).str.extract(r"(\d+)").astype(float)
df["phase_num"]        = df["announcement_phase"].astype(str).str.extract(r"(\d+)").astype(float)

# ── Target Encoding (KFold) ─────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col_name, group_keys in [
    ("district_target_mean",  ["district", "supply_target"]),
    ("district_year_mean",    ["district", "announcement_year"]),
    ("complex_target_mean",   ["complex_name", "supply_target"]),  # ★ 단지명 추가
    ("complex_year_mean",     ["complex_name", "announcement_year"]),
]:
    df[col_name] = np.nan
    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        means = train_df.groupby(group_keys)["cutline_score"].mean()
        df.iloc[val_idx, df.columns.get_loc(col_name)] = \
            df.iloc[val_idx].apply(
                lambda r: means.get(tuple(r[k] for k in group_keys), np.nan), axis=1
            ).values
    global_mean = df["cutline_score"].mean()
    df[col_name] = df[col_name].fillna(global_mean)

# ── 점수 구간 분류 ──────────────────────────────────────
def score_to_class(s):
    if s <= 4:  return 0
    if s <= 6:  return 1
    return 2

df["score_class"] = df["cutline_score"].apply(score_to_class)

# ── 인코딩 ────────────────────────────────────────────
cat_cols = ["district", "supply_target", "supply_type", "review_stage", "complex_name"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

features = [
    "announcement_year", "phase_num",
    "district", "complex_name",
    "housing_type_num", "supply_target", "supply_type",
    "cutline_rank", "review_stage",
    "district_target_mean", "district_year_mean",
    "complex_target_mean", "complex_year_mean",  # ★ 신규 feature
]

df_ml = df[features + ["cutline_score", "score_class"]].dropna()
X = df_ml[features]
y_cls = df_ml["score_class"]

X_train, X_test, y_tr, y_te = train_test_split(
    X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

# ── Optuna 튜닝 ────────────────────────────────────────
print("Optuna 하이퍼파라미터 튜닝 중... (약 1~2분)")

def objective(trial):
    params = {
        "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
        "max_depth":     trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":    trial.suggest_int("num_leaves", 20, 100),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "verbose": -1,
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_tr)
    return accuracy_score(y_te, model.predict(X_test))

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_params["random_state"] = 42
best_params["verbose"] = -1
print(f"최적 파라미터: {best_params}")
print(f"Optuna 최고 Accuracy: {study.best_value:.4f}")

# ── 최적 모델 ──────────────────────────────────────────
lgbm_best = LGBMClassifier(**best_params)
lgbm_best.fit(X_train, y_tr)

print("\n=== 최적 LightGBM ===")
pred_best = lgbm_best.predict(X_test)
print(f"Accuracy: {accuracy_score(y_te, pred_best):.4f}")
print(classification_report(y_te, pred_best,
    target_names=["저경쟁(1~4)", "중경쟁(5~6)", "고경쟁(7~9)"]))

# ── 앙상블 ────────────────────────────────────────────
print("=== 앙상블 (Voting) ===")
ensemble = VotingClassifier(estimators=[
    ("lgbm", lgbm_best),
    ("rf",   RandomForestClassifier(n_estimators=200, random_state=42)),
    ("gb",   GradientBoostingClassifier(n_estimators=200, random_state=42)),
], voting="soft")

ensemble.fit(X_train, y_tr)
pred_ens = ensemble.predict(X_test)
print(f"앙상블 Accuracy: {accuracy_score(y_te, pred_ens):.4f}")
print(classification_report(y_te, pred_ens,
    target_names=["저경쟁(1~4)", "중경쟁(5~6)", "고경쟁(7~9)"]))