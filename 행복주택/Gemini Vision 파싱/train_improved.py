import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from lightgbm import LGBMRegressor, LGBMClassifier

df = pd.read_csv("cutline_gemini.csv", encoding="utf-8-sig")
df = df[df["cutline_score"].notna()].copy()

# 정규화
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

df["supply_target"] = df["supply_target"].apply(normalize_target)
df["supply_type"]   = df["supply_type"].apply(normalize_supply)
df["housing_type_num"] = df["housing_type"].astype(str).str.extract(r"(\d+)").astype(float)
df["phase_num"] = df["announcement_phase"].astype(str).str.extract(r"(\d+)").astype(float)

# ── Target Encoding (KFold로 데이터 누수 방지) ──────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
df["district_target_mean"] = np.nan
df["district_year_mean"]   = np.nan

for train_idx, val_idx in kf.split(df):
    train_df = df.iloc[train_idx]

    # 자치구 + 신청자격 평균
    d_t_mean = train_df.groupby(["district", "supply_target"])["cutline_score"].mean()
    df.iloc[val_idx, df.columns.get_loc("district_target_mean")] = \
        df.iloc[val_idx].apply(
            lambda r: d_t_mean.get((r["district"], r["supply_target"]), np.nan), axis=1
        ).values

    # 자치구 + 연도 평균
    d_y_mean = train_df.groupby(["district", "announcement_year"])["cutline_score"].mean()
    df.iloc[val_idx, df.columns.get_loc("district_year_mean")] = \
        df.iloc[val_idx].apply(
            lambda r: d_y_mean.get((r["district"], r["announcement_year"]), np.nan), axis=1
        ).values

# 빈칸은 전체 평균으로 채우기
df["district_target_mean"].fillna(df["cutline_score"].mean(), inplace=True)
df["district_year_mean"].fillna(df["cutline_score"].mean(), inplace=True)

# 점수 구간 분류 추가
def score_to_class(s):
    if s <= 4:  return 0  # 저경쟁
    if s <= 6:  return 1  # 중경쟁
    return 2               # 고경쟁

df["score_class"] = df["cutline_score"].apply(score_to_class)

# 인코딩
cat_cols = ["district", "supply_target", "supply_type", "review_stage"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ── 피처 정의 ──────────────────────────────────────────
features_base = [
    "announcement_year", "phase_num", "district",
    "housing_type_num", "supply_target", "supply_type",
    "cutline_rank", "review_stage"
]
features_improved = features_base + ["district_target_mean", "district_year_mean"]

df_ml = df[features_improved + ["cutline_score", "score_class"]].dropna()

from sklearn.model_selection import train_test_split
X = df_ml[features_improved]
y_reg = df_ml["cutline_score"]
y_cls = df_ml["score_class"]

X_train, X_test, y_tr, y_te, y_cls_tr, y_cls_te = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42
)

# ── 1. 회귀 모델 비교 ──────────────────────────────────
print("=== 회귀 R2 비교 (Target Encoding 추가 후) ===")
lgbm_reg = LGBMRegressor(random_state=42, verbose=-1)
lgbm_reg.fit(X_train, y_tr)
pred = lgbm_reg.predict(X_test)
print(f"전체 LightGBM  R2: {r2_score(y_te, pred):.4f}  MAE: {mean_absolute_error(y_te, pred):.4f}")

# ── 2. 고령자 전용 모델 ────────────────────────────────
print("\n=== 고령자 전용 모델 (Target Encoding 추가) ===")
mask_e = encoders["supply_target"].inverse_transform(df_ml["supply_target"]) == "고령자"
df_e = df_ml[mask_e]
X_e = df_e[features_improved]
y_e = df_e["cutline_score"]

X_etr, X_ete, y_etr, y_ete = train_test_split(X_e, y_e, test_size=0.2, random_state=42)
lgbm_e = LGBMRegressor(random_state=42, verbose=-1)
lgbm_e.fit(X_etr, y_etr)
pred_e = lgbm_e.predict(X_ete)
print(f"고령자 R2:  {r2_score(y_ete, pred_e):.4f}  MAE: {mean_absolute_error(y_ete, pred_e):.4f}")

# ── 3. 분류 모델 (구간 예측) ───────────────────────────
print("\n=== 구간 분류 모델 ===")
print("0: 저경쟁(1~4점)  1: 중경쟁(5~6점)  2: 고경쟁(7~9점)")
lgbm_cls = LGBMClassifier(random_state=42, verbose=-1)
lgbm_cls.fit(X_train, y_cls_tr)
pred_cls = lgbm_cls.predict(X_test)
print(f"전체 Accuracy: {accuracy_score(y_cls_te, pred_cls):.4f}")

from sklearn.metrics import classification_report
print(classification_report(y_cls_te, pred_cls,
    target_names=["저경쟁(1~4)", "중경쟁(5~6)", "고경쟁(7~9)"]))

# ── 4. 하이브리드 최종 평가 ────────────────────────────
print("=== 하이브리드 최종 R2 ===")
df_test_all = X_test.copy()
df_test_all["실제"] = y_te.values
df_test_all["supply_target_orig"] = encoders["supply_target"].inverse_transform(
    df_test_all["supply_target"]
)

def hybrid_predict(row, model):
    t = row["supply_target_orig"]
    if t in ["청년", "신혼부부"]:         return 6.0
    if t in ["대학생", "주거급여수급자"]:  return 3.0
    return model.predict(pd.DataFrame([row[features_improved]]))[0]

df_test_all["예측"] = df_test_all.apply(lambda r: hybrid_predict(r, lgbm_e), axis=1)
print(f"하이브리드 R2:  {r2_score(df_test_all['실제'], df_test_all['예측']):.4f}")
print(f"하이브리드 MAE: {mean_absolute_error(df_test_all['실제'], df_test_all['예측']):.4f}")