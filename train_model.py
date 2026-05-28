import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor

df = pd.read_csv("cutline_gemini.csv", encoding="utf-8-sig")

# 점수 없는 행 제거
df = df[df["cutline_score"].notna()].copy()

# 신청자격 정규화
def normalize_target(x):
    x = str(x).strip()
    if "대학생" in x:      return "대학생"
    if "주거급여" in x:    return "주거급여수급자"
    if "고령" in x:       return "고령자"
    if "신혼부부" in x:    return "신혼부부"
    if "청년" in x:       return "청년"
    if "일반" in x:       return "일반가구"
    return x

df["supply_target"] = df["supply_target"].apply(normalize_target)

# 공급구분 정규화
def normalize_supply(x):
    x = str(x).strip()
    if "우선" in x: return "우선"
    if "일반" in x: return "일반"
    return "기타"

df["supply_type"] = df["supply_type"].apply(normalize_supply)

# 주택형 숫자 추출 
df["housing_type_num"] = df["housing_type"].astype(str).str.extract(r"(\d+)").astype(float)

# 공고차수 숫자 추출 
df["phase_num"] = df["announcement_phase"].astype(str).str.extract(r"(\d+)").astype(float)

# 피처 선택 및 인코딩 
features = [
    "announcement_year",
    "phase_num",
    "district",
    "housing_type_num",
    "supply_target",
    "supply_type",
    "cutline_rank",
    "review_stage",
]

df_ml = df[features + ["cutline_score"]].dropna()

cat_cols = ["district", "supply_target", "supply_type", "review_stage"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    encoders[col] = le

X = df_ml[features]
y = df_ml["cutline_score"]

print(f"학습 데이터: {len(df_ml)}행")
print(f"점수 분포:\n{y.value_counts().sort_index()}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 학습 및 평가
models = {
    "LinearRegression":      LinearRegression(),
    "RandomForest":          RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting":      GradientBoostingRegressor(random_state=42),
    "LightGBM":              LGBMRegressor(random_state=42, verbose=-1),
}

print("=== 모델 성능 비교 ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2  = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"{name:25s} R2: {r2:.4f}  MAE: {mae:.4f}")

# 신청자격별 R2 분석
print("\n=== 신청자격별 분석 (LightGBM) ===")
lgbm = LGBMRegressor(random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)

df_test = X_test.copy()
df_test["실제"] = y_test.values
df_test["예측"] = lgbm.predict(X_test)
df_test["supply_target_orig"] = encoders["supply_target"].inverse_transform(df_test["supply_target"])

for cat in df_test["supply_target_orig"].unique():
    sub = df_test[df_test["supply_target_orig"] == cat]
    if len(sub) < 10:
        continue
    r2  = r2_score(sub["실제"], sub["예측"])
    mae = mean_absolute_error(sub["실제"], sub["예측"])
    print(f"{cat:12s} n={len(sub):4d}  R2: {r2:.4f}  MAE: {mae:.4f}")

# 고령자 전용 모델
print("\n=== 고령자 전용 모델 ===")
df_elderly = df_ml[
    encoders["supply_target"].inverse_transform(df_ml["supply_target"]) == "고령자"
].copy()

print(f"고령자 데이터: {len(df_elderly)}행")
print(f"점수 분포:\n{df_elderly['cutline_score'].value_counts().sort_index()}")

X_e = df_elderly[features]
y_e = df_elderly["cutline_score"]

X_tr, X_te, y_tr, y_te = train_test_split(X_e, y_e, test_size=0.2, random_state=42)

lgbm_e = LGBMRegressor(random_state=42, verbose=-1)
lgbm_e.fit(X_tr, y_tr)
pred_e = lgbm_e.predict(X_te)

print(f"\nR2:  {r2_score(y_te, pred_e):.4f}")
print(f"MAE: {mean_absolute_error(y_te, pred_e):.4f}")

fi = pd.Series(lgbm_e.feature_importances_, index=features).sort_values(ascending=False)
print(f"\nFeature Importance (고령자):\n{fi}")

# 하이브리드 전략 평가
print("\n=== 하이브리드 전략 평가 ===")
print("규칙 기반:")
print("  청년, 신혼부부 → 6점 예측")
print("  대학생, 주거급여수급자 → 3점 예측")
print("  고령자 → LightGBM 예측")

df_test_all = X_test.copy()
df_test_all["실제"] = y_test.values
df_test_all["supply_target_orig"] = encoders["supply_target"].inverse_transform(
    df_test_all["supply_target"]
)

def hybrid_predict(row, model):
    t = row["supply_target_orig"]
    if t in ["청년", "신혼부부"]:      return 6.0
    if t in ["대학생", "주거급여수급자"]: return 3.0
    return model.predict(pd.DataFrame([row[features]]))[0]

df_test_all["하이브리드예측"] = df_test_all.apply(
    lambda r: hybrid_predict(r, lgbm_e), axis=1
)

r2_hybrid  = r2_score(df_test_all["실제"], df_test_all["하이브리드예측"])
mae_hybrid = mean_absolute_error(df_test_all["실제"], df_test_all["하이브리드예측"])
print(f"\n하이브리드 R2:  {r2_hybrid:.4f}")
print(f"하이브리드 MAE: {mae_hybrid:.4f}")