import pandas as pd
import numpy as np
import pickle
import os
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 데이터 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "hope_housing_virtual.csv"))

# 피처 선택
FEATURES = [
    "is_priority1_eligible",
    "income_percent",
    "total_asset",
    "car_value",
    "has_car",
    "is_parents_homeless",
    "is_applicant_disabled",
    "is_parents_disabled",
    "is_recipient",
    "is_single_parent_family",
    "is_income_under50",
    "subscription_count",
    "rank"
]

X = df[FEATURES]
y = df["label"]

# 학습/테스트 분리 (8:2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")

# LightGBM 학습
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# 평가
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n 모델 성능 ")
print(f"정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1 Score: {f1:.4f}")
print(f"\n 상세 리포트 ")
print(classification_report(y_test, y_pred, target_names=["불합격", "합격"]))

# 피처 중요도
print(" 피처 중요도 ")
importance = pd.Series(model.feature_importances_, index=FEATURES)
print(importance.sort_values(ascending=False))

# 모델 저장
model_path = os.path.join(BASE_DIR, "model_hope_housing.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# 피처 목록 저장 (FastAPI에서 사용)
features_path = os.path.join(BASE_DIR, "features_hope_housing.pkl")
with open(features_path, "wb") as f:
    pickle.dump(FEATURES, f)

print(f"\n모델 저장 완료: {model_path}")