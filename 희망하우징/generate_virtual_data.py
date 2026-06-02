import pandas as pd
import numpy as np
import os

np.random.seed(42)
TOTAL = 100000
RANK1_CUTLINE = 10  # 1순위 커트라인 가정값

CUTLINE_DATA = {
    ("연남공공원룸텔", "여성"): {"rank": 2, "cutline": 8,  "competition_rate": 41.5},
    ("연남공공원룸텔", "남성"): {"rank": 2, "cutline": 2,  "competition_rate": 9.0},
    ("희망하우징(정릉)", "여성"): {"rank": 2, "cutline": 8,  "competition_rate": 37.5},
    ("희망하우징(정릉)", "남성"): {"rank": 3, "cutline": 0,  "competition_rate": 5.2},
}


def determine_rank(row):
    if row["is_priority1_eligible"]:
        return 1
    if (row["income_percent"] <= 100
            and row["total_asset"] <= 337000000
            and row["car_value"] <= 45630000):
        return 2
    if (row["income_percent"] <= 100
            and row["total_asset"] <= 104000000
            and not row["has_car"]):
        return 3
    return 0


def calculate_score(row):
    score = 0
    if row["is_parents_homeless"]:
        score += 2
    if row["is_applicant_disabled"]:
        score += 2
    if row["is_parents_disabled"]:
        score += 1
    if row["subscription_count"] >= 24:
        score += 3
    elif row["subscription_count"] >= 12:
        score += 2
    elif row["subscription_count"] >= 6:
        score += 1
    if row["rank"] == 1:
        if row["is_recipient"]:
            score += 3
        if row["is_single_parent_family"]:
            score += 3
    elif row["rank"] in (2, 3):
        if row["is_income_under50"]:
            score += 3
    return score


def generate_for_complex(complex_name, gender, cutline_info):
    n = TOTAL
    # 필수조건 모두 통과한 사람만 생성 (방법 A)
    data = {
        "complex_name":            [complex_name] * n,
        "gender":                  [gender] * n,
        # 필수조건 모두 통과 고정
        "is_home_owner":           [False] * n,
        "is_married":              [False] * n,
        "school_location":         ["서울"] * n,
        "is_graduate_student":     [False] * n,
        "is_graduated":            [False] * n,
        # 순위 판단용
        "is_priority1_eligible":   np.random.choice([True, False], n, p=[0.15, 0.85]),
        "income_percent":          np.random.uniform(20, 120, n).round(1),
        "total_asset":             np.random.randint(10000000, 400000000, n),
        "car_value":               np.random.randint(0, 60000000, n),
        "has_car":                 np.random.choice([True, False], n, p=[0.3, 0.7]),
        # 가산점
        "is_parents_homeless":     np.random.choice([True, False], n, p=[0.4, 0.6]),
        "is_applicant_disabled":   np.random.choice([True, False], n, p=[0.1, 0.9]),
        "is_parents_disabled":     np.random.choice([True, False], n, p=[0.1, 0.9]),
        "is_recipient":            np.random.choice([True, False], n, p=[0.15, 0.85]),
        "is_single_parent_family": np.random.choice([True, False], n, p=[0.1, 0.90]),
        "is_income_under50":       np.random.choice([True, False], n, p=[0.3, 0.7]),
        "subscription_count":      np.random.randint(0, 37, n),
    }

    df = pd.DataFrame(data)

    # 순위 계산
    df["rank"] = df.apply(determine_rank, axis=1)
    df = df[df["rank"] > 0].reset_index(drop=True)

    # 점수 계산
    df["user_score"] = df.apply(calculate_score, axis=1)

    # 커트라인 기준으로 합격/불합격
    def get_cutline(rank):
        if rank == 1:
            return RANK1_CUTLINE
        return cutline_info["cutline"]

    df["cutline"] = df["rank"].apply(get_cutline)
    df["competition_rate"] = cutline_info["competition_rate"]
    df["label"] = (df["user_score"] >= df["cutline"]).astype(int)

    # 분포 확인
    print(f"\n[{complex_name} {gender}] 레이블 분포:")
    print(df["label"].value_counts())

    # 언더샘플링 (합격:불합격 = 2500:2500)
    df_pass = df[df["label"] == 1]
    df_fail = df[df["label"] == 0]
    sample_size = min(2500, len(df_pass), len(df_fail))

    df_balanced = pd.concat([
        df_pass.sample(sample_size, random_state=42),
        df_fail.sample(sample_size, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"균형 후: {len(df_balanced)}개 (합격 {sample_size} / 불합격 {sample_size})")
    return df_balanced


# 단지별 생성
all_dfs = []
for (complex_name, gender), cutline_info in CUTLINE_DATA.items():
    df = generate_for_complex(complex_name, gender, cutline_info)
    all_dfs.append(df)

df_final = pd.concat(all_dfs).reset_index(drop=True)

print(f"\n=== 최종 데이터 ===")
print(f"전체: {len(df_final)}개")
print(df_final["label"].value_counts())

# 저장
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hope_housing_virtual.csv")
df_final.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {output_path}")