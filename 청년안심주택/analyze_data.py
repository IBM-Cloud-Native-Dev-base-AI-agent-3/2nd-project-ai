import pandas as pd

df = pd.read_csv("youth_cutline_raw.csv")

print(f"전체 행수: {len(df)}")
print(f"\n=== 연도별 ===")
print(df["announcement_year"].value_counts().sort_index())
print(f"\n=== 신청자격별 ===")
print(df["supply_target"].value_counts())
print(f"\n=== 심사단계별 ===")
print(df["review_stage"].value_counts())
print(f"\n=== 점수 분포 ===")
print(df["cutline_score"].describe())
print(f"\n=== 자치구별 평균 커트라인 Top10 ===")
print(df.groupby("district")["cutline_score"].mean().sort_values(ascending=False).head(10))