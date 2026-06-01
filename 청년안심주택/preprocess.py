# preprocess.py 전체 교체

import pandas as pd
import json
import re

df = pd.read_csv("youth_cutline_raw.csv")

# ── 1. 추첨 제거 ─────────────────────────────────────────
df = df[df["cutline_score"].notna()].copy()
print(f"학습 데이터: {len(df)}행")
print(df["supply_target"].value_counts())

# ── 2. 주택형 숫자 추출 ──────────────────────────────────
def extract_num(x):
    if not x: return None
    m = re.search(r"[\d.]+", str(x))
    return float(m.group()) if m else None

df["housing_type_num"] = df["housing_type"].apply(extract_num)

# ── 3. 공고 차수 숫자화 ──────────────────────────────────
df["phase_num"] = df["announcement_phase"].str.extract(r"(\d+)").astype(float)

# ── 4. 계절 피처 (차수 기반 근사) ────────────────────────
def get_season(phase):
    if pd.isna(phase): return 1
    p = int(phase)
    if p == 1: return 0   # 봄 (1~4월)
    elif p == 2: return 1  # 여름 (7~8월)
    else: return 2         # 가을/겨울 (10~12월)

df["season"] = df["phase_num"].apply(get_season)

# ── 5. 카테고리 인코딩 ───────────────────────────────────
cat_map = {}
for col in ["district", "supply_target", "applicant_rank", "review_stage"]:
    cat = pd.Categorical(df[col])
    df[col + "_code"] = cat.codes
    cat_map[col] = dict(enumerate(cat.categories))

# ── 6. Target Encoding ────────────────────────────────────
district_mean = df.groupby("district")["cutline_score"].mean()
df["district_mean"] = df["district"].map(district_mean)

# complex_mean: Leave-One-Out 방식으로 데이터 누수 방지
loo_means = []
for idx, row in df.iterrows():
    others = df[(df["complex_name"] == row["complex_name"]) & (df.index != idx)]
    if len(others) > 0:
        loo_means.append(others["cutline_score"].mean())
    else:
        loo_means.append(district_mean.get(row["district"], df["cutline_score"].mean()))

df["complex_mean_loo"] = loo_means

# ── 7. 경쟁률 + 공급 세대수 피처 ─────────────────────────
comp = pd.read_csv("youth_competition_raw.csv")

# 경쟁률
comp_rate = comp.groupby(
    ["district", "supply_target", "announcement_year", "announcement_phase"]
)["competition_rate"].mean().reset_index()
comp_rate.columns = ["district", "supply_target", "announcement_year",
                    "announcement_phase", "avg_competition_rate"]

# 공급 세대수
comp_supply = comp.groupby(
    ["district", "supply_target", "announcement_year", "announcement_phase"]
)["supply_count"].mean().reset_index()
comp_supply.columns = ["district", "supply_target", "announcement_year",
                    "announcement_phase", "avg_supply_count"]

df = df.merge(comp_rate,   on=["district", "supply_target",
            "announcement_year", "announcement_phase"], how="left")
df = df.merge(comp_supply, on=["district", "supply_target",
            "announcement_year", "announcement_phase"], how="left")

# fallback: 자치구+신청자격 평균
rate_fb   = comp.groupby(["district", "supply_target"])["competition_rate"].mean()
supply_fb = comp.groupby(["district", "supply_target"])["supply_count"].mean()

df["avg_competition_rate"] = df["avg_competition_rate"].fillna(
    df.apply(lambda r: rate_fb.get((r["district"], r["supply_target"]), 50), axis=1))
df["avg_supply_count"] = df["avg_supply_count"].fillna(
    df.apply(lambda r: supply_fb.get((r["district"], r["supply_target"]), 10), axis=1))

print(f"\n경쟁률 커버리지:  {df['avg_competition_rate'].notna().sum()}/{len(df)}")
print(f"공급세대수 커버리지: {df['avg_supply_count'].notna().sum()}/{len(df)}")

# ── 8. 경쟁 구간 분류 ────────────────────────────────────
def classify(s):
    if s <= 4:  return 0
    elif s <= 7: return 1
    else:        return 2

df["score_class"] = df["cutline_score"].apply(classify)

labels = {0: "저경쟁(0~4점)", 1: "중경쟁(5~7점)", 2: "고경쟁(8점+)"}
print(f"\n=== 경쟁 구간 분포 ===")
for k, v in df["score_class"].value_counts().sort_index().items():
    print(f"{labels[k]}: {v}행")

# ── 9. 저장 ─────────────────────────────────────────────
df.to_csv("youth_cutline_ml.csv", index=False, encoding="utf-8-sig")

enc_data = {
    "district_mean":  district_mean.to_dict(),
    "rate_fallback":  {str(k): v for k, v in rate_fb.items()},
    "supply_fallback": {str(k): v for k, v in supply_fb.items()},
    "cat_map":        cat_map,
}
with open("encodings.json", "w", encoding="utf-8") as f:
    json.dump(enc_data, f, ensure_ascii=False, indent=2)

print("\n저장 완료: youth_cutline_ml.csv / encodings.json")