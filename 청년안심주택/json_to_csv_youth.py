import json
import pandas as pd

# ── 커트라인 CSV ────────────────────────
with open("parsed_youth_cutline.json") as f:
    data = json.load(f)

rows = []
for page, page_rows in data.items():
    for row in page_rows:
        row["source_file"] = page
        rows.append(row)

df_cut = pd.DataFrame(rows)

# ffill: 연도/차수 빈칸 채우기
df_cut["announcement_year"]  = df_cut["announcement_year"].ffill()
df_cut["announcement_phase"] = df_cut["announcement_phase"].ffill()

# 정규화
target_map = {
    "신혼부부계층": "신혼부부Ⅰ",
    "신혼Ⅰ": "신혼부부Ⅰ",
    "신혼Ⅱ": "신혼부부Ⅱ",
    "신혼부부I": "신혼부부Ⅰ",
    "신혼부부II": "신혼부부Ⅱ",
}
df_cut["supply_target"] = df_cut["supply_target"].replace(target_map)

# 주택형 숫자 추출
import re
df_cut["housing_type_num"] = df_cut["housing_type"].apply(
    lambda x: float(re.sub(r"[^0-9.]", "", str(x))[:4]) if x else None
)

# 점수 있는 행만 필터 (ML 학습용)
df_cut_ml = df_cut[df_cut["cutline_score"].notna()].copy()

df_cut.to_csv("youth_cutline_all.csv", index=False, encoding="utf-8-sig")
df_cut_ml.to_csv("youth_cutline_ml.csv", index=False, encoding="utf-8-sig")
print(f"전체: {len(df_cut)}행 | ML 학습용: {len(df_cut_ml)}행")

# 경쟁률 CSV
with open("parsed_youth_competition.json") as f:
    data2 = json.load(f)

rows2 = []
for page, page_rows in data2.items():
    for row in page_rows:
        rows2.append(row)

df_comp = pd.DataFrame(rows2)
df_comp["announcement_year"]  = df_comp["announcement_year"].ffill()
df_comp["announcement_phase"] = df_comp["announcement_phase"].ffill()

# 경쟁률 계산 (null인 경우)
mask = df_comp["competition_rate"].isna()
df_comp.loc[mask, "competition_rate"] = (
    df_comp.loc[mask, "applicant_count"] / df_comp.loc[mask, "supply_count"]
)

df_comp.to_csv("youth_competition.csv", index=False, encoding="utf-8-sig")
print(f"경쟁률 데이터: {len(df_comp)}행")