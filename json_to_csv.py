import json
import pandas as pd

def json_to_df(json_path, stage):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for page_rows in data.values():
        for row in page_rows:
            row["review_stage"] = stage
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_doc   = json_to_df("parsed_document.json", stage="서류")
    df_final = json_to_df("parsed_final.json",    stage="최종")

    df = pd.concat([df_doc, df_final], ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)

    # 연도/차수 빈칸을 앞 값으로 채우기
    df["announcement_year"]  = df["announcement_year"].ffill()
    df["announcement_phase"] = df["announcement_phase"].ffill()

    print(f"전체 행 수: {len(df)}")
    print(f"\n연도 분포:")
    print(df["announcement_year"].value_counts().sort_index())
    print(f"\n신청자격 분포:")
    print(df["supply_target"].value_counts())
    print(f"\nreview_stage 분포:")
    print(df["review_stage"].value_counts())
    print(f"\n빈칸 확인:")
    print(df[["announcement_year", "announcement_phase", "complex_name", "cutline_score"]].isna().sum())

    df.to_csv("cutline_gemini.csv", index=False, encoding="utf-8-sig")
    print("\n저장 완료: cutline_gemini.csv")