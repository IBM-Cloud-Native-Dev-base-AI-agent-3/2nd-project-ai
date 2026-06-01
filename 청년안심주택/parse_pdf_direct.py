import pdfplumber
import pandas as pd
import re
import os


def normalize_target(t):
    if not t:
        return None
    t = str(t).strip()

    if any(x in t for x in ["팀형", "셰어형(팀)"]):
        return "청년(팀형)"
    if "청년" in t:
        return "청년"
    if any(x in t for x in ["신혼Ⅱ", "신혼II"]):
        return "신혼Ⅱ"
    if any(x in t for x in ["신혼Ⅰ", "신혼I", "신혼부부"]):
        return "신혼Ⅰ"
    return t


def parse_cutline_pdf(pdf_path):
    rows = []
    fname = os.path.basename(pdf_path)

    year_match  = re.search(r"(\d{4})년", fname)
    phase_match = re.search(r"(\d+)차", fname)
    year  = int(year_match.group(1))  if year_match  else None
    phase = f"{phase_match.group(1)}차" if phase_match else None

    if "서류심사" in fname:
        review_stage = "서류"
    elif "최종당첨자" in fname:
        review_stage = "최종"
    elif "당첨자" in fname:
        review_stage = "최종"
    else:
        review_stage = "서류"

    with pdfplumber.open(pdf_path) as pdf:
        current_complex  = None
        current_district = None
        current_housing  = None  # 팀형 행에서 이전 주택형 참조용

        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # 헤더/공고 행 스킵
                if any(kw in line for kw in [
                    "단지명", "공고", "순위 →", "서류심사대상자", "당첨자 커트라인",
                    "가점사항", "무작위 추첨", "서류심사대상자 비율"
                ]):
                    continue

                # 자치구 추출 (단지명 행)
                district_match = re.search(r"\((.{2,3}구)", line)
                if district_match:
                    current_district = district_match.group(1)
                    complex_part = line.split("(")[0].strip()
                    if complex_part:
                        current_complex = complex_part
                    continue

                # 일반 행
                # "16 청년 1순위 가점합계 8점"
                # "16 청년 1순위 총점 8점"
                data_match = re.match(
                    r"^([\d\w./]+)\s+"
                    r"(청년[\w()남여/팀형]*|신혼[\w부부ⅠⅡI()]*)\s+"
                    r"(\d순위)\s+"
                    r"(?:가점합계|총점)\s+(\d+)점",
                    line
                )
                if data_match:
                    current_housing = data_match.group(1)
                    rows.append({
                        "announcement_year":  year,
                        "announcement_phase": phase,
                        "review_stage":       review_stage,
                        "complex_name":       current_complex,
                        "district":           current_district,
                        "housing_type":       data_match.group(1),
                        "supply_target":      normalize_target(data_match.group(2)),
                        "applicant_rank":     data_match.group(3),
                        "cutline_score":      int(data_match.group(4)),
                    })
                    continue

                # 셰어형 행
                # "37B 셰어형 청년(남) 2순위 총점 5점"
                share_match = re.match(
                    r"^([\d\w./]+)\s+셰어형\s+"
                    r"(청년[\w()남여/팀형]*)\s+"
                    r"(\d순위)\s+"
                    r"(?:가점합계|총점)\s+(\d+)점",
                    line
                )
                if share_match:
                    current_housing = share_match.group(1)
                    rows.append({
                        "announcement_year":  year,
                        "announcement_phase": phase,
                        "review_stage":       review_stage,
                        "complex_name":       current_complex,
                        "district":           current_district,
                        "housing_type":       share_match.group(1),
                        "supply_target":      normalize_target(share_match.group(2)),
                        "applicant_rank":     share_match.group(3),
                        "cutline_score":      int(share_match.group(4)),
                    })
                    continue

                # 팀형 단독 행
                # "청년(팀형) 1순위 총점 10점"  (주택형이 이전 행에 있는 경우)
                team_match = re.match(
                    r"^(청년\(팀형\)|셰어형\(팀\))\s+"
                    r"(\d순위)\s+"
                    r"(?:가점합계|총점)\s+(\d+)점",
                    line
                )
                if team_match:
                    rows.append({
                        "announcement_year":  year,
                        "announcement_phase": phase,
                        "review_stage":       review_stage,
                        "complex_name":       current_complex,
                        "district":           current_district,
                        "housing_type":       current_housing,  # 이전 주택형 사용
                        "supply_target":      normalize_target(team_match.group(1)),
                        "applicant_rank":     team_match.group(2),
                        "cutline_score":      int(team_match.group(3)),
                    })
                    continue

                # 추첨 / 전원합격 / 적격자없음
                special_match = re.match(
                    r"^([\d\w./]+)\s+"
                    r"(청년[\w()남여/팀형]*|신혼[\w부부ⅠⅡI()]*)\s+"
                    r"(추첨|전원합격|적격자 없음|-)",
                    line
                )
                if special_match:
                    current_housing = special_match.group(1)
                    rows.append({
                        "announcement_year":  year,
                        "announcement_phase": phase,
                        "review_stage":       review_stage,
                        "complex_name":       current_complex,
                        "district":           current_district,
                        "housing_type":       special_match.group(1),
                        "supply_target":      normalize_target(special_match.group(2)),
                        "applicant_rank":     None,
                        "cutline_score":      None,
                    })
                    continue

    return rows


# 전체 PDF 처리
PDF_DIR  = "청년안심주택"
all_rows = []
skip     = ["경쟁률", "2019", "2020"]

for fname in sorted(os.listdir(PDF_DIR)):
    if not fname.endswith(".pdf"):
        continue
    if any(k in fname for k in skip):
        continue

    pdf_path = os.path.join(PDF_DIR, fname)
    rows = parse_cutline_pdf(pdf_path)
    print(f"{fname}: {len(rows)}행")
    all_rows.extend(rows)

df = pd.DataFrame(all_rows)
print(f"\n전체: {len(df)}행")
print("\n=== 신청자격별 분포 ===")
print(df["supply_target"].value_counts())
print("\n=== 심사단계별 분포 ===")
print(df["review_stage"].value_counts())
print("\n=== 점수 있는 행 / 추첨 행 ===")
print(f"점수 있음: {df['cutline_score'].notna().sum()}")
print(f"추첨(null): {df['cutline_score'].isna().sum()}")

df.to_csv("youth_cutline_raw.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: youth_cutline_raw.csv")