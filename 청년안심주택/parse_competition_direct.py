import pdfplumber
import pandas as pd
import re
import os
from collections import defaultdict


def normalize_target(t):
    if not t: return None
    t = str(t).strip()
    if any(x in t for x in ["팀형", "셰어형(팀)", "팀 청약"]): return "청년(팀형)"
    if "청년" in t: return "청년"
    if any(x in t for x in ["신혼Ⅱ", "신혼II"]): return "신혼Ⅱ"
    if any(x in t for x in ["신혼Ⅰ", "신혼I", "신혼 I", "신혼부부"]): return "신혼Ⅰ"
    if "신혼" in t: return "신혼Ⅰ"
    return None


TARGET_PAT = r"청년[\w()남여/]*|신혼[\w부부ⅠⅡI()]*"


def get_lines_by_position(page, y_tol=4):
    words = page.extract_words(keep_blank_chars=False)
    if not words:
        return []
    row_map = defaultdict(list)
    for w in words:
        y_key = round(w["top"] / y_tol) * y_tol
        row_map[y_key].append(w)
    result = []
    for y_key in sorted(row_map.keys()):
        line_words = sorted(row_map[y_key], key=lambda w: w["x0"])
        line_text = " ".join(w["text"] for w in line_words).strip()
        if line_text:
            result.append(line_text)
    return result


def parse_competition_pdf(pdf_path):
    rows = []
    fname = os.path.basename(pdf_path)

    year_match  = re.search(r"(\d{4})년", fname)
    phase_match = re.search(r"(\d+)차", fname)
    year  = int(year_match.group(1))  if year_match  else None
    phase = f"{phase_match.group(1)}차" if phase_match else None

    with pdfplumber.open(pdf_path) as pdf:
        current_complex  = None
        current_district = None
        current_housing  = None
        current_target   = None
        current_supply   = None
        pending_housing  = None
        pending_supply   = None
        pending_target   = None  # target 먼저 보이면 저장

        for page in pdf.pages:
            lines = get_lines_by_position(page)

            for line in lines:
                line = line.replace("신혼 I", "신혼I").replace("신혼 II", "신혼II")
                # 헤더/공고 스킵
                if any(kw in line for kw in [
                    "단지명", "공급유형", "신청자격", "공급호수",
                    "입주자모집공고", "총계", "신청자수", "경쟁률"
                ]):
                    continue

                # 순위 행 스킵
                if re.match(r"^\d순위", line):
                    continue

                # 자치구 추출
                d_match = re.search(r"\((.{2,3}구)", line)
                if d_match:
                    current_district = d_match.group(1)
                    cp = line.split("(")[0].strip()
                    if cp and not re.match(r"^[\d.]+$", cp):
                        current_complex = cp

                # 소계 행 → 저장
                s_match = re.search(r"소계\s+([\d,]+)\s+([\d,.]+)", line)
                if s_match:
                    h = current_housing or pending_housing
                    t = current_target  or pending_target
                    s = current_supply  or pending_supply
                    if h and s:
                        try:
                            rows.append({
                                "announcement_year":  year,
                                "announcement_phase": phase,
                                "complex_name":       current_complex,
                                "district":           current_district,
                                "housing_type":       h,
                                "supply_target":      t,
                                "supply_count":       s,
                                "total_applicants":   int(s_match.group(1).replace(",", "")),
                                "competition_rate":   float(s_match.group(2).replace(",", "")),
                            })
                        except:
                            pass
                    # 소계 후 초기화 (pending은 다음 유닛에 쓸 수 있으므로 유지)
                    current_housing = None
                    current_target  = None
                    current_supply  = None
                    pending_housing = None
                    pending_supply  = None
                    continue

                # Pattern A: housing + target + supply [+ 순위데이터]
                # "36B 신혼II 67" / "골든노블레스 18 청년 17" / "37M 청년(남) 3"
                a_match = re.search(
                    r"\b([\d][\d\w./]*)\s+(" + TARGET_PAT + r")\s+(\d+)(?=\s*(?:\d순위|\s*$))",
                    line
                )
                if a_match and "소계" not in line:
                    prefix = line[:a_match.start()].strip()
                    if prefix and not re.match(r"^[\d.]+$", prefix):
                        d2 = re.search(r"\((.{2,3}구)", prefix)
                        if d2:
                            current_district = d2.group(1)
                        cp2 = prefix.split("(")[0].strip()
                        if cp2 and not re.match(r"^\d", cp2):
                            current_complex = cp2
                    current_housing = a_match.group(1)
                    current_target  = normalize_target(a_match.group(2))
                    current_supply  = int(a_match.group(3))
                    pending_housing = None
                    pending_supply  = None
                    pending_target  = None
                    continue

                # Pattern B: housing + target + 순위 + supply + 신청자수
                # "36A 신혼I 2순위 67 256" / "천호역... 35A 신혼I 2순위 5 263"
                b_match = re.search(
                    r"\b([\d][\d\w./]*)\s+(" + TARGET_PAT + r")\s+\d순위\s+(\d+)\s+[\d,]+",
                    line
                )
                if b_match and "소계" not in line:
                    prefix = line[:b_match.start()].strip()
                    if prefix and not re.match(r"^[\d.]+$", prefix):
                        d2 = re.search(r"\((.{2,3}구)", prefix)
                        if d2:
                            current_district = d2.group(1)
                        cp2 = prefix.split("(")[0].strip()
                        if cp2 and not re.match(r"^\d", cp2):
                            current_complex = cp2
                    current_housing = b_match.group(1)
                    current_target  = normalize_target(b_match.group(2))
                    current_supply  = int(b_match.group(3))
                    pending_housing = None
                    pending_supply  = None
                    pending_target  = None
                    continue

                # Pattern C: target 단독 행
                # "청년" / "신혼I" / "청년(남)"
                c_match = re.match(r"^(" + TARGET_PAT + r")$", line)
                if c_match:
                    t = normalize_target(c_match.group(1))
                    if pending_housing and pending_supply:
                        current_housing = pending_housing
                        current_target  = t
                        current_supply  = pending_supply
                        pending_housing = None
                        pending_supply  = None
                        pending_target  = None
                    else:
                        pending_target = t
                    continue

                # Pattern D: complex_name + target (supply 없음)
                # "천호역 효성해링턴타워(성내동 609) 청년"
                d2_match = re.match(r"^(.+?)\s+(" + TARGET_PAT + r")$", line)
                if d2_match and not re.search(r"\d+\s*$", d2_match.group(1)):
                    t = normalize_target(d2_match.group(2))
                    if pending_housing and pending_supply:
                        current_housing = pending_housing
                        current_target  = t
                        current_supply  = pending_supply
                        pending_housing = None
                        pending_supply  = None
                        pending_target  = None
                    else:
                        pending_target = t
                    continue

                # Pattern E: housing + supply 만 있는 행
                # "16 54" / "26A 1" / "33 14"
                e_match = re.match(r"^([\d][\d\w./]*)\s+(\d+)$", line)
                if e_match and "소계" not in line:
                    h = e_match.group(1)
                    s = int(e_match.group(2))
                    if pending_target:
                        current_housing = h
                        current_target  = pending_target
                        current_supply  = s
                        pending_target  = None
                    else:
                        pending_housing = h
                        pending_supply  = s
                    continue

    return rows


# 전체 처리
COMP_DIR = "청년안심주택/경쟁률"
all_rows = []

for fname in sorted(os.listdir(COMP_DIR)):
    if not fname.endswith(".pdf"):
        continue
    pdf_path = os.path.join(COMP_DIR, fname)
    rows = parse_competition_pdf(pdf_path)
    print(f"{fname}: {len(rows)}행")
    all_rows.extend(rows)

df = pd.DataFrame(all_rows)
print(f"\n전체: {len(df)}행")
print("\n=== 신청자격별 ===")
print(df["supply_target"].value_counts())
print("\n=== 자치구 커버리지 ===")
print(f"자치구 수: {df['district'].nunique()}")
print(df["district"].value_counts().head(10))

df.to_csv("youth_competition_raw.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: youth_competition_raw.csv")