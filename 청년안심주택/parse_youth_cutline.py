from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Optional
import json, os, time
from PIL import Image

# API 키 직접 입력 (환경변수 대신)
client = genai.Client(api_key="AIzaSyCmCkfv0Nz9LMwK8vG4QGtN3eOQOJupY6o")

# 스키마 정의
class CutlineRow(BaseModel):
    announcement_year: Optional[int]    # 공고연도
    announcement_phase: Optional[str]    # 공고차수 (1차, 2차)
    complex_name: str              # 단지명
    district: str              # 자치구 (자동 추론)
    housing_type: str              # 주택형 (19, 36A 등)
    supply_target: str              # 청년 / 신혼부부Ⅰ / 신혼부부Ⅱ
    applicant_rank: Optional[str]    # 1순위 / 2순위 / 3순위
    supply_type: Optional[str]    # 우선 / 일반
    cutline_rank: Optional[int]    # 순위 (1, 2...)
    cutline_score: Optional[float]  # 커트라인 점수
    tie_breaker: Optional[str]    # 동점자 기준
    review_stage: str              # 서류 / 최종

class CutlinePage(BaseModel):
    rows: list[CutlineRow]

# 프롬프트
PROMPT = """
이 이미지는 서울시 청년안심주택 서류심사대상자 또는 당첨자 커트라인 표입니다.

다음 정보를 추출해주세요:
- 공고연도, 공고차수 (파일명 또는 페이지 상단에서 추론)
- 단지명 → 자치구 자동 추론 (예: 세이지움 태릉입구역 → 노원구)
- 주택형 (숫자만, 예: 19, 36, 59)
- 신청자격: "청년", "신혼부부Ⅰ", "신혼부부Ⅱ" 중 하나
- 신청순위: "1순위", "2순위", "3순위"
- 공급구분: "우선", "일반"
- 커트라인 순위 (숫자)
- 커트라인 점수 (숫자, 추첨/전원합격이면 null)
- 동점자 기준
- 심사단계: "서류" 또는 "최종"
"""

PNG_DIR = "png_youth_cutline/"
OUTPUT_JSON = "parsed_youth_cutline.json"

if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, encoding="utf-8") as f:
        results = json.load(f)
else:
    results = {}

png_files = sorted([f for f in os.listdir(PNG_DIR) if f.endswith(".png")])
print(f"전체 PNG: {len(png_files)}개")

for idx, fname in enumerate(png_files):
    if fname in results and len(results[fname]) > 0:
        print(f"스킵: {fname}")
        continue

    img_path = os.path.join(PNG_DIR, fname)
    img = Image.open(img_path)

    for retry in range(5):  
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",  
                contents=[img, PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CutlinePage,
                )
            )
            parsed = json.loads(response.text)
            results[fname] = parsed.get("rows", [])
            print(f"[{idx+1}/{len(png_files)}] 완료: {fname} → {len(results[fname])}행")
            break
        except Exception as e:
            wait = 60 * (retry + 1)  # 60초, 120초, 180초... 점점 늘어남
            print(f"  재시도 {retry+1}/5 ({wait}초 대기): {e}")
            time.sleep(wait)
    else:
        results[fname] = []
        print(f"[{idx+1}/{len(png_files)}] 실패: {fname}")

    time.sleep(5)
    
print("파싱 완료")