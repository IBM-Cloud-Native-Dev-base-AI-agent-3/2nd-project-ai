import os
import json
import time
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
from PIL import Image

load_dotenv(find_dotenv())  # .env 파일에서 환경변수 로드
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

class CutlineRow(BaseModel):
    announcement_year: Optional[int] = Field(None, description="공고연도 (예: 2021, 2022)")
    announcement_phase: Optional[str] = Field(None, description="공고차수 (예: '1차', '2차')")
    complex_name: str = Field(description="단지명")
    district: str = Field(description="자치구 (예: 강남구, 마포구)")
    housing_type: str = Field(description="주택형 면적 (예: '59', '39S')")
    supply_target: str = Field(description="신청자격 (청년/신혼부부/고령자/대학생/주거급여수급자)")
    supply_type: str = Field(description="구분: '우선' 또는 '일반'")
    cutline_rank: Optional[int] = Field(None, description="커트라인 순위 (1, 2, 3). 없으면 null")
    cutline_score:Optional[float] = Field(None, description="커트라인 점수. 없으면 null")
    tie_breaker: Optional[str] = Field(None, description="동점자 기준 (추첨/전입일자/거주기간/전원합격 등)")
    tie_breaker_value: Optional[str] = Field(None, description="동점자 기준값 날짜/기간. 없으면 null")

class CutlinePage(BaseModel):
    rows: List[CutlineRow]

PROMPT = """
이 이미지는 SH/LH 행복주택 커트라인 표입니다.
표의 모든 행을 빠짐없이 JSON으로 변환하세요.

[규칙]
1. 공고연도/차수: 페이지 상단 제목에서 추출 (예: '2021년 1차' → year=2021, phase='1차')
2. 자치구: 단지명으로 유추 (예: 디에이치포레센트 → 강남구)
3. 커트라인 분리:
'1순위 6점(자치구 전입일: 2010.01.25)'
→ rank=1, score=6, tie_breaker='전입일자', value='2010.01.25'

'1순위 중 추첨' → rank=1, score=null, tie_breaker='추첨'
'무작위 추첨'   → rank=null, score=null, tie_breaker='추첨'
'전원'         → rank=null, score=null, tie_breaker='전원합격'
4. 추첨/전원/미달인 행도 포함하되 score=null로 처리
5. 행이 누락되지 않도록 모든 행 포함
"""

def parse_images(image_dir, stage, output_path):
    # 기존 진행 결과 로드
    results = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"기존 진행 {len(results)}페이지 로드")

    files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    print(f"총 {len(files)}페이지 처리 예정")

    for i, filename in enumerate(files):
        if filename in results:
            continue

        filepath = os.path.join(image_dir, filename)
        print(f"[{i+1}/{len(files)}] {filename} 파싱 중...")

        for retry in range(3):
            try:
                img = Image.open(filepath)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[img, PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=CutlinePage,
                        temperature=0.0
                    )
                )
                page_data = json.loads(response.text)
                
                # 심사단계 추가
                for row in page_data["rows"]:
                    row["review_stage"] = stage
                
                results[filename] = page_data["rows"]
                
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                print(f"  → {len(page_data['rows'])}행 추출 완료")
                break

            except Exception as e:
                print(f"  오류 ({retry+1}/3): {e}")
                if retry < 2:
                    time.sleep(5)

        time.sleep(4)  # rate limit 방지

    print(f"파싱 완료: {output_path}")
    return results


if __name__ == "__main__":
    # 서류심사 파싱
    parse_images(
        image_dir="png_document",
        stage="서류",
        output_path="parsed_document.json"
    )

    # 최종당첨 파싱
    parse_images(
        image_dir="png_final",
        stage="최종",
        output_path="parsed_final.json"
    )