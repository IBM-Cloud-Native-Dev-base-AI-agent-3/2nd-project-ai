class CompetitionRow(BaseModel):
    announcement_year: Optional[int]
    announcement_phase: Optional[str]
    complex_name: str
    district: str
    housing_type: str
    supply_target: str    # 청년 / 신혼부부Ⅰ / 신혼부부Ⅱ
    applicant_rank: Optional[str]
    supply_count: Optional[int]    # 공급세대수
    applicant_count: Optional[int]   # 신청자수
    competition_rate: Optional[float] # 경쟁률

class CompetitionPage(BaseModel):
    rows: list[CompetitionRow]

PROMPT_COMP = """
이 이미지는 서울시 청년안심주택 공급 경쟁률 표입니다.

다음 정보를 추출해주세요:
- 공고연도, 공고차수
- 단지명 → 자치구 자동 추론
- 주택형
- 신청자격: "청년", "신혼부부Ⅰ", "신혼부부Ⅱ"
- 신청순위
- 공급세대수 (supply_count)
- 신청자수 (applicant_count)
- 경쟁률 = 신청자수 / 공급세대수 (소수점 2자리)
  * 경쟁률이 표에 없으면 직접 계산
"""