from fastapi import APIRouter
from pydantic import BaseModel, Field
from enum import Enum

router = APIRouter()

class ComplexName(str, Enum):
    연남 = "연남공공원룸텔"
    정릉 = "희망하우징(정릉)"

class Gender(str, Enum):
    여성 = "여성"
    남성 = "남성"

class ApplicantRank(int, Enum):
    first = 1
    second = 2
    third = 3    

# 이전 공고 커트라인 데이터 (PDF에서 추출)
CUTLINE_DATA = {
    ("연남공공원룸텔", "여성"): {"cutline": 8, "competition_rate": 41.5},
    ("연남공공원룸텔", "남성"): {"cutline": 2, "competition_rate": 9.0},
    ("희망하우징(정릉)", "여성"): {"cutline": 8, "competition_rate": 37.5},
    ("희망하우징(정릉)", "남성"): {"cutline": 0, "competition_rate": 5.2},
}


class HopeHousingRequest(BaseModel):
    complex_name: ComplexName # "연남공공원룸텔" or "희망하우징(정릉)"
    gender: Gender # "여성" or "남성"
    applicant_rank: ApplicantRank # 1, 2, 3
    is_parents_homeless: bool
    is_applicant_disabled: bool
    is_parents_disabled: bool
    subscription_count: int = Field(ge=0) #0 이상만 허용
    # 1순위 전용
    is_recipient: bool = False
    is_single_parent_family: bool = False
    # 2·3순위 전용
    is_income_under50: bool = False


def calculate_score(req: HopeHousingRequest) -> int:
    score = 0

    # 공통 가점
    if req.is_parents_homeless:
        score += 2
    if req.is_applicant_disabled:
        score += 2
    if req.is_parents_disabled:
        score += 1

    # 청약저축 가점
    if req.subscription_count >= 24:
        score += 3
    elif req.subscription_count >= 12:
        score += 2
    elif req.subscription_count >= 6:
        score += 1

    # 순위별 추가 가점
    if req.applicant_rank == 1:
        if req.is_recipient:
            score += 3
        if req.is_single_parent_family:
            score += 3
    else:  # 2순위, 3순위
        if req.is_income_under50:
            score += 3

    return score


def calculate_win_probability(user_score: int, cutline: int) -> float:
    diff = user_score - cutline
    if diff >= 2:
        return 90.0
    elif diff == 1:
        return 70.0
    elif diff == 0:
        return 50.0
    elif diff == -1:
        return 20.0
    else:
        return 5.0


@router.post("/hope-housing")
def predict_hope_housing(req: HopeHousingRequest):
    key = (req.complex_name, req.gender)
    if key not in CUTLINE_DATA:
        return {"error": f"해당 단지/성별 데이터 없음: {req.complex_name} {req.gender}"}

    data = CUTLINE_DATA[key]
    cutline = data["cutline"]
    competition_rate = data["competition_rate"]

    user_score = calculate_score(req)
    win_probability = calculate_win_probability(user_score, cutline)

    return {
        "complex_name": req.complex_name,
        "gender": req.gender,
        "user_score": user_score,
        "previous_cutline": cutline,
        "competition_rate": competition_rate,
        "win_probability": win_probability,
        "result": "통과 예상" if user_score >= cutline else "미달 예상"
    }