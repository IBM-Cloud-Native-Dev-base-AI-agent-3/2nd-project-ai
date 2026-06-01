from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# FastAPI 애플리케이션 생성
app = FastAPI(title="임대주택 커트라인 예측 API")

# 모델 로딩
try:
    model = joblib.load("cutoff_model.pkl")
    target_encoding_map = joblib.load("target_encoding_map.pkl")
    print("로딩 완료")
except Exception as e:
    print(f"모델 로딩 실패: {e}")
    
# LabelEncoder 세팅
le_eligibility = LabelEncoder()
le_eligibility.fit(['청년', '신혼부부', '신혼부부I',
                    '신혼부부II', '고령자', '주거급여수급자',
                    '대학생', '일반가구'])