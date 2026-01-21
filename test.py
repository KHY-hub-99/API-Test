import google.generativeai as genai
import json
import pandas as pd
import random
import time
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# =================================================================
# API 테스트
# =================================================================

load_dotenv()
API = os.getenv("API_KEY")

genai.configure(api_key=API)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

df = pd.read_excel("places_3000.xlsx")

filtered_spot = df[(df["area"] == "종로구") & (df["category"] != "식당")][["name", "lat", "lng"]]
print(f"필터링된 장소 개수 : {len(filtered_spot)}")
filtered_restaurant = df[(df["area"] == "종로구") & (df["category"] == "식당")][["name", "lat", "lng"]]
filtered_accom = df[(df["area"] == "종로구") & (df["category"] == "숙박")][["name", "lat", "lng"]]

places = filtered_spot.to_dict(orient="records")
restaurants = filtered_restaurant.to_dict(orient="records")
accommodations = filtered_accom.to_dict(orient="records")

start_date = "2026-01-21"
end_date = "2026-01-22"
start = datetime.strptime(start_date, "%Y-%m-%d")
end = datetime.strptime(end_date, "%Y-%m-%d")
days = (end - start).days + 1
print(f"총 일수 : {days}")


schema = """
{
  "plans": {
    "day1": {
      "route": [
        {"name": "...", "category": "...", "lat": 0.0, "lng": 0.0}
      ],
      "restaurants": [
        {"name": "...", "category": "식당", "lat": 0.0, "lng": 0.0}
      ],
      "accommodations": [
        {"name": "...", "category": "숙박", "lat": 0.0, "lng": 0.0}
      ]
    },
    "day2": {
      "route": [],
      "restaurants": [],
      "accommodations": []
    }
  }
}
"""
system_prompt = f"""
너는 서울 여행 경로 생성기다.

반드시 아래 JSON 스키마 형식으로만 출력한다.

{schema}

규칙:
- 입력된 days 만큼 day1, day2, ... 생성
- 여행 시작 일자 : {start_date}, 여행 종료 일자 : {end_date}
- 매일 관광지 4~6곳 + 식당 2곳 구성
- route에는 places 목록에서만 선택
- restaurants에는 restaurants 목록에서만 선택
- accommodations에는 accommodations 목록에서만 선택
- route는 이동 동선을 고려하여 방문 순서 최적화
- restaurants는 해당 day의 마지막 관광지와 가까운 순서로 2곳 선택
- accommodations는 해당 day의 마지막 관광지와 가까운 순서로 1곳 선택
- {days}일차에는 accommodations 포함하지 않음
- 설명 문장은 출력하지 않는다
- 반드시 JSON만 출력한다
"""

user_prompt = {
    "days": days,
    "start_location": {"lat": 37.5547, "lng": 126.9706},
    "places": places[:6*days*3],
    "restraurants": restaurants[:3*days*3],
    "accommodations": accommodations[:days*3]
}

prompt = system_prompt + "\n\n" + json.dumps(user_prompt, ensure_ascii=False)

start_time = time.time()
response = model.generate_content(prompt)
elapsed = time.time() - start_time

print("⏱ Gemini 응답 시간:", round(elapsed, 3), "초")
print("\n====== Gemini 응답 ======\n")

# JSON 추출
def extract_json(text):
    if not text or text.strip() == "":
        raise ValueError("Gemini 응답이 비어있습니다.")

    text = text.strip()

    # 코드블록 제거
    if text.startswith("```"):
        text = text.split("```")[1]

    # JSON 시작/끝 찾기
    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == -1:
        raise ValueError("JSON 형식이 아닙니다:\n" + text)

    return json.loads(text[start:end])

result = extract_json(response.text)
print(json.dumps(result, ensure_ascii=False, indent=2))

## 교통 혼잡도 모델 (BPR 함수)
# T = T0 × (1 + α × (V/C)^β)
#T0 : 자유 흐름 이동시간
# V/C : 교통량 / 도로용량
# α, β : 실험으로 추정된 계수
# 미국 교통부(USDOT), 서울시 교통정책과에서도 사용

## 관광 체류시간 왜곡 연구
# 대표 논문
# Neuts & Nijkamp (2012)
# 관광지 혼잡이 방문 만족도와 체류시간에 미치는 영향

## 실제_소요시간 = 
# (이동시간 × 교통혼잡_가중치) +
# (체류시간 × 인구혼잡_가중치)