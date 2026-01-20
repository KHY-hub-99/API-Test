import google.generativeai as genai
import json
import pandas as pd
import random
import time
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# areas = ["종로구", "강남구", "마포구", "성수동", "홍대", "이태원", "잠실"]
# categories = ["관광지", "카페", "식당", "박물관", "공원", "시장"]
# mood_pool = ["감성", "조용한", "데이트", "힐링", "혼자", "활동적", "야경", "가족"]

# stay_time_map = {
#     "관광지": 90,
#     "카페": 50,
#     "식당": 70,
#     "박물관": 120,
#     "공원": 60,
#     "시장": 80
# }

# def random_place_name(category, area):
#     suffix = {
#         "관광지": ["명소", "전망대", "거리", "마을"],
#         "카페": ["감성카페", "뷰카페", "로스터리", "브런치"],
#         "식당": ["맛집", "식당", "국밥집", "파스타집"],
#         "박물관": ["박물관", "전시관"],
#         "공원": ["공원", "산책로"],
#         "시장": ["시장", "야시장"]
#     }
#     return f"{area} {random.choice(suffix[category])}"

# rows = []

# for i in range(1, 3001):
#     category = random.choice(categories)
#     area = random.choice(areas)
#     moods = random.sample(mood_pool, k=random.randint(2, 4))

#     row = {
#         "place_id": f"P{i:04d}",
#         "name": random_place_name(category, area),
#         "category": category,
#         "lat": round(37.4 + random.random() * 0.3, 6),
#         "lng": round(126.8 + random.random() * 0.4, 6),
#         "avg_stay_min": stay_time_map[category],
#         "mood_tags": ", ".join(moods),
#         "crowd_level": round(random.uniform(0.1, 0.9), 2),
#         "area": area
#     }

#     rows.append(row)

# df = pd.DataFrame(rows)

# df.to_excel("places_3000.xlsx", index=False)

# =================================================================
# API 테스트
# =================================================================

load_dotenv()
API = os.getenv("API_KEY")

genai.configure(api_key=API)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

df = pd.read_excel("places_3000.xlsx")
filtered_spot = df[(df["area"] == "종로구") & (df["category"] != "식당")][["name", "category", "lat", "lng"]]
print(f"필터링된 장소 개수 : {len(filtered_spot)}")
filtered_restaurant = df[(df["area"] == "종로구") & (df["category"] == "식당")][["name", "category", "lat", "lng"]]
places = filtered_spot.to_dict(orient="records")
restaurants = filtered_restaurant.to_dict(orient="records")

start_date = "2026-01-21"
end_date = "2026-01-22"
start = datetime.strptime(start_date, "%Y-%m-%d")
end = datetime.strptime(end_date, "%Y-%m-%d")
days = (end - start).days + 1
print(f"총 일수 : {days}")

system_prompt = f"""
너는 서울 여행 경로 생성기다.

반드시 아래 JSON 스키마 형식으로만 출력한다.
{
  "plans": {
    "day1": {
      "route": [
        {"name": "...", "category": "...", "lat": 0.0, "lng": 0.0}
      ],
      "restaurants": [
        {"name": "...", "category": "식당", "lat": 0.0, "lng": 0.0}
      ]
    },
    "day2": {
      "route": [],
      "restaurants": []
    }
  }
}

규칙:
- 입력된 days 만큼 day1, day2, ... 생성
- 여행 시작 일자 : {start_date}, 여행 종료 일자 : {end_date}
- 매일 관광지 4~6곳 + 식당 2곳 구성
- route에는 places 목록에서만 선택
- restaurants에는 restaurants 목록에서만 선택
- route는 이동 동선을 고려하여 방문 순서 최적화
- restaurants는 해당 day의 마지막 관광지와 가까운 순서로 2곳 선택
- 설명 문장은 출력하지 않는다
- 반드시 JSON만 출력한다
"""

user_prompt = {
    "days": 2,
    "start_location": {"lat": 37.5547, "lng": 126.9706},
    "places": places[:6*days*3],
    "restraurants": restaurants[:3*days*3],
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

# # 거리 계산
# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371  # km
#     phi1 = math.radians(lat1)
#     phi2 = math.radians(lat2)
#     dphi = math.radians(lat2 - lat1)
#     dlambda = math.radians(lon2 - lon1)

#     a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
#     return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# # 주변 음식점 찾기
# def find_near_restaurants(place, restaurants, k=2):
#     lat, lng = place["lat"], place["lng"]
#     dists = []

#     for r in restaurants:
#         d = haversine(lat, lng, r["lat"], r["lng"])
#         dists.append((d, r))

#     dists.sort(key=lambda x: x[0])
#     return [r for _, r in dists[:k]]

# # 반드시 plans 구조로 접근
# if "plans" not in result:
#     raise ValueError("Gemini JSON에 plans 필드가 없습니다.")

# plans = result["plans"]

# final_result = {}

# for day_key, day_places in plans.items():

#     # 방어 코드
#     if not isinstance(day_places, list):
#         print(f"⚠ {day_key} 구조가 리스트가 아닙니다. 건너뜁니다.")
#         continue

#     day_restaurants = []

#     for place in day_places:
#         if "lat" not in place or "lng" not in place:
#             continue

#         near = find_near_restaurants(place, restaurants)
#         day_restaurants.extend(near)

#     # 중복 제거 + 하루 2곳
#     unique = {r["name"]: r for r in day_restaurants}.values()
#     day_restaurants = list(unique)[:2]

#     final_result[day_key] = {
#         "route": day_places,
#         "restaurants": day_restaurants
#     }

# print("\n====== 최종 결과 ======\n")
# print(json.dumps(final_result, ensure_ascii=False, indent=2))

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