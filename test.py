import google.generativeai as genai
import json
import pandas as pd
import time
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# ============================================================
# API 설정
# ============================================================

load_dotenv()
API = os.getenv("API_KEY")

genai.configure(api_key=API)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# ============================================================
# 데이터 로드
# ============================================================

df = pd.read_excel("places_3000.xlsx")

filtered_spot = df[(df["area"] == "종로구") & (df["category"] != "식당")][["name", "category", "lat", "lng"]]
filtered_restaurant = df[(df["area"] == "종로구") & (df["category"] == "식당")][["name", "category", "lat", "lng"]]
filtered_accom = df[(df["area"] == "종로구") & (df["category"] == "숙박")][["name", "category", "lat", "lng"]]

places = filtered_spot.to_dict(orient="records")
restaurants = filtered_restaurant.to_dict(orient="records")
accommodations = filtered_accom.to_dict(orient="records")

# ============================================================
# 날짜 계산
# ============================================================

start_date = "2026-01-21"
end_date = "2026-01-22"

start = datetime.strptime(start_date, "%Y-%m-%d")
end = datetime.strptime(end_date, "%Y-%m-%d")
days = (end - start).days + 1

print(f"총 여행 일수: {days}")

# ============================================================
# 프롬프트
# ============================================================

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
- 마지막 날에는 accommodations 포함하지 않음
- 설명 문장은 출력하지 않는다
- 반드시 JSON만 출력한다
"""

user_prompt = {
    "days": days,
    "start_location": {"lat": 37.5547, "lng": 126.9706},
    "places": places[:6 * days * 3],
    "restaurants": restaurants[:3 * days * 3],
    "accommodations": accommodations[:days * 3]
}

prompt = system_prompt + "\n\n" + json.dumps(user_prompt, ensure_ascii=False)

# ============================================================
# Gemini 호출
# ============================================================

start_time = time.time()
response = model.generate_content(prompt)
elapsed = time.time() - start_time

print("⏱ Gemini 응답 시간:", round(elapsed, 3), "초")

# ============================================================
# JSON 추출
# ============================================================

def extract_json(text):
    if not text:
        raise ValueError("Gemini 응답이 비어있습니다.")

    text = text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == -1:
        raise ValueError("JSON 파싱 실패:\n" + text)

    return json.loads(text[start:end])


result = extract_json(response.text)

print("\n====== Gemini 결과 ======\n")
print(json.dumps(result, ensure_ascii=False, indent=2))

# with open("result.json", "w", encoding="utf-8") as f:
#     json.dump(result, f, ensure_ascii=False, indent=2)

# ============================================================
# 설정
# ============================================================

START_TIME = datetime.strptime("09:00", "%H:%M")
LUNCH_WINDOW = ("12:00", "13:00")
DINNER_WINDOW = ("18:00", "19:00")

stay_time_map = {
    "관광지": 90,
    "카페": 50,
    "식당": 70,
    "박물관": 120,
    "공원": 60,
    "시장": 80,
    "숙박": 0
}

# ============================================================
# 유틸
# ============================================================

def parse_time(t):
    return datetime.strptime(t, "%H:%M")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def travel_minutes(p1, p2):
    dist = haversine(p1["lat"], p1["lng"], p2["lat"], p2["lng"])
    return int(dist / 30 * 60)  # 평균 30km/h

# ============================================================
# 노드 생성
# ============================================================

def build_nodes(places, restaurants):
    nodes = []
    for p in places:
        nodes.append({
            "name": p["name"],
            "category": p["category"],
            "lat": p["lat"],
            "lng": p["lng"],
            "stay": stay_time_map.get(p["category"], 60),
            "type": "spot"
        })

    # 점심, 저녁 식당 1곳씩만 사용
    nodes.append({
        "name": restaurants[0]["name"],
        "category": "식당",
        "lat": restaurants[0]["lat"],
        "lng": restaurants[0]["lng"],
        "stay": 70,
        "type": "lunch"
    })

    nodes.append({
        "name": restaurants[1]["name"],
        "category": "식당",
        "lat": restaurants[1]["lat"],
        "lng": restaurants[1]["lng"],
        "stay": 70,
        "type": "dinner"
    })

    return nodes

# ============================================================
# Time Window 설정
# ============================================================

def build_time_windows(nodes):
    windows = []

    for n in nodes:
        if n["type"] == "lunch":
            windows.append((
                int((parse_time(LUNCH_WINDOW[0]) - START_TIME).total_seconds() / 60),
                int((parse_time(LUNCH_WINDOW[1]) - START_TIME).total_seconds() / 60)
            ))
        elif n["type"] == "dinner":
            windows.append((
                int((parse_time(DINNER_WINDOW[0]) - START_TIME).total_seconds() / 60),
                int((parse_time(DINNER_WINDOW[1]) - START_TIME).total_seconds() / 60)
            ))
        else:
            windows.append((0, 12 * 60))  # 09:00~21:00

    return windows

# ============================================================
# OR-Tools 모델
# ============================================================

def optimize_day(places, restaurants, start_location):
    nodes = build_nodes(places, restaurants)
    n = len(nodes)

    # 거리 행렬
    time_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            time_matrix[i][j] = travel_minutes(nodes[i], nodes[j]) + nodes[j]["stay"]

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return time_matrix[i][j]

    transit_callback = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

    routing.AddDimension(
        transit_callback,
        30,        # slack
        12*60,     # 하루 12시간
        False,
        "Time"
    )

    time_dim = routing.GetDimensionOrDie("Time")

    time_windows = build_time_windows(nodes)

    for i, window in enumerate(time_windows):
        idx = manager.NodeToIndex(i)
        time_dim.CumulVar(idx).SetRange(window[0], window[1])

    # 시작 시간 = 09:00
    time_dim.CumulVar(routing.Start(0)).SetValue(0)

    # 탐색 전략
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_params)

    if not solution:
        raise Exception("해결 불가")

    # 결과 추출
    index = routing.Start(0)
    timeline = []

    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        t = solution.Value(time_dim.CumulVar(index))
        visit_time = START_TIME + timedelta(minutes=t)

        timeline.append({
            "name": nodes[node]["name"],
            "category": nodes[node]["category"],
            "time": visit_time.strftime("%H:%M")
        })

        index = solution.Value(routing.NextVar(index))

    return timeline

plans = result["plans"]
for day_key, day_data in plans.items():
    print(f"\n📅 {day_key} 일정 최적화")

    day_places = day_data["route"]
    day_restaurants = day_data["restaurants"]

    if len(day_restaurants) < 2:
        print("❌ 식당이 2곳 미만입니다. 스킵합니다.")
        continue
    
    timeline = optimize_day(day_places, day_restaurants, start_location={
        "lat": 37.5547,
        "lng": 126.9706
    })

    print(f"\n📌 {day_key} 최종 타임라인")
    for t in timeline:
        print(f"{t['time']} - {t['name']} ({t['category']})")