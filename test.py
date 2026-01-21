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

area = input("여행할 지역을 입력하세요 (예: 종로구): ")

filtered_spot = df[(df["area"] == f"{area}") & (df["category"] != "식당")][["name", "lat", "lng"]]
filtered_restaurant = df[(df["area"] == f"{area}") & (df["category"] == "식당")][["name", "lat", "lng"]]
filtered_accom = df[(df["area"] == f"{area}") & (df["category"] == "숙박")][["name", "lat", "lng"]]

places = filtered_spot.to_dict(orient="records")
restaurants = filtered_restaurant.to_dict(orient="records")
accommodations = filtered_accom.to_dict(orient="records")

# ============================================================
# 날짜 계산
# ============================================================

start_date = input("여행 시작 일자 (예: 2026-01-20): ")
end_date = input("여행 종료 일자 (예: 2026-01-25): ")

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
- 매일 관광지 4~5곳 + 식당 2곳 구성
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

# with open("result.json", "w", encoding="utf-8") as f:
#     json.dump(result, f, ensure_ascii=False, indent=2)

# ============================================================
# 설정
# ============================================================

START_TIME = datetime.strptime("09:00", "%H:%M")
LUNCH_WINDOW = ("12:00", "13:00")
DINNER_WINDOW = ("18:00", "19:00")

FIXED_EVENTS = []

has_fixed = input("고정 일정이 있나요? (y/n): ").strip().lower()

if has_fixed == "y":
    while True:
        FIXED_DATE = input("고정 일정 날짜 (예: 2026-01-21): ")
        TITLE = input("고정 일정 제목 (예: 공연): ")
        FIXED_START = input("시작 시간 (예: 15:00): ")
        FIXED_END = input("종료 시간 (예: 16:30): ")

        FIXED_EVENTS.append({
            "date": FIXED_DATE,
            "title": TITLE,
            "start": FIXED_START,
            "end": FIXED_END
        })

        if input("계속 추가하시겠습니까? (y/n): ").strip().lower() != "y":
            break


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
    if p1["lat"] is None or p2["lat"] is None:
        return 0
    dist = haversine(p1["lat"], p1["lng"], p2["lat"], p2["lng"])
    return int(dist / 30 * 60)  # 평균 30km/h

def get_fixed_events_for_day(fixed_events, target_date):
    return [e for e in fixed_events if e["date"] == target_date]

# ============================================================
# 노드 생성
# ============================================================

def build_fixed_nodes(fixed_events):
    nodes = []

    for event in fixed_events:
        start_min = int((parse_time(event["start"]) - START_TIME).total_seconds() / 60)
        end_min = int((parse_time(event["end"]) - START_TIME).total_seconds() / 60)
        duration = end_min - start_min

        nodes.append({
            "name": event["title"],
            "category": "고정일정",
            "lat": None,
            "lng": None,
            "stay": duration,
            "type": "fixed",
            "window": (start_min, start_min + 5)  # 시작 시간 고정
        })

    return nodes

def build_nodes(places, restaurants, fixed_events):
    nodes = []

    # 관광지
    for p in places:
        nodes.append({
            "name": p["name"],
            "category": p["category"],
            "lat": p["lat"],
            "lng": p["lng"],
            "stay": stay_time_map.get(p["category"], 60),
            "type": "spot"
        })

    # 점심
    nodes.append({
        "name": restaurants[0]["name"],
        "category": "식당",
        "lat": restaurants[0]["lat"],
        "lng": restaurants[0]["lng"],
        "stay": 70,
        "type": "lunch"
    })

    # 저녁
    nodes.append({
        "name": restaurants[1]["name"],
        "category": "식당",
        "lat": restaurants[1]["lat"],
        "lng": restaurants[1]["lng"],
        "stay": 70,
        "type": "dinner"
    })

    # 고정 일정
    fixed_nodes = build_fixed_nodes(fixed_events)
    nodes.extend(fixed_nodes)

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

        elif n["type"] == "fixed":
            # ✅ 고정 일정은 입력된 시작 시간에만 방문 가능
            windows.append(n["window"])

        else:
            windows.append((0, 12 * 60))  # 09:00~21:00

    return windows

# ============================================================
# OR-Tools 모델
# ============================================================

def optimize_with_auto_trim(places, restaurants, start_location, fixed_events=None, max_retry=10):
    if fixed_events is None:
        fixed_events = []

    spots = places.copy()

    for attempt in range(max_retry):
        try:
            print(f"🔁 시도 {attempt+1}회 — 관광지 {len(spots)}곳")
            timeline = optimize_day(spots, restaurants, start_location, fixed_events)
            return timeline
        except Exception:
            print("⚠ 일정 불가능 → 관광지 1곳 제거 후 재시도")

            if len(spots) <= 2:
                raise Exception("관광지를 더 이상 줄일 수 없습니다.")

            spots.pop()

    raise Exception("일정 생성 실패")

def optimize_day(places, restaurants, start_location, fixed_events):
    nodes = build_nodes(places, restaurants, fixed_events)
    n = len(nodes)

    time_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue

            if nodes[i]["type"] == "fixed" or nodes[j]["type"] == "fixed":
                time_matrix[i][j] = nodes[j]["stay"]
            else:
                time_matrix[i][j] = travel_minutes(nodes[i], nodes[j]) + nodes[j]["stay"]

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return time_matrix[i][j]

    transit_callback = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

    routing.AddDimension(transit_callback, 30, 12*60, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    time_windows = build_time_windows(nodes)

    for i, window in enumerate(time_windows):
        idx = manager.NodeToIndex(i)
        time_dim.CumulVar(idx).SetRange(window[0], window[1])

    time_dim.CumulVar(routing.Start(0)).SetValue(0)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_params)

    if not solution:
        raise Exception("해결 불가")

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

# ============================================================
# 일정 타임라인 json에 추가
# ============================================================

plans = result["plans"]

current_date = start

for day_key, day_data in plans.items():
    print(f"\n📅 {day_key} 일정 최적화")

    day_places = day_data["route"]
    day_restaurants = day_data["restaurants"]

    day_str = current_date.strftime("%Y-%m-%d")
    day_fixed_events = get_fixed_events_for_day(FIXED_EVENTS, day_str)

    timeline = optimize_with_auto_trim(
    day_places,
    day_restaurants,
    start_location={"lat": 37.5547, "lng": 126.9706},
    fixed_events=day_fixed_events
    )

    result["plans"][day_key]["timeline"] = timeline

    for t in timeline:
        print(f"{t['time']} - {t['name']} ({t['category']})")

    current_date += timedelta(days=1)

print("\n====== Gemini 결과 ======\n")
print(json.dumps(result, ensure_ascii=False, indent=2))