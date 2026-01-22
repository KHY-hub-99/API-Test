from google import genai
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

client = genai.Client(api_key=API)

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
- 매일 관광지 5곳 + 식당 2곳 구성
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
response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
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

with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# ============================================================
# 설정
# ============================================================

LUNCH_WINDOW = ("11:20", "13:20")
DINNER_WINDOW = ("17:40", "19:30")
first_day_start_str = input("여행 첫날 시작 시간 (예: 14:00) : ").strip()
last_day_end_str = input("여행 마지막 날 종료 시간 (예: 18:00) : ").strip()

default_start_str = "10:00"
default_end_str = "21:00"

if not first_day_start_str: first_day_start_str = default_start_str
if not last_day_end_str: last_day_end_str = default_end_str

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

def build_fixed_nodes(fixed_events, day_start_dt):
    nodes = []
    BUFFER = 15

    for event in fixed_events:
        event_start = parse_time(event["start"])
        event_end = parse_time(event["end"])

        # [핵심] '그날의 시작 시간'과의 차이를 분(minute)으로 계산
        # 예: 시작 14:00, 이벤트 15:00 -> 60분 지점
        orig_start_min = int((event_start - day_start_dt).total_seconds() / 60)
        orig_end_min = int((event_end - day_start_dt).total_seconds() / 60)

        raw_start_min = orig_start_min - BUFFER
        buffered_start_min = max(0, raw_start_min)

        orig_duration = orig_end_min - orig_start_min
        secured_front_buffer = orig_start_min - buffered_start_min

        final_stay = secured_front_buffer + orig_duration + BUFFER

        nodes.append({
            "name": event["title"],
            "category": "고정일정",
            "lat": None,
            "lng": None,
            "stay": final_stay,
            "type": "fixed",
            "window": (buffered_start_min, buffered_start_min + 10), # 시작 시간 엄수
            "orig_time_str": f"{event['start']} - {event['end']}" 
        })

    return nodes

def build_nodes(places, restaurants, fixed_events, day_start_dt):
    nodes = []
    
    # [수정] 출발지 자동 설정 (입력 데이터가 비어있을 경우 대비)
    if places:
        first_place = places[0]
    else:
        # 예외 처리: 장소가 하나도 없으면 임의 좌표 사용
        first_place = {"lat": 37.5665, "lng": 126.9780} 

    nodes.append({
        "name": "시작점",  # 내부용 (출력 안 됨)
        "category": "출발",
        "lat": first_place["lat"],
        "lng": first_place["lng"],
        "stay": 0,
        "type": "depot"
    })

    # 1. 관광지
    for p in places:
        nodes.append({
            "name": p["name"],
            "category": p["category"],
            "lat": p["lat"],
            "lng": p["lng"],
            "stay": stay_time_map.get(p["category"], 60),
            "type": "spot"
        })

    # 2. 식당
    if restaurants:
        nodes.append({ "name": restaurants[0]["name"], "category": "식당", "lat": restaurants[0]["lat"], "lng": restaurants[0]["lng"], "stay": 70, "type": "lunch" })
        nodes.append({ "name": restaurants[1]["name"], "category": "식당", "lat": restaurants[1]["lat"], "lng": restaurants[1]["lng"], "stay": 70, "type": "dinner" })

    # 3. 고정 일정
    fixed_nodes = build_fixed_nodes(fixed_events, day_start_dt)
    nodes.extend(fixed_nodes)

    return nodes

# ============================================================
# Time Window 설정
# ============================================================

def build_time_windows(nodes, day_start_dt):
    windows = []

    # 윈도우 계산 헬퍼: 현재 날짜 시작 시간(day_start_dt) 기준 상대 분(min) 반환
    def get_relative_window(time_str):
        target_time = parse_time(time_str)
        diff_min = int((target_time - day_start_dt).total_seconds() / 60)
        return diff_min

    lunch_start = get_relative_window(LUNCH_WINDOW[0])
    lunch_end = get_relative_window(LUNCH_WINDOW[1])
    dinner_start = get_relative_window(DINNER_WINDOW[0])
    dinner_end = get_relative_window(DINNER_WINDOW[1])

    for n in nodes:
        if n["type"] == "lunch":
            # 만약 여행 시작(14:00)보다 점심(12:00)이 빠르면? -> 윈도우가 음수가 됨
            # OR-Tools가 처리할 수 있게 하거나, Disjunction으로 인해 드랍되도록 둠
            windows.append((lunch_start, lunch_end))
        
        elif n["type"] == "dinner":
            windows.append((dinner_start, dinner_end))
        
        elif n["type"] == "fixed":
            windows.append(n["window"])
        
        else:
            # 일반 관광지는 시간 제약 없음 (0 ~ 24시간)
            windows.append((0, 24 * 60))

    return windows

# ============================================================
# OR-Tools 모델 (수정됨)
# ============================================================

def optimize_day(places, restaurants, fixed_events, start_time_str, end_time_str=None):
    # 1. 기준 시간 설정
    day_start_dt = datetime.strptime(start_time_str, "%H:%M")
    
    # 2. 하루의 최대 길이(Horizon) 계산
    if end_time_str:
        day_end_dt = datetime.strptime(end_time_str, "%H:%M")
        max_horizon_minutes = int((day_end_dt - day_start_dt).total_seconds() / 60)
        if max_horizon_minutes < 0: max_horizon_minutes = 24 * 60 
    else:
        max_horizon_minutes = 24 * 60 

    # 3. 노드 생성
    nodes = build_nodes(places, restaurants, fixed_events, day_start_dt)
    n = len(nodes)

    # 4. 시간 매트릭스 생성
    time_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            travel_val = travel_minutes(nodes[i], nodes[j])
            
            # 고정 일정이 포함된 이동인가?
            is_fixed_involved = (nodes[i]["type"] == "fixed" or nodes[j]["type"] == "fixed")
            
            if is_fixed_involved:
                # [수정] 출발지(Depot)에서 고정 일정으로 바로 가는 경우 (오픈런)
                # "거기서 여행 시작"으로 간주하여 이동 시간을 0으로 만듦
                if nodes[i]["type"] == "depot" and nodes[j]["type"] == "fixed":
                    travel_val = 0 
                else:
                    # 그 외의 경우(관광지->고정, 고정->관광지)는 20분 여유 확보
                    travel_val = max(travel_val, 20)

            time_matrix[i][j] = nodes[i]["stay"] + travel_val

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return time_matrix[i][j]

    transit_callback = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

    routing.AddDimension(transit_callback, 30, max_horizon_minutes, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # [중요] 페널티 설정
    penalty_spot = 100000    # 관광지는 못 가면 아쉬움 (10만점)
    penalty_meal = 1000000   # 식사는 시간 되면 꼭 가라 (100만점)

    # 솔버 객체 미리 가져오기 (제약조건 추가용)
    solver = routing.solver()

    for i, node in enumerate(nodes):
        index = manager.NodeToIndex(i)
        
        # Depot(출발점)는 패스
        if node["type"] == "depot":
            continue

        time_windows = build_time_windows(nodes, day_start_dt)
        window = time_windows[i] # 예: Lunch [-160, -100] (이미 지남)

        # ------------------------------------------------
        # 1. 고정 일정 (Fixed)
        # ------------------------------------------------
        if node["type"] == "fixed":
            # 고정 일정은 약간 잘리더라도 최대한 방문하도록 보정
            safe_start = max(0, min(window[0], max_horizon_minutes))
            safe_end = max(safe_start, min(window[1], max_horizon_minutes))
            
            if safe_end < safe_start: safe_end = safe_start + 10
            
            time_dim.CumulVar(index).SetRange(safe_start, safe_end)
            continue 

        # -------------------------------------------------
        # 2. 일반 관광지 및 식당 (엄격한 시간 검사)
        # ---------------------------------------------------
        raw_start = window[0]
        raw_end = window[1]

        overlap_start = max(0, raw_start)
        overlap_end = min(max_horizon_minutes, raw_end)
        has_overlap = overlap_start <= overlap_end

        # 식당인데 시간이 안 맞으면? -> 제외
        if not has_overlap:
            routing.AddDisjunction([index], 0) 
            solver.Add(routing.VehicleVar(index) == -1)
            continue

        # 시간 설정
        time_dim.CumulVar(index).SetRange(overlap_start, overlap_end)
        
        if node["type"] == "spot":
            routing.AddDisjunction([index], penalty_spot)
        elif node["type"] in ["lunch", "dinner"]:
            routing.AddDisjunction([index], penalty_meal)

    # 검색 설정
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 1

    solution = routing.SolveWithParameters(search_params)

    if not solution:
        return []

    index = routing.Start(0)
    timeline = []

    while not routing.IsEnd(index):
        node_idx = manager.IndexToNode(index)
        node = nodes[node_idx]

        if node["type"] == "depot":
            index = solution.Value(routing.NextVar(index))
            continue

        if node["type"] == "fixed":
            time_str = node["orig_time_str"]
        else:
            t = solution.Value(time_dim.CumulVar(index))
            visit_start = day_start_dt + timedelta(minutes=t)
            visit_end = visit_start + timedelta(minutes=node["stay"])
            time_str = f"{visit_start.strftime('%H:%M')} - {visit_end.strftime('%H:%M')}"

        timeline.append({
            "name": node["name"],
            "category": node["category"],
            "time": time_str
        })
        index = solution.Value(routing.NextVar(index))

    return timeline
# ============================================================
# 일정 타임라인 json에 추가 (실행부 수정)
# ============================================================

# result = json.load(open("result.json", "r", encoding="utf-8"))
# 또는 기존 result 사용
plans = result["plans"]
current_date = start

# 전체 날짜 리스트 확인
day_keys = list(plans.keys())
total_days = len(day_keys)

for i, day_key in enumerate(day_keys):
    print(f"\n📅 {day_key} 일정 최적화")

    day_data = plans[day_key]
    day_places = day_data["route"]
    day_restaurants = day_data["restaurants"]
    day_str = current_date.strftime("%Y-%m-%d")
    day_fixed_events = get_fixed_events_for_day(FIXED_EVENTS, day_str)

    # 1. 시작 시간 결정
    if i == 0:
        # 첫째 날
        todays_start = first_day_start_str
    else:
        # 그 외 날짜
        todays_start = default_start_str

    # 2. 종료 시간 제한 결정
    if i == total_days - 1:
        # 마지막 날
        todays_end = last_day_end_str
    else:
        todays_end = default_end_str

    timeset = f"{todays_start} 시작" + (f" ~ {todays_end} 종료" if todays_end else "")
    print(timeset)

    # 3. 최적화 실행
    timeline = optimize_day(
        places=day_places,
        restaurants=day_restaurants,
        fixed_events=day_fixed_events,
        start_time_str=todays_start,       # 시작 시간 전달
        end_time_str=todays_end      # 종료 시간(마지막날용) 전달
    )

    result["plans"][day_key]["timeset"] = timeset
    result["plans"][day_key]["timeline"] = timeline

    if not timeline:
        print("   ⚠ 조건 만족하는 일정 생성 실패")
    else:
        for t in timeline:
            print(f"   {t['time']}  {t['name']} ({t['category']})")

    current_date += timedelta(days=1)

print("\n====== 최종 결과 ======\n")
# JSON 파일로 저장
file_path = "result_timeline.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"✅ 일정이 '{file_path}' 파일로 저장되었습니다.")