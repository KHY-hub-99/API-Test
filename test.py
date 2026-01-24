import os
os.environ["JAVA_OPTS"] = "-Xmx8G"
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-23"

from google import genai
import zipfile
import json
import pandas as pd
import geopandas as gpd
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import io
from r5py import TransportNetwork, TravelTimeMatrix, DetailedItineraries

# # GTFS 파일 경로 (본인의 경로로 수정)
# gtfs_path = "./data/south_korea_gtfs.zip"

# try:
#     with zipfile.ZipFile(gtfs_path) as z:
#         # 1. feed_info.txt 확인 (가장 정확함)
#         if "feed_info.txt" in z.namelist():
#             with z.open("feed_info.txt") as f:
#                 df = pd.read_csv(f)
#                 print("📅 [feed_info.txt] 데이터 유효 기간:")
#                 if 'feed_start_date' in df.columns:
#                     print(f"   시작일: {df['feed_start_date'].iloc[0]}")
#                     print(f"   종료일: {df['feed_end_date'].iloc[0]}")
#                 else:
#                     print("   날짜 정보 컬럼 없음")
        
#         # 2. calendar.txt 확인 (운행 스케줄 기준)
#         elif "calendar.txt" in z.namelist():
#             with z.open("calendar.txt") as f:
#                 df = pd.read_csv(f)
#                 print("📅 [calendar.txt] 운행 스케줄 범위:")
#                 print(f"   가장 이른 날짜: {df['start_date'].min()}")
#                 print(f"   가장 늦은 날짜: {df['end_date'].max()}")
        
#         # 3. calendar_dates.txt 확인 (예외 운행일 기준)
#         elif "calendar_dates.txt" in z.namelist():
#             with z.open("calendar_dates.txt") as f:
#                 df = pd.read_csv(f)
#                 print("📅 [calendar_dates.txt] 운행일 범위:")
#                 print(f"   가장 이른 날짜: {df['date'].min()}")
#                 print(f"   가장 늦은 날짜: {df['date'].max()}")
#         else:
#             print("⚠️ 날짜 정보를 확인할 파일이 없습니다.")

# except Exception as e:
#     print(f"❌ 오류: {e}")

# # ============================================================
# # API 설정
# # ============================================================

# load_dotenv()
# API = os.getenv("API_KEY")

# client = genai.Client(api_key=API)

# # ============================================================
# # 데이터 로드
# # ============================================================

# df = pd.read_excel("places_3000.xlsx")

# area = input("여행할 지역을 입력하세요 (예: 종로구): ")

# filtered_spot = df[(df["area"] == f"{area}") & (df["category"] != "식당")][["name", "lat", "lng"]]
# filtered_restaurant = df[(df["area"] == f"{area}") & (df["category"] == "식당")][["name", "lat", "lng"]]
# filtered_accom = df[(df["area"] == f"{area}") & (df["category"] == "숙박")][["name", "lat", "lng"]]

# places = filtered_spot.to_dict(orient="records")
# restaurants = filtered_restaurant.to_dict(orient="records")
# accommodations = filtered_accom.to_dict(orient="records")

# ============================================================
# 날짜 계산
# ============================================================

start_date = input("여행 시작 일자 (예: 2026-01-20): ")
end_date = input("여행 종료 일자 (예: 2026-01-25): ")

start = datetime.strptime(start_date, "%Y-%m-%d")
end = datetime.strptime(end_date, "%Y-%m-%d")
days = (end - start).days + 1

print(f"총 여행 일수: {days}")

# # ============================================================
# # 프롬프트
# # ============================================================

# schema = """
# {
#   "plans": {
#     "day1": {
#       "route": [
#         {"name": "...", "category": "...", "lat": 0.0, "lng": 0.0}
#       ],
#       "restaurants": [
#         {"name": "...", "category": "식당", "lat": 0.0, "lng": 0.0}
#       ],
#       "accommodations": [
#         {"name": "...", "category": "숙박", "lat": 0.0, "lng": 0.0}
#       ]
#     }
#   }
# }
# """

# system_prompt = f"""
# 너는 서울 여행 경로 생성기다.

# 반드시 아래 JSON 스키마 형식으로만 출력한다.

# {schema}

# 규칙:
# - 입력된 days 만큼 day1, day2, ... 생성
# - 여행 시작 일자 : {start_date}, 여행 종료 일자 : {end_date}
# - 매일 관광지 5곳 + 식당 2곳 구성
# - route에는 places 목록에서만 선택
# - restaurants에는 restaurants 목록에서만 선택
# - accommodations에는 accommodations 목록에서만 선택
# - route는 이동 동선을 고려하여 방문 순서 최적화
# - restaurants는 해당 day의 마지막 관광지와 가까운 순서로 2곳 선택
# - accommodations는 해당 day의 마지막 관광지와 가까운 순서로 1곳 선택
# - 마지막 날에는 accommodations 포함하지 않음
# - 설명 문장은 출력하지 않는다
# - 반드시 JSON만 출력한다
# """

# user_prompt = {
#     "days": days,
#     "start_location": {"lat": 37.5547, "lng": 126.9706},
#     "places": places[:6 * days * 3],
#     "restaurants": restaurants[:3 * days * 3],
#     "accommodations": accommodations[:days * 3]
# }

# prompt = system_prompt + "\n\n" + json.dumps(user_prompt, ensure_ascii=False)

# # ============================================================
# # Gemini 호출
# # ============================================================

# start_time = time.time()
# response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
# elapsed = time.time() - start_time

# print("⏱ Gemini 응답 시간:", round(elapsed, 3), "초")

# # ============================================================
# # JSON 추출
# # ============================================================

# def extract_json(text):
#     if not text:
#         raise ValueError("Gemini 응답이 비어있습니다.")

#     text = text.strip()

#     if text.startswith("```"):
#         text = text.split("```")[1]

#     start = text.find("{")
#     end = text.rfind("}") + 1

#     if start == -1 or end == -1:
#         raise ValueError("JSON 파싱 실패:\n" + text)

#     return json.loads(text[start:end])

# # ============================================================
# # 설정
# # ============================================================

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
# r5py 네트워크 초기화 (스크립트 시작 시 한 번만 실행) / Java 설치 필수
# ============================================================
transport_network = TransportNetwork("./data/south-korea_V2.osm.pbf", ["./data/south_korea_gtfs.zip"])

# ============================================================
# r5py 기반 이동 시간 계산 함수 (수정됨)
# ============================================================
def get_r5py_matrix(nodes, departure_time):
    """
    모든 노드 간의 대중교통 이동 시간 행렬을 한꺼번에 계산합니다.
    """
    # 1. 좌표가 있는 노드만 추출
    valid_nodes = [n for n in nodes if n["lat"] is not None]
    # 노드가 2개 미만이면 계산할 거리가 없음
    if len(valid_nodes) < 2: return {}

    df = pd.DataFrame(valid_nodes)
    gdf = gpd.GeoDataFrame(
        valid_nodes,
        geometry=gpd.points_from_xy([n['lng'] for n in valid_nodes], [n['lat'] for n in valid_nodes]),
        crs="EPSG:4326"
    )

    # 2. [변경점] TravelTimeMatrixComputer -> TravelTimeMatrix
    # .compute_travel_times() 호출 삭제 (인스턴스 생성 시 바로 계산됨)
    try:
        matrix = TravelTimeMatrix(
            transport_network,
            origins=gdf,
            destinations=gdf,
            departure=departure_time,
            transport_modes=["WALK", "TRANSIT"]
        )
    except Exception as e:
        print(f"⚠️ 행렬 계산 중 오류: {e}")
        return {}
    
    # 3. 결과 변환 (DataFrame -> Dictionary)
    # 최적화 로직에서 (from, to)로 바로 찾기 위해 딕셔너리로 변환
    r5_travel_times = {}
    
    # 결과 컬럼: 'from_id', 'to_id', 'travel_time'
    for row in matrix.itertuples():
        t = row.travel_time
        if not pd.isna(t):
            # from_id와 to_id는 정수형 인덱스로 변환하여 저장
            r5_travel_times[(int(row.from_id), int(row.to_id))] = int(t)
            
    return r5_travel_times

# ============================================================
# 상세 경로 추출 함수 (수정됨)
# ============================================================
def get_detailed_path_info(start_node, end_node, departure_time):
    # 1. 디버깅 헤더 출력
    print(f"\n{'='*60}")
    print(f"🔍 [디버그 시작] {start_node['name']} ➡️ {end_node['name']}")
    print(f"⏰ 검색 기준 시간(r5py 전달): {departure_time}")
    
    if start_node['lat'] is None or end_node['lat'] is None: 
        print("⚠️ 좌표 없음 -> 함수 종료")
        return ""
    if start_node['lat'] == end_node['lat'] and start_node['lng'] == end_node['lng']: 
        return "이동 없음"

    origin = gpd.GeoDataFrame([start_node], geometry=gpd.points_from_xy([start_node['lng']], [start_node['lat']]), crs="EPSG:4326")
    dest = gpd.GeoDataFrame([end_node], geometry=gpd.points_from_xy([end_node['lng']], [end_node['lat']]), crs="EPSG:4326")

    try:
        # 2. r5py 계산 시도
        print("🚀 r5py 경로 탐색 요청 중...")
        computer = DetailedItineraries(
            transport_network,
            origins=origin,
            destinations=dest,
            departure=departure_time, # 이 시간이 2024년인지 꼭 확인하세요!
            transport_modes=["WALK", "TRANSIT"]
        )
        
        if hasattr(computer, "compute_itineraries"): itineraries = computer.compute_itineraries()
        elif hasattr(computer, "itineraries"): itineraries = computer.itineraries
        else: itineraries = computer

        if itineraries.empty: 
            print("❌ 결과 없음 (Empty DataFrame)")
            return "도보 또는 경로 없음"
        
        # 3. 결과 데이터프레임 분석
        print(f"✅ 경로 발견! (총 {len(itineraries)}개 옵션)")
        print(f"📋 발견된 컬럼 목록: {itineraries.columns.tolist()}")
        
        # 최적 경로 (option 0)
        if 'option' in itineraries.columns:
            best_route = itineraries[itineraries['option'] == 0]
        else:
            best_route = itineraries

        details = []
        
        # 4. 세부 구간(Leg) 데이터 뜯어보기
        for idx, (_, leg) in enumerate(best_route.iterrows()):
            print(f"\n--- [구간 {idx+1}] 데이터 분석 ---")
            
            # (A) 모드 확인
            mode_obj = leg.get('transport_mode')
            mode = str(leg.get('mode', mode_obj))
            print(f"   🔹 Mode (Raw): {mode}")
            
            # (B) 시간 확인
            duration_val = leg.get('travel_time') # 우선순위 1
            if pd.isna(duration_val): duration_val = leg.get('duration') # 우선순위 2
            
            print(f"   🔹 Duration (Raw): {duration_val} (Type: {type(duration_val)})")
            
            # 시간 변환 로직
            if pd.isna(duration_val): duration = 1
            elif hasattr(duration_val, 'total_seconds'): duration = int(duration_val.total_seconds() / 60)
            else: 
                try: duration = int(float(duration_val))
                except: duration = 1
            if duration < 1: duration = 1
            print(f"   🔹 변환된 분(min): {duration}분")

            # (C) 노선 정보 확인
            route_short = leg.get('route_short_name')
            route_id = leg.get('route_id')
            route = leg.get('route')
            print(f"   🔹 Route Info: short='{route_short}', id='{route_id}', route='{route}'")

            # (D) 최종 문자열 조합
            if 'WALK' in str(mode).upper():
                details.append(f"🚶도보 {duration}분")
            else:
                final_route = route_short if not pd.isna(route_short) else (route_id if not pd.isna(route_id) else '대중교통')
                
                stop = leg.get('from_stop_name')
                if pd.isna(stop): stop = leg.get('start_stop_name')
                if pd.isna(stop): stop = '정류장'
                
                details.append(f"🚌[{final_route}] {stop} 승차({duration}분)")
                
        print(f"📝 최종 파싱 결과: {' -> '.join(details)}")
        print(f"{'='*60}\n")
        
        return " -> ".join(details)

    except Exception as e:
        print(f"🔥 [에러 발생] {e}")
        import traceback
        traceback.print_exc()
        return "경로 에러"

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

def optimize_day(places, restaurants, fixed_events, start_time_str, target_date_str, end_time_str=None):
    # 1. 기준 시간 설정
    day_start_dt = datetime.strptime(start_time_str, "%H:%M")
    
    # [핵심] r5py에게 전달할 '가짜 날짜' (GTFS 데이터가 있는 2024년 평일)
    # 2024년 5월 20일 (월요일)로 고정합니다.
    SAFE_GTFS_DATE = "2024-05-20"  
    r5_date_obj = datetime.strptime(SAFE_GTFS_DATE, "%Y-%m-%d")
    r5_departure_dt = datetime.combine(r5_date_obj, day_start_dt.time())

    # 사용자에게 보여줄 '진짜 날짜' (출력용)
    display_date_obj = datetime.strptime(target_date_str, "%Y-%m-%d")
    display_start_dt = datetime.combine(display_date_obj, day_start_dt.time())

    # 2. Horizon 계산
    if end_time_str:
        day_end_dt = datetime.strptime(end_time_str, "%H:%M")
        max_horizon_minutes = int((day_end_dt - day_start_dt).total_seconds() / 60)
        if max_horizon_minutes < 0: max_horizon_minutes = 24 * 60 
    else:
        max_horizon_minutes = 24 * 60 

    # 3. 노드 생성
    nodes = build_nodes(places, restaurants, fixed_events, day_start_dt)
    n = len(nodes)
    for idx, node in enumerate(nodes): node['id'] = idx

    # 4. Matrix 계산 (2024년 날짜 사용)
    # 여기서 2024년 버스 시간을 가져오므로 이동 시간이 줄어듭니다.
    r5_travel_times = get_r5py_matrix(nodes, r5_departure_dt)

    # 5. OR-Tools Matrix 채우기
    time_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            travel_val = r5_travel_times.get((i, j))
            
            # 대중교통이 없으면 도보 시간으로 대체
            if travel_val is None:
                travel_val = travel_minutes(nodes[i], nodes[j])
            
            is_fixed_involved = (nodes[i]["type"] == "fixed" or nodes[j]["type"] == "fixed")
            if is_fixed_involved:
                if nodes[i]["type"] == "depot" and nodes[j]["type"] == "fixed": travel_val = 0 
                else: travel_val = max(travel_val, 20)
            
            time_matrix[i][j] = nodes[i]["stay"] + int(travel_val)

    # 6. Solver 설정
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def time_callback(from_idx, to_idx): return time_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]
    transit_callback = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)
    routing.AddDimension(transit_callback, 30, max_horizon_minutes, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    
    penalty_spot = 100000
    penalty_meal = 1000000 
    solver = routing.solver()

    for i, node in enumerate(nodes):
        index = manager.NodeToIndex(i)
        if node["type"] == "depot": continue
        time_windows = build_time_windows(nodes, day_start_dt)
        window = time_windows[i]
        
        if node["type"] == "fixed":
            safe_start = max(0, min(window[0], max_horizon_minutes))
            safe_end = max(safe_start, min(window[1], max_horizon_minutes))
            if safe_end < safe_start: safe_end = safe_start + 10
            time_dim.CumulVar(index).SetRange(safe_start, safe_end)
            continue 
            
        raw_start, raw_end = window[0], window[1]
        overlap_start = max(0, raw_start)
        overlap_end = min(max_horizon_minutes, raw_end)
        
        if not (overlap_start <= overlap_end):
            routing.AddDisjunction([index], 0) 
            solver.Add(routing.VehicleVar(index) == -1)
            continue
        time_dim.CumulVar(index).SetRange(overlap_start, overlap_end)
        
        if node["type"] == "spot": routing.AddDisjunction([index], penalty_spot)
        elif node["type"] in ["lunch", "dinner"]: routing.AddDisjunction([index], penalty_meal)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 2 

    solution = routing.SolveWithParameters(search_params)
    if not solution: return []

    # 7. 결과 생성
    index = routing.Start(0)
    timeline = []
    prev_node = None
    
    # [매우 중요] 상세 경로 검색을 위한 시간 변수를 2024년으로 초기화
    current_r5_dt = r5_departure_dt

    while not routing.IsEnd(index):
        node_idx = manager.IndexToNode(index)
        node = nodes[node_idx]
        
        t_start_min = solution.Value(time_dim.CumulVar(index))

        # (1) 화면 표시용 시간 (2025/2026년 기준)
        if node["type"] == "fixed":
            time_str = node["orig_time_str"]
        elif node["type"] == "depot":
            time_str = "출발"
        else:
            visit_start = display_start_dt + timedelta(minutes=t_start_min)
            visit_end = visit_start + timedelta(minutes=node["stay"])
            time_str = f"{visit_start.strftime('%H:%M')} - {visit_end.strftime('%H:%M')}"

        # (2) 상세 경로 검색 (2024년 날짜 사용)
        transit_info = ""
        if prev_node:
            # 여기서 current_r5_dt (2024년)를 넘겨줘야 버스가 나옵니다!
            transit_info = get_detailed_path_info(prev_node, node, current_r5_dt)
        
        timeline.append({
            "name": node["name"],
            "category": node["category"],
            "time": time_str,
            "transit_info": transit_info 
        })

        # (3) 다음 이동을 위해 2024년 기준 시간 업데이트
        # r5py 출발 시간 = 2024년 시작 시간 + 누적 경과 시간 + 체류 시간
        current_r5_dt = r5_departure_dt + timedelta(minutes=t_start_min + node["stay"])
        
        prev_node = node
        index = solution.Value(routing.NextVar(index))

    return timeline

# ============================================================
# 일정 타임라인 json에 추가 (실행부 수정)
# ============================================================

if __name__ == "__main__":
    # result = extract_json(response.text)
    # with open("result.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 또는 기존 result 사용
    
    result = json.load(open("result.json", "r", encoding="utf-8"))
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
        
        # [중요] 현재 루프의 날짜 문자열 (YYYY-MM-DD)
        day_str = current_date.strftime("%Y-%m-%d")
        day_fixed_events = get_fixed_events_for_day(FIXED_EVENTS, day_str)

        # 1. 시작/종료 시간 결정
        if i == 0:
            todays_start = first_day_start_str
        else:
            todays_start = default_start_str

        if i == total_days - 1:
            todays_end = last_day_end_str
        else:
            todays_end = default_end_str

        timeset = f"{todays_start} 시작" + (f" ~ {todays_end} 종료" if todays_end else "")
        print(timeset)

        # 2. 최적화 실행 (target_date_str 추가 전달)
        timeline = optimize_day(
            places=day_places,
            restaurants=day_restaurants,
            fixed_events=day_fixed_events,
            start_time_str=todays_start,
            target_date_str=day_str,  # [수정] 날짜 정보 전달
            end_time_str=todays_end
        )

        result["plans"][day_key]["timeset"] = timeset
        result["plans"][day_key]["timeline"] = timeline

        if not timeline:
            print("   ⚠ 조건 만족하는 일정 생성 실패")
        else:
            for t in timeline:
                transit_msg = f" -> {t['transit_info']}" if t['transit_info'] else ""
                print(f"   [{t['time']}] {t['name']} ({t['category']}){transit_msg}")

        current_date += timedelta(days=1)

    print("\n====== 최종 결과 ======\n")
    file_path = "result_timeline.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 일정이 '{file_path}' 파일로 저장되었습니다.")