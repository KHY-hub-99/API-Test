import os
os.environ["JAVA_OPTS"] = "-Xmx8G"
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-21.0.10"

from google import genai
import zipfile
import json
import pandas as pd
import geopandas as gpd
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import time
from r5py import TransportNetwork, TravelTimeMatrix, DetailedItineraries, TransportMode

# ============================================================
# 1. 환경 설정 및 전역 상수
# ============================================================ 

# API 키 설정
load_dotenv()
API_KEY = os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY)

# 캐시 저장소
DETAILED_PATH_CACHE = {}

# 도보 이동 제한 (km -> 분 환산 기준 등)
WALK_ONLY_THRESHOLD_MIN = 12   
WALK_ONLY_THRESHOLD_MAX = 18   

MAX_TRANSFERS = 3
MAX_TRAVEL_TIME_MIN = 90

# 시간 윈도우 설정
LUNCH_WINDOW = ("11:20", "13:20")
DINNER_WINDOW = ("17:40", "19:30")

# 장소별 체류 시간
stay_time_map = {
    "관광지": 90, "카페": 50, "식당": 70, 
    "박물관": 120, "공원": 60, "시장": 80, "숙박": 0
}

# 데이터 파일 경로
osm_file = "./data/seoul_osm_v.pbf"
gtfs_files = ["./data/seoul_area_gtfs.zip"]

# ============================================================
# 2. 유틸리티 함수
# ============================================================
def parse_time(t):
    return datetime.strptime(t, "%H:%M")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def approx_walk_minutes(start, end):
    dist_km = haversine(start["lat"], start["lng"], end["lat"], end["lng"])
    return dist_km * 12

def dynamic_walk_threshold(dist_km):
    if dist_km < 0.6: return WALK_ONLY_THRESHOLD_MAX
    elif dist_km < 1.2: return 15
    else: return WALK_ONLY_THRESHOLD_MIN

def travel_minutes(p1, p2):
    if p1["lat"] is None or p2["lat"] is None: return 0
    dist = haversine(p1["lat"], p1["lng"], p2["lat"], p2["lng"])
    return int(dist / 30 * 60)

def get_fixed_events_for_day(fixed_events, target_date):
    return [e for e in fixed_events if e["date"] == target_date]

def duration_to_minutes(val):
    if val is None or pd.isna(val): return 0
    if hasattr(val, "total_seconds"): return int(val.total_seconds() / 60)
    if isinstance(val, str) and "day" in val:
        try: return int(pd.to_timedelta(val).total_seconds() / 60)
        except: return 0
    try: return int(float(val))
    except: return 0

def extract_json(text):
    if not text:
        raise ValueError("Gemini 응답이 비어있습니다.")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == -1:
        raise ValueError("JSON 파싱 실패:\n" + text)
    return json.loads(text[start:end])

# ============================================================
# 3. 교통 데이터 로드 (GTFS & OSM)
# ============================================================

# 3-1. TransportNetwork (r5py)
pickle_path = "./data/seoul_tn_cached.pkl"
if os.path.exists(pickle_path):
    print(f"📦 Pickle 파일 로드: {pickle_path}")
    start_load = time.time()
    transport_network = TransportNetwork.__new__(TransportNetwork)
    transport_network._transport_network = TransportNetwork._load_pickled_transport_network(self=TransportNetwork, path=pickle_path)
    print(f"⏱ 로드 완료: {round(time.time() - start_load, 2)}초")
else:
    print("🚀 TransportNetwork 생성 중... (시간 소요)")
    start_tn = time.time()
    transport_network = TransportNetwork(osm_file, gtfs_files)
    transport_network._save_pickled_transport_network(path=pickle_path, transport_network=transport_network)
    print(f"⏱ 생성 완료: {round(time.time() - start_tn, 2)}초")

# 3-2. Stops 로드 & 매핑
print("🚏 정류장 데이터 로드 중...")
with zipfile.ZipFile(gtfs_files[0]) as z:
    with z.open("stops.txt") as f:
        stops_df = pd.read_csv(f, dtype={'stop_id': str})

STOP_ID_TO_NAME = {str(row['stop_id']).strip(): str(row['stop_name']).strip() for _, row in stops_df.iterrows()}

def get_stop_name(stop_id):
    if pd.isna(stop_id): return None
    safe_id = str(stop_id).strip()
    try: safe_id = str(int(float(stop_id))).strip()
    except: pass
    name = STOP_ID_TO_NAME.get(safe_id)
    if not name and len(safe_id) < 5: name = STOP_ID_TO_NAME.get(safe_id.zfill(5))
    return name

# 3-3. Routes & Types 로드 (간선/지선 필터링용)
ROUTE_TYPE_MAP = {
    11: "간선", 12: "지선", 13: "순환", 14: "광역", 15: "마을",
    3: "버스", 2: "지하철", 109: "지하철"
}

def get_route_type_str(type_code):
    return ROUTE_TYPE_MAP.get(type_code, "")

print("🚌 노선 데이터 로드 중...")
with zipfile.ZipFile(gtfs_files[0]) as z:
    with z.open("routes.txt") as f:
        routes_df = pd.read_csv(f)

# ID -> 이름
ROUTE_ID_TO_NAME = dict(zip(routes_df["route_id"].astype(str), routes_df["route_short_name"].astype(str)))
# ID -> 타입 (숫자)
ROUTE_ID_TO_TYPE = dict(zip(routes_df["route_id"].astype(str), routes_df["route_type"].fillna(3).astype(int)))

def get_route_name(route_id):
    if pd.isna(route_id): return None
    try: safe_id = str(int(float(route_id)))
    except: safe_id = str(route_id)
    return ROUTE_ID_TO_NAME.get(safe_id)

# 3-4. 정류장별 노선 매핑 (병렬 노선 탐색용)
STOP_ROUTE_MAP = {}
try:
    print("🔄 정류장-노선 매핑 데이터 생성 중...")
    start_map = time.time()
    with zipfile.ZipFile(gtfs_files[0]) as z:
        with z.open("trips.txt") as f:
            trips = pd.read_csv(f, usecols=["route_id", "trip_id"])
        with z.open("stop_times.txt") as f:
            stop_times = pd.read_csv(f, usecols=["trip_id", "stop_id"], dtype={"stop_id": str})
    
    merged = stop_times.merge(trips, on="trip_id")
    grouped = merged.groupby("stop_id")["route_id"].unique()
    STOP_ROUTE_MAP = {str(k).strip(): set(v) for k, v in grouped.items()}
    print(f"✅ 매핑 완료 ({round(time.time() - start_map, 2)}초)")
except Exception as e:
    print(f"⚠️ 정류장 매핑 실패: {e}")

# ============================================================
# 4. 경로 계산 및 상세화 (r5py)
# ============================================================
def get_r5py_matrix(nodes, departure_time):
    valid_nodes = [n for n in nodes if n["lat"] is not None]
    if len(valid_nodes) < 2: return {}

    gdf = gpd.GeoDataFrame(
        valid_nodes,
        geometry=gpd.points_from_xy([n['lng'] for n in valid_nodes], [n['lat'] for n in valid_nodes]),
        crs="EPSG:4326"
    )

    try:
        matrix = TravelTimeMatrix(
            transport_network, origins=gdf, destinations=gdf, departure=departure_time,
            transport_modes=[TransportMode.WALK, TransportMode.TRANSIT]
        )
        r5_travel_times = {}
        for row in matrix.itertuples():
            if not pd.isna(row.travel_time):
                r5_travel_times[(int(row.from_id), int(row.to_id))] = int(row.travel_time)
        return r5_travel_times
    except Exception as e:
        print(f"⚠️ 행렬 계산 오류: {e}")
        return {}

def make_cache_key(start_node, end_node, departure_time):
    return (start_node.get("id"), end_node.get("id"), departure_time.hour)

def get_all_detailed_paths(trip_legs, departure_time):
    if not trip_legs: return {}
    path_map = {}
    origins_list, dests_list = [], []

    # 1. 선 필터링 (너무 가까우면 도보 처리) -> 리스트로 저장
    for start_node, end_node in trip_legs:
        if start_node['id'] == end_node['id']: continue
        
        dist_val = haversine(start_node["lat"], start_node["lng"], end_node["lat"], end_node["lng"])
        approx_min = dist_val * 15
        # [판단] 거리가 짧으면(dynamic_walk_threshold 이하), 비싼 r5py 계산을 안 하고 바로 결정해버림
        if approx_min <= dynamic_walk_threshold(dist_val):
            path_map[(start_node['id'], end_node['id'])] = [f"도보 : {round(approx_min)}분"]
            continue

        cache_key = make_cache_key(start_node, end_node, departure_time)
        if cache_key in DETAILED_PATH_CACHE:
            path_map[(start_node['id'], end_node['id'])] = DETAILED_PATH_CACHE[cache_key]
            continue

        origins_list.append(start_node)
        dests_list.append(end_node)

    if not origins_list: return path_map

    # 2. r5py 상세 경로 요청 (기존 코드 유지)
    origins_gdf = gpd.GeoDataFrame(origins_list, geometry=gpd.points_from_xy([n['lng'] for n in origins_list], [n['lat'] for n in origins_list]), crs="EPSG:4326")
    origins_gdf["id"] = [n["id"] for n in origins_list]
    dests_gdf = gpd.GeoDataFrame(dests_list, geometry=gpd.points_from_xy([n['lng'] for n in dests_list], [n['lat'] for n in dests_list]), crs="EPSG:4326")
    dests_gdf["id"] = [n["id"] for n in dests_list]

    try:
        computer = DetailedItineraries(
            transport_network, origins=origins_gdf, destinations=dests_gdf, departure=departure_time,
            transport_modes=[TransportMode.WALK, TransportMode.TRANSIT],
            max_public_transport_rides=MAX_TRANSFERS, max_time=timedelta(minutes=MAX_TRAVEL_TIME_MIN)
        )
    except: return path_map

    if computer.empty: return path_map

    mode_col = 'transport_mode' if 'transport_mode' in computer.columns else 'mode'

    def get_val(row, candidates, default=None):
        for c in candidates:
            if c in row.index and pd.notna(row[c]): return str(row[c]).strip()
        return default

    # 3. 상세 경로 파싱
    for (from_id, to_id), group in computer.groupby(['from_id', 'to_id']):
        best_route, best_time = None, float("inf")
        # 가장 빠른 경로 선택 로직 (기존 유지)
        for _, opt in group.groupby("option"):
            total = sum(max(1, duration_to_minutes(get_val(leg, ['travel_time', 'duration'], 0))) for _, leg in opt.iterrows())
            if total < best_time: best_time, best_route = total, opt
        
        if best_route is None: continue

        segments = [] # 개별 스텝을 담을 리스트
        for _, leg in best_route.iterrows():
            raw_mode = str(leg[mode_col]).upper()
            dur = max(1, duration_to_minutes(get_val(leg, ['travel_time', 'duration'], 0)))

            if 'WALK' in raw_mode:
                segments.append(f"도보 : {dur}분")
                continue

            from_stop_id = str(get_val(leg, ['start_stop_id', 'from_stop_id'])).strip()
            to_stop_id = str(get_val(leg, ['end_stop_id', 'to_stop_id'])).strip()
            from_stop = get_stop_name(from_stop_id) or "정류장"
            to_stop = get_stop_name(to_stop_id) or "정류장"
            
            current_route_id = str(get_val(leg, ['route_id'])).strip()
            current_route_name = get_route_name(current_route_id) or '대중교통'
            mode_label = "지하철" if any(x in raw_mode for x in ['SUBWAY', 'RAIL', 'METRO']) else "버스"
            
            final_route_str = ""

            # [수정 요청 1] 같은 구간을 가는 다른 모든 버스 번호 찾기
            if mode_label == "버스" and STOP_ROUTE_MAP:
                routes_at_start = STOP_ROUTE_MAP.get(from_stop_id, set())
                routes_at_end = STOP_ROUTE_MAP.get(to_stop_id, set())
                common_route_ids = routes_at_start.intersection(routes_at_end)
                
                # 현재 탑승한 노선도 포함 보장
                if current_route_id not in common_route_ids: common_route_ids.add(current_route_id)

                bus_names = []
                for rid in common_route_ids:
                    rname = get_route_name(rid)
                    if rname:
                        bus_names.append(rname)
                
                # 번호순 정렬 (깔끔한 출력을 위해)
                bus_names.sort()
                
                if not bus_names:
                    final_route_str = current_route_name
                else:
                    # [종로02, 1020, 7025] 형태로 나열
                    final_route_str = ", ".join(bus_names)
            else:
                final_route_str = current_route_name

            segments.append(f"[{mode_label}][{final_route_str}] : {from_stop} → {to_stop} : {dur}분")

        # [수정 요청 2] 문자열 join을 하지 않고 리스트 그대로 저장
        DETAILED_PATH_CACHE[make_cache_key({"id":from_id}, {"id":to_id}, departure_time)] = segments
        path_map[(int(from_id), int(to_id))] = segments

    return path_map

# ============================================================
# 5. 노드 빌더 & 최적화 (OR-Tools)
# ============================================================
def build_fixed_nodes(fixed_events, day_start_dt):
    nodes = []
    BUFFER = 15
    for event in fixed_events:
        event_start = parse_time(event["start"])
        event_end = parse_time(event["end"])
        orig_start_min = int((event_start - day_start_dt).total_seconds() / 60)
        orig_end_min = int((event_end - day_start_dt).total_seconds() / 60)
        
        raw_start_min = orig_start_min - BUFFER
        buffered_start_min = max(0, raw_start_min)
        final_stay = (orig_end_min - orig_start_min) + (orig_start_min - buffered_start_min) + BUFFER

        nodes.append({
            "name": event["title"], "category": "고정일정", "lat": None, "lng": None,
            "stay": final_stay, "type": "fixed", "window": (buffered_start_min, buffered_start_min + 10),
            "orig_time_str": f"{event['start']} - {event['end']}"
        })
    return nodes

def build_nodes(places, restaurants, fixed_events, day_start_dt):
    nodes = []
    first_place = places[0] if places else {"lat": 37.5665, "lng": 126.9780}
    nodes.append({"name": "시작점", "category": "출발", "lat": first_place["lat"], "lng": first_place["lng"], "stay": 0, "type": "depot"})

    for p in places:
        nodes.append({"name": p["name"], "category": p["category"], "lat": p["lat"], "lng": p["lng"], "stay": stay_time_map.get(p["category"], 60), "type": "spot"})

    if restaurants:
        nodes.append({"name": restaurants[0]["name"], "category": "식당", "lat": restaurants[0]["lat"], "lng": restaurants[0]["lng"], "stay": 70, "type": "lunch"})
        nodes.append({"name": restaurants[1]["name"], "category": "식당", "lat": restaurants[1]["lat"], "lng": restaurants[1]["lng"], "stay": 70, "type": "dinner"})

    nodes.extend(build_fixed_nodes(fixed_events, day_start_dt))
    return nodes

def build_time_windows(nodes, day_start_dt):
    windows = []
    def get_rel(t_str): return int((parse_time(t_str) - day_start_dt).total_seconds() / 60)
    
    l_s, l_e = get_rel(LUNCH_WINDOW[0]), get_rel(LUNCH_WINDOW[1])
    d_s, d_e = get_rel(DINNER_WINDOW[0]), get_rel(DINNER_WINDOW[1])

    for n in nodes:
        if n["type"] == "lunch": windows.append((l_s, l_e))
        elif n["type"] == "dinner": windows.append((d_s, d_e))
        elif n["type"] == "fixed": windows.append(n["window"])
        else: windows.append((0, 24 * 60))
    return windows

def optimize_day(places, restaurants, fixed_events, start_time_str, target_date_str, end_time_str=None):
    TRAVEL_BUFFER = 5
    day_start_dt = datetime.strptime(start_time_str, "%H:%M")
    
    # r5py 계산 기준 날짜 설정
    SAFE_GTFS_DATE = target_date_str
    r5_departure_dt = datetime.combine(datetime.strptime(SAFE_GTFS_DATE, "%Y-%m-%d"), datetime.strptime("11:00", "%H:%M").time())
    display_start_dt = datetime.combine(datetime.strptime(target_date_str, "%Y-%m-%d"), day_start_dt.time())

    # Horizon 계산
    max_horizon_minutes = 24 * 60
    if end_time_str:
        diff = int((datetime.strptime(end_time_str, "%H:%M") - day_start_dt).total_seconds() / 60)
        if diff > 0: max_horizon_minutes = diff

    nodes = build_nodes(places, restaurants, fixed_events, day_start_dt)
    for idx, node in enumerate(nodes): node["id"] = idx
    n = len(nodes)

    # 매트릭스 계산
    r5_travel_times = get_r5py_matrix(nodes, r5_departure_dt)
    time_matrix = [[0]*n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            val = r5_travel_times.get((i, j))
            if val is None: val = travel_minutes(nodes[i], nodes[j])
            
            # 고정일정 이동시간 보정
            if (nodes[i]["type"]=="fixed" or nodes[j]["type"]=="fixed"):
                if not (nodes[i]["type"]=="depot" and nodes[j]["type"]=="fixed"):
                    val = max(val, 20)
            
            time_matrix[i][j] = nodes[i]["stay"] + int(val)

    # OR-Tools Solver
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        return time_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_callback = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)
    routing.AddDimension(transit_callback, 30, max_horizon_minutes, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    time_windows = build_time_windows(nodes, day_start_dt)
    solver = routing.solver()

    for i, node in enumerate(nodes):
        index = manager.NodeToIndex(i)
        if node["type"] == "depot": continue
        
        window = time_windows[i]
        if node["type"] == "fixed":
            time_dim.CumulVar(index).SetRange(max(0, window[0]), min(max_horizon_minutes, window[1]))
            continue

        overlap_start, overlap_end = max(0, window[0]), min(max_horizon_minutes, window[1])
        if overlap_start > overlap_end:
            routing.AddDisjunction([index], 0)
            solver.Add(routing.VehicleVar(index) == -1)
        else:
            time_dim.CumulVar(index).SetRange(overlap_start, overlap_end)
            penalty = 1000000 if node["type"] in ["lunch", "dinner"] else 100000
            routing.AddDisjunction([index], penalty)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 1

    solution = routing.SolveWithParameters(search_params)
    if not solution: return []

    index = routing.Start(0)
    visited_nodes = []
    while not routing.IsEnd(index):
        node_idx = manager.IndexToNode(index)
        nodes[node_idx]['arrival_min'] = solution.Value(time_dim.CumulVar(index))
        visited_nodes.append(nodes[node_idx])
        index = solution.Value(routing.NextVar(index))

    trip_legs = [(visited_nodes[i], visited_nodes[i+1]) for i in range(len(visited_nodes)-1)]
    
    print("🚀 상세 경로 계산 중...")
    start_path_time = time.time()
    path_map = get_all_detailed_paths(trip_legs, r5_departure_dt)
    end_path_time = time.time()
    print(f"⏱ 상세 경로 계산 완료: {round(end_path_time - start_path_time, 2)}초")

    timeline = []
    actual_visits = [n for n in visited_nodes if n["type"] != "depot"]

    # 첫 장소의 시작 시간 보장
    current_time_cursor = display_start_dt + timedelta(minutes=actual_visits[0]['arrival_min'])

    for i, node in enumerate(actual_visits):
        transit_info = []
        travel_min = 0 # 실제 텍스트상 이동 시간
        
        # 1. 이동 시간 및 텍스트 계산
        if i > 0:
            prev = actual_visits[i-1]
            dist = haversine(prev['lat'], prev['lng'], node['lat'], node['lng'])
            
            # 상세 경로 가져오기 (List[str])
            r5_path_list = path_map.get((prev['id'], node['id']))
            
            # 이동 시간 파싱 (텍스트에서 분 추출) 또는 거리 기반 계산
            if r5_path_list:
                transit_info = r5_path_list
                # 텍스트 내의 모든 "X분"을 합산 (예: "도보 4분", "버스 10분" 등)
                import re
                for segment in r5_path_list:
                    # "4분", "12분" 등 숫자 추출
                    mins = re.findall(r'(\d+)분', segment)
                    for m in mins:
                        travel_min += int(m)
            else:
                # 경로가 없으면 직선거리 기준
                travel_min = int(dist * 12)
                if dist < 0.1:
                    transit_info = ["도보 이동 (100m 이내)"]
                    travel_min = 0 # 건물 내 이동은 시간 거의 안 씀
                else:
                    transit_info = [f"도보 : {travel_min}분"]

        # 2. 타임라인 시간 확정 (Logic: 이전 종료 + 이동 시간)
        if node["type"] == "fixed":
            # 고정 일정은 원래 시간 엄수
            time_parts = node["orig_time_str"].split(" - ")
            start_dt = datetime.strptime(f"{target_date_str} {time_parts[0]}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{target_date_str} {time_parts[1]}", "%Y-%m-%d %H:%M")
            
            # 만약 도착했는데 시간이 남으면 '대기' 발생
            wait_min = int((start_dt - current_time_cursor).total_seconds() / 60)
            if wait_min > 0:
                transit_info.append(f"(대기 {wait_min}분)")
            
            current_time_cursor = end_dt # 종료 시간으로 커서 이동
            time_str = node["orig_time_str"]
            
        else:
            start_dt = current_time_cursor + timedelta(minutes=travel_min)
            end_dt = start_dt + timedelta(minutes=node["stay"])
            
            time_str = f"{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}"
            current_time_cursor = end_dt # 다음을 위해 커서 업데이트

        # 결과 저장
        timeline.append({
            "name": node["name"], 
            "category": node["category"], 
            "time": time_str, 
            "transit_to_here": transit_info 
        })

    return timeline

# ============================================================
# 6. 메인 실행부 (통합)
# ============================================================
if __name__ == "__main__":
    # # 1. 엑셀 및 기본 정보 로드
    # print("📂 장소 데이터 로드 중 (places_3000.xlsx)...")
    # try:
    #     df = pd.read_excel("places_3000.xlsx")
    # except FileNotFoundError:
    #     print("❌ 'places_3000.xlsx' 파일이 없습니다.")
    #     exit()

    # area = input("여행할 지역을 입력하세요 (예: 종로구): ")
    
    # # 2. 장소 필터링
    # filtered_spot = df[(df["area"] == f"{area}") & (df["category"] != "식당")][["name", "lat", "lng"]]
    # filtered_restaurant = df[(df["area"] == f"{area}") & (df["category"] == "식당")][["name", "lat", "lng"]]
    # filtered_accom = df[(df["area"] == f"{area}") & (df["category"] == "숙박")][["name", "lat", "lng"]]

    # places = filtered_spot.to_dict(orient="records")
    # restaurants = filtered_restaurant.to_dict(orient="records")
    # accommodations = filtered_accom.to_dict(orient="records")

    # 3. 날짜 입력
    start_date = input("여행 시작 일자 (예: 2026-01-20): ")
    end_date = input("여행 종료 일자 (예: 2026-01-25): ")
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days + 1
    print(f"총 여행 일수: {days}일")

    # # 4. Gemini API 호출 (1차 계획 생성)
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
    # 너는 서울 여행 장소 추천기다. 반드시 아래 JSON 스키마 형식으로만 출력한다.
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

    # print("🤖 Gemini가 초기 계획을 생성하고 있습니다...")
    # prompt = system_prompt + "\n\n" + json.dumps(user_prompt, ensure_ascii=False)
    
    # start_time = time.time()
    # response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
    # print(f"⏱ Gemini 응답 시간: {round(time.time() - start_time, 3)}초")

    # try:
    #     result = extract_json(response.text)
    #     # result.json 저장 (백업용)
    #     with open("result.json", "w", encoding="utf-8") as f:
    #         json.dump(result, f, ensure_ascii=False, indent=2)
    # except Exception as e:
    #     print(f"❌ JSON 파싱 실패: {e}")
    #     exit()

    result = json.load(open("result.json", "r", encoding="utf-8"))

    # 5. 세부 일정 설정
    first_day_start_str = input("여행 첫날 시작 시간 (예: 14:00) : ").strip() or "10:00"
    last_day_end_str = input("여행 마지막 날 종료 시간 (예: 18:00) : ").strip() or "21:00"
    default_start_str = "10:00"
    default_end_str = "21:00"

    FIXED_EVENTS = []
    if input("고정 일정이 있나요? (y/n): ").strip().lower() == "y":
        while True:
            FIXED_EVENTS.append({
                "date": input("날짜 (YYYY-MM-DD): "),
                "title": input("제목: "),
                "start": input("시작(HH:MM): "),
                "end": input("종료(HH:MM): ")
            })
            if input("더 추가하시겠습니까? (y/n): ").lower() != "y": break

    # 6. 최적화 실행 (Day loop)
    plans = result["plans"]
    current_date = start
    day_keys = list(plans.keys())

    for i, day_key in enumerate(day_keys):
        print(f"\n📅 {day_key} ({current_date.strftime('%Y-%m-%d')}) 최적화 진행...")
        
        todays_start = first_day_start_str if i == 0 else default_start_str
        todays_end = last_day_end_str if i == len(day_keys)-1 else default_end_str
        
        start_opt_time = time.time()
        timeline = optimize_day(
            places=plans[day_key]["route"],
            restaurants=plans[day_key]["restaurants"],
            fixed_events=get_fixed_events_for_day(FIXED_EVENTS, current_date.strftime("%Y-%m-%d")),
            start_time_str=todays_start,
            target_date_str=current_date.strftime("%Y-%m-%d"),
            end_time_str=todays_end
        )
        end_opt_time = time.time()
        print(f"⏱ {day_key} 최적화 완료: {round(end_opt_time - start_opt_time, 2)}초")
        
        result["plans"][day_key]["timeline"] = timeline
        
        if timeline:
            for t in timeline:
                if t.get('transit_to_here'):
                    # 만약 문자열로 들어왔다면 리스트로 변환해서 처리 (안전장치)
                    infos = t['transit_to_here']
                    if isinstance(infos, str):
                        infos = [infos]
                        
                    for step in infos:
                        print(f"    ▼ {step}")
                
                print(f"  [{t['time']}] {t['name']} ({t['category']})")

        current_date += timedelta(days=1)

    # 7. 최종 결과 저장
    with open("result_timeline.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 최종 일정이 'result_timeline.json' 파일로 저장되었습니다.")