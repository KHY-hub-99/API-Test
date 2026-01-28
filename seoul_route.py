import os
import multiprocessing

available_cores = multiprocessing.cpu_count()
JAVA_PARALLELISM = 2
if available_cores > JAVA_PARALLELISM:
    JAVA_PARALLELISM = JAVA_PARALLELISM
else:
    JAVA_PARALLELISM = available_cores
print(f"⚙️  설정된 사용 코어 수: {JAVA_PARALLELISM}개")
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-21.0.10"
os.environ["JAVA_OPTS"] = f"-Xmx8G -Djava.util.concurrent.ForkJoinPool.common.parallelism={JAVA_PARALLELISM}"

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
import re
from r5py import TransportNetwork, TravelTimeMatrix, DetailedItineraries, TransportMode
import pickle
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# 1. 환경 설정 및 전역 상수
# ============================================================ 
# API 키 설정
load_dotenv()
API_KEY = os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY)

# 캐시 저장소
DETAILED_PATH_CACHE = {}

# 폴백(좌표 없는 경우) 이동 시간 설정(분)
FALLBACK_MOVE_MIN = 30

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
    # start/end가 좌표 없음(None)일 수 있으므로 안전 처리
    if not start or not end or start.get("lat") is None or end.get("lat") is None:
        return FALLBACK_MOVE_MIN
    dist_km = haversine(start["lat"], start["lng"], end["lat"], end["lng"])
    return dist_km * 15

def dynamic_walk_threshold(dist_km):
    if dist_km < 0.6: return WALK_ONLY_THRESHOLD_MAX
    elif dist_km < 1.2: return 15
    else: return WALK_ONLY_THRESHOLD_MIN

def travel_minutes(p1, p2):
    # 좌표가 없으면 0을 반환(상위 로직에서 고정일정 보정으로 최소 시간 적용됨)
    if p1 is None or p2 is None or p1.get("lat") is None or p2.get("lat") is None: return 0
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

# 3-1. TransportNetwork (기존 유지)
pickle_path = "./data/seoul_tn_cached.pkl"
if os.path.exists(pickle_path):
    print(f"📦 TransportNetwork 로드 중...")
    transport_network = TransportNetwork.__new__(TransportNetwork)
    transport_network._transport_network = TransportNetwork._load_pickled_transport_network(self=TransportNetwork, path=pickle_path)
else:
    print("🚀 TransportNetwork 생성 중...")
    transport_network = TransportNetwork(osm_file, gtfs_files)
    transport_network._save_pickled_transport_network(path=pickle_path, transport_network=transport_network)

# 3-2 & 3-3. 메타데이터(Stop/Route) 고속 로드 (Pickle 적용)
meta_cache_path = "./data/metadata_cache.pkl"

if os.path.exists(meta_cache_path):
    print("⚡ 메타데이터 캐시 로드 중...")
    with open(meta_cache_path, "rb") as f:
        meta_data = pickle.load(f)
        STOP_ID_TO_NAME = meta_data["stops"]
        ROUTE_ID_TO_NAME = meta_data["routes"]
        STOP_ROUTE_MAP = meta_data["stop_route_map"]
else:
    print("🐢 메타데이터 생성 중 (최초 1회만 느림)...")
    # Stops
    with zipfile.ZipFile(gtfs_files[0]) as z:
        with z.open("stops.txt") as f:
            stops_df = pd.read_csv(f, dtype={'stop_id': str})
    STOP_ID_TO_NAME = {str(row['stop_id']).strip(): str(row['stop_name']).strip() for _, row in stops_df.iterrows()}
    
    # Routes
    with zipfile.ZipFile(gtfs_files[0]) as z:
        with z.open("routes.txt") as f:
            routes_df = pd.read_csv(f)
    ROUTE_ID_TO_NAME = dict(zip(routes_df["route_id"].astype(str), routes_df["route_short_name"].astype(str)))
    
    # Stop-Route Map
    try:
        with zipfile.ZipFile(gtfs_files[0]) as z:
            with z.open("trips.txt") as f:
                trips = pd.read_csv(f, usecols=["route_id", "trip_id"])
            with z.open("stop_times.txt") as f:
                stop_times = pd.read_csv(f, usecols=["trip_id", "stop_id"], dtype={"stop_id": str})
        merged = stop_times.merge(trips, on="trip_id")[["stop_id", "route_id"]].drop_duplicates()
        grouped = merged.groupby("stop_id")["route_id"].apply(set)
        STOP_ROUTE_MAP = grouped.to_dict()
    except Exception as e:
        print(f"⚠️ 매핑 실패: {e}")
        STOP_ROUTE_MAP = {}

    # 캐시 저장
    with open(meta_cache_path, "wb") as f:
        pickle.dump({
            "stops": STOP_ID_TO_NAME,
            "routes": ROUTE_ID_TO_NAME,
            "stop_route_map": STOP_ROUTE_MAP
        }, f)

# Helper 함수들 (기존 유지)
def get_stop_name(stop_id):
    if pd.isna(stop_id): return None
    safe_id = str(stop_id).strip()
    try: safe_id = str(int(float(stop_id))).strip()
    except: pass
    name = STOP_ID_TO_NAME.get(safe_id)
    if not name and len(safe_id) < 5: name = STOP_ID_TO_NAME.get(safe_id.zfill(5))
    return name

def get_route_name(route_id):
    if pd.isna(route_id): return None
    try: safe_id = str(int(float(route_id)))
    except: safe_id = str(route_id)
    return ROUTE_ID_TO_NAME.get(safe_id)

# ============================================================
# 4. 경로 계산 및 상세화 (r5py) - 안전성 보강
# ============================================================

def get_r5py_matrix(nodes, departure_time):
    valid_nodes = [n for n in nodes if n.get("lat") is not None]
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
    # 수정 전: ID만 사용 -> 날짜가 달라도 ID가 같으면 충돌 발생
    # return (s_id, e_id, int(departure_time.hour))

    # 수정 후: '장소 이름'을 포함하여 유일성 보장
    s_name = start_node.get("name", str(start_node.get("id")))
    e_name = end_node.get("name", str(end_node.get("id")))
    
    # 고정 일정 등의 경우 좌표가 없을 수 있으므로 이름 기반으로 구분
    return (s_name, e_name, int(departure_time.hour))


def get_all_detailed_paths(trip_legs, departure_time):
    """
    trip_legs: [(start_node, end_node), ...]
    안전 조치:
      - 좌표가 없는 노드(예: 고정일정)는 r5py 요청 대상에서 제외
      - 좌표 없는 구간에 대해선 폴백 경로(fallback)를 만들어 path_map에 넣음
    """
    if not trip_legs: return {}
    path_map = {}
    origins_list, dests_list = [], []

    # 1) 요청할 (좌표 있는) 쌍만 수집하고, 좌표 없는 쌍은 폴백으로 처리
    for start_node, end_node in trip_legs:
        if start_node['id'] == end_node['id']: continue

        cache_key = make_cache_key(start_node, end_node, departure_time)
        if cache_key in DETAILED_PATH_CACHE:
            path_map[(int(start_node['id']), int(end_node['id']))] = DETAILED_PATH_CACHE[cache_key]
            continue

        # 좌표가 없으면 r5 요청을 만들지 않고 폴백으로 채움
        if start_node.get('lat') is None or end_node.get('lat') is None:
            fallback_entry = {"fastest": [f"이동(좌표없음) : {FALLBACK_MOVE_MIN}분"], "min_transfer": [f"이동(좌표없음) : {FALLBACK_MOVE_MIN}분"]}
            DETAILED_PATH_CACHE[cache_key] = fallback_entry
            path_map[(int(start_node['id']), int(end_node['id']))] = fallback_entry
            continue

        # 좌표가 모두 있으면 r5 요청 대상에 추가
        origins_list.append(start_node)
        dests_list.append(end_node)

    # 2) 좌표 있는 쌍만 r5py로 상세 경로 요청
    if origins_list:
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
        except Exception as e:
            print(f"⚠️ DetailedItineraries 호출 실패: {e}")
            computer = None

        if computer is not None and not computer.empty:
            mode_col = 'transport_mode' if 'transport_mode' in computer.columns else 'mode'

            def get_val(row, candidates, default=None):
                for c in candidates:
                    if c in row.index and pd.notna(row[c]): return str(row[c]).strip()
                return default

            def parse_route_to_segments(route_df):
                segs = []
                for _, leg in route_df.iterrows():
                    raw_mode = str(leg[mode_col]).upper() if mode_col in leg.index else ''

                    # 시간 파싱
                    ride_time = max(1, duration_to_minutes(get_val(leg, ['travel_time', 'duration'], 0)))
                    wait_time = duration_to_minutes(get_val(leg, ['wait_time', 'wait'], 0))

                    if wait_time > 0:
                        segs.append(f"대기 : {wait_time}분")

                    if 'WALK' in raw_mode:
                        segs.append(f"도보 : {ride_time}분")
                        continue

                    f_id, t_id = str(get_val(leg, ['start_stop_id', 'from_stop_id'])), str(get_val(leg, ['end_stop_id', 'to_stop_id']))
                    f_stop, t_stop = get_stop_name(f_id) or "정류장", get_stop_name(t_id) or "정류장"
                    c_rid = str(get_val(leg, ['route_id']))
                    mode_lbl = "지하철" if any(x in raw_mode for x in ['SUBWAY', 'RAIL', 'METRO']) else "버스"

                    if mode_lbl == "버스" and STOP_ROUTE_MAP:
                        common = STOP_ROUTE_MAP.get(f_id, set()).intersection(STOP_ROUTE_MAP.get(t_id, set()))
                        common.add(c_rid)
                        b_names = sorted([n for n in [get_route_name(rid) for rid in common] if n])
                        r_str = ", ".join(b_names) if b_names else (get_route_name(c_rid) or '대중교통')
                    else:
                        r_str = get_route_name(c_rid) or '대중교통'

                    segs.append(f"[{mode_lbl}][{r_str}] : {f_stop} → {t_stop} : {ride_time}분")

                return segs

            # 3) 그룹별 옵션 분석
            for (from_id, to_id), group in computer.groupby(['from_id', 'to_id']):
                options_data = []
                for _, opt in group.groupby("option"):
                    t_min = sum(max(1, duration_to_minutes(get_val(leg, ['travel_time', 'duration'], 0))) for _, leg in opt.iterrows())
                    t_count = sum(1 for _, leg in opt.iterrows() if 'WALK' not in str(leg[mode_col]).upper())
                    options_data.append({"route": opt, "time": t_min, "transfers": t_count})

                if not options_data:
                    continue

                fastest_opt = min(options_data, key=lambda x: (x['time'], x['transfers']))
                result_entry = {"fastest": parse_route_to_segments(fastest_opt['route'])}

                walk_opts = [o for o in options_data if o['transfers'] == 0]
                best_walk = min(walk_opts, key=lambda x: x['time']) if walk_opts else None

                transit_opts = [o for o in options_data if o['transfers'] > 0]
                transit_opts.sort(key=lambda x: (x['transfers'], x['time']))
                best_transit = transit_opts[0] if transit_opts else None

                winner_opt = None
                if best_walk and best_transit:
                    if best_walk['time'] <= best_transit['time']:
                        winner_opt = best_walk
                    else:
                        winner_opt = best_transit
                elif best_transit:
                    winner_opt = best_transit
                else:
                    winner_opt = best_walk

                if winner_opt:
                    result_entry["min_transfer"] = parse_route_to_segments(winner_opt['route'])
                else:
                    # 드물게 옵션이 비어있으면 폴백 적용
                    result_entry["min_transfer"] = [f"도보 : {FALLBACK_MOVE_MIN}분"]

                cache_key = (int(from_id), int(to_id), int(departure_time.hour))
                DETAILED_PATH_CACHE[cache_key] = result_entry
                path_map[(int(from_id), int(to_id))] = result_entry

    return path_map

# ============================================================
# 5. 노드 빌더 & 최적화 (OR-Tools)
# ============================================================

def build_fixed_nodes(fixed_events, day_start_dt):
    nodes = []
    BUFFER = 15
    for idx, event in enumerate(fixed_events):
        event_start = parse_time(event["start"]) if event.get("start") else day_start_dt
        event_end = parse_time(event["end"]) if event.get("end") else day_start_dt
        orig_start_min = int((event_start - day_start_dt).total_seconds() / 60)
        orig_end_min = int((event_end - day_start_dt).total_seconds() / 60)

        raw_start_min = orig_start_min - BUFFER
        buffered_start_min = max(0, raw_start_min)
        final_stay = (orig_end_min - orig_start_min) + (orig_start_min - buffered_start_min) + BUFFER

        # 고정일정은 좌표가 없을 수도 있으므로 lat/lng는 None으로 둠
        nodes.append({
            "name": event.get("title", "고정일정"), "category": "고정일정", "lat": None, "lng": None,
            "stay": final_stay, "type": "fixed", "window": (buffered_start_min, buffered_start_min + 10),
            "orig_time_str": f"{event.get('start','00:00')} - {event.get('end','00:00')}"
        })
    return nodes

def build_nodes(places, restaurants, fixed_events, day_start_dt):
    nodes = []
    first_place = places[0] if places else {"lat": 37.5665, "lng": 126.9780}
    nodes.append({"name": "시작점", "category": "출발", "lat": first_place["lat"], "lng": first_place["lng"], "stay": 0, "type": "depot"})

    for p in places:
        nodes.append({"name": p["name"], "category": p["category"], "lat": p.get("lat"), "lng": p.get("lng"), "stay": stay_time_map.get(p["category"], 60), "type": "spot"})

    if restaurants:
        nodes.append({"name": restaurants[0]["name"], "category": "식당", "lat": restaurants[0].get("lat"), "lng": restaurants[0].get("lng"), "stay": 70, "type": "lunch"})
        nodes.append({"name": restaurants[1]["name"], "category": "식당", "lat": restaurants[1].get("lat"), "lng": restaurants[1].get("lng"), "stay": 70, "type": "dinner"})

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

# optimize_day 함수는 기존 로직을 유지하되, get_all_detailed_paths에서 좌표 없는 구간을 안전하게 처리하므로
# 이후 로직은 대부분 그대로 동작한다.

def optimize_day(places, restaurants, fixed_events, start_time_str, target_date_str, end_time_str=None):
    TRAVEL_BUFFER = 5
    day_start_dt = datetime.strptime(start_time_str, "%H:%M")
    
    SAFE_GTFS_DATE = target_date_str
    r5_departure_dt = datetime.combine(datetime.strptime(SAFE_GTFS_DATE, "%Y-%m-%d"), datetime.strptime("11:00", "%H:%M").time())
    display_start_dt = datetime.combine(datetime.strptime(target_date_str, "%Y-%m-%d"), day_start_dt.time())

    max_horizon_minutes = 24 * 60
    if end_time_str:
        diff = int((datetime.strptime(end_time_str, "%H:%M") - day_start_dt).total_seconds() / 60)
        if diff > 0: max_horizon_minutes = diff

    nodes = build_nodes(places, restaurants, fixed_events, day_start_dt)
    for idx, node in enumerate(nodes): node["id"] = idx
    n = len(nodes)

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
                    val = max(val, 30)
            
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
    # search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # search_params.time_limit.seconds = 1
    # search_params.log_search = False

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

    def build_timeline_by_type(path_type):
        timeline = []
        actual_visits = [n for n in visited_nodes if n["type"] != "depot"]
        cursor = display_start_dt + timedelta(minutes=actual_visits[0]['arrival_min'])

        for i, node in enumerate(actual_visits):
            transit_info = []
            travel_min = 0
            
            if i > 0:
                prev = actual_visits[i-1]
                path_options = path_map.get((prev['id'], node['id']))
                
                if path_options:
                    chosen_path = path_options.get(path_type, path_options.get('fastest', []))
                    transit_info = chosen_path
                    for segment in chosen_path:
                        mins = re.findall(r'(\d+)분', segment)
                        for m in mins: travel_min += int(m)
                else:
                    # path_map에 정보가 없으면 좌표 유무 기반 폴백 적용
                    if prev.get('lat') is None or node.get('lat') is None:
                        travel_min = FALLBACK_MOVE_MIN
                        transit_info = [f"도보 : {FALLBACK_MOVE_MIN}분"]
                    else:
                        dist = haversine(prev['lat'], prev['lng'], node['lat'], node['lng'])
                        travel_min = int(dist * 15)
                        transit_info = [f"도보 : {travel_min}분"]

            if node["type"] == "fixed":
                time_parts = node.get("orig_time_str", "00:00 - 00:00").split(" - ")
                start_dt = datetime.strptime(f"{target_date_str} {time_parts[0]}", "%Y-%m-%d %H:%M")
                end_dt = datetime.strptime(f"{target_date_str} {time_parts[1]}", "%Y-%m-%d %H:%M")
                cursor = end_dt
                time_str = node["orig_time_str"]
            else:
                start_dt = cursor + timedelta(minutes=travel_min)
                end_dt = start_dt + timedelta(minutes=node["stay"])
                time_str = f"{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}"
                cursor = end_dt

            timeline.append({
                "name": node["name"],
                "category": node["category"],
                "time": time_str,
                "transit_to_here": transit_info
            })
        return timeline

    return {
        "fastest_version": build_timeline_by_type("fastest"),
        "min_transfer_version": build_timeline_by_type("min_transfer")
    }

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
    day_keys = list(plans.keys())

    print(f"\n🚀 병렬 최적화 시작: {len(day_keys)}일치 일정을 동시에 계산합니다.")
    start_total_opt = time.time()

    # [내부 함수] 병렬 처리를 위한 래퍼 함수
    def process_day_wrapper(args):
        day_key, date_obj, is_first, is_last = args
        
        todays_start = first_day_start_str if is_first else default_start_str
        todays_end = last_day_end_str if is_last else default_end_str
        current_date_str = date_obj.strftime("%Y-%m-%d")
        
        print(f"   ▶ {day_key} 최적화 시작...")
        
        # 실제 최적화 수행
        day_res = optimize_day(
            places=plans[day_key]["route"],
            restaurants=plans[day_key]["restaurants"],
            fixed_events=get_fixed_events_for_day(FIXED_EVENTS, current_date_str),
            start_time_str=todays_start,
            target_date_str=current_date_str,
            end_time_str=todays_end
        )
        return day_key, day_res

    # 6-1. 병렬 실행 인자(Task) 준비
    tasks = []
    curr = start
    for i, day_key in enumerate(day_keys):
        tasks.append((day_key, curr, i==0, i==len(day_keys)-1))
        curr += timedelta(days=1)

    # 6-2. ThreadPoolExecutor로 병렬 실행
    processed_results = {}

    with ThreadPoolExecutor(max_workers=JAVA_PARALLELISM) as executor:
        for day_key, day_res in executor.map(process_day_wrapper, tasks):
            processed_results[day_key] = day_res
            print(f"   ✅ {day_key} 완료")

    print(f"⏱ 전체 최적화 완료: {round(time.time() - start_total_opt, 2)}초")
    
    # 3. 결과 취합 및 화면 출력
    curr = start
    for i, day_key in enumerate(day_keys):
        # 결과 저장
        result["plans"][day_key]["timelines"] = processed_results[day_key]
        day_results = processed_results[day_key]
        
        print(f"\n📅 {day_key} ({curr.strftime('%Y-%m-%d')})")

        # 두 가지 버전(최단 시간, 최소 환승) 모두 출력
        for ver_key, label in [("fastest_version", "최단 시간"), ("min_transfer_version", "최소 환승")]:
            timeline = day_results[ver_key]
            
            separator = "-" * 60
            print(f"\n[{label} 기준 일정] {day_key}")
            print(separator)

            for t in timeline:
                if t.get('transit_to_here'):
                    # 리스트 형태의 경로를 화살표로 연결하여 출력
                    path_str = " -> ".join([s for s in t['transit_to_here']])
                    print(f"  [TRANSIT] {path_str}")
                print(f"  [{t['time']}] {t['name']} ({t['category']})")
            
            print(separator)

        # 날짜 카운터 증가
        curr += timedelta(days=1)

    # 7. 모든 루프가 끝난 후 최종 파일 저장 (루프 외부)
    with open("result_timeline.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n전체 일정이 'result_timeline.json' 파일로 저장되었습니다.")