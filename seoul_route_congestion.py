import os, pickle, re, time, math, json, zipfile, joblib
import multiprocessing

available_cores = multiprocessing.cpu_count()
JAVA_PARALLELISM = max(2, available_cores // 2)
print(f"⚙️  Java 내부 병렬성 설정: {JAVA_PARALLELISM}개")
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-21.0.10"
os.environ["JAVA_OPTS"] = f"-Xmx8G -Djava.util.concurrent.ForkJoinPool.common.parallelism={JAVA_PARALLELISM}"

from google import genai
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from r5py import TransportNetwork, TravelTimeMatrix, DetailedItineraries, TransportMode
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# [NEW] 0. 혼잡도 모델 및 설정 로드
# ============================================================
print("🧠 혼잡도 예측 모델 로드 중...")
try:
    CONGESTION_MODEL = joblib.load('./model/congestion_model_latlon.pkl')
    # 이전 단계에서 한국어 컬럼명으로 학습했으므로 순서를 맞춰줍니다.
    # ['month', 'day', 'hour', 'dayofweek', 'is_holiday', 'is_weekend', '위도', '경도']
    print("✅ 모델 로드 성공")
except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")
    CONGESTION_MODEL = None

# 공휴일 정의 (모델 학습때와 동일하게)
KOREAN_HOLIDAYS_2026 = [
    '20260101', # 신정 (목)
    '20260216', '20260217', '20260218', # 설날 연휴 (월, 화, 수)
    '20260301', # 삼일절 (일)
    '20260302', # 삼일절 대체공휴일 (월)
    '20260505', # 어린이날 (화)
    '20260524', # 부처님오신날 (일)
    '20260525', # 부처님오신날 대체공휴일 (월)
    '20260606', # 현충일 (토)
    '20260608', # 현충일 대체공휴일 (월) - *관공서 공휴일 규정에 따라 적용 예상
    '20260815', # 광복절 (토)
    '20260817', # 광복절 대체공휴일 (월)
    '20260924', '20260925', '20260926', # 추석 연휴 (목, 금, 토)
    '20261003', # 개천절 (토)
    '20261005', # 개천절 대체공휴일 (월)
    '20261009', # 한글날 (금)
    '20261225'  # 크리스마스 (금)
]

def get_congestion_level(lat, lng, dt):
    """
    위치와 시간을 받아 혼잡도(0:Low, 1:Med, 2:High)를 반환
    """
    if CONGESTION_MODEL is None or lat is None or lng is None:
        return 0 
    
    # 파생 변수 생성
    month = dt.month
    day = dt.day
    hour = dt.hour
    
    # [수정] datetime 객체는 .dayofweek 속성이 없으므로 .weekday() 메서드 사용
    # 월요일=0, ... 일요일=6 (Pandas dayofweek와 동일)
    dayofweek = dt.weekday() 
    
    date_str = dt.strftime('%Y%m%d')
    
    # [수정] 2026년 공휴일 리스트 참조 확인
    is_holiday = 1 if date_str in KOREAN_HOLIDAYS_2026 else 0
    is_weekend = 1 if dayofweek >= 5 else 0
    
    # 입력 데이터 프레임 생성
    input_vector = pd.DataFrame([[
        month, day, hour, dayofweek, is_holiday, is_weekend, lat, lng
    ]], columns=['month', 'day', 'hour', 'dayofweek', 'is_holiday', 'is_weekend', '위도', '경도'])
    
    return CONGESTION_MODEL.predict(input_vector)[0]

def get_stay_weight(level):
    """
    혼잡도 등급에 따른 시간 가중치 반환
    0 (Low) -> 1.0 (변화 없음)
    1 (Med) -> 1.1 (10% 증가)
    2 (High) -> 1.3 (30% 증가)
    """
    if level == 2: return 1.3
    elif level == 1: return 1.1
    else: return 1.0

def get_wait_weight(level):
    """
    대기 시간 전용 가중치
    0 (Low) -> 1.0 (변화 없음)
    1 (Med) -> 1.5 (50% 증가)
    2 (High) -> 2.0 (2배 증가)
    """
    if level == 2: return 2.0
    elif level == 1: return 1.5
    else: return 1.0

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

MAX_TRANSFERS = 2
MAX_TRAVEL_TIME_MIN = 90

# 시간 윈도우 설정
LUNCH_WINDOW = ("11:20", "13:20")
DINNER_WINDOW = ("17:40", "19:30")

# 장소별 체류 시간
stay_time_map = {
    "관광지": 90, "카페": 50, "음식점": 70, 
    "박물관": 120, "공원": 60, "시장": 80, "숙박": 0
}

# 데이터 파일 경로
osm_file = "./data/seoul_osm_v.pbf"
gtfs_files = ["./data/seoul_area_gtfs.zip"]

# 서울 구별 중심 좌표
SEOUL_GU_COORDS = {
"강남구": {"lat": 37.514575, "lon": 127.0495556},
"강동구": {"lat": 37.52736667, "lon": 127.1258639},
"강북구": {"lat": 37.63695556, "lon": 127.0277194},
"강서구": {"lat": 37.54815556, "lon": 126.851675},
"관악구": {"lat": 37.47538611, "lon": 126.9538444},
"광진구": {"lat": 37.53573889, "lon": 127.0845333},
"구로구": {"lat": 37.49265, "lon": 126.8895972},
"금천구": {"lat": 37.44910833, "lon": 126.9041972},
"노원구": {"lat": 37.65146111, "lon": 127.0583889},
"도봉구": {"lat": 37.66583333, "lon": 127.0495222},
"동대문구": {"lat": 37.571625, "lon": 127.0421417},
"동작구": {"lat": 37.50965556, "lon": 126.941575},
"마포구": {"lat": 37.56070556, "lon": 126.9105306},
"서대문구": {"lat": 37.57636667, "lon": 126.9388972},
"서초구": {"lat": 37.48078611, "lon": 127.0348111},
"성동구": {"lat": 37.56061111, "lon": 127.039},
"성북구": {"lat": 37.58638333, "lon": 127.0203333},
"송파구": {"lat": 37.51175556, "lon": 127.1079306},
"양천구": {"lat": 37.51423056, "lon": 126.8687083},
"영등포구": {"lat": 37.52361111, "lon": 126.8983417},
"용산구": {"lat": 37.53609444, "lon": 126.9675222},
"은평구": {"lat": 37.59996944, "lon": 126.9312417},
"종로구": {"lat": 37.57037778, "lon": 126.9816417},
"중구": {"lat": 37.56100278, "lon": 126.9996417},
"중랑구": {"lat": 37.60380556, "lon": 127.0947778},
}

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
pickle_path = "./data/seoul_tn_cached.pkl"
if os.path.exists(pickle_path):
    print(f"📦 TransportNetwork 캐시 로드 중...")
    try:
        # 안전한 로딩을 위해 클래스 메서드 대신 직접 로드 시도
        with open(pickle_path, 'rb') as f:
            transport_network = pickle.load(f)
    except Exception:
        # 구버전 pickle 호환 문제 시 재생성
        transport_network = TransportNetwork.__new__(TransportNetwork)
        transport_network._transport_network = TransportNetwork._load_pickled_transport_network(transport_network, pickle_path)
else:
    print("🚀 TransportNetwork 생성 중 (최초 1회)...")
    transport_network = TransportNetwork(osm_file, gtfs_files)
    try:
        # 최신 r5py 방식 저장
        transport_network.save(pickle_path)
    except:
        pass

meta_cache_path = "./data/metadata_cache_v2.pkl" # 파일명 v2로 변경 (캐시 갱신을 위해)

STOP_COORDS = {} # 전역 변수

if os.path.exists(meta_cache_path):
    print("⚡ 메타데이터(좌표포함) 캐시 로드 중...")
    with open(meta_cache_path, "rb") as f:
        meta_data = pickle.load(f)
        STOP_ID_TO_NAME = meta_data["stops"]
        ROUTE_ID_TO_NAME = meta_data["routes"]
        STOP_ROUTE_MAP = meta_data["stop_route_map"]
        STOP_COORDS = meta_data["coords"] # [NEW] 좌표 로드
else:
    print("🐢 메타데이터 생성 중 (좌표 포함)...")
    # Stops
    with zipfile.ZipFile(gtfs_files[0]) as z:
        with z.open("stops.txt") as f:
            # [NEW] stop_lat, stop_lon 컬럼 추가 로드
            stops_df = pd.read_csv(f, dtype={'stop_id': str}, usecols=['stop_id', 'stop_name', 'stop_lat', 'stop_lon'])
    
    STOP_ID_TO_NAME = {str(row['stop_id']).strip(): str(row['stop_name']).strip() for _, row in stops_df.iterrows()}
    
    # [NEW] 정류장 ID -> 좌표 매핑 생성
    for _, row in stops_df.iterrows():
        s_id = str(row['stop_id']).strip()
        STOP_COORDS[s_id] = {'lat': row['stop_lat'], 'lng': row['stop_lon']}
    
    # Routes (기존 동일)
    with zipfile.ZipFile(gtfs_files[0]) as z:
        with z.open("routes.txt") as f:
            routes_df = pd.read_csv(f)
    ROUTE_ID_TO_NAME = dict(zip(routes_df["route_id"].astype(str), routes_df["route_short_name"].astype(str)))
    
    # Stop-Route Map (기존 동일)
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
        STOP_ROUTE_MAP = {}

    # 캐시 저장
    with open(meta_cache_path, "wb") as f:
        pickle.dump({
            "stops": STOP_ID_TO_NAME,
            "routes": ROUTE_ID_TO_NAME,
            "stop_route_map": STOP_ROUTE_MAP,
            "coords": STOP_COORDS # [NEW] 저장
        }, f)

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
    """캐시 키 생성을 일관성 있게 관리"""
    s_id = start_node.get("id")
    e_id = end_node.get("id")
    # 좌표 기반 유니크성 확보를 위해 ID와 시간대 조합
    return (s_id, e_id, departure_time.hour)


def get_all_detailed_paths(trip_legs, departure_time):
    if not trip_legs: return {}
    path_map = {}
    origins_list, dests_list = [], []

    # 1) 요청할 (좌표 있는) 쌍만 수집하고, 좌표 없는 쌍은 폴백으로 처리
    for start_node, end_node in trip_legs:
        if start_node['id'] == end_node['id']: continue

        ckey = make_cache_key(start_node, end_node, departure_time)
        if ckey in DETAILED_PATH_CACHE:
            path_map[(start_node['id'], end_node['id'])] = DETAILED_PATH_CACHE[ckey]
            continue

        # 좌표가 없으면 r5 요청을 만들지 않고 폴백으로 채움
        if start_node.get('lat') is None or end_node.get('lat') is None:
            fallback = {"fastest": [f"이동(좌표없음) : {FALLBACK_MOVE_MIN}분"], 
                        "min_transfer": [f"이동(좌표없음) : {FALLBACK_MOVE_MIN}분"]}
            path_map[(start_node['id'], end_node['id'])] = fallback
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

                    # 출발 정류장 ID (대기하는 곳)
                    f_id = str(get_val(leg, ['start_stop_id', 'from_stop_id'])).strip()
                    t_id = str(get_val(leg, ['end_stop_id', 'to_stop_id'])).strip()

                    if wait_time > 0:
                        # [핵심 수정] 대기 텍스트 뒤에 정류장 ID를 몰래 심어둡니다.
                        # 예: "대기 : 5분 [STOP:1000023]"
                        segs.append(f"대기 : {wait_time}분 [STOP:{f_id}]")

                    if 'WALK' in raw_mode:
                        segs.append(f"도보 : {ride_time}분")
                        continue

                    f_stop, t_stop = get_stop_name(f_id) or "정류장", get_stop_name(t_id) or "정류장"
                    c_rid = str(get_val(leg, ['route_id']))
                    mode_lbl = "지하철" if any(x in raw_mode for x in ['SUBWAY', 'RAIL', 'METRO']) else "버스"

                    # ... (버스 노선명 찾는 로직 기존 동일) ...
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
        nodes.append({
            "name": p["name"], 
            "category": p["category"], 
            "category2": p.get("category2", ""), # category2 추가
            "lat": p.get("lat"), 
            "lng": p.get("lng"), 
            "stay": stay_time_map.get(p["category"], 60), 
            "type": "spot"
        })

    if len(restaurants) >= 2:
        nodes.append({
            "name": restaurants[0]["name"], 
            "category": "음식점", 
            "category2": restaurants[0].get("category2", "식당"), # category2 추가
            "lat": restaurants[0].get("lat"), 
            "lng": restaurants[0].get("lng"), 
            "stay": 70, 
            "type": "lunch"
        })
        dinner_idx = 1 if restaurants[0]["name"] != restaurants[1]["name"] else 2
        if len(restaurants) > dinner_idx:
            nodes.append({
                "name": restaurants[1]["name"], 
                "category": "음식점", 
                "category2": restaurants[1].get("category2", "식당"), # category2 추가
                "lat": restaurants[1].get("lat"), 
                "lng": restaurants[1].get("lng"), 
                "stay": 70, 
                "type": "dinner"
            })

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
    routing.AddDimension(transit_callback, 480, max_horizon_minutes, False, "Time")
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
        
        # 첫 번째 장소 도착 시간 기준
        cursor_dt = display_start_dt + timedelta(minutes=actual_visits[0]['arrival_min'])

        for i, node in enumerate(actual_visits):
            transit_info = []
            travel_min = 0
            
            # ============================================================
            # 1. 이동 경로 및 대기 시간 계산 (정류장 혼잡도 반영)
            # ============================================================
            if i > 0:
                prev = actual_visits[i-1]
                path_options = path_map.get((prev['id'], node['id']))
                if path_options:
                    chosen_path = path_options.get(path_type, path_options.get('fastest', []))
                    
                    for segment in chosen_path:
                        seg_mins = sum(int(m) for m in re.findall(r'(\d+)분', segment))
                        
                        if "대기" in segment:
                            target_lat, target_lng = None, None
                            
                            stop_match = re.search(r'\[STOP:(.*?)\]', segment)
                            if stop_match:
                                s_id = stop_match.group(1).strip()
                                if s_id in STOP_COORDS:
                                    target_lat = STOP_COORDS[s_id]['lat']
                                    target_lng = STOP_COORDS[s_id]['lng']
                            
                            if target_lat is None:
                                target_lat = prev.get('lat')
                                target_lng = prev.get('lng')

                            cong_level = get_congestion_level(target_lat, target_lng, cursor_dt)
                            weight = get_wait_weight(cong_level) # 대기 시간 가중치 사용
                            
                            weighted_wait = int(seg_mins * weight)
                            added_wait = weighted_wait - seg_mins
                            
                            icons = {0: "🟢", 1: "🟡", 2: "🔴"}
                            cong_icon = icons.get(cong_level, "")

                            clean_segment = re.sub(r'\s*\[STOP:.*?\]', '', segment) 
                            clean_segment += f" {cong_icon}"
                            
                            if added_wait > 0:
                                seg_mins = weighted_wait
                                clean_segment += f"(+{added_wait}분)"
                            
                            segment = clean_segment
                                
                        transit_info.append(segment)
                        travel_min += seg_mins
                else:
                    dist = haversine(prev['lat'], prev['lng'], node['lat'], node['lng']) if prev.get('lat') else 0
                    travel_min = int(dist * 15) if dist > 0 else FALLBACK_MOVE_MIN
                    transit_info.append(f"도보 : {travel_min}분")

            # ============================================================
            # 2. 도착 시간 확정 (이동 시간 반영)
            # ============================================================
            arrival_dt = cursor_dt + timedelta(minutes=travel_min)
            
            # 식사 시간 윈도우 체크 (너무 일찍 도착하면 대기)
            if node["type"] in ["lunch", "dinner"]:
                window_start_min, _ = build_time_windows([node], display_start_dt)[0]
                window_start_dt = display_start_dt + timedelta(minutes=window_start_min)
                earliest_start_dt = window_start_dt - timedelta(minutes=20) # 20분 전까진 허용
                
                if arrival_dt < earliest_start_dt:
                    wait_min = int((window_start_dt - arrival_dt).total_seconds() / 60)
                    transit_info.append(f"현장 대기 : {wait_min}분")
                    arrival_dt = window_start_dt

            # ============================================================
            # 3. [핵심] 체류 시간 계산 (혼잡도 가중치 적용)
            # ============================================================
            final_stay_min = node["stay"]
            congestion_label = ""
            
            # (A) 고정 일정 및 출발지가 아닌 경우 혼잡도 계산
            if node["type"] not in ["fixed", "depot"]:
                cong_level = get_congestion_level(node.get('lat'), node.get('lng'), arrival_dt)
                
                labels = {0: "🟢여유", 1: "🟡보통", 2: "🔴혼잡"}
                congestion_label = labels.get(cong_level, "정보없음")
                
                # (B) 모든 장소는 체류 시간 늘리기
                weight = get_stay_weight(cong_level) # 체류 시간 가중치 사용
                
                original_stay = node["stay"]
                final_stay_min = int(original_stay * weight)
                
                # 시간이 늘어났으면 로그(디버깅용) 혹은 결과에 표시할 수도 있음
                if final_stay_min > original_stay:
                    # 예: "🔴혼잡(+18분)" 처럼 표시하고 싶다면 아래 주석 해제
                    # congestion_label += f"(+{final_stay_min - original_stay}분)"
                    pass

            elif node["type"] == "fixed":
                congestion_label = "📅고정"

            # ============================================================
            # 4. 종료 시간 계산 및 커서 업데이트
            # ============================================================
            if node["type"] == "fixed":
                time_str = node.get("orig_time_str", "00:00 - 00:00")
                time_parts = time_str.split(" - ")
                # 고정 일정은 정해진 시간에 끝나므로 커서를 강제로 맞춤
                cursor_dt = datetime.strptime(f"{target_date_str} {time_parts[1]}", "%Y-%m-%d %H:%M")
            else:
                # 일반 장소는 늘어난 체류시간(final_stay_min)만큼 머물고 출발
                end_dt = arrival_dt + timedelta(minutes=final_stay_min)
                time_str = f"{arrival_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}"
                cursor_dt = end_dt

            # ============================================================
            # 5. 결과 저장
            # ============================================================
            timeline.append({
                "name": node['name'], 
                "category": node["category"],
                "category2": node.get("category2", node["category"]),
                "time": time_str,
                "transit_to_here": transit_info,
                "congestion_level": congestion_label
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
    # 1. 엑셀 및 기본 정보 로드
    print("📂 장소 데이터 로드 중...")
    try:
        df = pd.read_excel("./data/place_전체_통합_진짜최종.xlsx")
    except FileNotFoundError:
        print("❌ 'places_전체_통합.xlsx' 파일이 없습니다.")
        exit()

    area = input("여행할 지역을 입력하세요 (예: 종로구): ")

    if area not in SEOUL_GU_COORDS:
        raise ValueError("서울 구 이름이 아닙니다.")
    
    center_lat = SEOUL_GU_COORDS[area]["lat"]
    center_lon = SEOUL_GU_COORDS[area]["lon"]

    df["distance_km"] = df.apply(lambda r: haversine(center_lat, center_lon, r["lat"], r["lng"]), axis=1)
    RADIUS_KM = 6
    
    # 2. 장소 필터링
    area_mask = df[df["distance_km"] <= RADIUS_KM].copy()
    print(f"\n📍 {area} 중심 반경 {RADIUS_KM}km 이내 장소 수: {len(area_mask)}")

    dist_mask = df["distance_km"] <= RADIUS_KM

    filtered_spot = df[dist_mask & (df["category"] != "음식점") & (df["category"] != "숙박")][["name", "lat", "lng", "category", "category2"]]

    avg_lat = filtered_spot["lat"].mean()
    avg_lng = filtered_spot["lng"].mean()

    # 관광지 중심 1.5km 이내 식당만 추출 (훨씬 타이트한 동선)
    df["dist_to_center"] = df.apply(lambda r: haversine(avg_lat, avg_lng, r["lat"], r["lng"]), axis=1)
    filtered_restaurant = df[(df["dist_to_center"] <= 3) & (df["category"] == "음식점")][["name", "lat", "lng", "category", "category2"]]

    filtered_accom = df[dist_mask & (df["category"] == "숙박")][["name", "lat", "lng", "category", "category2"]]

    places = filtered_spot.to_dict(orient="records")
    print(len(places), "개의 관광지가 선택되었습니다.")

    restaurants = filtered_restaurant.to_dict(orient="records")
    print(len(restaurants), "개의 음식점이 선택되었습니다.")

    accommodations = filtered_accom.to_dict(orient="records")
    print(len(accommodations), "개의 숙박 시설이 선택되었습니다.")

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
    #         {"name": "...", "category": "...", "category2": "...", "lat": 0.0, "lng": 0.0}
    #       ],
    #       "restaurants": [
    #         {"name": "...", "category": "...", "category2": "...", "lat": 0.0, "lng": 0.0}
    #       ],
    #       "accommodations": [
    #         {"name": "...", "category": "...", "category2": "...", "lat": 0.0, "lng": 0.0}
    #       ]
    #     }
    #   }
    # }
    # """
    
    # system_prompt = f"""
    # 너는 '서울 여행 장소 추천 전문가'이다. 반드시 제공된 데이터만을 사용하여 계획을 세운다.
    # {schema}
    # [절대 규칙]
    # 1. 모든 장소의 이름, 좌표(lat, lng), 카테고리는 입력된 데이터와 100% 일치해야 한다. 절대 값을 수정하거나 새로운 좌표를 생성하지 마라.
    # 2. 'route' 배열: 오직 제공된 'places' 목록에서 5개를 선택하여 담는다.
    # 3. 'restaurants' 배열: 오직 제공된 'restaurants' 목록에서 2개를 선택한다.
    # 4. 'accommodations' 배열: 오직 제공된 'accommodations' 목록에서 1개를 선택한다. (마지막 날은 빈 배열 []로 출력)
    # 5. 할루시네이션 방지: 목록에 없는 장소나 좌표를 출력할 경우 시스템 오류로 간주한다.
    # 6. 출력 형식: 반드시 순수 JSON 데이터만 출력하며, 설명이나 추가 텍스트를 절대 포함하지 않는다.
    # """

    # user_prompt = {
    #     "days": days,
    #     "start_location": {"lat": 37.5547, "lng": 126.9706},
    #     "places": places, # [:6 * days * 4]
    #     "restaurants": restaurants, # [:3 * days * 4]
    #     "accommodations": accommodations # [:days * 4]
    # }

    # print("🤖 Gemini가 초기 계획을 생성하고 있습니다...")
    # prompt = system_prompt + "\n\n" + json.dumps(user_prompt, ensure_ascii=False)
    
    # start_time = time.time()
    # response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt, config={"temperature": 0})
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

    # 6-1. 병렬 실행 인자(Task) 준비
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

    tasks = []
    curr = start
    for i, day_key in enumerate(day_keys):
        tasks.append((day_key, curr, i==0, i==len(day_keys)-1))
        curr += timedelta(days=1)

    # 6-2. ThreadPoolExecutor로 병렬 실행
    processed_results = {}

    max_workers = min(days, 4)
    print(f"⚙️ 최대 {max_workers}개 코어로 병렬 처리 중...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_day_wrapper, tasks))
        for day_key, day_res in results:
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
                    path_str = " -> ".join([s for s in t['transit_to_here']])
                    print(f"  [TRANSIT] {path_str}")
                
                # category 대신 category2 출력
                display_cat = t.get('category2', t['category'])
                
                # [수정된 출력 포맷]
                # 기존: [{t['time']}] {t['name']} ({display_cat})
                # 변경: [{t['time']}] {t['name']} ({display_cat}) {t['congestion_level']}
                print(f"  [{t['time']}] {t['name']} ({display_cat}) {t['congestion_level']}")
            
            print(separator)

        # 날짜 카운터 증가
        curr += timedelta(days=1)

    # 7. 모든 루프가 끝난 후 최종 파일 저장 (루프 외부)
    with open("result_timeline.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n전체 일정이 'result_timeline.json' 파일로 저장되었습니다.")