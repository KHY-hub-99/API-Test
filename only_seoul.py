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
# 전역 상수 / 캐시
# ============================================================
DETAILED_PATH_CACHE = {}

WALK_ONLY_THRESHOLD_MIN = 12   # 최소값
WALK_ONLY_THRESHOLD_MAX = 18   # 최대값

MAX_TRANSFERS = 2
MAX_TRAVEL_TIME_MIN = 90

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
    R = 6371  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def approx_walk_minutes(start, end):
    dist_km = haversine(
        start["lat"], start["lng"],
        end["lat"], end["lng"]
    )
    return dist_km * 12  # 1km ≈ 12분

def dynamic_walk_threshold(dist_km):
    if dist_km < 0.6:
        return WALK_ONLY_THRESHOLD_MAX   # 18
    elif dist_km < 1.2:
        return 15
    else:
        return WALK_ONLY_THRESHOLD_MIN   # 12

def travel_minutes(p1, p2):
    if p1["lat"] is None or p2["lat"] is None:
        return 0
    dist = haversine(p1["lat"], p1["lng"], p2["lat"], p2["lng"])
    return int(dist / 30 * 60)  # 평균 30km/h

def get_fixed_events_for_day(fixed_events, target_date):
    return [e for e in fixed_events if e["date"] == target_date]

def duration_to_minutes(val):
    if val is None or pd.isna(val):
        return 0

    # 1️⃣ pandas Timedelta
    if hasattr(val, "total_seconds"):
        return int(val.total_seconds() / 60)

    # 2️⃣ 문자열 "0 days HH:MM:SS"
    if isinstance(val, str) and "day" in val:
        try:
            td = pd.to_timedelta(val)
            return int(td.total_seconds() / 60)
        except Exception:
            return 0

    # 3️⃣ 숫자
    try:
        return int(float(val))
    except Exception:
        return 0

# ============================================================
# r5py 변수(스크립트 시작 시 한 번만 실행) / Java 설치 필수
# ============================================================
def load_transport_network(osm_path, gtfs_paths, pickle_path="seoul_tn_cached.pkl"):
    # pickle이 존재하고 재생성 옵션이 꺼져 있으면 불러오기
    if os.path.exists(pickle_path):
        print(f"📦 Pickle 파일 '{pickle_path}' 로드 중...")
        tn = TransportNetwork.__new__(TransportNetwork)
        tn._transport_network = TransportNetwork._load_pickled_transport_network(self=TransportNetwork, path=pickle_path)
        print("✅ 로드 완료")
        return tn

    # pickle 없거나 force_rebuild=True 면 새로 생성
    print("🚀 TransportNetwork 새로 생성 중... (시간 걸림)")
    tn = TransportNetwork(osm_path, gtfs_paths)

    # 생성 후 pickle 저장
    try:
        tn._save_pickled_transport_network(path=pickle_path, transport_network=tn)
        print(f"💾 생성 완료 후 pickle 저장: '{pickle_path}'")
    except Exception as e:
        print(f"⚠️ pickle 저장 실패: {e}")

    return tn

osm_file = "./data/seoul_osm_v.pbf"
gtfs_files = ["./data/seoul_area_gtfs.zip"]

start_tn = time.time()
transport_network = load_transport_network(osm_file, gtfs_files)
end_tn = time.time()
print(f"⏱ TransportNetwork 로드/생성 시간: {round(end_tn - start_tn, 2)}초")

# ============================================================
# stops, routes 매칭
# ============================================================
# stops.txt 로드 부분 수정
with zipfile.ZipFile("./data/seoul_area_gtfs.zip") as z:
    with z.open("stops.txt") as f:
        stops_df = pd.read_csv(f, dtype={'stop_id': str}) # ID를 처음부터 문자열로 읽기

# 딕셔너리 생성 시 공백 제거 및 확실한 문자열 처리
STOP_ID_TO_NAME = {
    str(row['stop_id']).strip(): str(row['stop_name']).strip() 
    for _, row in stops_df.iterrows()
}

with zipfile.ZipFile("./data/seoul_area_gtfs.zip") as z:
    with z.open("routes.txt") as f:
        routes_df = pd.read_csv(f)

ROUTE_ID_TO_NAME = dict(
    zip(routes_df["route_id"].astype(str), routes_df["route_short_name"].astype(str))
)

def get_stop_name(stop_id):
    if pd.isna(stop_id):
        return None
    
    # 1. 숫자/문자열 혼용 대응: "10100001.0" 같은 데이터를 "10100001"로 변환
    try:
        # 소수점이 포함된 경우를 대비해 float -> int -> str 순으로 변환
        safe_id = str(int(float(stop_id))).strip()
    except (ValueError, TypeError):
        safe_id = str(stop_id).strip()

    # 2. 딕셔너리 조회
    name = STOP_ID_TO_NAME.get(safe_id)
    
    # 3. 서울 GTFS 특성: 앞자리에 0이 포함된 5자리 ID 대응 (예: "05123")
    if not name and len(safe_id) < 5:
        name = STOP_ID_TO_NAME.get(safe_id.zfill(5))
        
    return name

def safe_stop_name(val):
    if pd.isna(val):
        return None
    try:
        return get_stop_name(str(int(float(val))))
    except Exception:
        return get_stop_name(str(val))

def get_route_name(route_id):
    if pd.isna(route_id):
        return None
    try:
        safe_id = str(int(float(route_id)))
    except Exception:
        safe_id = str(route_id)
        
    return ROUTE_ID_TO_NAME.get(safe_id)

# ============================================================
# r5py 기반 이동 시간 계산 함수 (수정됨)
# ============================================================
def get_r5py_matrix(nodes, departure_time):
    """
    모든 노드 간의 대중교통 이동 시간 행렬을 한꺼번에 계산합니다.
    """
    valid_nodes = [n for n in nodes if n["lat"] is not None]
    if len(valid_nodes) < 2:
        return {}

    gdf = gpd.GeoDataFrame(
        valid_nodes,
        geometry=gpd.points_from_xy(
            [n['lng'] for n in valid_nodes],
            [n['lat'] for n in valid_nodes]
        ),
        crs="EPSG:4326"
    )

    try:
        matrix = TravelTimeMatrix(
            transport_network,
            origins=gdf,
            destinations=gdf,
            departure=departure_time,
            transport_modes=[TransportMode.WALK, TransportMode.TRANSIT]
        )
    except Exception as e:
        print(f"⚠️ 행렬 계산 중 오류: {e}")
        return {}

    r5_travel_times = {}
    for row in matrix.itertuples():
        if not pd.isna(row.travel_time):
            r5_travel_times[(int(row.from_id), int(row.to_id))] = int(row.travel_time)

    return r5_travel_times

# ============================================================
# 상세 경로 추출 함수 (수정됨)
# ============================================================

def make_cache_key(start_node, end_node, departure_time):
    """
    정류장 단위 + 시간 버킷 캐시 키
    """
    hour_bucket = departure_time.hour

    return (
        start_node.get("nearest_stop_id") or start_node.get("id"),
        end_node.get("nearest_stop_id") or end_node.get("id"),
        hour_bucket
    )

def get_all_detailed_paths(trip_legs, departure_time):
    """
    필요한 구간만 상세 경로 계산 + 캐싱 + 선 컷 최적화
    """
    if not trip_legs:
        return {}

    path_map = {}
    origins_list, dests_list = [], []
    valid_pairs = []

    # ===============================
    # 1️⃣ 선 필터링
    # ===============================
    for start_node, end_node in trip_legs:
        if start_node['lat'] is None or end_node['lat'] is None:
            continue
        if start_node['id'] == end_node['id']:
            continue

        # 거리 기반 도보 컷
        approx_min = approx_walk_minutes(start_node, end_node)
        if approx_min <= dynamic_walk_threshold(
            haversine(
                start_node["lat"], start_node["lng"],
                end_node["lat"], end_node["lng"]
            )
        ):
            path_map[(start_node['id'], end_node['id'])] = f"도보 {round(approx_min)}분"
            continue

        # 캐시 키 (정류장 + 시간 버킷)
        cache_key = make_cache_key(start_node, end_node, departure_time)
        if cache_key in DETAILED_PATH_CACHE:
            path_map[(start_node['id'], end_node['id'])] = DETAILED_PATH_CACHE[cache_key]
            continue

        origins_list.append(start_node)
        dests_list.append(end_node)
        valid_pairs.append((start_node['id'], end_node['id'], cache_key))

    if not origins_list:
        return path_map

    # ===============================
    # 2️⃣ GeoDataFrame 생성
    # ===============================
    origins_gdf = gpd.GeoDataFrame(
        origins_list,
        geometry=gpd.points_from_xy(
            [n['lng'] for n in origins_list],
            [n['lat'] for n in origins_list]
        ),
        crs="EPSG:4326"
    )
    origins_gdf["id"] = [n["id"] for n in origins_list]

    dests_gdf = gpd.GeoDataFrame(
        dests_list,
        geometry=gpd.points_from_xy(
            [n['lng'] for n in dests_list],
            [n['lat'] for n in dests_list]
        ),
        crs="EPSG:4326"
    )
    dests_gdf["id"] = [n["id"] for n in dests_list]

    # ===============================
    # 3️⃣ r5py 호출 (제한 적용)
    # ===============================
    try:
        computer = DetailedItineraries(
            transport_network,
            origins=origins_gdf,
            destinations=dests_gdf,
            departure=departure_time,
            transport_modes=[TransportMode.WALK, TransportMode.TRANSIT],
            max_public_transport_rides=MAX_TRANSFERS,
            max_time=timedelta(minutes=MAX_TRAVEL_TIME_MIN)
        )
    except Exception as e:
        print(f"⚠️ 상세 경로 계산 오류: {e}")
        return path_map

    if computer.empty:
        return path_map

    # ===============================
    # 4️⃣ Helper
    # ===============================
    def get_val(row, candidates, default=None):
        for c in candidates:
            if c in row.index and pd.notna(row[c]):
                v = str(row[c]).strip()
                if v:
                    return v
        return default

    mode_col = 'transport_mode' if 'transport_mode' in computer.columns else 'mode'

    # ===============================
    # 5️⃣ 결과 파싱
    # ===============================
    for (from_id, to_id), group in computer.groupby(['from_id', 'to_id']):
        best_route, best_time = None, float("inf")

        for _, opt in group.groupby("option"):
            total = sum(
                max(1, duration_to_minutes(get_val(leg, ['travel_time', 'duration'], 0)))
                for _, leg in opt.iterrows()
            )
            if total < best_time:
                best_time, best_route = total, opt

        if best_route is None:
            continue

        segments = []
        for _, leg in best_route.iterrows():
            raw_mode = str(leg[mode_col]).upper()
            dur = max(1, duration_to_minutes(get_val(leg, ['travel_time', 'duration'], 0)))

            if 'WALK' in raw_mode:
                segments.append(f"도보 {dur}분")
                continue

            # [수정] 실제 존재하는 컬럼명인 start_stop_id와 end_stop_id를 사용합니다.
            # r5py 버전이나 설정에 따라 컬럼명이 다를 수 있으므로 유연하게 대처합니다.
            from_stop_id = get_val(leg, ['start_stop_id', 'from_stop_id'])
            to_stop_id = get_val(leg, ['end_stop_id', 'to_stop_id'])

            # 딕셔너리에서 이름 조회
            from_stop = get_stop_name(from_stop_id) or f"미확정 정류장({from_stop_id})"
            to_stop = get_stop_name(to_stop_id) or f"미확정 정류장({to_stop_id})"
            
            route_id = get_val(leg, ['route_id'])
            route_name = (
                get_val(leg, ['route_short_name']) or 
                get_route_name(route_id) or 
                '대중교통'
            )

            mode_label = "지하철" if any(x in raw_mode for x in ['SUBWAY', 'RAIL', 'METRO']) else "버스"
            segments.append(f"[{mode_label}][{route_name}] {from_stop} → {to_stop} ({dur}분)")

        path_text = " > ".join(segments)

        # 캐시 저장
        cache_key = make_cache_key(
            {"nearest_stop_id": str(from_id)},
            {"nearest_stop_id": str(to_id)},
            departure_time
        )
        DETAILED_PATH_CACHE[cache_key] = path_text
        path_map[(int(from_id), int(to_id))] = path_text

    return path_map

# ============================================================
# 노드 생성
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
            "window": (buffered_start_min, buffered_start_min + 10),
            "orig_time_str": f"{event['start']} - {event['end']}"
        })

    return nodes


def build_nodes(places, restaurants, fixed_events, day_start_dt):
    nodes = []

    if places:
        first_place = places[0]
    else:
        first_place = {"lat": 37.5665, "lng": 126.9780}

    nodes.append({
        "name": "시작점",
        "category": "출발",
        "lat": first_place["lat"],
        "lng": first_place["lng"],
        "stay": 0,
        "type": "depot"
    })

    for p in places:
        nodes.append({
            "name": p["name"],
            "category": p["category"],
            "lat": p["lat"],
            "lng": p["lng"],
            "stay": stay_time_map.get(p["category"], 60),
            "type": "spot"
        })

    if restaurants:
        nodes.append({ "name": restaurants[0]["name"], "category": "식당", "lat": restaurants[0]["lat"], "lng": restaurants[0]["lng"], "stay": 70, "type": "lunch" })
        nodes.append({ "name": restaurants[1]["name"], "category": "식당", "lat": restaurants[1]["lat"], "lng": restaurants[1]["lng"], "stay": 70, "type": "dinner" })

    nodes.extend(build_fixed_nodes(fixed_events, day_start_dt))
    return nodes


# ============================================================
# Time Window 설정
# ============================================================

def build_time_windows(nodes, day_start_dt):
    windows = []

    def get_relative_window(time_str):
        target_time = parse_time(time_str)
        return int((target_time - day_start_dt).total_seconds() / 60)

    lunch_start = get_relative_window(LUNCH_WINDOW[0])
    lunch_end = get_relative_window(LUNCH_WINDOW[1])
    dinner_start = get_relative_window(DINNER_WINDOW[0])
    dinner_end = get_relative_window(DINNER_WINDOW[1])

    for n in nodes:
        if n["type"] == "lunch":
            windows.append((lunch_start, lunch_end))
        elif n["type"] == "dinner":
            windows.append((dinner_start, dinner_end))
        elif n["type"] == "fixed":
            windows.append(n["window"])
        else:
            windows.append((0, 24 * 60))

    return windows


# ============================================================
# OR-Tools 모델
# ============================================================

def optimize_day(places, restaurants, fixed_events, start_time_str, target_date_str, end_time_str=None):
    # ==========================================
    # 1. 초기 설정 및 데이터 준비
    # ==========================================
    TRAVEL_BUFFER = 5  # 이동 후 여유 시간 (분)

    day_start_dt = datetime.strptime(start_time_str, "%H:%M")

    # r5py 계산용 날짜 (GTFS 데이터 유효 기간 내의 날짜 사용)
    # start_date는 전역 변수라고 가정
    SAFE_GTFS_DATE = start_date 
    r5_date_obj = datetime.strptime(SAFE_GTFS_DATE, "%Y-%m-%d")
    r5_departure_dt = datetime.combine(r5_date_obj, datetime.strptime("11:00", "%H:%M").time())

    # 결과 출력용 실제 날짜
    display_date_obj = datetime.strptime(target_date_str, "%Y-%m-%d")
    display_start_dt = datetime.combine(display_date_obj, day_start_dt.time())

    # 하루 최대 시간(분) 계산
    if end_time_str:
        day_end_dt = datetime.strptime(end_time_str, "%H:%M")
        max_horizon_minutes = int((day_end_dt - day_start_dt).total_seconds() / 60)
        if max_horizon_minutes < 0:
            max_horizon_minutes = 24 * 60
    else:
        max_horizon_minutes = 24 * 60

    # 노드 생성 및 ID 부여
    nodes = build_nodes(places, restaurants, fixed_events, day_start_dt)
    for idx, node in enumerate(nodes):
        node["id"] = idx

    n = len(nodes)

    # ==========================================
    # 2. 이동 시간 행렬(Matrix) 계산
    # ==========================================
    # r5py로 대중교통 시간 계산
    r5_travel_times = get_r5py_matrix(nodes, r5_departure_dt)

    # OR-Tools용 최종 행렬 생성 (r5py 실패 시 직선거리 대체 로직 포함)
    time_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            travel_val = r5_travel_times.get((i, j))
            if travel_val is None:
                travel_val = travel_minutes(nodes[i], nodes[j]) # 하버사인 거리 기반 예비 계산

            # 고정 일정 관련 이동 시간 보정 (출발지->고정일정은 0분 등)
            is_fixed_involved = (nodes[i]["type"] == "fixed" or nodes[j]["type"] == "fixed")
            if is_fixed_involved:
                if nodes[i]["type"] == "depot" and nodes[j]["type"] == "fixed":
                    travel_val = 0
                else:
                    travel_val = max(travel_val, 20) # 고정일정 이동 최소 시간 보장

            # 이동시간 + 체류시간 + 버퍼
            time_matrix[i][j] = nodes[i]["stay"] + int(travel_val) + TRAVEL_BUFFER

    # ==========================================
    # 3. OR-Tools 모델 설정
    # ==========================================
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_idx, to_idx):
        return time_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_callback = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)
    
    # Time Dimension 추가 (여기서 time_dim 변수 생성됨)
    routing.AddDimension(transit_callback, 30, max_horizon_minutes, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # 페널티 및 타임 윈도우 설정
    penalty_spot = 100000
    penalty_meal = 1000000
    solver = routing.solver()

    time_windows = build_time_windows(nodes, day_start_dt)

    for i, node in enumerate(nodes):
        index = manager.NodeToIndex(i)
        if node["type"] == "depot":
            continue

        window = time_windows[i]

        # 고정 일정 처리
        if node["type"] == "fixed":
            safe_start = max(0, min(window[0], max_horizon_minutes))
            safe_end = max(safe_start, min(window[1], max_horizon_minutes))
            time_dim.CumulVar(index).SetRange(safe_start, safe_end)
            continue

        # 일반 일정 처리
        overlap_start = max(0, window[0])
        overlap_end = min(max_horizon_minutes, window[1])

        if overlap_start > overlap_end:
            routing.AddDisjunction([index], 0)
            solver.Add(routing.VehicleVar(index) == -1)
            continue

        time_dim.CumulVar(index).SetRange(overlap_start, overlap_end)

        if node["type"] == "spot":
            routing.AddDisjunction([index], penalty_spot)
        elif node["type"] in ["lunch", "dinner"]:
            routing.AddDisjunction([index], penalty_meal)

    # ==========================================
    # 4. 솔루션 탐색 (Solve)
    # ==========================================
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 1

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return []

    # ==========================================
    # 5. 결과 처리 (속도 개선된 부분)
    # ==========================================
    
    # 5-1. 방문 순서 및 시간 정보 먼저 추출
    index = routing.Start(0)
    visited_nodes = []
    
    while not routing.IsEnd(index):
        node_idx = manager.IndexToNode(index)
        t_start_min = solution.Value(time_dim.CumulVar(index))
        node = nodes[node_idx]
        
        # 계산된 도착 시간(분)을 노드 객체에 임시 저장
        node['arrival_min'] = t_start_min
        visited_nodes.append(node)
        
        index = solution.Value(routing.NextVar(index))

    # 5-2. 이동 구간(Leg) 리스트 생성
    trip_legs = []
    for i in range(len(visited_nodes) - 1):
        start_node = visited_nodes[i]
        end_node = visited_nodes[i+1]
        trip_legs.append((start_node, end_node))

    # 5-3. 상세 경로 '일괄' 계산 (Batch Processing)
    # 이 부분이 기존 루프 방식보다 훨씬 빠릅니다.
    print("🚀 전체 상세 경로 일괄 계산 중...")
    batch_start = time.time()
    
    # 앞서 정의한 get_all_detailed_paths 함수 호출
    path_map = get_all_detailed_paths(trip_legs, r5_departure_dt)
    
    print(f"(상세 경로 일괄 계산 시간: {round(time.time() - batch_start, 2)}초)")

    # 5-4. 최종 타임라인 조립 (수정 버전)
    timeline = []
    
    # 시작점(depot)을 제외한 실제 방문지들만 추출
    actual_visits = [n for n in visited_nodes if n["type"] != "depot"]

    for i, node in enumerate(actual_visits):
        # 1. 방문 시간 계산
        if node["type"] == "fixed":
            time_str = node["orig_time_str"]
        else:
            visit_start = display_start_dt + timedelta(minutes=node['arrival_min'])
            visit_end = visit_start + timedelta(minutes=node["stay"])
            time_str = f"{visit_start.strftime('%H:%M')} - {visit_end.strftime('%H:%M')}"

        # 2. 이전 장소로부터의 경로 계산
        transit_info = ""
        if i > 0:
            prev_node = actual_visits[i-1]
            dist = haversine(prev_node['lat'], prev_node['lng'], node['lat'], node['lng'])
            
            # [핵심] 거리가 100m 미만이면 건물 내 이동으로 간주
            if dist < 0.1: 
                transit_info = "도보 이동 (건물 내 이동)"
            else:
                # r5py 경로가 있으면 그것을 쓰고, 없으면 도보 계산
                r5_path = path_map.get((prev_node['id'], node['id']))
                if r5_path:
                    transit_info = r5_path
                else:
                    walk_min = approx_walk_minutes(prev_node, node)
                    transit_info = f"도보 이동 (약 {round(walk_min)}분)"

        timeline.append({
            "name": node["name"],
            "category": node["category"],
            "time": time_str,
            "transit_to_here": transit_info
        })

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
        start_opt = time.time()
        timeline = optimize_day(
            places=day_places,
            restaurants=day_restaurants,
            fixed_events=day_fixed_events,
            start_time_str=todays_start,
            target_date_str=day_str,  # [수정] 날짜 정보 전달
            end_time_str=todays_end
        )
        end_opt = time.time()
        print(f"⏱ optimize_day 실행 시간: {round(end_opt - start_opt, 2)}초")

        result["plans"][day_key]["timeset"] = timeset
        result["plans"][day_key]["timeline"] = timeline

        if not timeline:
            print("   ⚠ 조건 만족하는 일정 생성 실패")
        else:
            for i, t in enumerate(timeline):
                # 1. 경로 정보를 먼저 출력 (장소 사이의 '연결고리' 역할)
                if t.get('transit_to_here') and "이동 없음" not in t['transit_to_here']:
                    # 경로가 도보인지 대중교통인지에 따라 화살표 느낌 추가
                    prefix = "    ▼ " 
                    print(f"{prefix}{t['transit_to_here']}")

                # 2. 그 다음 장소 정보를 출력
                print(f"  [{t['time']}] {t['name']} ({t['category']})")

        current_date += timedelta(days=1)

    print("\n====== 최종 결과 ======\n")
    file_path = "result_timeline.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 일정이 '{file_path}' 파일로 저장되었습니다.")