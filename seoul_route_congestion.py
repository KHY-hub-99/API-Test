import os
import multiprocessing
import joblib  # 모델 로드를 위해 추가
import numpy as np # 데이터 처리를 위해 추가

available_cores = multiprocessing.cpu_count()
JAVA_PARALLELISM = 1
print(f"⚙️  설정된 사용 코어 수: {JAVA_PARALLELISM}개")
# JAVA_HOME 경로는 사용자 환경에 맞게 확인 필요
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
load_dotenv()
API_KEY = os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY)

# 캐시 저장소
DETAILED_PATH_CACHE = {}

# 폴백(좌표 없는 경우) 이동 시간 설정(분)
FALLBACK_MOVE_MIN = 30

# 도보 이동 제한
WALK_ONLY_THRESHOLD_MIN = 12   
WALK_ONLY_THRESHOLD_MAX = 18   

MAX_TRANSFERS = 2
MAX_TRAVEL_TIME_MIN = 90

# 시간 윈도우 설정
LUNCH_WINDOW = ("11:20", "13:20")
DINNER_WINDOW = ("17:40", "19:30")

stay_time_map = {
    "관광지": 90, "카페": 50, "음식점": 70, 
    "박물관": 120, "공원": 60, "시장": 80, "숙박": 0
}

osm_file = "./data/seoul_osm_v.pbf"
gtfs_files = ["./data/seoul_area_gtfs.zip"]

# [추가] 혼잡도 모델 관련 상수 및 가중치
MODEL_PATH = "./model/seoul_congestion_model.pkl"
TRAFFIC_WEIGHTS = {0: 1.0, 1: 1.2, 2: 1.5}  # Low, Medium, High (이동 시간 가중치)
CROWD_WEIGHTS = {0: 1.0, 1: 1.1, 2: 1.3}    # Low, Medium, High (대기 시간 가중치)
LEVEL_MAP = {0: "Low", 1: "Medium", 2: "High"} # 로그 출력을 위한 매핑

# [NEW] 제공해주신 교통 지점 좌표 데이터 (유효하지 않은 0.0 좌표 제외)
TRAFFIC_NODE_COORDS = {
    "성산로(금화터널)": (37.56859, 126.94844),
    "사직로(사직터널)": (37.57231, 126.96325),
    "자하문로(자하문터널)": (37.58883, 126.96855),
    "대사관로(삼청터널)": (37.59636, 126.98421),
    "율곡로(안국역)": (37.57600, 126.98434),
    "창경궁로(서울여자대학교)": (37.58253, 126.99801),
    "대학로(한국방송통신대학교)": (37.57820, 127.00202),
    "종로(동묘앞역)": (37.57345, 127.01676),
    "퇴계로(신당역)": (37.56571, 127.02091),
    "동호로(장충체육관)": (37.55879, 127.00725),
    "장충단로(장충단공원)": (37.55689, 127.00466),
    "퇴계로(회현역)": (37.55743, 126.97649),
    "세종대로(서울역)": (37.55912, 126.97454),
    "새문안로(서울역사박물관)": (37.56965, 126.97144),
    "종로(종로3가역)": (37.57053, 126.99096),
    "서소문로(시청역)": (37.56272, 126.97307),
    "세종대로(시청역2)": (37.56750, 126.97721),
    "을지로(을지로3가역)": (37.56632, 126.98910),
    "칠패로(숭례문)": (37.55959, 126.97252),
    "남산1호터널": (37.54241, 127.00136),
    "남산2호터널": (37.55532, 126.98977),
    "남산3호터널": (37.54478, 126.98835),
    "소월로(회현역)": (37.55704, 126.97636),
    "소파로(숭의여자대학교)": (37.55490, 126.98352),
    "도봉로(도봉산역)": (37.69179, 127.04509),
    "동일로(의정부IC)": (37.68857, 127.05538),
    "아차산로(워커힐)": (37.55010, 127.10841),
    "망우로(망우리공원)": (37.60148, 127.11567),
    "경춘북로(중랑경찰서)": (37.61994, 127.10533),
    "화랑로(조선왕릉)": (37.63088, 127.09911),
    "북부간선도로(신내IC)": (37.61400, 127.10836),
    "서하남로(서하남IC)": (37.51684, 127.14690),
    "천호대로(상일IC)": (37.54746, 127.17522),
    "올림픽대로(강일IC)": (37.56708, 127.14080),
    "경부고속도로(양재IC)": (37.46516, 127.03868),
    "송파대로(복정역)": (37.46816, 127.12651),
    "밤고개로(세곡동사거리)": (37.46242, 127.10788),
    "분당수서로(성남시계)": (37.47103, 127.12321),
    "과천대로(남태령역)": (37.46329, 126.98815),
    "양재대로(양재IC)": (37.46009, 127.03018),
    "반포대로(우면산터널)": (37.48372, 127.01184),
    "시흥대로(석수역)": (37.43701, 126.90281),
    "금오로(광명시계)": (37.48243, 126.84189),
    "오리로(광명시계)": (37.48261, 126.84311),
    "개봉로(개봉교)": (37.48618, 126.85650),
    "광명대교(광명시계)": (37.48504, 126.87330),
    "철산교(광명시계)": (37.47510, 126.87835),
    "금천교(광명시계)": (37.46517, 126.88425),
    "금하로(광명시계)": (37.45135, 126.89169),
    "오정로(부천시계)": (37.54278, 126.80941),
    "화곡로(화곡로입구)": (37.53935, 126.82308),
    "경인고속국도(신월IC)": (37.52485, 126.83186),
    "경인로(유한공고)": (37.48861, 126.82279),
    "신정로(작동터널)": (37.50607, 126.82458),
    "김포대로(개화교)": (37.58531, 126.79557),
    "올림픽대로(개화IC)": (37.58776, 126.81297),
    "통일로(고양시계)": (37.64469, 126.91127),
    "서오릉로(고양시계)": (37.61788, 126.90626),
    "수색로(고양시계)": (37.58747, 126.88641),
    "강변북로(난지한강공원)": (37.57089, 126.87256),
    "강변북로(구리시계)": (37.55827, 127.11420),
    "동부간선도로(상도지하차도)": (37.68325, 127.05253),
    "행주대교": (37.59812, 126.80993),
    "월드컵대교": (37.55647, 126.88551),
    "가양대교": (37.57157, 126.86273),
    "성산대교": (37.54820, 126.88895),
    "양화대교": (37.54279, 126.90349),
    "서강대교": (37.53736, 126.92526),
    "마포대교": (37.53360, 126.93658),
    "원효대교": (37.52424, 126.94046),
    "한강대교": (37.51811, 126.95929),
    "동작대교": (37.50976, 126.98181),
    "반포대교": (37.51462, 126.99667),
    "잠수교": (37.50826, 126.99974),
    "한남대교": (37.52711, 127.01328),
    "동호대교": (37.53814, 127.02001),
    "성수대교": (37.53685, 127.03511),
    "영동대교": (37.53041, 127.05746),
    "청담대교": (37.52840, 127.06544),
    "잠실대교": (37.52409, 127.09204),
    "올림픽대교": (37.53387, 127.10423),
    "천호대교": (37.54267, 127.11288),
    "광진교": (37.54415, 127.11526),
    "진흥로(구기터널)": (37.60869, 126.95531),
    "평창문화로(북악터널)": (37.61155, 126.97931),
    "동호로(금호터널)": (37.55178, 127.01320),
    "서빙고로(한남역)": (37.52720, 127.00470),
    "천호대로(군자교)": (37.56072, 127.06857),
    "뚝섬로(용비교)": (37.54201, 127.02067),
    "동일로(군자교)": (37.55381, 127.07050),
    "화랑로(상월곡역)": (37.60422, 127.04427),
    "동소문로(길음교사거리)": (37.59921, 127.02177),
    "화랑로(화랑대역)": (37.61950, 127.08093),
    "도봉로(쌍문역)": (37.64571, 127.03317),
    "동부간선도로(월계1교)": (37.63148, 127.06358),
    "동일로(노원역)": (37.65247, 127.06093),
    "증산로(디지털미디어시티역)": (37.57968, 126.90512),
    "통일로(산골고개정류장)": (37.59493, 126.94025),
    "성산로(연희IC)": (37.56351, 126.93010),
    "연희로(연희IC)": (37.56626, 126.93054),
    "남부순환로(화곡로입구 교차로)": (37.53997, 126.82539),
    "남부순환로(신월IC)": (37.52277, 126.83643),
    "강서로(화곡터널)": (37.53446, 126.84511),
    "공항대로(발산역)": (37.55902, 126.83178),
    "경인로(오류IC)": (37.49798, 126.85170),
    "경인로(거리공원입구교차로)": (37.50644, 126.88438),
    "시흥대로(시흥IC)": (37.47753, 126.89904),
    "영등포로(오목교)": (37.52318, 126.88373),
    "시흥대로(구로디지털단지역)": (37.48741, 126.90545),
    "국회대로(여의2교)": (37.52665, 126.91333),
    "경인로(서울교)": (37.52019, 126.91490),
    "여의대방로(여의교)": (37.51684, 126.92838),
    "양녕로(상도터널)": (37.51120, 126.95347),
    "동작대로(총신대입구역)": (37.49435, 126.98291),
    "문성로(난곡터널)": (37.47905, 126.92425),
    "남부순환로(낙성대역)": (37.47771, 126.96224),
    "남부순환로(예술의전당)": (37.47622, 127.00482),
    "강남대로(강남역-신분당)": (37.49069, 127.03116),
    "사평대로(고속터미널역)": (37.50323, 127.00596),
    "반포대로(서초역)": (37.49624, 127.00555),
    "언주로(매봉터널)": (37.49201, 127.04797),
    "남부순환로(수서IC)": (37.49610, 127.09103),
    "헌릉로(세곡동사거리)": (37.46516, 127.10576),
    "노들로(여의하류IC)": (37.52965, 126.90915),
    "테헤란로(선릉역)": (37.50548, 127.05213),
    "강남대로(신사역)": (37.51480, 127.02013),
    "백제고분로(종합운동장)": (37.51064, 127.07856),
    "송파대로(송파역)": (37.50003, 127.11218),
    "서부간선도로(지상)": (37.52097, 126.88183),
    "올림픽대로": (37.50600, 126.97375),
    "강변북로": (37.51700, 126.97412),
    "내부순환로": (37.60868, 126.99888),
    "북부간선로": (37.60856, 127.05258),
    "동부간선도로": (37.56869, 127.07602),
    "경부고속도로": (37.49321, 127.02252),
    "분당수서로": (37.49770, 127.08720),
    "강남순환로(관악터널)": (37.44910, 126.92617),
    "서부간선지하도로": (37.46894, 126.88367),
    "신월여의지하도로": (37.52932, 126.86228)
}

KOREAN_HOLIDAYS_2025 = {
    '20250101': '신정', '20250128': '설날연휴', '20250129': '설날', '20250130': '설날연휴',
    '20250301': '삼일절', '20250303': '대체공휴일', '20250505': '어린이날',
    '20250506': '대체공휴일', '20250606': '현충일', '20250815': '광복절',
    '20251003': '개천절', '20251005': '추석연휴', '20251006': '추석',
    '20251007': '추석연휴', '20251008': '대체공휴일', '20251009': '한글날',
    '20251225': '크리스마스'
}

SEOUL_WEATHER_2025 = {
    1: {'temp': -2, 'rain_prob': 15}, 2: {'temp': 1, 'rain_prob': 15},
    3: {'temp': 6, 'rain_prob': 25}, 4: {'temp': 13, 'rain_prob': 30},
    5: {'temp': 18, 'rain_prob': 35}, 6: {'temp': 23, 'rain_prob': 50},
    7: {'temp': 26, 'rain_prob': 60}, 8: {'temp': 27, 'rain_prob': 45},
    9: {'temp': 22, 'rain_prob': 35}, 10: {'temp': 15, 'rain_prob': 25},
    11: {'temp': 7, 'rain_prob': 20}, 12: {'temp': 0, 'rain_prob': 20}
}

ROAD_CAPACITY_DEFAULT = {
    '터널': 2500, '대로': 2000, '로': 1500, '역': 1800, 'default': 1600
}

# [추가] 전역 변수로 모델 로드
try:
    if os.path.exists(MODEL_PATH):
        print(f"📦 혼잡도 모델 로드 중: {MODEL_PATH}")
        loaded_package = joblib.load(MODEL_PATH)
        TRAFFIC_MODEL = loaded_package['traffic_model']
        CROWD_MODEL = loaded_package['crowd_model']
        LOCATION_MAP = loaded_package['location_map']
        print("✅ 모델 로드 완료")
    else:
        print("⚠️ 경고: 혼잡도 모델 파일이 없습니다. 가중치가 적용되지 않습니다.")
        TRAFFIC_MODEL, CROWD_MODEL, LOCATION_MAP = None, None, {}
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    TRAFFIC_MODEL, CROWD_MODEL, LOCATION_MAP = None, None, {}

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
    if not start or not end or start.get("lat") is None or end.get("lat") is None:
        return FALLBACK_MOVE_MIN
    dist_km = haversine(start["lat"], start["lng"], end["lat"], end["lng"])
    return dist_km * 15

def dynamic_walk_threshold(dist_km):
    if dist_km < 0.6: return WALK_ONLY_THRESHOLD_MAX
    elif dist_km < 1.2: return 15
    else: return WALK_ONLY_THRESHOLD_MIN

def travel_minutes(p1, p2):
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
    if not text: raise ValueError("Gemini 응답이 비어있습니다.")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == -1: raise ValueError("JSON 파싱 실패:\n" + text)
    return json.loads(text[start:end])

# [추가] 혼잡도 예측을 위한 헬퍼 함수들
def get_road_capacity_val(name):
    name = str(name)
    if '터널' in name: return 2500
    if '대로' in name: return 2000
    if '역' in name: return 1800
    if '로' in name: return 1500
    return 1600

# [NEW] 가장 가까운 교통 지점 찾기
def find_nearest_traffic_node(target_lat, target_lng, max_dist_km=2.0):
    if target_lat is None or target_lng is None:
        return None
        
    nearest_node = None
    min_dist = float('inf')

    for name, (node_lat, node_lng) in TRAFFIC_NODE_COORDS.items():
        # 모델에 존재하는 지점인지 확인
        if TRAFFIC_MODEL is not None and LOCATION_MAP and name not in LOCATION_MAP:
            continue
            
        dist = haversine(target_lat, target_lng, node_lat, node_lng)
        
        if dist < min_dist:
            min_dist = dist
            nearest_node = name

    if min_dist <= max_dist_km:
        return nearest_node
    else:
        return None

# [MODIFIED] 예측 함수 (좌표 기반 매핑 추가)
def predict_congestion_weights(location_name, current_dt, lat=None, lng=None):
    if TRAFFIC_MODEL is None or not LOCATION_MAP:
        return 1.0, 1.0, "Unknown", "Unknown"

    target_name = None

    # 1. 이름 일치 확인
    if location_name in LOCATION_MAP:
        target_name = location_name
    
    # 2. 좌표 기반 매핑 (이름 불일치 시)
    if target_name is None and lat is not None and lng is not None:
        found_node = find_nearest_traffic_node(lat, lng)
        if found_node:
            target_name = found_node

    if target_name is None:
        return 1.0, 1.0, "Unknown", "Unknown"

    hour = current_dt.hour
    month = current_dt.month
    day_of_week = current_dt.weekday() 
    date_str = current_dt.strftime("%Y%m%d")
    
    is_holiday = 1 if date_str in KOREAN_HOLIDAYS_2025 else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    
    weather = SEOUL_WEATHER_2025.get(month, SEOUL_WEATHER_2025[1])
    rain_prob = weather['rain_prob']
    temp = weather['temp']
    
    rng = np.random.RandomState(month * 100 + hour)
    is_raining = 1 if rng.rand() < (rain_prob / 100) else 0
    weather_impact = 1.3 if is_raining else 1.0
    if is_raining: rain_prob = 80

    road_cap = get_road_capacity_val(target_name)
    loc_code = LOCATION_MAP[target_name]

    input_data = pd.DataFrame([{
        'location_code': loc_code,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'temperature': temp,
        'rain_prob': rain_prob,
        'weather_impact': weather_impact,
        'road_capacity': road_cap
    }])

    try:
        t_level = TRAFFIC_MODEL.predict(input_data)[0]
        c_level = CROWD_MODEL.predict(input_data)[0]
        
        t_weight = TRAFFIC_WEIGHTS.get(t_level, 1.0)
        c_weight = CROWD_WEIGHTS.get(c_level, 1.0)
        
        t_str = LEVEL_MAP.get(t_level, "Unknown")
        c_str = LEVEL_MAP.get(c_level, "Unknown")
        
        if target_name != location_name:
             t_str = f"{t_str}({target_name})" # 매핑된 정보 표시
        
        return t_weight, c_weight, t_str, c_str
    except Exception as e:
        return 1.0, 1.0, "Error", "Error"

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
    s_name = start_node.get("name", str(start_node.get("id")))
    e_name = end_node.get("name", str(end_node.get("id")))
    s_coord = f"{start_node.get('lat')}_{start_node.get('lng')}"
    e_coord = f"{end_node.get('lat')}_{end_node.get('lng')}"
    return (f"{s_name}_{s_coord}", f"{e_name}_{e_coord}", int(departure_time.hour))

def get_all_detailed_paths(trip_legs, departure_time):
    if not trip_legs: return {}
    path_map = {}
    origins_list, dests_list = [], []

    # 1) 요청 대상 수집
    for start_node, end_node in trip_legs:
        if start_node['id'] == end_node['id']: continue
        
        # 캐싱된 키에도 departure_time.hour가 포함되어 있으므로 시간대별 혼잡도가 캐시됨
        cache_key = make_cache_key(start_node, end_node, departure_time)
        if cache_key in DETAILED_PATH_CACHE:
            path_map[(int(start_node['id']), int(end_node['id']))] = DETAILED_PATH_CACHE[cache_key]
            continue

        if start_node.get('lat') is None or end_node.get('lat') is None:
            fallback = {"fastest": [f"이동(좌표없음) : {FALLBACK_MOVE_MIN}분"], "min_transfer": [f"이동(좌표없음) : {FALLBACK_MOVE_MIN}분"]}
            DETAILED_PATH_CACHE[cache_key] = fallback
            path_map[(int(start_node['id']), int(end_node['id']))] = fallback
            continue

        origins_list.append(start_node)
        dests_list.append(end_node)

    # 2) r5py 요청
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

            # [핵심 수정] 파싱 시 가중치 적용 함수
            def parse_route_to_segments_with_congestion(route_df, current_dt):
                segs = []
                total_weighted_min = 0

                for _, leg in route_df.iterrows():
                    raw_mode = str(leg[mode_col]).upper() if mode_col in leg.index else ''
                    
                    ride_time = max(1, duration_to_minutes(get_val(leg, ['travel_time', 'duration'], 0)))
                    wait_time = duration_to_minutes(get_val(leg, ['wait_time', 'wait'], 0))
                    
                    f_id = str(get_val(leg, ['start_stop_id', 'from_stop_id']))
                    f_stop_name = get_stop_name(f_id) or "정류장"
                    
                    # [가중치 및 레벨 예측]
                    t_weight, c_weight, t_stat, c_stat = predict_congestion_weights(f_stop_name, current_dt)
                    
                    final_ride_time = ride_time
                    final_wait_time = wait_time
                    note = ""

                    if 'WALK' in raw_mode:
                        pass
                    else:
                        is_bus = 'BUS' in raw_mode
                        if is_bus:
                            final_ride_time = int(ride_time * t_weight)
                        
                        if wait_time > 0:
                            final_wait_time = int(wait_time * c_weight)
                        
                        # 지연 발생 시 로그에 상세 표시
                        diff_time = (final_ride_time - ride_time) + (final_wait_time - wait_time)
                        if diff_time > 0:
                            note = f"(T:{t_stat}/C:{c_stat} +{diff_time}분)"

                    if final_wait_time > 0:
                        segs.append(f"대기 : {final_wait_time}분")

                    if 'WALK' in raw_mode:
                        segs.append(f"도보 : {final_ride_time}분")
                    else:
                        t_id = str(get_val(leg, ['end_stop_id', 'to_stop_id']))
                        t_stop = get_stop_name(t_id) or "정류장"
                        c_rid = str(get_val(leg, ['route_id']))
                        mode_lbl = "지하철" if any(x in raw_mode for x in ['SUBWAY', 'RAIL', 'METRO']) else "버스"
                        
                        r_str = get_route_name(c_rid) or '대중교통'
                        if mode_lbl == "버스" and STOP_ROUTE_MAP:
                            common = STOP_ROUTE_MAP.get(f_id, set()).intersection(STOP_ROUTE_MAP.get(t_id, set()))
                            common.add(c_rid)
                            b_names = sorted([n for n in [get_route_name(rid) for rid in common] if n])
                            if b_names: r_str = ", ".join(b_names)

                        segs.append(f"[{mode_lbl}][{r_str}] : {f_stop_name} → {t_stop} : {final_ride_time}분 {note}")

                    total_weighted_min += (final_ride_time + final_wait_time)
                    current_dt += timedelta(minutes=final_ride_time + final_wait_time)

                return segs, total_weighted_min

            # 3) 그룹별 옵션 분석
            for (from_id, to_id), group in computer.groupby(['from_id', 'to_id']):
                options_data = []
                for _, opt in group.groupby("option"):
                    # 가중치 적용된 시간 계산
                    raw_time = sum(max(1, duration_to_minutes(get_val(leg, ['travel_time', 'duration'], 0))) for _, leg in opt.iterrows())
                    t_count = sum(1 for _, leg in opt.iterrows() if 'WALK' not in str(leg[mode_col]).upper())
                    options_data.append({"route": opt, "time": raw_time, "transfers": t_count})

                if not options_data: continue

                # 최적 옵션 선정 (기존 로직: 최단시간, 최소환승)
                fastest_opt = min(options_data, key=lambda x: (x['time'], x['transfers']))
                
                # [수정] 선정된 경로에 대해 가중치 적용하여 문자열 생성
                segs_fast, _ = parse_route_to_segments_with_congestion(fastest_opt['route'], departure_time)
                result_entry = {"fastest": segs_fast}

                walk_opts = [o for o in options_data if o['transfers'] == 0]
                best_walk = min(walk_opts, key=lambda x: x['time']) if walk_opts else None

                transit_opts = [o for o in options_data if o['transfers'] > 0]
                transit_opts.sort(key=lambda x: (x['transfers'], x['time']))
                best_transit = transit_opts[0] if transit_opts else None

                winner_opt = best_transit if best_transit else best_walk
                if best_walk and best_transit and best_walk['time'] <= best_transit['time']:
                    winner_opt = best_walk

                if winner_opt:
                    segs_min, _ = parse_route_to_segments_with_congestion(winner_opt['route'], departure_time)
                    result_entry["min_transfer"] = segs_min
                else:
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
        nodes.append({"name": restaurants[0]["name"], "category": "음식점", "lat": restaurants[0].get("lat"), "lng": restaurants[0].get("lng"), "stay": 70, "type": "lunch"})
        nodes.append({"name": restaurants[1]["name"], "category": "음식점", "lat": restaurants[1].get("lat"), "lng": restaurants[1].get("lng"), "stay": 70, "type": "dinner"})

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
    # (기존 로직 유지)
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
    
    # [수정 고려] OR-Tools 매트릭스 생성 시에도 예측 가중치를 적용할 수 있으나, 
    # N*N 호출 비용 문제로 여기서는 평균적인 보수값(1.0~1.1)만 적용하거나 기존 유지를 권장.
    # 일단 기존 로직을 유지하되, 상세 경로(get_all_detailed_paths)에서만 정확한 보정을 수행합니다.
    for i in range(n):
        for j in range(n):
            if i == j: continue
            val = r5_travel_times.get((i, j))
            if val is None: val = travel_minutes(nodes[i], nodes[j])
            
            if (nodes[i]["type"]=="fixed" or nodes[j]["type"]=="fixed"):
                if not (nodes[i]["type"]=="depot" and nodes[j]["type"]=="fixed"):
                    val = max(val, 30)
            
            time_matrix[i][j] = nodes[i]["stay"] + int(val)

    # (OR-Tools Solver 기존과 동일)
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
    # 상세 경로 계산 시 predict_congestion_weights가 내부적으로 호출됨
    path_map = get_all_detailed_paths(trip_legs, r5_departure_dt) 
    
    # [타임라인 생성]
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
                    # 파싱된 문자열에서 분 추출하여 합산 (이미 가중치 반영된 시간임)
                    for segment in chosen_path:
                        mins = re.findall(r'(\d+)분', segment)
                        for m in mins: travel_min += int(m)
                else:
                    # Fallback 시에도 간단히 가중치 적용 가능
                    t_w, _ = predict_congestion_weights(prev['name'], cursor)
                    if prev.get('lat') is None or node.get('lat') is None:
                        base_min = FALLBACK_MOVE_MIN
                        travel_min = int(base_min * t_w)
                        transit_info = [f"이동(좌표없음) : {travel_min}분"]
                    else:
                        dist = haversine(prev['lat'], prev['lng'], node['lat'], node['lng'])
                        base_min = int(dist * 15)
                        travel_min = int(base_min * t_w)
                        transit_info = [f"도보/이동 : {travel_min}분"]
            
            # [핵심 추가] 각 방문 장소(Spot) 도착 시점의 혼잡도 예측
            _, _, t_stat, c_stat = predict_congestion_weights(
                node["name"], 
                cursor, 
                lat=node.get("lat"), 
                lng=node.get("lng")
            )
            congestion_info = f"Traffic:{t_stat}, Crowd:{c_stat}"

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
                "transit_to_here": transit_info,
                "congestion": congestion_info
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
        df = pd.read_excel("./data/place_전체_통합.xlsx")
        df.columns = ["category", "name", "place_id", "area", "lat", "lng"]
    except FileNotFoundError:
        print("❌ 'places_전체_통합.xlsx' 파일이 없습니다.")
        exit()

    area = input("여행할 지역을 입력하세요 (예: 종로구): ")
    
    # 2. 장소 필터링
    area_mask = df["area"].str.contains(area, na=False)

    filtered_spot = df[area_mask & (df["category"] != "음식점") & (df["category"] != "숙박")][["name", "lat", "lng"]]

    filtered_restaurant = df[area_mask & (df["category"] == "음식점")][["name", "lat", "lng"]]

    filtered_accom = df[area_mask & (df["category"] == "숙박")][["name", "lat", "lng"]]

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
    #         {"name": "...", "category": "...", "lat": 0.0, "lng": 0.0}
    #       ],
    #       "restaurants": [
    #         {"name": "...", "category": "음식점", "lat": 0.0, "lng": 0.0}
    #       ],
    #       "accommodations": [
    #         {"name": "...", "category": "숙박", "lat": 0.0, "lng": 0.0}
    #       ]
    #     }
    #   }
    # }
    # """
    
    # system_prompt = f"""
    # 너는 '서울 여행 장소 추천 전문가'이다. 반드시 제공된 데이터만을 사용하여 계획을 세운다.
    # {schema}
    # [절대 규칙]
    # 1. 모든 장소의 이름, 카테고리, 좌표(lat, lng)는 입력된 데이터와 100% 일치해야 한다. 절대 값을 수정하거나 새로운 좌표를 생성하지 마라.
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
    
    if available_cores >= days * 2:
        available_cores = days * 2
    else:
        available_cores = available_cores - 2

    print(f"⚙️ 최대 {available_cores}개 코어로 병렬 처리 중...")

    with ThreadPoolExecutor(max_workers=available_cores) as executor:
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
                congestion_log = t.get('congestion', 'N/A')
                print(f"  [{t['time']}] {t['name']} ({t['category']}) - 📊 {congestion_log}")
            
            print(separator)

        # 날짜 카운터 증가
        curr += timedelta(days=1)

    # 7. 모든 루프가 끝난 후 최종 파일 저장 (루프 외부)
    with open("result_timeline_congestion.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n전체 일정이 'result_timeline_congestion.json' 파일로 저장되었습니다.")