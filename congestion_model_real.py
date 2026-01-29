# Seoul Dual Congestion Prediction Model (Traffic & Crowd)
# Features: Traffic, Floating Population, Day of Week, Holidays, Weather, Road Capacity

import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import glob
import os
import dotenv
import time
import joblib

dotenv.load_dotenv()

np.random.seed(42)

# ============================================
# CONFIGURATION
# ============================================
SDOT_API_KEY = os.getenv("SDOT_API_KEY")
SDOT_API_URL = f"http://openapi.seoul.go.kr:8088/{SDOT_API_KEY}/xml/IotVdata018"
TRAFFIC_DATA_PATH = "./data"

# ============================================
# REAL-TIME WEATHER FUNCTION
# ============================================
def get_current_weather_seoul():
    """Fetch current weather for Seoul from wttr.in (free, no API key needed)"""
    try:
        url = "https://wttr.in/Seoul?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        weather_data = response.json()
        current = weather_data['current_condition'][0]

        temp = int(current['temp_C'])
        humidity = int(current['humidity'])
        weather_desc = current['weatherDesc'][0]['value'].lower()

        rain_keywords = ['rain', 'drizzle', 'shower', 'thunderstorm', 'sleet']
        is_raining = any(keyword in weather_desc for keyword in rain_keywords)

        snow_keywords = ['snow', 'blizzard', 'ice']
        is_snowing = any(keyword in weather_desc for keyword in snow_keywords)

        return {
            'temp': temp,
            'humidity': humidity,
            'description': weather_desc,
            'is_raining': is_raining,
            'is_snowing': is_snowing,
            'success': True
        }
    except Exception as e:
        return {
            'temp': None, 'humidity': None, 'description': 'unknown',
            'is_raining': False, 'is_snowing': False, 'success': False, 'error': str(e)
        }

# ============================================
# KOREAN HOLIDAYS 2025 (공휴일)
# ============================================
KOREAN_HOLIDAYS_2025 = {
    '20250101': '신정', '20250128': '설날연휴', '20250129': '설날', '20250130': '설날연휴',
    '20250301': '삼일절', '20250303': '대체공휴일', '20250505': '어린이날',
    '20250506': '대체공휴일', '20250606': '현충일', '20250815': '광복절',
    '20251003': '개천절', '20251005': '추석연휴', '20251006': '추석',
    '20251007': '추석연휴', '20251008': '대체공휴일', '20251009': '한글날',
    '20251225': '크리스마스'
}

# ============================================
# ROAD CAPACITY DATA (도로 용량)
# ============================================
ROAD_CAPACITY = {
    '터널': 2500, '대로': 2000, '로': 1500, '역': 1800, 'default': 1600
}

# ============================================
# WEATHER PATTERNS FOR SEOUL 2025
# ============================================
SEOUL_WEATHER_2025 = {
    1: {'temp': -2, 'rain_prob': 15}, 2: {'temp': 1, 'rain_prob': 15},
    3: {'temp': 6, 'rain_prob': 25}, 4: {'temp': 13, 'rain_prob': 30},
    5: {'temp': 18, 'rain_prob': 35}, 6: {'temp': 23, 'rain_prob': 50},
    7: {'temp': 26, 'rain_prob': 60}, 8: {'temp': 27, 'rain_prob': 45},
    9: {'temp': 22, 'rain_prob': 35}, 10: {'temp': 15, 'rain_prob': 25},
    11: {'temp': 7, 'rain_prob': 20}, 12: {'temp': 0, 'rain_prob': 20}
}

WEATHER_IMPACT = {
    'clear': 1.0, 'cloudy': 1.05, 'rain': 1.3,
    'heavy_rain': 1.5, 'snow': 1.4, 'cold': 1.1, 'hot': 1.1
}

# ============================================
# STEP 1: Load Traffic Volume Data from Excel
# ============================================
def load_traffic_data():
    print("=== Loading Traffic Volume Data (교통량 데이터 로딩) ===")
    excel_files = glob.glob(os.path.join(TRAFFIC_DATA_PATH, "*.xlsx"))
    all_traffic_data = []

    for file in sorted(excel_files):
        print(f"  Loading: {os.path.basename(file)}")
        xlsx = pd.ExcelFile(file)
        data_sheets = [s for s in xlsx.sheet_names if "2025년" in s]
        if data_sheets:
            df = pd.read_excel(xlsx, sheet_name=data_sheets[0])
            all_traffic_data.append(df)

    if all_traffic_data:
        traffic_df = pd.concat(all_traffic_data, ignore_index=True)
        print(f"\nTotal traffic records loaded: {len(traffic_df):,}")
        return traffic_df
    return None

# ============================================
# STEP 2: Fetch Floating Population from S-DoT API
# ============================================
def fetch_floating_population(start=1, end=5000):
    print(f"\n=== Fetching Floating Population Data (유동인구 데이터 수집) ===")
    print(f"  Target Range: {start} to {end}")
    
    all_data = []
    current_start = start
    batch_size = 1000  

    while current_start <= end:
        current_end = min(current_start + batch_size - 1, end)
        print(f"  Requesting batch: {current_start} ~ {current_end}...")
        
        url = f"{SDOT_API_URL}/{current_start}/{current_end}/"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            result_code = root.find('.//CODE')
            if result_code is not None and result_code.text != 'INFO-000':
                print(f"  API Warning at batch {current_start}: {result_code.text}")
                break

            rows = root.findall('.//row')
            if not rows: break

            for row in rows:
                all_data.append({
                    'sensing_time': row.findtext('SENSING_TIME'),
                    'region': row.findtext('REGION'),
                    'visitor_count': int(row.findtext('VISITOR_COUNT') or 0),
                })
            
            current_start += batch_size
            time.sleep(0.1) 

        except Exception as e:
            print(f"  API request failed at {current_start}: {e}")
            break

    if all_data:
        df = pd.DataFrame(all_data)
        print(f"  ✅ Successfully fetched {len(df)} total records.")
        return df
    else:
        print("  ❌ Failed to fetch any data.")
        return None

# ============================================
# STEP 3: Feature Engineering Functions
# ============================================
def get_day_of_week(date_str):
    try: return datetime.strptime(str(date_str), '%Y%m%d').weekday()
    except: return 0

def get_day_name_korean(day_str):
    """ '월' -> 0, '화' -> 1 로 변환 """
    day_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
    try: return day_map.get(str(day_str).strip(), 0)
    except: return 0

def is_weekend(day_of_week): return 1 if day_of_week >= 5 else 0
def is_holiday(date_str): return 1 if str(date_str) in KOREAN_HOLIDAYS_2025 else 0
def get_month(date_str):
    try: return int(str(date_str)[4:6])
    except: return 1

def get_road_capacity(location_name):
    location_name = str(location_name)
    if '터널' in location_name: return ROAD_CAPACITY['터널']
    elif '대로' in location_name: return ROAD_CAPACITY['대로']
    elif '역' in location_name: return ROAD_CAPACITY['역']
    elif '로' in location_name: return ROAD_CAPACITY['로']
    else: return ROAD_CAPACITY['default']

def get_weather_features(month, hour):
    weather = SEOUL_WEATHER_2025.get(month, SEOUL_WEATHER_2025[1])
    np.random.seed(month * 100 + hour)
    rain_prob = weather['rain_prob']
    if 14 <= hour <= 18: rain_prob *= 1.2
    
    is_raining = 1 if np.random.random() < (rain_prob / 100) else 0
    impact = WEATHER_IMPACT['clear']
    
    if is_raining:
        impact = WEATHER_IMPACT['heavy_rain'] if rain_prob > 50 else WEATHER_IMPACT['rain']
    elif weather['temp'] < 0: impact = WEATHER_IMPACT['cold']
    elif weather['temp'] > 30: impact = WEATHER_IMPACT['hot']

    return {
        'temperature': weather['temp'], 'rain_prob': rain_prob,
        'is_raining': is_raining, 'weather_impact': impact
    }

# ============================================
# STEP 4: Process and Prepare Training Data (고속 최적화)
# ============================================
def prepare_training_data(traffic_df, population_df=None):
    print("\n=== Preparing Training Data (훈련 데이터 준비 - 고속 최적화 버전) ===")

    # 1. 교통 데이터 전처리
    print("  1. Transforming traffic data (Melting)...")
    hour_cols = [f"{i}시" for i in range(24)]
    available_hour_cols = [col for col in hour_cols if col in traffic_df.columns]
    
    traffic_long = traffic_df.melt(
        id_vars=['일자', '요일', '지점명'], 
        value_vars=available_hour_cols,
        var_name='시간', value_name='traffic_volume'
    ).dropna()
    
    traffic_long['hour'] = traffic_long['시간'].str.replace('시', '').astype(int)
    print(f"     -> Rows to process: {len(traffic_long):,}")

    # 2. 기본 특성 추가
    print("  2. Adding basic features...")
    day_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
    traffic_long['day_of_week'] = traffic_long['요일'].map(lambda x: day_map.get(str(x).strip(), 0))
    
    traffic_long['is_weekend'] = traffic_long['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    traffic_long['is_holiday'] = traffic_long['일자'].apply(is_holiday)
    traffic_long['month'] = traffic_long['일자'].astype(str).str[4:6].astype(int)
    
    # 도로 용량 매핑
    print("  3. Mapping road capacity...")
    def fast_capacity(name):
        if '터널' in name: return 2500
        if '대로' in name: return 2000
        if '역' in name: return 1800
        if '로' in name: return 1500
        return 1600
    traffic_long['road_capacity'] = traffic_long['지점명'].apply(fast_capacity)
    
    # 3. 날씨 특성 추가 (Lookup Table 방식)
    print("  4. Calculating weather features (Optimized)...")
    weather_lookup = []
    for m in range(1, 13):
        for h in range(24):
            w = get_weather_features(m, h)
            w['month'] = m
            w['hour'] = h
            weather_lookup.append(w)
    
    weather_df = pd.DataFrame(weather_lookup)
    traffic_long = traffic_long.merge(weather_df, on=['month', 'hour'], how='left')

    # 4. 데이터 집계
    print("  5. Aggregating data...")
    group_cols = ['지점명', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday', 'road_capacity']
    mean_cols = ['traffic_volume', 'temperature', 'rain_prob', 'weather_impact']
    aggregated = traffic_long.groupby(group_cols)[mean_cols].mean().reset_index()

    # 5. 유동인구 병합
    print("  6. Merging floating population...")
    if population_df is not None and not population_df.empty:
        population_df['sensing_str'] = population_df['sensing_time'].astype(str)
        population_df['hour'] = population_df['sensing_str'].str.extract(r'_(\d{2}):').astype(float).fillna(0).astype(int)
        
        pop_agg = population_df.groupby('hour')['visitor_count'].mean().reset_index()
        pop_agg['floating_population'] = (pop_agg['visitor_count'] * 100).astype(int)
        aggregated = aggregated.merge(pop_agg, on='hour', how='left')
        aggregated['floating_population'] = aggregated['floating_population'].ffill().fillna(30000)
    else:
        aggregated['floating_population'] = np.random.randint(5000, 100000, size=len(aggregated))

    # 6. Location Encoding
    locations = aggregated['지점명'].unique()
    location_to_num = {loc: i for i, loc in enumerate(locations)}
    aggregated['location_code'] = aggregated['지점명'].map(location_to_num)

    # 7. 정답(Label) 생성
    print("  7. Generating labels...")
    aggregated['capacity_ratio'] = aggregated['traffic_volume'] / aggregated['road_capacity']
    
    thresholds = 0.85 / aggregated['weather_impact']
    conditions = [
        (aggregated['capacity_ratio'] > thresholds),
        (aggregated['capacity_ratio'] > thresholds * 0.6)
    ]
    choices = [2, 1] 
    aggregated['traffic_level'] = np.select(conditions, choices, default=0)

    max_pop_df = aggregated.groupby('지점명')['floating_population'].max().reset_index()
    max_pop_df.columns = ['지점명', 'max_pop']
    aggregated = aggregated.merge(max_pop_df, on='지점명', how='left')
    aggregated['pop_ratio'] = aggregated['floating_population'] / aggregated['max_pop']
    
    pop_conditions = [
        (aggregated['pop_ratio'] > 0.8),
        (aggregated['pop_ratio'] > 0.5)
    ]
    aggregated['crowd_level'] = np.select(pop_conditions, [2, 1], default=0)

    print(f"  ✅ Dataset Ready: {len(aggregated)} rows processed.")
    return aggregated, location_to_num, locations

# ============================================
# STEP 5: Train Model (수정됨 - Train/Test 분리 및 상세 평가)
# ============================================
def train_model(data):
    print("\n=== Training Dual Models (Gradient Boosting: 성능 개선 버전) ===")

    feature_cols = [
        'location_code', 'hour', 'day_of_week', 'month', 
        'is_weekend', 'is_holiday', 
        'temperature', 'rain_prob', 'weather_impact', 'road_capacity'
    ]
    
    X = data[feature_cols]
    y_traffic = data['traffic_level'] 
    y_crowd = data['crowd_level']     

    # 데이터 분리
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X, y_traffic, test_size=0.2, random_state=42, stratify=y_traffic
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_crowd, test_size=0.2, random_state=42, stratify=y_crowd
    )

    print(f"  📊 Data Split: Train {len(X_train_t):,} / Test {len(X_test_t):,}")

    # ==========================================
    # 3. 모델 1: 교통 혼잡도 (부스팅 모델 적용)
    # ==========================================
    print("\n  🚗 [1] Training Traffic Model (Gradient Boosting)...")
    
    # [핵심 변경] RandomForest -> HistGradientBoostingClassifier
    # 이 모델은 틀린 문제를 집중적으로 학습합니다.
    model_traffic = HistGradientBoostingClassifier(
        max_iter=500,            # 학습 횟수를 대폭 늘림 (기존 200 -> 500)
        learning_rate=0.05,      # [중요] 횟수가 늘어난 만큼 학습 속도를 늦춤 (기존 0.1 -> 0.05)
        max_depth=12,            # 나무 깊이를 제한하여 과적합 방지
        min_samples_leaf=40,     # 한 노드에 최소 40개의 데이터가 있어야 분기 (노이즈 무시)
        l2_regularization=1.5,   # 규제 강도 증가 (일반화 성능 향상)
        early_stopping=True,     # [필수] 더 이상 좋아지지 않으면 500번 다 안 채우고 멈춤 (시간 절약)
        random_state=42,
        class_weight='balanced'  # 데이터 불균형 해소
    )
    model_traffic.fit(X_train_t, y_train_t)
    
    y_pred_t = model_traffic.predict(X_test_t)
    acc_t = accuracy_score(y_test_t, y_pred_t)
    
    print(f"    👉 Accuracy: {acc_t*100:.1f}%")
    print("    👉 Classification Report:")
    print(classification_report(y_test_t, y_pred_t, target_names=['Low', 'Medium', 'High']))

    # ==========================================
    # 4. 모델 2: 인구 혼잡도
    # ==========================================
    print("\n  👥 [2] Training Crowd Model (Gradient Boosting)...")
    model_crowd = HistGradientBoostingClassifier(
        max_iter=500,            # 학습 횟수를 대폭 늘림 (기존 200 -> 500)
        learning_rate=0.05,      # [중요] 횟수가 늘어난 만큼 학습 속도를 늦춤 (기존 0.1 -> 0.05)
        max_depth=12,            # 나무 깊이를 제한하여 과적합 방지
        min_samples_leaf=40,     # 한 노드에 최소 40개의 데이터가 있어야 분기 (노이즈 무시)
        l2_regularization=1.5,   # 규제 강도 증가 (일반화 성능 향상)
        early_stopping=True,     # [필수] 더 이상 좋아지지 않으면 500번 다 안 채우고 멈춤 (시간 절약)
        random_state=42,
        class_weight='balanced'  # 데이터 불균형 해소
    )
    model_crowd.fit(X_train_c, y_train_c)
    
    y_pred_c = model_crowd.predict(X_test_c)
    acc_c = accuracy_score(y_test_c, y_pred_c)
    
    print(f"    👉 Accuracy: {acc_c*100:.1f}%")
    print("    👉 Classification Report:")
    print(classification_report(y_test_c, y_pred_c, target_names=['Low', 'Medium', 'High']))

    # (참고) HistGradientBoosting은 feature_importances_ 속성이 없습니다.
    # 대신 permutation importance를 써야 하지만, 여기서는 생략하고 컬럼 목록만 반환합니다.
    
    return model_traffic, model_crowd, feature_cols

def save_models(model_t, model_c, loc_map, features, filepath="seoul_congestion_model.pkl"):
    # [NEW] model 폴더가 없으면 자동으로 만듭니다.
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"📂 Created directory: {directory}")

    print(f"\n💾 Saving models to {filepath}...")
    package = {
        'traffic_model': model_t,
        'crowd_model': model_c,
        'location_map': loc_map,
        'features': features,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(package, filepath)
    print("  ✅ Save complete!")

def load_models(filepath="seoul_congestion_model.pkl"):
    """저장된 모델 불러오기"""
    if not os.path.exists(filepath):
        return None
    
    print(f"\n📂 Loading models from {filepath}...")
    try:
        package = joblib.load(filepath)
        print(f"  ✅ Load complete! (Saved at: {package.get('timestamp', 'Unknown')})")
        return package
    except Exception as e:
        print(f"  ❌ Error loading file: {e}")
        return None

# ============================================
# STEP 6: Make Predictions
# ============================================
def predict_congestion(model_traffic, model_crowd, feature_cols, location_to_num, 
                       location_name, hour, day_of_week, month, is_weekend, is_holiday, 
                       road_cap, temp, rain_prob, weather_impact):
    
    if location_name not in location_to_num:
        return None, None

    input_data = pd.DataFrame([{
        'location_code': location_to_num[location_name],
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

    t_pred = model_traffic.predict(input_data)[0]
    c_pred = model_crowd.predict(input_data)[0]

    level_map = {0: "Low (쾌적)", 1: "Medium (보통)", 2: "High (혼잡)"}
    return level_map[t_pred], level_map[c_pred]

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("=" * 70)
    print("Seoul Dual Congestion Prediction Model")
    print("=" * 70)

    # [수정] 모델 파일 경로를 'model' 폴더로 변경
    model_file_path = "./model/seoul_congestion_model.pkl"
    
    # 1. 모델 불러오기 시도
    saved_package = load_models(model_file_path)

    if saved_package:
        # 파일이 있으면 바로 로드
        model_t = saved_package['traffic_model']
        model_c = saved_package['crowd_model']
        location_to_num = saved_package['location_map']
        features = saved_package['features']
        locations = list(location_to_num.keys()) # 장소 리스트 복원
        
    else:
        # 2. 파일이 없으면 학습 시작
        print("🚀 No saved model found. Starting training process...")
        
        # Step 1: 데이터 로드
        traffic_df = load_traffic_data()
        if traffic_df is None: exit(1)

        # Step 2: 유동인구 데이터 확인 (여기는 data 폴더 유지)
        data_dir = "./data"
        pop_file = os.path.join(data_dir, "floating_population.xlsx")
        
        if os.path.exists(pop_file):
            print(f"📂 Loading existing population data...")
            population_df = pd.read_excel(pop_file)
        else:
            population_df = fetch_floating_population(start=1, end=5000)
            if population_df is not None:
                os.makedirs(data_dir, exist_ok=True)
                population_df.to_excel(pop_file, index=False)

        # Step 3: 데이터 전처리
        data, location_to_num, locations = prepare_training_data(traffic_df, population_df)

        # Step 4: 학습
        model_t, model_c, features = train_model(data)
        
        # Step 5: [중요] 학습 후 'model' 폴더에 저장
        save_models(model_t, model_c, location_to_num, features, model_file_path)

    # ============================================
    # INTERACTIVE MODE
    # ============================================
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE (대화형 예측 모드)")
    print("=" * 70)
    print("✓ 교통(Traffic)과 인구(Crowd) 혼잡도를 각각 예측합니다.")
    print("✓ 미래 예측 시 교통량을 입력할 필요가 없습니다. (패턴 기반 예측)")
    print("\nEnter 'q' to quit (종료하려면 'q' 입력)\n")

    location_list = list(locations)
    for i, loc in enumerate(location_list[:10]): print(f"  {loc}")
    if len(location_list) > 10: print(f"  ... and {len(location_list) - 10} more")
    print("\n  💡 Tip: 검색할 지점명을 입력하세요 (예: 강남)")
    print()

    while True:
        try:
            print("-" * 50)
            now = datetime.now()
            day_names_en = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            # 1. 장소 선택
            loc_input = input("Location (지점명 검색): ").strip()
            if loc_input.lower() == 'q': break
            if not loc_input: loc_input = location_list[0]

            matches = [loc for loc in location_list if loc_input.lower() in loc.lower()]
            if not matches:
                print(f"  ❌ 찾을 수 없습니다: '{loc_input}'")
                continue
            elif len(matches) == 1: selected_loc = matches[0]
            else:
                for i, loc in enumerate(matches[:10]): print(f"    {i}: {loc}")
                choice = input("  Select number [0]: ").strip()
                if choice.lower() == 'q': break
                choice_idx = int(choice) if choice else 0
                selected_loc = matches[choice_idx]

            road_cap = get_road_capacity(selected_loc)
            print(f"  ✓ Selected: {selected_loc}")

            # 2. 시간 선택
            time_choice = input(f"Use current time? ({now.hour}:00) [Y/n]: ").strip().lower()
            if time_choice == 'q': break

            if time_choice == 'n':
                month = int(input("Month (1-12): ").strip())
                day = int(input("Day (1-31): ").strip())
                hour = int(input("Hour (0-23): ").strip())
                
                sel_date = datetime(2025, month, day)
                dow = sel_date.weekday()
                date_str = f"2025{month:02d}{day:02d}"
                is_hol = 1 if date_str in KOREAN_HOLIDAYS_2025 else 0
                is_wknd = 1 if dow >= 5 else 0
            else:
                hour = now.hour
                dow = now.weekday()
                month = now.month
                today_str = now.strftime('%Y%m%d')
                is_hol = 1 if today_str in KOREAN_HOLIDAYS_2025 else 0
                is_wknd = 1 if dow >= 5 else 0

            # 3. 날씨 정보 가져오기
            print("\n  🌤️ 날씨 정보 확인 중...")
            weather = get_current_weather_seoul()
            if weather['success']:
                temp = weather['temp']
                is_rain = 1 if weather['is_raining'] else 0
            else:
                temp = SEOUL_WEATHER_2025.get(month, {}).get('temp', 15)
                is_rain = 0
            
            w_impact = 1.3 if is_rain else 1.0
            rain_prob = 80 if is_rain else 10

            # 4. 예측 수행
            t_res, c_res = predict_congestion(
                model_t, model_c, features, location_to_num, selected_loc,
                hour, dow, month, is_wknd, is_hol,
                road_cap, temp, rain_prob, w_impact
            )

            # 5. 결과 출력
            print("\n" + "="*50)
            print(f"📍 {selected_loc} 예측 리포트")
            print(f"📅 {month}월 {day_names_en[dow]} {hour}시 | 🌡️ {temp}°C")
            print("-" * 50)
            
            icon_t = "🟢" if "Low" in t_res else ("🔴" if "High" in t_res else "🟡")
            print(f"🚗 [교통 혼잡도] : {icon_t} {t_res}")
            
            icon_c = "🟢" if "Low" in c_res else ("🔴" if "High" in c_res else "🟡")
            print(f"👥 [인구 혼잡도] : {icon_c} {c_res}")
            
            if "High" in t_res and "High" in c_res:
                print("\n🚨 경고: 최악의 혼잡 시간대입니다! 우회 권장.")
            elif "High" in t_res:
                print("\n💡 팁: 도로는 막히지만 사람은 적습니다. 대중교통 이용 추천!")
            elif "High" in c_res:
                print("\n💡 팁: 도로는 괜찮으나 장소가 붐빕니다.")
                
            print("="*50 + "\n")

        except Exception as e:
            print(f"  Error: {e}")
            break

    print("\nThank you for using Seoul Dual Congestion Predictor!")