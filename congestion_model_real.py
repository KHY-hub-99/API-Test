# Seoul Congestion Prediction Model using Real Data (Enhanced)
# Features: Traffic, Floating Population, Day of Week, Holidays, Weather, Road Capacity

import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import glob
import os
import dotenv
import time

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
        # wttr.in provides free weather data in JSON format
        url = "https://wttr.in/Seoul?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        weather_data = response.json()
        current = weather_data['current_condition'][0]

        temp = int(current['temp_C'])
        humidity = int(current['humidity'])
        weather_desc = current['weatherDesc'][0]['value'].lower()

        # Determine if it's raining
        rain_keywords = ['rain', 'drizzle', 'shower', 'thunderstorm', 'sleet']
        is_raining = any(keyword in weather_desc for keyword in rain_keywords)

        # Determine if it's snowing
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
            'temp': None,
            'humidity': None,
            'description': 'unknown',
            'is_raining': False,
            'is_snowing': False,
            'success': False,
            'error': str(e)
        }

# ============================================
# KOREAN HOLIDAYS 2025 (공휴일)
# ============================================
KOREAN_HOLIDAYS_2025 = {
    # 신정 (New Year's Day)
    '20250101': '신정',
    # 설날 (Lunar New Year) - Jan 28-30
    '20250128': '설날연휴',
    '20250129': '설날',
    '20250130': '설날연휴',
    # 삼일절 (Independence Movement Day)
    '20250301': '삼일절',
    # 대체공휴일 (Substitute holiday)
    '20250303': '대체공휴일',
    # 어린이날 (Children's Day)
    '20250505': '어린이날',
    # 부처님오신날 (Buddha's Birthday) - May 5
    '20250505': '부처님오신날',
    # 대체공휴일
    '20250506': '대체공휴일',
    # 현충일 (Memorial Day)
    '20250606': '현충일',
    # 광복절 (Liberation Day)
    '20250815': '광복절',
    # 추석 (Chuseok) - Oct 5-7
    '20251005': '추석연휴',
    '20251006': '추석',
    '20251007': '추석연휴',
    '20251008': '대체공휴일',
    # 개천절 (National Foundation Day)
    '20251003': '개천절',
    # 한글날 (Hangul Day)
    '20251009': '한글날',
    # 크리스마스 (Christmas)
    '20251225': '크리스마스',
}

# ============================================
# ROAD CAPACITY DATA (도로 용량)
# Based on road type: tunnel, main road, intersection
# ============================================
ROAD_CAPACITY = {
    # Tunnels (터널) - typically higher capacity
    '터널': 2500,
    # Main roads (대로)
    '대로': 2000,
    # Regular roads (로)
    '로': 1500,
    # Intersections/stations (역, 교차로)
    '역': 1800,
    # Default
    'default': 1600
}

# ============================================
# WEATHER PATTERNS FOR SEOUL 2025 (서울 날씨 패턴)
# Monthly averages: temp (°C), rain probability (%), conditions
# ============================================
SEOUL_WEATHER_2025 = {
    1: {'temp': -2, 'rain_prob': 15, 'condition': 'cold'},
    2: {'temp': 1, 'rain_prob': 15, 'condition': 'cold'},
    3: {'temp': 6, 'rain_prob': 25, 'condition': 'mild'},
    4: {'temp': 13, 'rain_prob': 30, 'condition': 'mild'},
    5: {'temp': 18, 'rain_prob': 35, 'condition': 'warm'},
    6: {'temp': 23, 'rain_prob': 50, 'condition': 'rainy'},  # 장마 start
    7: {'temp': 26, 'rain_prob': 60, 'condition': 'rainy'},  # 장마 peak
    8: {'temp': 27, 'rain_prob': 45, 'condition': 'hot'},
    9: {'temp': 22, 'rain_prob': 35, 'condition': 'mild'},
    10: {'temp': 15, 'rain_prob': 25, 'condition': 'mild'},
    11: {'temp': 7, 'rain_prob': 20, 'condition': 'cold'},
    12: {'temp': 0, 'rain_prob': 20, 'condition': 'cold'},
}

# Weather condition impact on congestion (multiplier)
WEATHER_IMPACT = {
    'clear': 1.0,
    'cloudy': 1.05,
    'rain': 1.3,      # Rain increases congestion by 30%
    'heavy_rain': 1.5,
    'snow': 1.4,
    'cold': 1.1,      # Extreme cold
    'hot': 1.1,       # Extreme heat
}

# ============================================
# STEP 1: Load Traffic Volume Data from Excel
# ============================================
def load_traffic_data():
    """Load all monthly traffic volume Excel files"""
    print("=== Loading Traffic Volume Data (교통량 데이터 로딩) ===")

    excel_files = glob.glob(os.path.join(TRAFFIC_DATA_PATH, "*.xlsx"))
    all_traffic_data = []

    for file in sorted(excel_files):
        print(f"  Loading: {os.path.basename(file)}")
        xlsx = pd.ExcelFile(file)

        # Get the data sheet (second sheet, named like "2025년 01월")
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
    """
    S-DoT API에서 1,000건씩 끊어서 데이터를 수집하고 합칩니다.
    """
    print(f"\n=== Fetching Floating Population Data (유동인구 데이터 수집) ===")
    print(f"  Target Range: {start} to {end}")
    
    all_data = []
    current_start = start
    batch_size = 1000  # API 제한

    while current_start <= end:
        current_end = min(current_start + batch_size - 1, end)
        print(f"  Requesting batch: {current_start} ~ {current_end}...")
        
        url = f"{SDOT_API_URL}/{current_start}/{current_end}/"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            root = ET.fromstring(response.content)
            
            # 에러 체크
            result_code = root.find('.//CODE')
            if result_code is not None and result_code.text != 'INFO-000':
                print(f"  API Warning at batch {current_start}: {result_code.text}")
                # 데이터가 없으면 루프 중단 (예: 요청 범위가 실제 데이터보다 클 때)
                break

            rows = root.findall('.//row')
            if not rows:
                break

            for row in rows:
                all_data.append({
                    'sensing_time': row.findtext('SENSING_TIME'),
                    'region': row.findtext('REGION'),
                    'district': row.findtext('AUTONOMOUS_DISTRICT'),
                    'dong': row.findtext('ADMINISTRATIVE_DISTRICT'),
                    'visitor_count': int(row.findtext('VISITOR_COUNT') or 0),
                })
            
            # 다음 배치를 위해 인덱스 증가
            current_start += batch_size
            time.sleep(0.1)  # 서버 부하 방지용 짧은 대기

        except Exception as e:
            print(f"  API request failed at {current_start}: {e}")
            break

    # 데이터가 하나라도 모였으면 DataFrame 생성
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
    """Convert date string to day of week (0=Monday, 6=Sunday)"""
    try:
        date = datetime.strptime(str(date_str), '%Y%m%d')
        return date.weekday()
    except:
        return 0

def get_day_name_korean(day_code):
    """Get Korean day name from code"""
    day_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
    return day_map.get(day_code, 0)

def is_weekend(day_of_week):
    """Check if day is weekend (Saturday=5, Sunday=6)"""
    return 1 if day_of_week >= 5 else 0

def is_holiday(date_str):
    """Check if date is a Korean holiday"""
    return 1 if str(date_str) in KOREAN_HOLIDAYS_2025 else 0

def get_holiday_name(date_str):
    """Get holiday name if applicable"""
    return KOREAN_HOLIDAYS_2025.get(str(date_str), None)

def get_month(date_str):
    """Extract month from date string"""
    try:
        return int(str(date_str)[4:6])
    except:
        return 1

def get_road_capacity(location_name):
    """Estimate road capacity based on location name"""
    location_name = str(location_name)

    if '터널' in location_name:
        return ROAD_CAPACITY['터널']
    elif '대로' in location_name:
        return ROAD_CAPACITY['대로']
    elif '역' in location_name:
        return ROAD_CAPACITY['역']
    elif '로' in location_name:
        return ROAD_CAPACITY['로']
    else:
        return ROAD_CAPACITY['default']

def get_weather_features(month, hour):
    """Get weather features based on month and time"""
    weather = SEOUL_WEATHER_2025.get(month, SEOUL_WEATHER_2025[1])

    # Simulate daily weather variation
    np.random.seed(month * 100 + hour)

    # Base rain probability adjusted by hour (higher in afternoon)
    rain_prob = weather['rain_prob']
    if 14 <= hour <= 18:
        rain_prob *= 1.2

    # Determine if it's raining
    is_raining = 1 if np.random.random() < (rain_prob / 100) else 0

    # Weather impact multiplier
    if is_raining:
        if rain_prob > 50:
            impact = WEATHER_IMPACT['heavy_rain']
        else:
            impact = WEATHER_IMPACT['rain']
    elif weather['condition'] == 'cold' and weather['temp'] < 0:
        impact = WEATHER_IMPACT['cold']
    elif weather['condition'] == 'hot' and weather['temp'] > 30:
        impact = WEATHER_IMPACT['hot']
    else:
        impact = WEATHER_IMPACT['clear']

    return {
        'temperature': weather['temp'],
        'rain_prob': rain_prob,
        'is_raining': is_raining,
        'weather_impact': impact
    }

# ============================================
# STEP 4: Process and Prepare Training Data
# ============================================
def prepare_training_data(traffic_df, population_df=None):
    """Prepare data for model training with all features"""
    print("\n=== Preparing Training Data (훈련 데이터 준비) ===")

    # Process traffic data - reshape from wide to long format
    hour_cols = [f"{i}시" for i in range(24)]

    # Melt the dataframe to get hourly data
    id_vars = ['일자', '요일', '지점명', '지점번호', '방향', '구분']
    available_hour_cols = [col for col in hour_cols if col in traffic_df.columns]

    traffic_long = traffic_df.melt(
        id_vars=id_vars,
        value_vars=available_hour_cols,
        var_name='시간',
        value_name='traffic_volume'
    )

    # Extract hour as integer
    traffic_long['hour'] = traffic_long['시간'].str.replace('시', '').astype(int)

    # Drop rows with missing traffic volume
    traffic_long = traffic_long.dropna(subset=['traffic_volume'])

    print(f"  Total hourly records: {len(traffic_long):,}")

    # ============================================
    # ADD NEW FEATURES
    # ============================================
    print("\n  Adding features...")

    # 1. Day of week (요일)
    print("    - Day of week (요일)")
    traffic_long['day_of_week'] = traffic_long['요일'].apply(get_day_name_korean)

    # 2. Is weekend (주말)
    print("    - Weekend flag (주말)")
    traffic_long['is_weekend'] = traffic_long['day_of_week'].apply(is_weekend)

    # 3. Is holiday (공휴일)
    print("    - Holiday flag (공휴일)")
    traffic_long['is_holiday'] = traffic_long['일자'].apply(is_holiday)

    # 4. Month (월)
    print("    - Month (월)")
    traffic_long['month'] = traffic_long['일자'].apply(get_month)

    # 5. Road capacity (도로 용량)
    print("    - Road capacity (도로 용량)")
    traffic_long['road_capacity'] = traffic_long['지점명'].apply(get_road_capacity)

    # 6. Traffic to capacity ratio (용량 대비 교통량)
    traffic_long['capacity_ratio'] = traffic_long['traffic_volume'] / traffic_long['road_capacity']

    # 7. Weather features (날씨)
    print("    - Weather features (날씨)")
    weather_features = traffic_long.apply(
        lambda row: get_weather_features(row['month'], row['hour']), axis=1
    )
    traffic_long['temperature'] = weather_features.apply(lambda x: x['temperature'])
    traffic_long['is_raining'] = weather_features.apply(lambda x: x['is_raining'])
    traffic_long['weather_impact'] = weather_features.apply(lambda x: x['weather_impact'])

    # Create location encoding
    locations = traffic_long['지점명'].unique()
    location_to_num = {loc: i for i, loc in enumerate(locations)}
    traffic_long['location'] = traffic_long['지점명'].map(location_to_num)

    print(f"\n  Unique locations: {len(locations)}")

    # Aggregate by location, hour, day_of_week, month
    print("  Aggregating data...")
    aggregated = traffic_long.groupby(['지점명', 'hour', 'day_of_week', 'month']).agg({
        'traffic_volume': 'mean',
        'is_weekend': 'first',
        'is_holiday': 'max',  # If any day in group is holiday
        'road_capacity': 'first',
        'capacity_ratio': 'mean',
        'temperature': 'first',
        'is_raining': 'mean',  # Probability of rain
        'weather_impact': 'mean',
    }).reset_index()

    aggregated.columns = ['location_name', 'hour', 'day_of_week', 'month',
                          'avg_traffic', 'is_weekend', 'is_holiday',
                          'road_capacity', 'capacity_ratio',
                          'temperature', 'rain_probability', 'weather_impact']

    aggregated['location'] = aggregated['location_name'].map(location_to_num)

    # Add floating population
    if population_df is not None and len(population_df) > 0:
        population_df['hour'] = population_df['sensing_time'].str.extract(r'_(\d{2}):').astype(int)
        pop_by_hour = population_df.groupby('hour').agg({'visitor_count': 'mean'}).reset_index()
        pop_by_hour.columns = ['hour', 'avg_visitor_count']
        pop_by_hour['floating_population'] = (pop_by_hour['avg_visitor_count'] * 500).astype(int)
        aggregated = aggregated.merge(pop_by_hour[['hour', 'floating_population']], on='hour', how='left')
        aggregated['floating_population'] = aggregated['floating_population'].fillna(
            aggregated['floating_population'].mean()
        ).astype(int)
    else:
        aggregated['floating_population'] = (aggregated['avg_traffic'] * np.random.uniform(20, 50, len(aggregated))).astype(int)

    # ============================================
    # DEFINE CONGESTION LEVELS (using capacity ratio)
    # ============================================
    print("\n  Defining congestion levels based on capacity ratio...")

    # Using capacity ratio is more meaningful than raw traffic
    def assign_congestion(row):
        ratio = row['capacity_ratio']
        weather = row['weather_impact']

        # Adjust thresholds based on weather
        high_threshold = 0.85 / weather  # Lower threshold in bad weather
        med_threshold = 0.60 / weather

        if ratio > high_threshold:
            return 2  # High
        elif ratio > med_threshold:
            return 1  # Medium
        else:
            return 0  # Low

    aggregated['congestion_level'] = aggregated.apply(assign_congestion, axis=1)

    # Print statistics
    print(f"\n  Dataset statistics:")
    print(f"    Total samples: {len(aggregated):,}")
    print(f"    Weekend samples: {aggregated['is_weekend'].sum():,} ({aggregated['is_weekend'].mean()*100:.1f}%)")
    print(f"    Holiday samples: {aggregated['is_holiday'].sum():,} ({aggregated['is_holiday'].mean()*100:.1f}%)")
    print(f"    Rainy samples: {(aggregated['rain_probability'] > 0.5).sum():,}")

    print(f"\n  Congestion level distribution:")
    for level, name in [(0, 'Low (원활)'), (1, 'Medium (보통)'), (2, 'High (혼잡)')]:
        count = (aggregated['congestion_level'] == level).sum()
        pct = count / len(aggregated) * 100
        print(f"    {name}: {count:,} ({pct:.1f}%)")

    return aggregated, location_to_num, locations

# ============================================
# STEP 5: Train Model
# ============================================
def train_model(data):
    """Train Random Forest model with enhanced features"""
    print("\n=== Training Random Forest Model (랜덤 포레스트 훈련) ===")

    # Enhanced feature set
    feature_cols = [
        'location', 'hour', 'day_of_week', 'month',
        'is_weekend', 'is_holiday',
        'avg_traffic', 'road_capacity', 'capacity_ratio',
        'temperature', 'rain_probability', 'weather_impact',
        'floating_population'
    ]

    X = data[feature_cols]
    y = data['congestion_level']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {len(feature_cols)}")

    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  Model Accuracy: {accuracy * 100:.1f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

    # Feature importance
    print("\n  Feature Importance (특성 중요도):")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance_df.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"    {row['feature']:20s}: {row['importance']:.3f} {bar}")

    return model, feature_cols, X_test, y_test

# ============================================
# STEP 6: Make Predictions
# ============================================
def predict_congestion(model, feature_cols, location_to_num, location_name,
                       hour, day_of_week, month, is_weekend, is_holiday,
                       traffic, road_capacity, temperature, is_raining, population):
    """Predict congestion for given conditions"""
    if location_name not in location_to_num:
        return None, "Location not found"

    capacity_ratio = traffic / road_capacity
    weather_impact = WEATHER_IMPACT['rain'] if is_raining else WEATHER_IMPACT['clear']
    rain_prob = 0.7 if is_raining else 0.1

    data = pd.DataFrame({
        'location': [location_to_num[location_name]],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month],
        'is_weekend': [is_weekend],
        'is_holiday': [is_holiday],
        'avg_traffic': [traffic],
        'road_capacity': [road_capacity],
        'capacity_ratio': [capacity_ratio],
        'temperature': [temperature],
        'rain_probability': [rain_prob],
        'weather_impact': [weather_impact],
        'floating_population': [population]
    })

    pred = model.predict(data)[0]
    levels = {0: "Low (원활)", 1: "Medium (보통)", 2: "High (혼잡)"}
    return pred, levels[pred]

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("=" * 70)
    print("Seoul Congestion Prediction Model (Enhanced)")
    print("서울 혼잡도 예측 모델 (개선판)")
    print("=" * 70)
    print("\nFeatures included (포함된 특성):")
    print("  ✓ Traffic volume (교통량)")
    print("  ✓ Day of week (요일)")
    print("  ✓ Weekend/Weekday (주말/평일)")
    print("  ✓ Korean holidays (공휴일)")
    print("  ✓ Weather conditions (날씨)")
    print("  ✓ Road capacity (도로 용량)")
    print("  ✓ Floating population (유동인구)")
    print("=" * 70)

    # Step 1: Load traffic data
    traffic_df = load_traffic_data()

    if traffic_df is None:
        print("Failed to load traffic data!")
        exit(1)

    # Step 2: Fetch floating population
    data_dir = "./data"
    file_name = "floating_population.xlsx"
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        print(f"📂 기존 파일이 있어 로드합니다: {file_path}")
        population_df = pd.read_excel(file_path)
    else:
        print("🚀 파일이 없어 API를 호출합니다...")
        population_df = fetch_floating_population(start=1, end=5000)

        if population_df is not None:
            # ./data 폴더가 없으면 자동으로 생성 (에러 방지)
            os.makedirs("./data", exist_ok=True) 
            population_df.to_excel(file_path, index=False)
            print("💾 데이터 저장 완료")

    # Step 3: Prepare training data
    data, location_to_num, locations = prepare_training_data(traffic_df, population_df)

    # Step 4: Train model
    model, feature_cols, X_test, y_test = train_model(data)

    # Step 5: Sample predictions
    print("\n" + "=" * 70)
    print("Sample Predictions (예측 결과 샘플)")
    print("=" * 70)

    sample_loc = list(locations)[0]  # 성산로(금화터널)
    road_cap = get_road_capacity(sample_loc)

    test_scenarios = [
        # (location, hour, day_of_week, month, is_weekend, is_holiday, traffic, temp, is_raining, pop, description)
        (sample_loc, 8, 0, 3, 0, 0, 2200, 6, 0, 50000, "Monday morning rush (월요일 아침 러시)"),
        (sample_loc, 8, 0, 7, 0, 0, 2200, 26, 1, 50000, "Monday morning + rain (월요일 아침 + 비)"),
        (sample_loc, 14, 5, 5, 1, 0, 1500, 18, 0, 40000, "Saturday afternoon (토요일 오후)"),
        (sample_loc, 10, 2, 1, 0, 1, 800, -2, 0, 20000, "설날 holiday (설날 공휴일)"),
        (sample_loc, 18, 4, 6, 0, 0, 2500, 23, 1, 60000, "Friday evening + 장마 rain"),
        (sample_loc, 23, 6, 8, 1, 0, 400, 27, 0, 10000, "Sunday late night (일요일 심야)"),
    ]

    print(f"\nLocation: {sample_loc} (Road capacity: {road_cap} vehicles/hour)")
    print("-" * 70)

    for loc, hour, dow, month, weekend, holiday, traffic, temp, rain, pop, desc in test_scenarios:
        pred, level = predict_congestion(
            model, feature_cols, location_to_num, loc,
            hour, dow, month, weekend, holiday,
            traffic, road_cap, temp, rain, pop
        )
        rain_icon = "🌧️" if rain else "☀️"
        holiday_icon = "🎉" if holiday else ""
        weekend_icon = "📅" if weekend else ""
        print(f"  {hour:02d}:00 | {temp:3d}°C {rain_icon} {weekend_icon}{holiday_icon} | Traffic: {traffic:,} | → {level}")
        print(f"         {desc}")
        print()

    # Print Korean holidays
    print("=" * 70)
    print("Korean Holidays 2025 (2025년 공휴일)")
    print("=" * 70)
    for date, name in sorted(KOREAN_HOLIDAYS_2025.items())[:10]:
        print(f"  {date[:4]}-{date[4:6]}-{date[6:]}: {name}")
    print("  ...")

    print("\n" + "=" * 70)
    print("Model ready for predictions! (모델 준비 완료)")
    print("=" * 70)

    # ============================================
    # INTERACTIVE MODE (대화형 모드)
    # Uses historical data + real-time weather automatically!
    # ============================================
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE (대화형 예측 모드)")
    print("=" * 70)
    print("✓ Auto-lookup: Historical traffic data (과거 교통량)")
    print("✓ Auto-lookup: Real-time weather (실시간 날씨)")
    print("✓ Auto-detect: Current date/time (현재 날짜/시간)")
    print("\nEnter 'q' to quit (종료하려면 'q' 입력)\n")

    # Show available locations
    print("Available locations (사용 가능한 지점):")
    location_list = list(locations)
    for i, loc in enumerate(location_list[:10]):
        print(f"  {loc}")
    if len(location_list) > 10:
        print(f"  ... and {len(location_list) - 10} more")
    print("\n  💡 Tip: Type part of the name to search (예: '강남', '터널', 'gangnam')")
    print()

    while True:
        try:
            print("-" * 50)

            # Get current date/time as defaults
            now = datetime.now()
            current_hour = now.hour
            current_dow = now.weekday()
            current_month = now.month
            day_names_en = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            # Get location by name search
            loc_input = input("Location (지점명 검색): ").strip()
            if loc_input.lower() == 'q':
                break
            if not loc_input:
                loc_input = location_list[0]  # Default to first location

            # Search for matching locations
            matches = [loc for loc in location_list if loc_input.lower() in loc.lower()]

            if len(matches) == 0:
                print(f"  ❌ No locations found matching '{loc_input}'")
                print(f"  Try: 터널, 대로, 역, 강남, 종로, etc.")
                continue
            elif len(matches) == 1:
                selected_loc = matches[0]
            else:
                # Multiple matches - let user choose
                print(f"  Found {len(matches)} matches:")
                for i, loc in enumerate(matches[:10]):
                    print(f"    {i}: {loc}")
                if len(matches) > 10:
                    print(f"    ... and {len(matches) - 10} more (try a more specific search)")

                choice = input("  Select number [0]: ").strip()
                if choice.lower() == 'q':
                    break
                choice_idx = int(choice) if choice else 0
                if choice_idx < 0 or choice_idx >= len(matches):
                    print(f"  Invalid! Enter 0-{len(matches)-1}")
                    continue
                selected_loc = matches[choice_idx]

            road_cap = get_road_capacity(selected_loc)
            print(f"  ✓ Selected: {selected_loc} (capacity: {road_cap}/hr)")

            # Ask: use current time or custom?
            time_choice = input(f"Use current time? ({current_hour}:00 {day_names_en[current_dow]}) [Y/n]: ").strip().lower()
            if time_choice == 'q':
                break

            if time_choice == 'n':
                # Custom time input (ordered: Month → Day → Hour)
                month_input = input("Month (월) [1-12]: ").strip()
                if month_input.lower() == 'q':
                    break
                month = int(month_input) if month_input else current_month
                if month < 1 or month > 12:
                    print("  Invalid! Enter 1-12")
                    continue

                # Get day of month
                day_input = input(f"Day of month (일) [1-31]: ").strip()
                if day_input.lower() == 'q':
                    break
                day_of_month = int(day_input) if day_input else now.day
                if day_of_month < 1 or day_of_month > 31:
                    print("  Invalid! Enter 1-31")
                    continue

                # Calculate day of week from the date
                try:
                    selected_date = datetime(2025, month, day_of_month)
                    day_of_week = selected_date.weekday()
                    day_names_kr = ['월', '화', '수', '목', '금', '토', '일']
                    print(f"  📅 2025-{month:02d}-{day_of_month:02d} is a {day_names_en[day_of_week]} ({day_names_kr[day_of_week]}요일)")
                except ValueError:
                    print(f"  Invalid date! {month}월 {day_of_month}일 doesn't exist.")
                    continue

                # Check if this date is a holiday
                date_str = f"2025{month:02d}{day_of_month:02d}"
                is_hol = 1 if date_str in KOREAN_HOLIDAYS_2025 else 0
                if is_hol:
                    holiday_name = KOREAN_HOLIDAYS_2025.get(date_str, 'Holiday')
                    print(f"  🎉 This is a holiday: {holiday_name}")

                hour_input = input("Hour (시간) [0-23]: ").strip()
                if hour_input.lower() == 'q':
                    break
                hour = int(hour_input) if hour_input else current_hour
                if hour < 0 or hour > 23:
                    print("  Invalid! Enter 0-23")
                    continue
            else:
                # Use current time
                hour = current_hour
                day_of_week = current_dow
                month = current_month
                print(f"  Using current: {hour}:00, {day_names_en[day_of_week]}, Month {month}")

                # Check if today is a holiday
                today_str = now.strftime('%Y%m%d')
                is_hol = 1 if today_str in KOREAN_HOLIDAYS_2025 else 0
                if is_hol:
                    holiday_name = KOREAN_HOLIDAYS_2025.get(today_str, 'Holiday')
                    print(f"  🎉 Today is a holiday: {holiday_name}")

            is_wknd = 1 if day_of_week >= 5 else 0

            # ============================================
            # AUTOMATICALLY FETCH REAL-TIME WEATHER
            # ============================================
            print("\n  🌤️ Fetching current weather for Seoul...")
            weather = get_current_weather_seoul()

            if weather['success']:
                temp = weather['temp']
                is_rain = 1 if weather['is_raining'] else 0
                weather_desc = weather['description']
                rain_icon = "🌧️" if is_rain else "☀️"
                print(f"  {rain_icon} Current: {temp}°C, {weather_desc}")
                if weather['is_raining']:
                    print(f"  ⚠️ It's raining! Expect higher congestion.")
            else:
                print(f"  ⚠️ Could not fetch weather (using seasonal default)")
                temp = SEOUL_WEATHER_2025.get(month, {}).get('temp', 15)
                is_rain = 0

            # ============================================
            # AUTOMATICALLY LOOK UP TRAFFIC FROM HISTORICAL DATA
            # ============================================
            # Find matching records in the training data
            matching_data = data[
                (data['location_name'] == selected_loc) &
                (data['hour'] == hour) &
                (data['day_of_week'] == day_of_week) &
                (data['month'] == month)
            ]

            if len(matching_data) > 0:
                # Use historical average
                traffic = int(matching_data['avg_traffic'].mean())
                population = int(matching_data['floating_population'].mean())
                print(f"\n  📊 Found {len(matching_data)} historical records")
                print(f"  📈 Historical avg traffic: {traffic:,} vehicles/hour")
            else:
                # Fallback: use location + hour average
                fallback_data = data[
                    (data['location_name'] == selected_loc) &
                    (data['hour'] == hour)
                ]
                if len(fallback_data) > 0:
                    traffic = int(fallback_data['avg_traffic'].mean())
                    population = int(fallback_data['floating_population'].mean())
                    print(f"\n  📊 Using location+hour average (no exact match)")
                    print(f"  📈 Estimated traffic: {traffic:,} vehicles/hour")
                else:
                    # Last resort: location average
                    loc_data = data[data['location_name'] == selected_loc]
                    traffic = int(loc_data['avg_traffic'].mean()) if len(loc_data) > 0 else 1500
                    population = int(loc_data['floating_population'].mean()) if len(loc_data) > 0 else 30000
                    print(f"\n  📊 Using location average")
                    print(f"  📈 Estimated traffic: {traffic:,} vehicles/hour")

            # Get temperature from monthly data
            temp = SEOUL_WEATHER_2025.get(month, {}).get('temp', 15)

            # Adjust traffic for weather (rain increases congestion)
            if is_rain:
                traffic_adjusted = int(traffic * 0.9)  # Less cars but slower
                print(f"  🌧️ Rain adjustment: traffic reduced to {traffic_adjusted:,} (but slower)")
            else:
                traffic_adjusted = traffic

            # Adjust for holiday (less traffic)
            if is_hol:
                traffic_adjusted = int(traffic_adjusted * 0.6)
                print(f"  🎉 Holiday adjustment: traffic reduced to {traffic_adjusted:,}")

            # Make prediction
            pred, level = predict_congestion(
                model, feature_cols, location_to_num, selected_loc,
                hour, day_of_week, month, is_wknd, is_hol,
                traffic_adjusted, road_cap, temp, is_rain, population
            )

            # Display result
            day_names = ['월', '화', '수', '목', '금', '토', '일']
            rain_icon = "🌧️ Rain" if is_rain else "☀️ Clear"
            holiday_text = " 🎉 Holiday" if is_hol else ""
            weekend_text = " 📅 Weekend" if is_wknd else ""

            print()
            print("=" * 50)
            print(f"📍 Location: {selected_loc}")
            print(f"🕐 Time: {hour:02d}:00 ({day_names[day_of_week]}요일)")
            print(f"📅 Month: {month}월 | {temp}°C | {rain_icon}{holiday_text}{weekend_text}")
            print(f"🚗 Traffic: {traffic_adjusted:,} / {road_cap:,} capacity ({traffic_adjusted/road_cap*100:.0f}%)")
            print(f"👥 Population: {population:,}")
            print("-" * 50)

            # Color-coded result
            if pred == 0:
                print(f"✅ Prediction: {level}")
                print("   Traffic is flowing smoothly!")
            elif pred == 1:
                print(f"⚠️  Prediction: {level}")
                print("   Expect some delays.")
            else:
                print(f"🚨 Prediction: {level}")
                print("   Heavy congestion expected!")
            print("=" * 50)
            print()

        except ValueError:
            print("  Invalid input! Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break

    print("\nThank you for using Seoul Congestion Predictor! (이용해 주셔서 감사합니다!)")
