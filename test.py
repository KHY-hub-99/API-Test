import r5py
import pandas as pd
import geopandas as gpd # geopandas 임포트 필수
from shapely.geometry import Point
from datetime import datetime, timedelta

# # 1. TransportNetwork 초기화 (기존과 동일)
# transport_network = r5py.TransportNetwork(
#     "./data/seoul.osm.pbf",
#     ["./data/seoul_bus_gtfs_V.zip"]
# )

# # 2. 출발지/도착지 데이터 생성 (수정된 부분 ★)
# # 일반 DataFrame 생성
# origins_df = pd.DataFrame({
#     "id": [1],
#     "lat": [37.52901],
#     "lon": [126.934607],
#     "name": ["여의도선착장"]
# })

# destinations_df = pd.DataFrame({
#     "id": [2],
#     "lat": [37.55468],
#     "lon": [126.894245],
#     "name": ["망원선착장"]
# })

# # ★ 핵심: GeoDataFrame으로 변환 및 좌표계(CRS) 설정
# origins = gpd.GeoDataFrame(
#     origins_df,
#     geometry=gpd.points_from_xy(origins_df.lon, origins_df.lat),
#     crs="EPSG:4326" # 위경도 좌표계 명시
# )

# destinations = gpd.GeoDataFrame(
#     destinations_df,
#     geometry=gpd.points_from_xy(destinations_df.lon, destinations_df.lat),
#     crs="EPSG:4326" # 위경도 좌표계 명시
# )

# # 3. 여행 시간 매트릭스 계산
# travel_time_matrix = r5py.TravelTimeMatrix(
#     transport_network,
#     origins=origins,
#     destinations=destinations,
#     departure=datetime(2026, 2, 14, 8, 30),
#     transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK]
# )

# r5_travel_times = {}
# for row in travel_time_matrix.itertuples():
#     if not pd.isna(row.travel_time):
#         r5_travel_times[(int(row.from_id), int(row.to_id))] = int(row.travel_time)

# print(r5_travel_times)

# ==========================================
# 1. TransportNetwork 초기화 (기존과 동일)
# ==========================================
transport_network = r5py.TransportNetwork(
    "./data/seoul_osm_v.pbf",
    ["./data/seoul_subway_gtfs_V2.zip"]
)

# ==========================================
# 2. 출발/도착지 설정 (도로 위 좌표)
# ==========================================
# 강남역 10번 출구 앞 도로
origins_df = pd.DataFrame({
    "id": [1], "name": ["강남역"], "lat": [37.4985], "lon": [127.0275]
})
# 홍대입구역 2번 출구 앞 도로
destinations_df = pd.DataFrame({
    "id": [102], "name": ["홍대입구"], "lat": [37.5569], "lon": [126.9245]
})

origins = gpd.GeoDataFrame(
    origins_df, geometry=gpd.points_from_xy(origins_df.lon, origins_df.lat), crs="EPSG:4326"
)
destinations = gpd.GeoDataFrame(
    destinations_df, geometry=gpd.points_from_xy(destinations_df.lon, destinations_df.lat), crs="EPSG:4326"
)

# ==========================================
# 3. 상세 경로 계산 (DetailedItineraries)
# ==========================================
print("[2] 상세 경로 탐색 중...")

# 2026년 평일(수요일) 아침 8시 30분
test_date = datetime(2026, 1, 28, 8, 30)

# [수정됨] 최신 버전 클래스 사용
computer = r5py.DetailedItineraries(
    transport_network,
    origins=origins,
    destinations=destinations,
    departure=test_date,
    transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
    max_time_walking=timedelta(minutes=250), # 걷기 허용 시간 대폭 늘림
)

# 결과 계산 실행
itineraries = computer

# ==========================================
# 4. 결과 분석 출력
# ==========================================
print("\n[3] 상세 이동 경로 분석:\n")

if itineraries.empty:
    print("❌ 경로가 생성되지 않았습니다.")
else:
    # 첫 번째 추천 경로(option 0)만 추출
    path = itineraries[itineraries['option'] == 0].copy()
    
    total_minutes = 0
    step_count = 1
    has_subway = False

    for idx, row in path.iterrows():
        # [수정] 알려주신 컬럼명 'transport_mode' 사용
        mode = row['transport_mode'] 
        
        # travel_time이 Timedelta 객체이므로 분 단위 변환
        duration = row['travel_time']
        minutes = round(duration.total_seconds() / 60, 1)
        
        # 대기 시간 확인
        wait_min = 0
        if 'wait_time' in row and not pd.isna(row['wait_time']):
             wait_min = round(row['wait_time'].total_seconds() / 60, 1)

        # 노선 정보 (route_id)
        route_info = ""
        if 'route_id' in row and not pd.isna(row['route_id']):
            route_info = f"[노선: {row['route_id']}]"
            has_subway = True # 노선 정보가 있다는 건 대중교통을 탔다는 뜻

        # 출력
        print(f"▶ Step {step_count}: {mode}")
        print(f"   - 소요 시간: {minutes}분")
        if wait_min > 0:
            print(f"   - 대기 시간: {wait_min}분")
        if route_info:
            print(f"   - {route_info}")
            print(f"   - 구간: {row.get('start_stop_id', '?')} -> {row.get('end_stop_id', '?')}")
        
        print("-" * 30)
        
        total_minutes += minutes
        step_count += 1

    print(f"\n✅ 총 소요 시간: 약 {total_minutes}분")
    
    if has_subway or 'TRAM' in path['transport_mode'].values:
        print("\n🎉 [성공] 지하철(TRAM) 경로가 포함되었습니다!")
    else:
        print("\n⚠️ [실패] 지하철을 타지 않았습니다. (전 구간 도보)")