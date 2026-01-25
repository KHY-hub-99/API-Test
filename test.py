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
    ["./data/seoul_subway_gtfs_V1.zip"]
)

# ==========================================
# 3. 출발지/도착지 설정 (GeoDataFrame)
# ==========================================
# 테스트: 강남역(2호선) -> 홍대입구역(2호선)
# stops.txt에 있는 좌표 근처로 설정해야 도보 연결이 잘 됩니다.
origins_df = pd.DataFrame({
    "id": [1],
    "name": ["강남역"],
    "lat": [37.4985],
    "lon": [127.0275]
})

destinations_df = pd.DataFrame({
    "id": [102],
    "name": ["홍대입구역"],
    "lat": [37.5569],
    "lon": [126.9245]
})

# 위경도 좌표계(EPSG:4326) 명시하여 변환
origins = gpd.GeoDataFrame(
    origins_df, 
    geometry=gpd.points_from_xy(origins_df.lon, origins_df.lat), 
    crs="EPSG:4326"
)

destinations = gpd.GeoDataFrame(
    destinations_df, 
    geometry=gpd.points_from_xy(destinations_df.lon, destinations_df.lat), 
    crs="EPSG:4326"
)

# ==========================================
# 4. 여행 시간 계산 (TravelTimeMatrix)
# ==========================================
print("[2] 경로 탐색 계산 시작...")

# calendar.txt가 2025~2026년이므로 해당 기간의 평일로 설정
test_date = datetime(2026, 1, 14, 8, 30) # 2025년 6월 18일 (수요일) 아침 8:30

matrix_computer = r5py.TravelTimeMatrix(
    transport_network,
    origins=origins,
    destinations=destinations,
    departure=test_date,
    transport_modes=[r5py.TransportMode.TRANSIT],
    
    # [수정 포인트] 숫자가 아닌 timedelta 객체 사용 필수!
    max_time_walking=timedelta(minutes=1000),   # 역까지 걷는 시간 최대 30분
)

# ==========================================
# 5. 결과 확인
# ==========================================
print("\n[3] 계산 결과:")
if not matrix_computer.empty:
    print(matrix_computer)
    
    # 보기 좋게 출력 (소요시간 분 단위)
    for row in matrix_computer.itertuples():
        t_min = row.travel_time
        print(f"\n✅ {origins_df.iloc[0]['name']} -> {destinations_df.iloc[0]['name']}")
        print(f"   소요 시간: {t_min}분")
else:
    print("\n❌ 경로를 찾을 수 없습니다.")
    print("   1. OSM 파일 범위가 출발/도착지를 포함하는지 확인하세요.")
    print("   2. GTFS의 stops.txt 좌표와 출발/도착지 거리가 너무 멀지 않은지 확인하세요.")
