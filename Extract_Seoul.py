import pandas as pd
import zipfile
import os
import subprocess

# # 서울 bbox
# MIN_LAT, MAX_LAT = 37.413, 37.715
# MIN_LON, MAX_LON = 126.734, 127.269

# INPUT_GTFS = "./data/GTFS_DataSet.zip"
# TEMP_DIR = "./data/gtfs_tmp/GTFS_DataSet"
# OUTPUT_GTFS = "./data/seoul_gtfs.zip"

# os.makedirs(TEMP_DIR, exist_ok=True)

# # 1. unzip
# with zipfile.ZipFile(INPUT_GTFS) as z:
#     z.extractall(TEMP_DIR)

# # 2. stops 필터
# stops = pd.read_csv(f"{TEMP_DIR}/stops.txt")

# seoul_stops = stops[
#     (stops.stop_lat.between(MIN_LAT, MAX_LAT)) &
#     (stops.stop_lon.between(MIN_LON, MAX_LON))
# ]

# seoul_stop_ids = set(seoul_stops.stop_id)

# seoul_stops.to_csv(f"{TEMP_DIR}/stops.txt", index=False)

# # 3. 연결 테이블 필터 함수
# def filter_by_stop(df, cols):
#     return df[df[cols].isin(seoul_stop_ids).any(axis=1)]

# files = {
#     "stop_times.txt": ["stop_id"],
#     "trips.txt": [],
#     "routes.txt": [],
#     "calendar.txt": [],
#     "calendar_dates.txt": [],
# }

# # stop_times
# st = pd.read_csv(f"{TEMP_DIR}/stop_times.txt")
# st = st[st.stop_id.isin(seoul_stop_ids)]
# st.to_csv(f"{TEMP_DIR}/stop_times.txt", index=False)

# # trips → stop_times 기준으로
# valid_trip_ids = set(st.trip_id)

# trips = pd.read_csv(f"{TEMP_DIR}/trips.txt")
# trips = trips[trips.trip_id.isin(valid_trip_ids)]
# trips.to_csv(f"{TEMP_DIR}/trips.txt", index=False)

# # routes
# routes = pd.read_csv(f"{TEMP_DIR}/routes.txt")
# routes = routes[routes.route_id.isin(trips.route_id)]
# routes.to_csv(f"{TEMP_DIR}/routes.txt", index=False)

# # calendar
# calendar = pd.read_csv(f"{TEMP_DIR}/calendar.txt")
# calendar = calendar[calendar.service_id.isin(trips.service_id)]
# calendar.to_csv(f"{TEMP_DIR}/calendar.txt", index=False)

# # 4. zip 다시 만들기
# with zipfile.ZipFile(OUTPUT_GTFS, "w") as z:
#     for f in os.listdir(TEMP_DIR):
#         z.write(f"{TEMP_DIR}/{f}", f)

# print("✅ seoul_gtfs.zip 생성 완료")