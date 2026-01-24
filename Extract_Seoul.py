import pandas as pd
import zipfile
import io
import os

# 서울 bbox
MIN_LAT, MAX_LAT = 37.4703, 37.5875
MIN_LNG, MAX_LNG = 126.8602, 127.104

INPUT_GTFS = "./data/south_korea_gtfs.zip"
OUTPUT_GTFS = "./data/seoul_gtfs.zip"

SKIP_TRANSFERS = True 

def filter_gtfs():
    print(f"📂 {INPUT_GTFS} 로드 중... (Transfers 보존 모드)")
    
    dfs = {}
    valid_stop_ids = set()
    valid_trip_ids = set()
    valid_route_ids = set()
    
    with zipfile.ZipFile(INPUT_GTFS) as z:
        # ---------------------------------------------------------
        # 1. stops.txt (기준점)
        # ---------------------------------------------------------
        if "stops.txt" in z.namelist():
            with z.open("stops.txt") as f:
                # [핵심] 모든 ID를 문자열로 변환하고 공백 제거
                stops = pd.read_csv(f, dtype=str)
                if 'stop_id' in stops.columns:
                    stops['stop_id'] = stops['stop_id'].str.strip()
                
                # 좌표 필터링 (float 변환 필요)
                stops['stop_lat'] = stops['stop_lat'].astype(float)
                stops['stop_lon'] = stops['stop_lon'].astype(float)
                
                initial_len = len(stops)
                stops = stops[
                    (stops['stop_lat'] >= MIN_LAT) & (stops['stop_lat'] <= MAX_LAT) &
                    (stops['stop_lon'] >= MIN_LNG) & (stops['stop_lon'] <= MAX_LNG)
                ]
                dfs['stops.txt'] = stops
                valid_stop_ids = set(stops['stop_id'])
                print(f"✅ stops.txt: {initial_len} -> {len(stops)}")
        else:
            print("❌ stops.txt가 없습니다.")
            return

        # ---------------------------------------------------------
        # 2. transfers.txt (오류의 주범 -> 정밀 세척)
        # ---------------------------------------------------------
        if "transfers.txt" in z.namelist():
            with z.open("transfers.txt") as f:
                transfers = pd.read_csv(f, dtype=str)
                
                # 컬럼명에 공백이 있을 수 있으므로 공백 제거 (strip)
                transfers.columns = [c.strip() for c in transfers.columns]
                
                if 'from_stop_id' in transfers.columns and 'to_stop_id' in transfers.columns:
                    # 데이터 내 공백 제거
                    transfers['from_stop_id'] = transfers['from_stop_id'].str.strip()
                    transfers['to_stop_id'] = transfers['to_stop_id'].str.strip()
                    
                    initial_len = len(transfers)
                    
                    # [핵심 로직] 두 정류장이 모두 valid_stop_ids에 존재해야 함
                    transfers = transfers[
                        transfers['from_stop_id'].isin(valid_stop_ids) & 
                        transfers['to_stop_id'].isin(valid_stop_ids)
                    ]
                    dfs['transfers.txt'] = transfers
                    print(f"✅ transfers.txt: {initial_len} -> {len(transfers)} (유효한 환승만 남김)")
                else:
                    print("⚠️ transfers.txt에 필수 컬럼(from_stop_id, to_stop_id)이 없어 제외합니다.")

        # ---------------------------------------------------------
        # 3. stop_times.txt
        # ---------------------------------------------------------
        if "stop_times.txt" in z.namelist():
            with z.open("stop_times.txt") as f:
                st = pd.read_csv(f, dtype=str)
                st['stop_id'] = st['stop_id'].str.strip()
                st['trip_id'] = st['trip_id'].str.strip()
                
                initial_len = len(st)
                st = st[st['stop_id'].isin(valid_stop_ids)]
                dfs['stop_times.txt'] = st
                valid_trip_ids = set(st['trip_id'])
                print(f"✅ stop_times.txt: {initial_len} -> {len(st)}")

        # ---------------------------------------------------------
        # 4. trips.txt
        # ---------------------------------------------------------
        if "trips.txt" in z.namelist():
            with z.open("trips.txt") as f:
                trips = pd.read_csv(f, dtype=str)
                trips['trip_id'] = trips['trip_id'].str.strip()
                trips['route_id'] = trips['route_id'].str.strip()
                
                initial_len = len(trips)
                trips = trips[trips['trip_id'].isin(valid_trip_ids)]
                dfs['trips.txt'] = trips
                valid_route_ids = set(trips['route_id'])
                print(f"✅ trips.txt: {initial_len} -> {len(trips)}")

        # ---------------------------------------------------------
        # 5. routes.txt
        # ---------------------------------------------------------
        if "routes.txt" in z.namelist():
            with z.open("routes.txt") as f:
                routes = pd.read_csv(f, dtype=str)
                routes['route_id'] = routes['route_id'].str.strip()
                
                initial_len = len(routes)
                routes = routes[routes['route_id'].isin(valid_route_ids)]
                dfs['routes.txt'] = routes
                print(f"✅ routes.txt: {initial_len} -> {len(routes)}")

        # ---------------------------------------------------------
        # 6. 나머지 파일 복사
        # ---------------------------------------------------------
        for filename in z.namelist():
            if filename not in dfs and filename.endswith(".txt"):
                with z.open(filename) as f:
                    dfs[filename] = pd.read_csv(f)
                    print(f"ℹ️ {filename}: 그대로 복사")

    # 저장
    print(f"💾 {OUTPUT_GTFS} 저장 중...")
    with zipfile.ZipFile(OUTPUT_GTFS, 'w', zipfile.ZIP_DEFLATED) as z_out:
        for name, df in dfs.items():
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            z_out.writestr(name, buffer.getvalue())
            
    print("✨ 완료! 서울 GTFS 생성됨 (환승 정보 포함).")

if __name__ == "__main__":
    filter_gtfs()