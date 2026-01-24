import pandas as pd
import zipfile
import io
import os

# ==========================================
# ⚙️ 설정
# ==========================================
INPUT_GTFS = "./data/south_korea_gtfs.zip"   # KTDB에서 다운받은 파일명
OUTPUT_GTFS = "./data/seoul_gtfs.zip" # 결과 파일명

# 서울시 바운더리 (필요에 따라 조정 가능)
# 넉넉하게 잡은 서울 좌표입니다.
MIN_LAT, MAX_LAT = 37.4703, 37.5875
MIN_LNG, MAX_LNG = 126.8602, 127.104

def filter_gtfs_cascade():
    print("🚀 GTFS 데이터 연쇄 필터링 시작...")
    
    dfs = {}
    
    # GTFS 파일 읽기
    with zipfile.ZipFile(INPUT_GTFS, 'r') as z:
        
        # 1. Stops.txt 필터링 (가장 먼저!)
        # ------------------------------------------------
        if "stops.txt" not in z.namelist():
            print("❌ stops.txt가 없습니다.")
            return

        with z.open("stops.txt") as f:
            stops = pd.read_csv(f, dtype=str)
            # 공백 제거 및 좌표 변환
            stops['stop_id'] = stops['stop_id'].str.strip()
            stops['stop_lat'] = stops['stop_lat'].astype(float)
            stops['stop_lon'] = stops['stop_lon'].astype(float)
            
            initial_stops = len(stops)
            
            # 좌표 기준 필터링
            stops = stops[
                (stops['stop_lat'] >= MIN_LAT) & (stops['stop_lat'] <= MAX_LAT) &
                (stops['stop_lon'] >= MIN_LNG) & (stops['stop_lon'] <= MAX_LNG)
            ]
            
            # 살아남은 정류장 ID 목록 확보
            valid_stop_ids = set(stops['stop_id'])
            dfs['stops.txt'] = stops
            print(f"✅ 1. 정류장 필터링 완료: {initial_stops} -> {len(stops)}개")

        # 2. Stop_times.txt 필터링 (정류장 ID 기준)
        # ------------------------------------------------
        if "stop_times.txt" in z.namelist():
            with z.open("stop_times.txt") as f:
                # 데이터가 크므로 필요한 컬럼 위주로 읽기
                st = pd.read_csv(f, dtype=str)
                st['stop_id'] = st['stop_id'].str.strip()
                st['trip_id'] = st['trip_id'].str.strip()
                
                initial_st = len(st)
                
                # 살아있는 정류장에 포함된 시간표만 남김
                st = st[st['stop_id'].isin(valid_stop_ids)]
                
                # 살아남은 Trip ID 목록 확보
                valid_trip_ids = set(st['trip_id'])
                dfs['stop_times.txt'] = st
                print(f"✅ 2. 시간표 필터링 완료: {initial_st} -> {len(st)}개")

        # 3. Trips.txt 필터링 (Trip ID 기준)
        # ------------------------------------------------
        if "trips.txt" in z.namelist():
            with z.open("trips.txt") as f:
                trips = pd.read_csv(f, dtype=str)
                trips['trip_id'] = trips['trip_id'].str.strip()
                trips['route_id'] = trips['route_id'].str.strip()
                
                initial_trips = len(trips)
                
                # 살아있는 시간표를 가진 Trip만 남김
                trips = trips[trips['trip_id'].isin(valid_trip_ids)]
                
                # 살아남은 Route ID 목록 확보
                valid_route_ids = set(trips['route_id'])
                dfs['trips.txt'] = trips
                print(f"✅ 3. 운행정보 필터링 완료: {initial_trips} -> {len(trips)}개")

        # 4. Routes.txt 필터링 (Route ID 기준)
        # ------------------------------------------------
        if "routes.txt" in z.namelist():
            with z.open("routes.txt") as f:
                routes = pd.read_csv(f, dtype=str)
                routes['route_id'] = routes['route_id'].str.strip()
                
                initial_routes = len(routes)
                
                # 살아있는 Trip을 가진 노선만 남김
                routes = routes[routes['route_id'].isin(valid_route_ids)]
                dfs['routes.txt'] = routes
                print(f"✅ 4. 노선 필터링 완료: {initial_routes} -> {len(routes)}개")

        # 5. Transfers.txt 필터링 (Stop ID 기준 - 양쪽 모두 존재해야 함)
        # ------------------------------------------------
        if "transfers.txt" in z.namelist():
            with z.open("transfers.txt") as f:
                transfers = pd.read_csv(f, dtype=str)
                transfers['from_stop_id'] = transfers['from_stop_id'].str.strip()
                transfers['to_stop_id'] = transfers['to_stop_id'].str.strip()
                
                initial_trans = len(transfers)
                
                # from과 to 모두 서울 안에 있는 정류장이어야 함
                transfers = transfers[
                    transfers['from_stop_id'].isin(valid_stop_ids) & 
                    transfers['to_stop_id'].isin(valid_stop_ids)
                ]
                dfs['transfers.txt'] = transfers
                print(f"✅ 5. 환승정보 필터링 완료: {initial_trans} -> {len(transfers)}개")

        # 6. 나머지 파일들 (Calendar, Agency 등)
        # ------------------------------------------------
        # 엄밀하게 하려면 calendar도 service_id로 필터링해야 하지만,
        # r5py는 참조되지 않는 calendar가 있어도 에러를 내진 않으므로 그대로 둡니다.
        for filename in z.namelist():
            if filename not in dfs and filename.endswith(".txt"):
                with z.open(filename) as f:
                    # Agency 등은 그냥 복사 (Encoding 문제 방지를 위해 pandas 경유)
                    try:
                        temp_df = pd.read_csv(f, dtype=str)
                        dfs[filename] = temp_df
                        print(f"ℹ️  {filename}: 복사됨")
                    except:
                        print(f"⚠️ {filename} 읽기 실패, 건너뜀")

    # 7. 저장
    print(f"💾 {OUTPUT_GTFS} 저장 중...")
    with zipfile.ZipFile(OUTPUT_GTFS, 'w', zipfile.ZIP_DEFLATED) as z_out:
        for name, df in dfs.items():
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            z_out.writestr(name, buffer.getvalue())
            
    print("✨ 모든 작업 완료! 무결성이 확보된 서울 GTFS가 생성되었습니다.")

if __name__ == "__main__":
    filter_gtfs_cascade()